"""
Test 31: Find the true crossover point for StreamFuse vs DenseKeep.

Sweep across (C, H) space to find where SF beats Hybrid.
Key insight from K-sweep: H (spatial dim) matters more than C.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 50
N_ITER = 500
T, B = 4, 2

def bench(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e6  # microseconds

from catfuse.sparseflow.streamfuse_kernel import sparse_streamfuse_conv3x3_bn_lif
from catfuse.sparseflow.lif_seq_kernel import lif_sequential

print("=" * 95)
print("Policy Crossover: StreamFuse vs DenseKeep (Hybrid)")
print(f"T={T}, B={B}, K=T={T}")
print("=" * 95)
print(f"{'C':>5s} {'H':>5s} {'tiles':>6s} {'Hybrid(μs)':>12s} {'SF(μs)':>12s} {'ratio':>8s} {'winner':>8s}")
print("-" * 95)

results = []

for C in [64, 128, 256, 512]:
    for H in [4, 7, 8, 14, 16, 28, 32, 56]:
        conv = nn.Conv2d(C, C, 3, padding=1, bias=False).to(device)
        bn = nn.BatchNorm2d(C).to(device).eval()

        x = (torch.rand(T, B, C, H, H, device=device) > 0.5).float()

        # Pre-compute folded BN
        with torch.no_grad():
            inv_std = torch.rsqrt(bn.running_var + bn.eps)
            w_fused = conv.weight * (bn.weight * inv_std).view(-1, 1, 1, 1)
            b_fused = bn.bias - bn.running_mean * bn.weight * inv_std

        # Hybrid: cuDNN BatchFold conv + lif_sequential
        def hybrid_fn():
            with torch.no_grad():
                x_4d = x.reshape(T*B, C, H, H)
                z_4d = F.conv2d(x_4d, w_fused, b_fused, padding=1)
                z = z_4d.reshape(T, B, C, H, H)
                v_init = torch.zeros(B, C, H, H, device=device)
                _, _ = lif_sequential(z, v_init, tau=2.0, v_threshold=1.0, v_reset=0.0)
        t_hyb = bench(hybrid_fn)

        # StreamFuse: one kernel launch
        w_cl = conv.weight.half().permute(0, 2, 3, 1).contiguous()
        bn_scale = (bn.weight * inv_std).float().contiguous()
        bn_bias_f = (bn.bias - bn.running_mean * bn.weight * inv_std).float().contiguous()
        bias_arg = torch.empty(1, dtype=torch.float32, device=device)
        GSC = 16 if C <= 64 else (32 if C <= 256 else 64)
        NUM_GROUPS = triton.cdiv(C, GSC)
        BH, BW = 8, 16
        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(H, BW)
        N_TILES = B * GH * GW

        def sf_fn():
            with torch.no_grad():
                x_flat = x.reshape(T*B, C, H, H).contiguous()
                v_init = torch.zeros(B, C, H, H, device=device)
                spike_out = torch.empty(T*B, C, H, H, dtype=torch.float32, device=device)
                v_out = torch.empty_like(v_init)
                def _grid(META):
                    return (N_TILES, triton.cdiv(C, META["BLOCK_N"]))
                sparse_streamfuse_conv3x3_bn_lif[_grid](
                    x_flat, w_cl, bias_arg, bn_scale, bn_bias_f,
                    v_init, spike_out, v_out,
                    T, B,
                    C_IN=C, C_OUT=C,
                    H=H, W=H, H_OUT=H, W_OUT=H,
                    GH=GH, GW=GW,
                    HAS_BIAS=False, HAS_BN=True,
                    DECAY=0.5, RECIP_TAU=0.5,
                    V_TH=1.0, HAS_V_RESET=True, V_RESET=0.0,
                    GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
                )
        try:
            t_sf = bench(sf_fn)
        except Exception as e:
            t_sf = float('inf')

        ratio = t_hyb / t_sf if t_sf > 0 and t_sf != float('inf') else 0
        winner = "SF" if ratio > 1.0 else "Hybrid"
        tiles = B * GH * GW
        results.append((C, H, tiles, t_hyb, t_sf, ratio, winner))
        print(f"{C:>5d} {H:>5d} {tiles:>6d} {t_hyb:>11.1f} {t_sf:>11.1f} {ratio:>7.2f}x {winner:>8s}")

# Summary: find crossover
print("\n" + "=" * 95)
print("CROSSOVER ANALYSIS")
print("=" * 95)
print(f"\n{'C':>5s} {'H_crossover':>12s} {'min_tiles_for_SF':>18s}")
print("-" * 40)
for C in [64, 128, 256, 512]:
    c_results = [(h, ratio) for (c, h, _, _, _, ratio, _) in results if c == C and ratio > 0]
    sf_wins = [(h, ratio) for h, ratio in c_results if ratio > 1.0]
    if sf_wins:
        min_h = min(h for h, _ in sf_wins)
        print(f"{C:>5d} {min_h:>12d} {'(H≥'+str(min_h)+')':>18s}")
    else:
        print(f"{C:>5d} {'never':>12s} {'—':>18s}")

print("=" * 95)
