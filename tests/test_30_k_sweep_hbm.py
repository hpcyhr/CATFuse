"""
Test 30: K-sweep + HBM analytic formula validation (§3.9)

[Stage 8 note] This test predates the Stage 3-6 Implementation hierarchy.
Its `hybrid_fn` benchmark uses an INLINE Python loop for the LIF dynamics
rather than the production lif_sequential Triton kernel — so its wall-clock
numbers reflect "what if LIF were inline Python", NOT actual deployment
performance of PartialFusionConvBNLIF / STFusionConvBNLIF.

Kept for historical record. For the deployment-correct K-sweep targeting
§3.10 paper data, use tests/test_30b_k_sweep_real.py — it goes through
SparseFlow.forward_with_k (Stage 6) and DenseKeep.forward (Stage 3) using
the same kernels SEW-RN18 actually runs in production.

For a single Conv+BN+LIF layer:
  1. Compute analytic HBM bytes for reference and CTF at each K
  2. Measure wall-clock at each K
  3. Compare ratio curves

Three execution modes:
  - REF: SpikingJelly per-step Conv→BN→LIF (baseline)
  - CTF-Dense: PartialFusionConvBNLIF (cuDNN + fused BN+LIF)
                BUT: hybrid_fn uses inline Python LIF, NOT lif_sequential
  - CTF-SF: StreamFuse kernel (z stays in registers)
                this one IS realistic — calls sparse_streamfuse_conv3x3_bn_lif
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, layer

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 50
N_ITER = 200
N_REPEAT = 3
ELEM_SIZE = 4  # fp32

def bench(fn):
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1e6)  # microseconds
    return statistics.median(times)


def analytic_hbm(mode, T, K, step_bytes):
    """Compute analytic HBM bytes for a given mode/K.

    |step| = B * C * H * W * elem_size

    REF:     5T * |step|   (z_write, z_read, v_read, v_write, s_write per step)
    HYBRID:  (2 + 2/K) * T * |step| / 5 ... actually:
             Per step: z_write(1) + z_read(1) + s_write(1) = 3|step|
             Per block boundary: v_write(1) + v_read(1) = 2|step|
             Total: T*3|step| + (T/K)*2|step| = (3T + 2T/K)|step|
             Wait, let me re-derive for hybrid (cuDNN writes z to HBM):

    Actually from the paper:
    REF:  5T|step|  (z_write + z_read + v_read + v_write + s_write)

    CTF-Full (StreamFuse): z stays on-chip, v stays on-chip within block
      Per step: s_write = 1|step|
      Per block boundary: v_write + v_read = 2|step|
      Total = T*|step| + (T/K)*2*|step| = (T + 2T/K)|step|
      Ratio = (1 + 2/K) / 5

    CTF-Hybrid (cuDNN partial fusion): z goes through HBM
      Per step: z_write + z_read + s_write = 3|step|  (cuDNN writes z, LIF reads z, writes s)
      Per block boundary: v_write + v_read = 2|step|
      Total = T*3*|step| + (T/K)*2*|step| = (3T + 2T/K)|step|
      Ratio = (3 + 2/K) / 5

    But wait — in our PartialFusionConvBNLIF with BatchFold:
      cuDNN does conv on [T*B, ...] in one call → z[T*B, C, H, W] written once
      lif_seq_kernel reads z, writes s, carries v in registers for T steps
      v loaded once at start, stored once at end (not per block!)

    So with K=T (single block):
      z_write: T*|step| (cuDNN BatchFold writes all T*B outputs)
      z_read:  T*|step| (lif_seq reads all)
      s_write: T*|step| (lif_seq writes all)
      v_read:  1*|step| (once at start)
      v_write: 1*|step| (once at end)
      Total = (3T + 2)|step|

    With K<T (multiple blocks):
      Each block: cuDNN on [K*B,...], lif_seq on K steps
      z_write: T*|step|
      z_read:  T*|step|
      s_write: T*|step|
      v_read:  (T/K)*|step|
      v_write: (T/K)*|step|
      Total = (3T + 2T/K)|step|
      Ratio = (3 + 2/K) / 5
    """
    n_blocks = T // K if K > 0 else T

    if mode == 'ref':
        return 5 * T * step_bytes
    elif mode == 'hybrid':  # PartialFusionConvBNLIF
        return (3 * T + 2 * n_blocks) * step_bytes
    elif mode == 'streamfuse':  # StreamFuse kernel
        return (T + 2 * n_blocks) * step_bytes
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ================================================================
# Test configurations
# ================================================================
configs = [
    # label,        B,  Cin, Cout, H, sparsity
    ("Middle 3x3",   2, 256, 256,  14, 0.5),
    ("Deep 3x3",     2, 512, 512,   7, 0.5),
    ("Shallow 3x3",  2,  64,  64,  32, 0.5),
]

T_values = [4, 8, 16]
K_values_fn = lambda T: [k for k in [1, 2, 4, 8, 16, 32] if k <= T]

print("=" * 100)
print("§3.9 K-sweep + HBM analytic formula validation")
print("=" * 100)

for label, B, Cin, Cout, H, sp in configs:
    print(f"\n{'─' * 100}")
    print(f"Layer: {label}  (B={B}, Cin={Cin}, Cout={Cout}, H={H})")
    print(f"{'─' * 100}")

    step_bytes = B * Cout * H * H * ELEM_SIZE
    step_kb = step_bytes / 1024

    conv = nn.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(device)
    bn = nn.BatchNorm2d(Cout).to(device).eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0)
    lif = lif.to(device)
    functional.set_step_mode(lif, 'm')

    for T in T_values:
        K_values = K_values_fn(T)
        x = (torch.rand(T, B, Cin, H, H, device=device) > sp).float()

        print(f"\n  T={T}, |step|={step_kb:.1f} KB")
        print(f"  {'K':>4s} {'I/O_ref':>12s} {'I/O_hyb':>12s} {'I/O_sf':>12s} "
              f"{'ratio_hyb':>10s} {'ratio_sf':>10s} {'t_ref(us)':>10s} {'t_hyb(us)':>10s} {'t_sf(us)':>10s} "
              f"{'spdup_hyb':>10s} {'spdup_sf':>10s}")
        print(f"  {'─' * 96}")

        for K in K_values:
            # Analytic HBM
            io_ref = analytic_hbm('ref', T, K, step_bytes)
            io_hyb = analytic_hbm('hybrid', T, K, step_bytes)
            io_sf  = analytic_hbm('streamfuse', T, K, step_bytes)
            r_hyb  = io_hyb / io_ref
            r_sf   = io_sf / io_ref

            # Wall-clock: REF (SJ per-step)
            def ref_fn():
                functional.reset_net(lif)
                with torch.no_grad():
                    for t in range(T):
                        z = conv(x[t])
                        z = bn(z)
                        _ = lif(z)
            t_ref = bench(ref_fn)

            # Wall-clock: HYBRID (PartialFusionConvBNLIF with K blocks)
            # Fold BN into conv weights
            with torch.no_grad():
                inv_std = torch.rsqrt(bn.running_var + bn.eps)
                w_fused = conv.weight * (bn.weight * inv_std).view(-1, 1, 1, 1)
                b_fused = bn.bias - bn.running_mean * bn.weight * inv_std

            def hybrid_fn():
                n_blocks = T // K
                v = torch.zeros(B, Cout, H, H, device=device)
                with torch.no_grad():
                    for blk in range(n_blocks):
                        t0_blk = blk * K
                        # BatchFold conv for this block
                        x_block = x[t0_blk:t0_blk+K].reshape(K*B, Cin, H, H)
                        z_block = F.conv2d(x_block, w_fused, b_fused, padding=1)
                        z_block = z_block.reshape(K, B, Cout, H, H)
                        # LIF over K steps
                        for t in range(K):
                            v = v * 0.5 + z_block[t] * 0.5
                            spike = (v >= 1.0).float()
                            v = v * (1.0 - spike)
            t_hyb = bench(hybrid_fn)

            # Wall-clock: StreamFuse (single kernel per block, z in registers)
            # Use our StreamFuse kernel
            try:
                from catfuse.sparseflow.streamfuse_kernel import sparse_streamfuse_conv3x3_bn_lif
                import triton

                w_cl = conv.weight.half().permute(0, 2, 3, 1).contiguous()
                bn_scale = (bn.weight * inv_std).float().contiguous()
                bn_bias_folded = (bn.bias - bn.running_mean * bn.weight * inv_std).float().contiguous()
                bias_arg = torch.empty(1, dtype=torch.float32, device=device)
                GSC = 16 if Cin <= 64 else 32
                NUM_GROUPS = triton.cdiv(Cin, GSC)

                BH, BW = 8, 16
                GH = triton.cdiv(H, BH)
                GW = triton.cdiv(H, BW)

                def sf_fn():
                    n_blocks = T // K
                    v_init = torch.zeros(B, Cout, H, H, device=device)
                    with torch.no_grad():
                        for blk in range(n_blocks):
                            t0_blk = blk * K
                            x_block = x[t0_blk:t0_blk+K]
                            x_flat = x_block.reshape(K*B, Cin, H, H).contiguous()
                            spike_out = torch.empty(K*B, Cout, H, H, dtype=torch.float32, device=device)
                            v_out = torch.empty_like(v_init)
                            N_TILES = B * GH * GW

                            def _grid(META):
                                return (N_TILES, triton.cdiv(Cout, META["BLOCK_N"]))

                            sparse_streamfuse_conv3x3_bn_lif[_grid](
                                x_flat, w_cl, bias_arg, bn_scale, bn_bias_folded,
                                v_init, spike_out, v_out,
                                K, B,
                                C_IN=Cin, C_OUT=Cout,
                                H=H, W=H, H_OUT=H, W_OUT=H,
                                GH=GH, GW=GW,
                                HAS_BIAS=False, HAS_BN=True,
                                DECAY=0.5, RECIP_TAU=0.5,
                                V_TH=1.0, HAS_V_RESET=True, V_RESET=0.0,
                                GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
                            )
                            v_init = v_out
                t_sf = bench(sf_fn)
            except Exception as e:
                t_sf = float('inf')

            spdup_hyb = t_ref / t_hyb if t_hyb > 0 else 0
            spdup_sf  = t_ref / t_sf if t_sf > 0 and t_sf != float('inf') else 0

            print(f"  {K:>4d} {io_ref/1024:>10.1f}KB {io_hyb/1024:>10.1f}KB {io_sf/1024:>10.1f}KB "
                  f"  {r_hyb:>8.3f}   {r_sf:>8.3f} {t_ref:>9.1f}us {t_hyb:>9.1f}us {t_sf:>9.1f}us "
                  f"  {spdup_hyb:>8.2f}x  {spdup_sf:>8.2f}x")

print("\n" + "=" * 100)
print("LEGEND:")
print("  I/O_ref = 5T·|step|  (reference: z write+read, v read+write, s write)")
print("  I/O_hyb = (3T + 2T/K)·|step|  (hybrid: cuDNN writes z to HBM)")
print("  I/O_sf  = (T + 2T/K)·|step|   (StreamFuse: z stays in registers)")
print("  ratio   = I/O_X / I/O_ref")
print("=" * 100)
