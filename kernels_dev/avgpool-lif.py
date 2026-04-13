"""
v10: F4 — AvgPool → LIF fused kernel
=====================================

CTF transform: TimeBlock(K) ∘ StreamFuse(AvgPool, LIF) ∘ StateCarry(LIF)

AvgPool (2x2, stride=2) is a TSI operator — pure elementwise spatial
reduction, no temporal state. LIF is CSR. StreamFuse is applicable.

The fused kernel loads 4 input pixels per output position, averages them
in registers, feeds the average directly into the LIF update, writes only
the spike back to HBM. The pooled intermediate never materializes.

This corresponds to downsample points in Spiking-ResNet where pooling
is applied before a LIF layer.

Shape: B=32, C=128, H=W=16, pool 2x2 stride=2 → H_out=W_out=8
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import statistics
import json

from spikingjelly.activation_based import neuron, functional, surrogate, layer

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Shape
B = 32
C = 128
H = 16
W = 16
POOL_K = 2       # hardcoded 2x2 pool
POOL_STRIDE = 2  # hardcoded stride 2
H_out = H // POOL_STRIDE  # 8
W_out = W // POOL_STRIDE  # 8

# LIF
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

T_TOTAL = 16
K_LIST = [1, 2, 4, 8, 16]

N_WARMUP = 20
N_ITER = 100
N_REPEAT = 11


# ============================================================
# v10 fused AvgPool+LIF kernel
# ============================================================

@triton.jit
def avgpool_lif_block_kernel(
    x_ptr,          # [K, B, C, H, W]
    s_ptr,          # [K, B, C, H_out, W_out]
    v_carry_ptr,    # [B, C, H_out, W_out]
    K: tl.constexpr,
    B, C, H, W, H_out, W_out,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    stride_v_b, stride_v_c, stride_v_h, stride_v_w,
    BLOCK_M: tl.constexpr,  # tile along (B * H_out * W_out)
    BLOCK_N: tl.constexpr,  # tile along C
):
    """
    Each program handles a [BLOCK_M, BLOCK_N] tile of output neurons.
    For each output position, it reads the 4 corresponding input pixels
    (2x2 pool), averages them in registers, feeds into LIF.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    # Input coordinates for 2x2 pool (stride 2)
    h_in_base = h_out_idx * 2
    w_in_base = w_out_idx * 2

    # v_offs: indexing into [B, C, H_out, W_out] layout
    v_offs = (b_idx[:, None] * stride_v_b
              + offs_n[None, :] * stride_v_c
              + h_out_idx[:, None] * stride_v_h
              + w_out_idx[:, None] * stride_v_w)
    v_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C)

    v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

    for t in tl.static_range(K):
        # Load 4 input pixels for the 2x2 window
        # Positions: (h_in_base+0, w_in_base+0), (0,1), (1,0), (1,1)
        x_base = (t * stride_x_t
                  + b_idx[:, None] * stride_x_b
                  + offs_n[None, :] * stride_x_c)

        p00 = tl.load(x_ptr + x_base
                      + h_in_base[:, None] * stride_x_h
                      + w_in_base[:, None] * stride_x_w,
                      mask=v_mask, other=0.0)
        p01 = tl.load(x_ptr + x_base
                      + h_in_base[:, None] * stride_x_h
                      + (w_in_base[:, None] + 1) * stride_x_w,
                      mask=v_mask, other=0.0)
        p10 = tl.load(x_ptr + x_base
                      + (h_in_base[:, None] + 1) * stride_x_h
                      + w_in_base[:, None] * stride_x_w,
                      mask=v_mask, other=0.0)
        p11 = tl.load(x_ptr + x_base
                      + (h_in_base[:, None] + 1) * stride_x_h
                      + (w_in_base[:, None] + 1) * stride_x_w,
                      mask=v_mask, other=0.0)

        # StreamFuse: z_t is a register-only average
        z_t = (p00 + p01 + p10 + p11) * 0.25

        # LIF update
        v = v + (z_t - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        # Write spike
        s_offs = (t * stride_s_t
                  + b_idx[:, None] * stride_s_b
                  + offs_n[None, :] * stride_s_c
                  + h_out_idx[:, None] * stride_s_h
                  + w_out_idx[:, None] * stride_s_w)
        tl.store(s_ptr + s_offs, spike, mask=v_mask)

    # StateCarry
    tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


def triton_avgpool_lif_blocked(x_seq, K):
    """
    x_seq:        [T_total, B, C, H, W]
    returns s_seq:[T_total, B, C, H_out, W_out]
    """
    T_total = x_seq.shape[0]
    assert T_total % K == 0
    n_blocks = T_total // K

    s_seq = torch.empty(T_total, B, C, H_out, W_out,
                        device=device, dtype=torch.float32)
    v_carry = torch.zeros(B, C, H_out, W_out,
                          device=device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 32
    grid = (triton.cdiv(B * H_out * W_out, BLOCK_M), triton.cdiv(C, BLOCK_N))

    for block_idx in range(n_blocks):
        bs = block_idx * K
        x_block = x_seq[bs:bs + K].contiguous()
        s_block = s_seq[bs:bs + K]

        avgpool_lif_block_kernel[grid](
            x_block, s_block, v_carry,
            K=K,
            B=B, C=C, H=H, W=W, H_out=H_out, W_out=W_out,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_x_t=x_block.stride(0), stride_x_b=x_block.stride(1),
            stride_x_c=x_block.stride(2), stride_x_h=x_block.stride(3),
            stride_x_w=x_block.stride(4),
            stride_s_t=s_block.stride(0), stride_s_b=s_block.stride(1),
            stride_s_c=s_block.stride(2), stride_s_h=s_block.stride(3),
            stride_s_w=s_block.stride(4),
            stride_v_b=v_carry.stride(0), stride_v_c=v_carry.stride(1),
            stride_v_h=v_carry.stride(2), stride_v_w=v_carry.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
    return s_seq


# ============================================================
# Reference for bit-exact parity: torch avg_pool2d + py LIF
# ============================================================

def reference_avgpool_lif(x_seq, T):
    v = torch.zeros(B, C, H_out, W_out, device=device)
    spikes = []
    pool = nn.AvgPool2d(POOL_K, stride=POOL_STRIDE)
    for t in range(T):
        z_t = pool(x_seq[t])
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines
# ============================================================

def make_sj_pool_lif(backend):
    """
    SpikingJelly-style pool + LIF:
      SeqToANNContainer(AvgPool2d(2)) feeds into a multi-step LIFNode.
    This is the standard pattern in Spiking-ResNet definitions.
    """
    pool = layer.SeqToANNContainer(nn.AvgPool2d(POOL_K, stride=POOL_STRIDE))
    lif = neuron.LIFNode(
        tau=TAU,
        v_threshold=V_TH,
        v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m',
        backend=backend,
    ).to(device)
    return pool, lif


def sj_baseline(x_seq, pool, lif):
    functional.reset_net(lif)
    y_seq = pool(x_seq)
    s_seq = lif(y_seq)
    return s_seq


# ============================================================
# Timing
# ============================================================

def cuda_time_one_shot(fn, n_iter):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def cuda_time_stats(fn, n_iter=N_ITER, n_repeat=N_REPEAT):
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'max': max(samples),
        'stdev': statistics.stdev(samples),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("v10: F4 AvgPool→LIF fused kernel")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C}, H=W={H} → H_out=W_out={H_out}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    print()

    x_seq = torch.randn(T_TOTAL, B, C, H, W, device=device)

    # Step 1: parity vs local reference
    print("Step 1: Parity — v10 fused vs local torch reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<14} {'spike_rate':<12}")
    print('-' * 60)
    with torch.no_grad():
        s_ref = reference_avgpool_lif(x_seq, T_TOTAL)

    all_pass = True
    for K in K_LIST:
        s_tri = triton_avgpool_lif_blocked(x_seq, K)
        exact = torch.equal(s_ref, s_tri)
        max_d = (s_ref - s_tri).abs().max().item()
        n_diff = (s_ref != s_tri).sum().item()
        rate = s_tri.mean().item()
        print(f"{K:<4} {str(exact):<8} {max_d:<12.2e} "
              f"{n_diff:<14,} {rate:<12.4f}")
        if not exact:
            all_pass = False
    print()
    if not all_pass:
        print("FAIL: parity broken")
        return
    print("PASS: all K bit-exact vs local reference ✓")
    print()

    # Step 2: v10 vs SpikingJelly
    print("Step 2: v10 vs SpikingJelly (numerical equivalence)")
    pool_torch, lif_torch = make_sj_pool_lif('torch')
    with torch.no_grad():
        s_sj_torch = sj_baseline(x_seq, pool_torch, lif_torch)
    s_v10 = triton_avgpool_lif_blocked(x_seq, K=16)
    max_d = (s_sj_torch - s_v10).abs().max().item()
    n_diff_sj = (s_sj_torch != s_v10).sum().item()
    print(f"  v10 vs SJ(torch) max_diff: {max_d:.2e}, "
          f"diff_spikes: {n_diff_sj:,}/{s_v10.numel():,} "
          f"({n_diff_sj/s_v10.numel()*100:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_torch.mean().item():.4f}")
    print(f"  v10       spike rate: {s_v10.mean().item():.4f}")
    if max_d < 1e-4:
        print("  numerical equivalence OK")
    else:
        print("  ⚠ larger than expected")
    print()

    # Step 3: wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<22} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 78)
    results = []

    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline(x_seq, pool_torch, lif_torch)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_torch = cuda_time_stats(run_sj_torch)
    results.append(('sj_naive_torch', t_sj_torch))

    cupy_available = False
    try:
        pool_cupy, lif_cupy = make_sj_pool_lif('cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline(x_seq, pool_cupy, lif_cupy)
        for _ in range(N_WARMUP):
            run_sj_cupy()
        torch.cuda.synchronize()
        t_sj_cupy = cuda_time_stats(run_sj_cupy)
        results.append(('sj_naive_cupy', t_sj_cupy))
        cupy_available = True
    except Exception as e:
        print(f"  (cupy backend unavailable: {e})")
        t_sj_cupy = None

    for K in K_LIST:
        def run_v10(K=K):
            _ = triton_avgpool_lif_blocked(x_seq, K)
        for _ in range(N_WARMUP):
            run_v10()
        torch.cuda.synchronize()
        t_v10 = cuda_time_stats(run_v10)
        results.append((f'v10_K={K}', t_v10))

    t_torch_med = t_sj_torch['median']
    t_cupy_med = t_sj_cupy['median'] if cupy_available else float('nan')
    for name, stats in results:
        med = stats['median']
        std = stats['stdev']
        vs_torch = t_torch_med / med
        vs_cupy = t_cupy_med / med if cupy_available else float('nan')
        cupy_str = f"{vs_cupy:<12.2f}x" if cupy_available else "N/A"
        print(f"{name:<22} {med:<14.3f} {std:<10.3f} "
              f"{vs_torch:<12.2f}x {cupy_str}")
    print()

    out = {
        'kernel': 'F4_avgpool_lif',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C': C, 'H': H, 'W': W,
                  'H_out': H_out, 'W_out': W_out, 'pool_k': POOL_K},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v10_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v10_v100_baseline.json")


if __name__ == '__main__':
    main()