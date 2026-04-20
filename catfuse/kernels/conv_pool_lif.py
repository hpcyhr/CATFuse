"""
v15: F8 — Conv → Pool → LIF fused kernel (cuDNN conv + Triton pool+LIF fusion)
===============================================================================

CTF transform:
    TimeBlock(K) ∘ StreamFuse(Pool, LIF) ∘ StateCarry(LIF)

Conv is delegated to cuDNN (batched over T dim). The Triton kernel fuses
the AvgPool2d reduction together with the LIF multi-step dynamics, so the
pool intermediate y (shape [T, B, C, H_out/2, W_out/2]) never materializes
in HBM.

This is a "category A" fusion — unlike F1/F7/F13 where the fusion scope
is just LIF (giving only the ~1.22× Triton constant), F8 fuses Pool as
an additional TSI operator into the LIF kernel. Expected speedup: higher
than the 1.22× constant, closer to v10's 5.10× which also fused pool.

Baselines:
  - sj_torch:  SeqToANNContainer(Conv2d, AvgPool2d) → LIFNode(torch)
  - sj_cupy:   SeqToANNContainer(Conv2d, AvgPool2d) → LIFNode(cupy)  ← SOTA

Shape:
  x:  [T, B, C_in, H, W]           = [16, 32, 128, 16, 16]
  z:  [T, B, C_out, H, W]  (conv, stride=1, padding=1, preserves H,W)
  y:  [T, B, C, H/2, W/2]  (pool 2x2 stride 2, NOT materialized)
  s:  [T, B, C, H/2, W/2]           = [16, 32, 128,  8,  8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import statistics
import json

from spikingjelly.activation_based import neuron, functional as sj_func, surrogate, layer

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Shape
B = 32
C_in = 128
C_out = 128
H = 16
W = 16
KS = 3
PAD = 1
STRIDE = 1
H_conv = (H + 2 * PAD - KS) // STRIDE + 1  # = 16
W_conv = (W + 2 * PAD - KS) // STRIDE + 1  # = 16

POOL_K = 2
POOL_STRIDE = 2
H_out = H_conv // POOL_STRIDE  # 8
W_out = W_conv // POOL_STRIDE  # 8

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
# cuDNN conv (batched over T)
# ============================================================

def cudnn_conv_batched(x_seq, w):
    """
    x_seq: [T, B, C_in, H, W]
    w:     [C_out, C_in, KS, KS]
    returns z_seq: [T, B, C_out, H_conv, W_conv]
    """
    T_ = x_seq.shape[0]
    x_flat = x_seq.view(T_ * B, C_in, H, W)
    z_flat = F.conv2d(x_flat, w, bias=None, stride=STRIDE, padding=PAD)
    return z_flat.view(T_, B, C_out, H_conv, W_conv)


# ============================================================
# Triton fused AvgPool + LIF kernel (reads z_seq from HBM)
# ============================================================

@triton.jit
def pool_lif_fusion_kernel(
    z_ptr,          # [K, B, C, H_conv, W_conv] — conv output already in HBM
    s_ptr,          # [K, B, C, H_out, W_out]   — output spikes
    v_carry_ptr,    # [B, C, H_out, W_out]       — carried membrane potential
    K: tl.constexpr,
    B, C, H_conv, W_conv, H_out, W_out,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_z_t, stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    stride_v_b, stride_v_c, stride_v_h, stride_v_w,
    BLOCK_M: tl.constexpr,   # tile along (B * H_out * W_out)
    BLOCK_N: tl.constexpr,   # tile along C
):
    """
    Each program handles a [BLOCK_M, BLOCK_N] tile of output neurons.
    For each output position (b, c, h_out, w_out), read the 4 input
    pixels at (b, c, 2*h_out+[0,1], 2*w_out+[0,1]), average them in
    registers, feed into LIF. v lives in registers across K time steps.
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

    # Input (z) coordinates for 2x2 pool at stride 2
    h_in_base = h_out_idx * 2
    w_in_base = w_out_idx * 2

    # v layout: [B, C, H_out, W_out]
    v_offs = (b_idx[:, None] * stride_v_b
              + offs_n[None, :] * stride_v_c
              + h_out_idx[:, None] * stride_v_h
              + w_out_idx[:, None] * stride_v_w)
    v_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C)

    v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

    for t in tl.static_range(K):
        # Load the 4 z positions for the pool window
        z_base = (t * stride_z_t
                  + b_idx[:, None] * stride_z_b
                  + offs_n[None, :] * stride_z_c)

        p00 = tl.load(z_ptr + z_base
                      + h_in_base[:, None] * stride_z_h
                      + w_in_base[:, None] * stride_z_w,
                      mask=v_mask, other=0.0)
        p01 = tl.load(z_ptr + z_base
                      + h_in_base[:, None] * stride_z_h
                      + (w_in_base[:, None] + 1) * stride_z_w,
                      mask=v_mask, other=0.0)
        p10 = tl.load(z_ptr + z_base
                      + (h_in_base[:, None] + 1) * stride_z_h
                      + w_in_base[:, None] * stride_z_w,
                      mask=v_mask, other=0.0)
        p11 = tl.load(z_ptr + z_base
                      + (h_in_base[:, None] + 1) * stride_z_h
                      + (w_in_base[:, None] + 1) * stride_z_w,
                      mask=v_mask, other=0.0)

        # StreamFuse: pooled value is register-only
        y_t = (p00 + p01 + p10 + p11) * 0.25

        # LIF update
        v = v + (y_t - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        # Store spike
        s_offs = (t * stride_s_t
                  + b_idx[:, None] * stride_s_b
                  + offs_n[None, :] * stride_s_c
                  + h_out_idx[:, None] * stride_s_h
                  + w_out_idx[:, None] * stride_s_w)
        tl.store(s_ptr + s_offs, spike, mask=v_mask)

    tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


def triton_pool_lif_fusion(z_seq, K):
    """
    z_seq: [T_total, B, C, H_conv, W_conv]
    returns s_seq: [T_total, B, C, H_out, W_out]
    """
    T_total = z_seq.shape[0]
    assert T_total % K == 0
    n_blocks = T_total // K

    s_seq = torch.empty(T_total, B, C_out, H_out, W_out,
                        device=device, dtype=torch.float32)
    v_carry = torch.zeros(B, C_out, H_out, W_out,
                          device=device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 32
    grid = (triton.cdiv(B * H_out * W_out, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    for block_idx in range(n_blocks):
        bs = block_idx * K
        z_block = z_seq[bs:bs + K].contiguous()
        s_block = s_seq[bs:bs + K]

        pool_lif_fusion_kernel[grid](
            z_block, s_block, v_carry,
            K=K,
            B=B, C=C_out,
            H_conv=H_conv, W_conv=W_conv,
            H_out=H_out, W_out=W_out,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_z_t=z_block.stride(0), stride_z_b=z_block.stride(1),
            stride_z_c=z_block.stride(2), stride_z_h=z_block.stride(3),
            stride_z_w=z_block.stride(4),
            stride_s_t=s_block.stride(0), stride_s_b=s_block.stride(1),
            stride_s_c=s_block.stride(2), stride_s_h=s_block.stride(3),
            stride_s_w=s_block.stride(4),
            stride_v_b=v_carry.stride(0), stride_v_c=v_carry.stride(1),
            stride_v_h=v_carry.stride(2), stride_v_w=v_carry.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
    return s_seq


def v15_fusion_forward(x_seq, w, K):
    """
    Full F8 fusion forward:
      1. cuDNN conv (batched over T)
      2. Triton pool+LIF fusion kernel with StateCarry
    """
    z_seq = cudnn_conv_batched(x_seq, w)
    return triton_pool_lif_fusion(z_seq, K)


# ============================================================
# Reference for parity: cudnn conv + torch avg_pool2d + py LIF
# ============================================================

def reference_conv_pool_lif(x_seq, w, T):
    z_seq = cudnn_conv_batched(x_seq, w)
    pool = nn.AvgPool2d(POOL_K, stride=POOL_STRIDE)
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        y_t = pool(z_seq[t])
        v = v + (y_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines
# ============================================================

def make_sj_pipeline(conv, backend):
    """
    SJ-style pipeline:
        SeqToANNContainer(Conv2d, AvgPool2d) → LIFNode(backend)
    """
    conv_sj = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv_sj.weight.copy_(conv.weight)
    pool_sj = nn.AvgPool2d(POOL_K, stride=POOL_STRIDE)

    conv_pool_seq = layer.SeqToANNContainer(conv_sj, pool_sj)
    lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m', backend=backend,
    ).to(device)
    return conv_pool_seq, lif


def sj_baseline_forward(x_seq, conv_pool_seq, lif):
    sj_func.reset_net(lif)
    y_seq = conv_pool_seq(x_seq)
    return lif(y_seq)


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
    print("=" * 80)
    print("v15: F8 Conv→Pool→LIF fused kernel (cuDNN conv + Triton pool+LIF)")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C_in}→{C_out}, "
          f"H=W={H}→{H_conv}→{H_out}, k={KS}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print()

    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(3.5)
    w = conv.weight.contiguous()

    x_seq = torch.randn(T_TOTAL, B, C_in, H, W, device=device)

    # Step 1: parity
    print("Step 1: Parity — v15 fusion vs cuDNN-conv+torch-pool+py-LIF reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<18} {'spike_rate':<12}")
    print('-' * 68)
    with torch.no_grad():
        s_ref = reference_conv_pool_lif(x_seq, w, T_TOTAL)

    all_pass = True
    for K in K_LIST:
        s_tri = v15_fusion_forward(x_seq, w, K)
        exact = torch.equal(s_ref, s_tri)
        max_d = (s_ref - s_tri).abs().max().item()
        n_diff = (s_ref != s_tri).sum().item()
        rate = s_tri.mean().item()
        print(f"{K:<4} {str(exact):<8} {max_d:<12.2e} "
              f"{n_diff:,}/{s_tri.numel():,}   {rate:<12.4f}")
        if not exact:
            all_pass = False
    print()
    if all_pass:
        print("PASS: all K bit-exact ✓")
    else:
        print("⚠ not all K bit-exact — check pool indexing or v layout")
    print()

    # Step 2: v15 vs SJ
    print("Step 2: v15 vs SpikingJelly (numerical equivalence)")
    conv_pool_t, lif_t = make_sj_pipeline(conv, 'torch')
    with torch.no_grad():
        s_sj_t = sj_baseline_forward(x_seq, conv_pool_t, lif_t)
    s_v15 = v15_fusion_forward(x_seq, w, K=16)
    max_d = (s_sj_t - s_v15).abs().max().item()
    n_diff = (s_sj_t != s_v15).sum().item()
    flip_pct = n_diff / s_v15.numel() * 100
    print(f"  v15 vs SJ(torch) max_diff: {max_d:.2e}")
    print(f"  diff_spikes: {n_diff:,}/{s_v15.numel():,} ({flip_pct:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_t.mean().item():.4f}")
    print(f"  v15       spike rate: {s_v15.mean().item():.4f}")
    print()

    # Step 3: wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<24} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 80)
    results = []

    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline_forward(x_seq, conv_pool_t, lif_t)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_t = cuda_time_stats(run_sj_torch)
    results.append(('sj_torch', t_sj_t))

    cupy_available = False
    try:
        conv_pool_c, lif_c = make_sj_pipeline(conv, 'cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline_forward(x_seq, conv_pool_c, lif_c)
        for _ in range(N_WARMUP):
            run_sj_cupy()
        torch.cuda.synchronize()
        t_sj_c = cuda_time_stats(run_sj_cupy)
        results.append(('sj_cupy', t_sj_c))
        cupy_available = True
    except Exception as e:
        print(f"  (cupy backend unavailable: {e})")
        t_sj_c = None

    for K in K_LIST:
        def run_v15(K=K):
            _ = v15_fusion_forward(x_seq, w, K)
        for _ in range(N_WARMUP):
            run_v15()
        torch.cuda.synchronize()
        t_v15 = cuda_time_stats(run_v15)
        results.append((f'v15_K={K}', t_v15))

    t_torch_med = t_sj_t['median']
    t_cupy_med = t_sj_c['median'] if cupy_available else float('nan')
    for name, stats in results:
        med = stats['median']
        std = stats['stdev']
        vs_torch = t_torch_med / med
        vs_cupy = t_cupy_med / med if cupy_available else float('nan')
        cupy_str = f"{vs_cupy:<12.2f}x" if cupy_available else "N/A"
        print(f"{name:<24} {med:<14.3f} {std:<10.3f} "
              f"{vs_torch:<12.2f}x {cupy_str}")
    print()

    out = {
        'kernel': 'F8_conv_pool_lif',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C_in': C_in, 'C_out': C_out,
                  'H': H, 'W': W, 'KS': KS,
                  'H_conv': H_conv, 'W_conv': W_conv,
                  'H_out': H_out, 'W_out': W_out,
                  'pool_k': POOL_K, 'pool_stride': POOL_STRIDE},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v15_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v15_v100_baseline.json")


if __name__ == '__main__':
    main()