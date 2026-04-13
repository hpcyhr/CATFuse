"""
v13: F1 — Conv→LIF fused kernel (fair version: cuDNN conv + Triton LIF fusion)
===============================================================================

CTF transform: TimeBlock(K) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)

Fair-comparison version of F1, replacing the deprecated v7c (which tried to
write conv in Triton and lost 5× to cuDNN on sm_70).

Design (same principle as v12):
  - cuDNN does the conv. Both baseline and fusion use F.conv2d.
  - Triton kernel fuses LIF multi-step with StateCarry, reading z_seq
    that cuDNN has already written to HBM.
  - This is F1 in its simplest CTF realization — there is no BN to fold,
    no residual to add, no pool to reduce; just conv → LIF.

Expected result: modest speedup vs sj_cupy, because:
  - There is no TSI fusion beyond the single Conv (no BN fold, no add, etc.)
  - LIF fusion itself is comparable to sj_cupy's multi-step LIF kernel
  - Only wins: (1) one fewer kernel launch (we skip SJ's separate BN wrapper,
    even though there's no BN the SeqToANNContainer still adds overhead),
    (2) slightly tighter Triton LIF kernel

An outcome of ~1.0-1.1× would actually *confirm* CTF theory: when there is
no TSI fusion opportunity, CTF's HBM savings reduce to zero, and the fused
and non-fused pipelines perform similarly. This is a feature, not a failure.

Baselines:
  - sj_torch:  nn.Conv2d → sj LIFNode(torch)
  - sj_cupy:   nn.Conv2d → sj LIFNode(cupy)   ← SNN SOTA
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

# Shape — same as v7c / v11 / v12 for direct comparison
B = 32
C_in = 128
C_out = 128
H = 16
W = 16
KS = 3
PAD = 1
STRIDE = 1
H_out = (H + 2 * PAD - KS) // STRIDE + 1
W_out = (W + 2 * PAD - KS) // STRIDE + 1

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
# cuDNN conv (same as v12, no bias — there's no BN to fold)
# ============================================================

def cudnn_conv_batched(x_seq, w):
    """
    x_seq: [T, B, C_in, H, W]
    w:     [C_out, C_in, KS, KS]
    Returns z_seq: [T, B, C_out, H_out, W_out]
    """
    T_ = x_seq.shape[0]
    x_flat = x_seq.view(T_ * B, C_in, H, W)
    z_flat = F.conv2d(x_flat, w, bias=None, stride=STRIDE, padding=PAD)
    return z_flat.view(T_, B, C_out, H_out, W_out)


# ============================================================
# Triton LIF fusion kernel — identical to v12's lif_fusion_kernel
# (this is the shared template for all "cuDNN dense op + Triton LIF
# fusion" variants: F1, F7, F8, and eventually F11)
# ============================================================

@triton.jit
def lif_fusion_kernel(
    z_ptr,          # [K, numel]
    s_ptr,          # [K, numel]
    v_carry_ptr,    # [numel]
    K: tl.constexpr,
    numel,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_t,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    v = tl.load(v_carry_ptr + offs, mask=mask, other=0.0)

    for t in tl.static_range(K):
        z_t = tl.load(z_ptr + t * stride_t + offs, mask=mask, other=0.0)
        v = v + (z_t - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike
        tl.store(s_ptr + t * stride_t + offs, spike, mask=mask)

    tl.store(v_carry_ptr + offs, v, mask=mask)


def triton_lif_fusion(z_seq, K):
    T_total = z_seq.shape[0]
    assert T_total % K == 0
    n_blocks = T_total // K

    s_seq = torch.empty_like(z_seq)
    numel_per_step = B * C_out * H_out * W_out
    v_carry = torch.zeros(numel_per_step, device=device, dtype=torch.float32)

    BLOCK = 1024
    grid = (triton.cdiv(numel_per_step, BLOCK),)

    for block_idx in range(n_blocks):
        bs = block_idx * K
        z_block = z_seq[bs:bs + K].contiguous().view(K, -1)
        s_block = s_seq[bs:bs + K].view(K, -1)

        lif_fusion_kernel[grid](
            z_block, s_block, v_carry,
            K=K,
            numel=numel_per_step,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_t=numel_per_step,
            BLOCK=BLOCK,
        )
    return s_seq


def v13_fusion_forward(x_seq, w, K):
    """
    Full F1 fusion forward:
      1. cuDNN conv
      2. Triton LIF fusion kernel with TimeBlock(K) and StateCarry
    """
    z_seq = cudnn_conv_batched(x_seq, w)
    s_seq = triton_lif_fusion(z_seq, K)
    return s_seq


# ============================================================
# Reference for parity: cuDNN conv + py LIF
# (shares same cuDNN call as fusion path → bit-exact expected)
# ============================================================

def reference_conv_lif(x_seq, w, T):
    z_seq = cudnn_conv_batched(x_seq, w)
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        v = v + (z_seq[t] - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines
# ============================================================

def make_sj_pipeline(conv, backend):
    """
    SJ-style pipeline: SeqToANNContainer(Conv2d) → LIFNode(backend)
    Weights copied from the caller's conv for apples-to-apples comparison.
    """
    conv_sj = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv_sj.weight.copy_(conv.weight)

    conv_seq = layer.SeqToANNContainer(conv_sj)
    lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m', backend=backend,
    ).to(device)
    return conv_seq, lif


def sj_baseline_forward(x_seq, conv_seq, lif):
    sj_func.reset_net(lif)
    y_seq = conv_seq(x_seq)
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
    print("v13: F1 Conv→LIF — FAIR version (cuDNN conv + Triton LIF fusion)")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C_in}→{C_out}, H=W={H}, k={KS}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print()

    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(1.5)
    w = conv.weight.contiguous()

    x_seq = torch.randn(T_TOTAL, B, C_in, H, W, device=device)

    # Step 1: parity — bit-exact expected (same cuDNN call as reference)
    print("Step 1: Parity — v13 fusion vs cuDNN-conv+py-LIF reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<18} {'spike_rate':<12}")
    print('-' * 68)
    with torch.no_grad():
        s_ref = reference_conv_lif(x_seq, w, T_TOTAL)

    all_pass = True
    for K in K_LIST:
        s_tri = v13_fusion_forward(x_seq, w, K)
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
        print("⚠ not all K bit-exact")
    print()

    # Step 2: v13 vs SpikingJelly
    print("Step 2: v13 vs SpikingJelly (numerical equivalence)")
    conv_seq_t, lif_t = make_sj_pipeline(conv, 'torch')
    with torch.no_grad():
        s_sj_t = sj_baseline_forward(x_seq, conv_seq_t, lif_t)
    s_v13 = v13_fusion_forward(x_seq, w, K=16)
    max_d = (s_sj_t - s_v13).abs().max().item()
    n_diff = (s_sj_t != s_v13).sum().item()
    flip_pct = n_diff / s_v13.numel() * 100
    print(f"  v13 vs SJ(torch) max_diff: {max_d:.2e}")
    print(f"  diff_spikes: {n_diff:,}/{s_v13.numel():,} ({flip_pct:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_t.mean().item():.4f}")
    print(f"  v13       spike rate: {s_v13.mean().item():.4f}")
    print()

    # Step 3: wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<24} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 80)
    results = []

    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline_forward(x_seq, conv_seq_t, lif_t)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_t = cuda_time_stats(run_sj_torch)
    results.append(('sj_torch', t_sj_t))

    cupy_available = False
    try:
        conv_seq_c, lif_c = make_sj_pipeline(conv, 'cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline_forward(x_seq, conv_seq_c, lif_c)
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
        def run_v13(K=K):
            _ = v13_fusion_forward(x_seq, w, K)
        for _ in range(N_WARMUP):
            run_v13()
        torch.cuda.synchronize()
        t_v13 = cuda_time_stats(run_v13)
        results.append((f'v13_K={K}', t_v13))

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
        'kernel': 'F1_conv_lif_fair',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C_in': C_in, 'C_out': C_out,
                  'H': H, 'W': W, 'KS': KS},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v13_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v13_v100_baseline.json")


if __name__ == '__main__':
    main()