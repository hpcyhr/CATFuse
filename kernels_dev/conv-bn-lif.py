"""
v12: F7 — Conv→BN→LIF fused kernel (fair version: cudnn conv + Triton LIF fusion)
==================================================================================

CTF transform: TimeBlock(K) ∘ StreamFuse(ConvBias, LIF) ∘ StateCarry(LIF)

Design (per 2025-04-11 framing calibration):
  - Dense conv is delegated to cuDNN. We don't reimplement conv in Triton
    because our goal is to demonstrate that CTF's fusion brings a speedup
    over a non-fused pipeline that ALSO uses cuDNN — not to outperform
    cuDNN on conv itself.
  - BN is statically folded into conv weights + bias (standard inference
    optimization; legal in CTF because inference BN is a TSI operator
    and TSI operators can be statically merged without affecting the
    CTF schedule space).
  - The Triton kernel only fuses the LIF multi-step loop together with
    any register-level ops that CTF permits. For F7 the register-level
    content is just "load z_t → LIF update → store spike"; there's no
    add/pool to fuse.
  - StateCarry carries v across block boundaries via an HBM v_carry buffer.

This Triton kernel is called `lif_fusion_kernel` because it will be shared
across F1 (Conv→LIF), F7 (Conv→BN→LIF), F8 (Conv→Pool→LIF after the pool
is done by cudnn or absorbed), and in general any (cudnn-output)→LIF
fusion target. CTF's unification shows up here.

Baselines:
  - cudnn_torch_lif:  F.conv2d → nn.BatchNorm2d(eval) → sj LIFNode(torch)
  - cudnn_cupy_lif:   F.conv2d → nn.BatchNorm2d(eval) → sj LIFNode(cupy)
    ← the real SOTA — cupy-fused multi-step LIF after cuDNN's conv+BN
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
# BN folding utility
# ============================================================

def fold_bn_into_conv(conv_weight, bn):
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w_fold = conv_weight * scale.view(-1, 1, 1, 1)
    b_fold = beta - mean * scale
    return w_fold.contiguous(), b_fold.contiguous()


def make_conv_bn():
    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    bn = nn.BatchNorm2d(C_out).to(device)
    with torch.no_grad():
        conv.weight.mul_(1.5)
        bn.weight.copy_(torch.rand_like(bn.weight) * 0.5 + 0.8)
        bn.bias.copy_(torch.randn_like(bn.bias) * 0.1)
        bn.running_mean.copy_(torch.randn_like(bn.running_mean) * 0.1)
        bn.running_var.copy_(torch.rand_like(bn.running_var) * 0.3 + 0.8)
    bn.eval()
    return conv, bn


# ============================================================
# cudnn side: batched conv over T
# ============================================================

def cudnn_conv_batched(x_seq, w, bias=None):
    """
    x_seq: [T, B, C_in, H, W]
    w:     [C_out, C_in, KS, KS]
    bias:  [C_out] or None

    Returns z_seq: [T, B, C_out, H_out, W_out]

    Fuses the T dim into the batch dim so cuDNN sees a single large
    conv call. This is the same conv that the baseline will use,
    ensuring apples-to-apples comparison at the conv layer.
    """
    T_ = x_seq.shape[0]
    x_flat = x_seq.view(T_ * B, C_in, H, W)
    z_flat = F.conv2d(x_flat, w, bias=bias, stride=STRIDE, padding=PAD)
    return z_flat.view(T_, B, C_out, H_out, W_out)


# ============================================================
# Triton LIF fusion kernel — fused multi-step LIF + StateCarry
# (Shared template for F1/F7/F8 etc. since after cudnn does the dense
# conv, the fusion kernel is just: load z_t → LIF step → store spike.)
# ============================================================

@triton.jit
def lif_fusion_kernel(
    z_ptr,          # [K, numel] — flattened conv output for this block
    s_ptr,          # [K, numel] — output spikes
    v_carry_ptr,    # [numel]    — carried membrane potential
    K: tl.constexpr,
    numel,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_t,
    BLOCK: tl.constexpr,
):
    """
    Each program handles BLOCK neurons along the flattened (B,C,H,W) dim.
    v lives in registers across all K time steps in this block.

    CTF mapping:
      - StreamFuse(ConvBias, LIF): z_t is already in HBM (produced by cuDNN);
        the kernel reads it once and feeds directly into the LIF update,
        never writing any intermediate (v, charge, reset state) back.
      - StateCarry: v travels through HBM between blocks via v_carry_ptr.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    v = tl.load(v_carry_ptr + offs, mask=mask, other=0.0)

    for t in tl.static_range(K):
        z_t = tl.load(z_ptr + t * stride_t + offs, mask=mask, other=0.0)

        # LIF update
        v = v + (z_t - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        tl.store(s_ptr + t * stride_t + offs, spike, mask=mask)

    tl.store(v_carry_ptr + offs, v, mask=mask)


def triton_lif_fusion(z_seq, K):
    """
    z_seq: [T_total, B, C_out, H_out, W_out] — already computed by cuDNN
    K:     block size
    returns s_seq: [T_total, B, C_out, H_out, W_out]
    """
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


def v12_fusion_forward(x_seq, w_fold, b_fold, K):
    """
    Full F7 fusion forward:
      1. cuDNN conv (with BN already folded in)
      2. Triton LIF fusion kernel with TimeBlock(K) and StateCarry
    """
    z_seq = cudnn_conv_batched(x_seq, w_fold, bias=b_fold)
    s_seq = triton_lif_fusion(z_seq, K)
    return s_seq


# ============================================================
# Reference for parity: cudnn conv (with folded weights) + py LIF
# This shares the exact same cuDNN call as the fusion path, so
# bit-exact parity is expected.
# ============================================================

def reference_folded_conv_lif(x_seq, w_fold, b_fold, T):
    z_seq = cudnn_conv_batched(x_seq, w_fold, bias=b_fold)
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        v = v + (z_seq[t] - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines (true non-fused pipelines)
# ============================================================

def make_sj_pipeline(conv, bn, backend):
    """
    Build an SJ-style baseline pipeline:
        SeqToANNContainer(Conv2d, BatchNorm2d) → LIFNode(backend)
    Weights copied from the caller's conv/bn so comparison is apples-to-apples.
    """
    conv_sj = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    bn_sj = nn.BatchNorm2d(C_out).to(device)
    with torch.no_grad():
        conv_sj.weight.copy_(conv.weight)
        bn_sj.weight.copy_(bn.weight)
        bn_sj.bias.copy_(bn.bias)
        bn_sj.running_mean.copy_(bn.running_mean)
        bn_sj.running_var.copy_(bn.running_var)
    bn_sj.eval()

    conv_bn_seq = layer.SeqToANNContainer(conv_sj, bn_sj)
    lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m', backend=backend,
    ).to(device)
    return conv_bn_seq, lif


def sj_baseline_forward(x_seq, conv_bn_seq, lif):
    sj_func.reset_net(lif)
    y_seq = conv_bn_seq(x_seq)
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
    print("v12: F7 Conv→BN→LIF — FAIR version (cuDNN conv + Triton LIF fusion)")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C_in}→{C_out}, H=W={H}, k={KS}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print()

    conv, bn = make_conv_bn()
    w_fold, b_fold = fold_bn_into_conv(conv.weight.detach(), bn)

    x_seq = torch.randn(T_TOTAL, B, C_in, H, W, device=device)

    # Step 0: BN fold sanity
    print("Step 0: BN fold sanity check")
    with torch.no_grad():
        y_conv_bn = bn(conv(x_seq[0]))
        y_fold = F.conv2d(x_seq[0], w_fold, bias=b_fold, padding=PAD)
    fold_diff = (y_conv_bn - y_fold).abs().max().item()
    print(f"  |conv→bn - folded conv| max_diff: {fold_diff:.2e}")
    assert fold_diff < 1e-4
    print("  BN fold verified ✓")
    print()

    # Step 1: parity — v12 vs folded-conv reference (expect bit-exact)
    print("Step 1: Parity — v12 fusion vs folded-conv+py-LIF reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<18} {'spike_rate':<12}")
    print('-' * 68)
    with torch.no_grad():
        s_ref = reference_folded_conv_lif(x_seq, w_fold, b_fold, T_TOTAL)

    all_pass = True
    for K in K_LIST:
        s_tri = v12_fusion_forward(x_seq, w_fold, b_fold, K)
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
        print("PASS: all K bit-exact ✓ (reference and fusion share the same cuDNN call)")
    else:
        print("⚠ not all K bit-exact — check kernel or v_carry layout")
    print()

    # Step 2: v12 vs SpikingJelly (expect numerical equivalence)
    print("Step 2: v12 vs SpikingJelly (numerical equivalence)")
    conv_bn_t, lif_t = make_sj_pipeline(conv, bn, 'torch')
    with torch.no_grad():
        s_sj_t = sj_baseline_forward(x_seq, conv_bn_t, lif_t)
    s_v12 = v12_fusion_forward(x_seq, w_fold, b_fold, K=16)
    max_d = (s_sj_t - s_v12).abs().max().item()
    n_diff = (s_sj_t != s_v12).sum().item()
    flip_pct = n_diff / s_v12.numel() * 100
    print(f"  v12 vs SJ(torch) max_diff: {max_d:.2e}")
    print(f"  diff_spikes: {n_diff:,}/{s_v12.numel():,} ({flip_pct:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_t.mean().item():.4f}")
    print(f"  v12       spike rate: {s_v12.mean().item():.4f}")
    print()

    # Step 3: wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<24} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 80)
    results = []

    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline_forward(x_seq, conv_bn_t, lif_t)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_t = cuda_time_stats(run_sj_torch)
    results.append(('sj_torch', t_sj_t))

    cupy_available = False
    try:
        conv_bn_c, lif_c = make_sj_pipeline(conv, bn, 'cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline_forward(x_seq, conv_bn_c, lif_c)
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
        def run_v12(K=K):
            _ = v12_fusion_forward(x_seq, w_fold, b_fold, K)
        for _ in range(N_WARMUP):
            run_v12()
        torch.cuda.synchronize()
        t_v12 = cuda_time_stats(run_v12)
        results.append((f'v12_K={K}', t_v12))

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
        'kernel': 'F7_conv_bn_lif_fair',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C_in': C_in, 'C_out': C_out,
                  'H': H, 'W': W, 'KS': KS},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v12_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v12_v100_baseline.json")


if __name__ == '__main__':
    main()