"""
Conv → BN → LIF: fusion vs non-fusion, K=T only.

First chained-TSI fusion experiment. Tests whether §3.9's structural claim
extends from "single TSI → CSR" to "two TSI → CSR" by inserting an
inference-mode BatchNorm between Conv and LIF.

Inference-mode BN parameters are pre-computed into per-channel (scale, bias):
    scale[c] = gamma[c] / sqrt(running_var[c] + eps)
    bias[c]  = beta[c]  - running_mean[c] * scale[c]

So BN becomes a per-channel affine: z' = z * scale[c] + bias[c]

Key contrast with Conv → LIF:
  - non-fusion now uses 3 kernels (conv, BN, LIF) instead of 2
  - Each extra kernel boundary materializes one more T·step_out tensor in HBM
  - fusion eliminates BOTH intermediate tensors (z and z') in registers
  - Analytic savings goes from 48% (Conv→LIF) to ~66% (Conv→BN→LIF)

Analytic HBM bytes at K=T:
    non-fusion = T·step_in + |w| + 5·T·step_out + 8·C_out
    fusion     = T·step_in + |w| + T·step_out + step_out + 8·C_out

Where the 5·T·step_out comes from:
  conv writes z       : T·step_out
  BN reads z          : T·step_out
  BN writes z'        : T·step_out
  LIF reads z'        : T·step_out
  LIF writes spike    : T·step_out
  total               : 5·T·step_out

For default shape (T=16, B=64, C=128, H=W=28, k=3, s=1, p=1):
    no_fusion ≈ 392 + 0.56 + 5·392 = 2352.6 MB
    fusion    ≈ 392 + 0.56 + 392 + 24.5 = 809.1 MB
    ratio     = 0.344, savings 65.6%

Parity policy:
  - fusion vs no_fusion : require bit-exact (same Triton tl.dot order)
  - fusion vs reference : allow max_diff < 1e-3 and spike_match > 99.9%
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


# ============================================================
# Fusion kernel: implicit-GEMM conv + BN + LIF, all in registers
# ============================================================

@triton.jit
def _conv_bn_lif_fused_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, spike_ptr, v_carry_ptr,
    T,
    B, C_in, C_out,
    H_in, W_in,
    H_out, W_out,
    tau, v_th, v_reset,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_CONV: tl.constexpr,
    PADDING: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (B * H_out * W_out)

    wo = offs_m % W_out
    tmp = offs_m // W_out
    ho = tmp % H_out
    b = tmp // H_out

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out

    # Pre-load BN scale/bias for this BLOCK_N of channels
    # These stay in registers for the entire T loop (cheap reuse)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)   # [BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)     # [BLOCK_N]

    v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    inv_tau = 1.0 / tau

    K_TOTAL = C_in * K_CONV * K_CONV

    for t in range(T):
        # --- implicit-GEMM conv ---
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K_TOTAL, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_TOTAL

            kw = offs_k % K_CONV
            tmp_k = offs_k // K_CONV
            kh = tmp_k % K_CONV
            ci = tmp_k // K_CONV

            h_in = ho[:, None] * STRIDE + kh[None, :] - PADDING
            w_in = wo[:, None] * STRIDE + kw[None, :] - PADDING
            in_bounds = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            valid = in_bounds & mask_m[:, None] & mask_k[None, :]

            x_off = (t * stride_x_t
                     + b[:, None] * stride_x_b
                     + ci[None, :] * stride_x_c
                     + h_in * stride_x_h
                     + w_in * stride_x_w)
            x_tile = tl.load(x_ptr + x_off, mask=valid, other=0.0)

            w_off = (offs_n[None, :] * (C_in * K_CONV * K_CONV)
                     + ci[:, None] * (K_CONV * K_CONV)
                     + kh[:, None] * K_CONV
                     + kw[:, None])
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile)

        z = acc  # [BLOCK_M, BLOCK_N], conv output in registers

        # --- BN epilogue (per-channel affine, broadcast scale/bias over M) ---
        z_bn = z * scale[None, :] + bias[None, :]   # [BLOCK_M, BLOCK_N]

        # --- LIF ---
        v = v + (z_bn - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s

        # --- store spike ---
        s_off = (t * stride_s_t
                 + b[:, None] * stride_s_b
                 + offs_n[None, :] * stride_s_c
                 + ho[:, None] * stride_s_h
                 + wo[:, None] * stride_s_w)
        s_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(spike_ptr + s_off, s, mask=s_mask)

    # v_carry wasted write
    v_flat_off = (b[:, None] * (C_out * H_out * W_out)
                  + offs_n[None, :] * (H_out * W_out)
                  + ho[:, None] * W_out
                  + wo[:, None])
    v_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(v_carry_ptr + v_flat_off, v, mask=v_mask)


# ============================================================
# Non-fusion kernels: conv, BN, LIF as 3 separate Triton kernels
# ============================================================

@triton.jit
def _conv_only_kernel(
    x_ptr, w_ptr, z_ptr,
    T,
    B, C_in, C_out,
    H_in, W_in,
    H_out, W_out,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_z_t, stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_CONV: tl.constexpr,
    PADDING: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (B * H_out * W_out)

    wo = offs_m % W_out
    tmp = offs_m // W_out
    ho = tmp % H_out
    b = tmp // H_out

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out

    K_TOTAL = C_in * K_CONV * K_CONV

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        kw = offs_k % K_CONV
        tmp_k = offs_k // K_CONV
        kh = tmp_k % K_CONV
        ci = tmp_k // K_CONV

        h_in = ho[:, None] * STRIDE + kh[None, :] - PADDING
        w_in = wo[:, None] * STRIDE + kw[None, :] - PADDING
        in_bounds = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
        valid = in_bounds & mask_m[:, None] & mask_k[None, :]

        x_off = (pid_t * stride_x_t
                 + b[:, None] * stride_x_b
                 + ci[None, :] * stride_x_c
                 + h_in * stride_x_h
                 + w_in * stride_x_w)
        x_tile = tl.load(x_ptr + x_off, mask=valid, other=0.0)

        w_off = (offs_n[None, :] * (C_in * K_CONV * K_CONV)
                 + ci[:, None] * (K_CONV * K_CONV)
                 + kh[:, None] * K_CONV
                 + kw[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    z_off = (pid_t * stride_z_t
             + b[:, None] * stride_z_b
             + offs_n[None, :] * stride_z_c
             + ho[:, None] * stride_z_h
             + wo[:, None] * stride_z_w)
    z_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(z_ptr + z_off, acc, mask=z_mask)


@triton.jit
def _bn_only_kernel(
    z_ptr, scale_ptr, bias_ptr, z_out_ptr,
    N_total, C_per_pixel,
    BLOCK: tl.constexpr,
):
    """
    Per-channel affine: z_out[i] = z[i] * scale[c(i)] + bias[c(i)]
    Layout: z is [T, B, C, H, W] flat, channel index = (i // (H*W)) % C
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total

    # channel index
    c_idx = (offs // C_per_pixel) % 0  # placeholder
    # The above is wrong; we need to know spatial size per channel
    # We'll pass it from Python
    pass


@triton.jit
def _bn_only_kernel_v2(
    z_ptr, scale_ptr, bias_ptr, z_out_ptr,
    N_total,
    C_out, H_out, W_out,
    BLOCK: tl.constexpr,
):
    """
    z layout: [T, B, C_out, H_out, W_out] flat
    For each flat offset i, channel = (i // (H_out*W_out)) % C_out
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total

    HW = H_out * W_out
    c_idx = (offs // HW) % C_out

    z_val = tl.load(z_ptr + offs, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    z_out = z_val * scale + bias
    tl.store(z_out_ptr + offs, z_out, mask=mask)


@triton.jit
def _lif_only_kernel(
    z_ptr, spike_ptr,
    T, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial
    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau
    for t in range(T):
        base = t * stride_t + offs
        z = tl.load(z_ptr + base, mask=mask, other=0.0)
        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s
        tl.store(spike_ptr + base, s, mask=mask)


BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 64
BLOCK_K_DEFAULT = 32
BLOCK_BN_DEFAULT = 1024
BLOCK_LIF_DEFAULT = 1024


# ============================================================
# Wrappers
# ============================================================

def run_fusion(x, w, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0,
               BLOCK_M=BLOCK_M_DEFAULT, BLOCK_N=BLOCK_N_DEFAULT,
               BLOCK_K=BLOCK_K_DEFAULT,
               k_conv=3, padding=1, stride=1):
    assert x.is_contiguous() and w.is_contiguous()
    assert scale.is_contiguous() and bias.is_contiguous()
    T, B, C_in, H_in, W_in = x.shape
    C_out, _, kh, kw = w.shape
    assert kh == kw == k_conv

    H_out = (H_in + 2 * padding - k_conv) // stride + 1
    W_out = (W_in + 2 * padding - k_conv) // stride + 1

    spike = torch.empty(T, B, C_out, H_out, W_out,
                        device=x.device, dtype=x.dtype)
    v_carry = torch.empty(B * C_out * H_out * W_out,
                          device=x.device, dtype=x.dtype)

    sx_t = B * C_in * H_in * W_in
    sx_b = C_in * H_in * W_in
    sx_c = H_in * W_in
    sx_h = W_in
    sx_w = 1

    ss_t = B * C_out * H_out * W_out
    ss_b = C_out * H_out * W_out
    ss_c = H_out * W_out
    ss_h = W_out
    ss_w = 1

    grid = (
        triton.cdiv(B * H_out * W_out, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )

    _conv_bn_lif_fused_kernel[grid](
        x, w, scale, bias, spike, v_carry,
        T,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        tau, v_th, v_reset,
        sx_t, sx_b, sx_c, sx_h, sx_w,
        ss_t, ss_b, ss_c, ss_h, ss_w,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        K_CONV=k_conv, PADDING=padding, STRIDE=stride,
    )
    return spike


def run_no_fusion(x, w, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0,
                  BLOCK_M=BLOCK_M_DEFAULT, BLOCK_N=BLOCK_N_DEFAULT,
                  BLOCK_K=BLOCK_K_DEFAULT,
                  BLOCK_BN=BLOCK_BN_DEFAULT, BLOCK_LIF=BLOCK_LIF_DEFAULT,
                  k_conv=3, padding=1, stride=1):
    assert x.is_contiguous() and w.is_contiguous()
    T, B, C_in, H_in, W_in = x.shape
    C_out, _, kh, kw = w.shape
    H_out = (H_in + 2 * padding - k_conv) // stride + 1
    W_out = (W_in + 2 * padding - k_conv) // stride + 1

    z = torch.empty(T, B, C_out, H_out, W_out,
                    device=x.device, dtype=x.dtype)
    z_bn = torch.empty_like(z)
    spike = torch.empty_like(z)

    sx_t = B * C_in * H_in * W_in
    sx_b = C_in * H_in * W_in
    sx_c = H_in * W_in
    sx_h = W_in
    sx_w = 1

    sz_t = B * C_out * H_out * W_out
    sz_b = C_out * H_out * W_out
    sz_c = H_out * W_out
    sz_h = W_out
    sz_w = 1

    # Pass 1: conv → z
    grid_conv = (
        T,
        triton.cdiv(B * H_out * W_out, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )
    _conv_only_kernel[grid_conv](
        x, w, z,
        T,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        sx_t, sx_b, sx_c, sx_h, sx_w,
        sz_t, sz_b, sz_c, sz_h, sz_w,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        K_CONV=k_conv, PADDING=padding, STRIDE=stride,
    )

    # Pass 2: BN → z_bn
    N_total = T * B * C_out * H_out * W_out
    grid_bn = (triton.cdiv(N_total, BLOCK_BN),)
    _bn_only_kernel_v2[grid_bn](
        z, scale, bias, z_bn,
        N_total,
        C_out, H_out, W_out,
        BLOCK=BLOCK_BN,
    )

    # Pass 3: LIF → spike
    N_spatial = B * C_out * H_out * W_out
    grid_lif = (triton.cdiv(N_spatial, BLOCK_LIF),)
    _lif_only_kernel[grid_lif](
        z_bn, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        N_spatial,
        BLOCK=BLOCK_LIF,
    )
    return spike


def run_reference(x, w, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0,
                  k_conv=3, padding=1, stride=1):
    """torch.nn.functional.conv2d + BN as affine + manual LIF."""
    T, B, C_in, H_in, W_in = x.shape
    C_out = w.shape[0]
    x_flat = x.reshape(T * B, C_in, H_in, W_in)
    z_flat = torch.nn.functional.conv2d(x_flat, w,
                                        stride=stride, padding=padding)
    H_out, W_out = z_flat.shape[-2], z_flat.shape[-1]
    z = z_flat.reshape(T, B, C_out, H_out, W_out)

    # BN: per-channel affine. scale and bias are [C_out].
    z_bn = z * scale.view(1, 1, C_out, 1, 1) + bias.view(1, 1, C_out, 1, 1)

    v = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    spikes = torch.empty_like(z_bn)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z_bn[t] - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


def analytic_hbm_bytes(T, B, C_in, C_out, H_in, W_in, H_out, W_out,
                       k_conv, dtype_bytes=4):
    step_in = B * C_in * H_in * W_in * dtype_bytes
    step_out = B * C_out * H_out * W_out * dtype_bytes
    w_bytes = C_out * C_in * k_conv * k_conv * dtype_bytes
    bn_params = 2 * C_out * dtype_bytes  # scale + bias

    # non-fusion: conv writes z (T·step_out), BN reads z + writes z' (2·T·step_out),
    #             LIF reads z' + writes spike (2·T·step_out)
    # Plus: T·step_in for x reads, w_bytes for w, bn_params for scale+bias
    no_fusion = T * step_in + w_bytes + 5 * T * step_out + bn_params

    # fusion: T·step_in for x, w_bytes for w, bn_params for scale+bias,
    #         T·step_out for spike write, step_out for v_carry wasted write
    fusion = T * step_in + w_bytes + T * step_out + step_out + bn_params

    return {
        'step_in_MB': step_in / 1024 / 1024,
        'step_out_MB': step_out / 1024 / 1024,
        'w_MB': w_bytes / 1024 / 1024,
        'bn_params_bytes': bn_params,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


def parity_check(x, w, scale, bias, tau, v_th, v_reset, k_conv, padding, stride):
    out_ref = run_reference(x, w, scale, bias, tau, v_th, v_reset,
                            k_conv=k_conv, padding=padding, stride=stride)
    out_fus = run_fusion(x, w, scale, bias, tau, v_th, v_reset,
                         k_conv=k_conv, padding=padding, stride=stride)
    out_nof = run_no_fusion(x, w, scale, bias, tau, v_th, v_reset,
                            k_conv=k_conv, padding=padding, stride=stride)

    def diff(a, b):
        d = (a - b).abs()
        return {
            'max_diff': d.max().item(),
            'mean_abs_diff': d.mean().item(),
            'n_diff': (a != b).sum().item(),
            'n_total': a.numel(),
            'bit_exact': bool((a == b).all().item()),
        }

    def spike_match_pct(a, b):
        return ((a == b).float().mean().item()) * 100

    return {
        'spike_rate_reference': out_ref.mean().item(),
        'spike_rate_fusion': out_fus.mean().item(),
        'spike_rate_no_fusion': out_nof.mean().item(),
        'fusion_vs_reference': {
            **diff(out_fus, out_ref),
            'spike_match_pct': spike_match_pct(out_fus, out_ref),
        },
        'no_fusion_vs_reference': {
            **diff(out_nof, out_ref),
            'spike_match_pct': spike_match_pct(out_nof, out_ref),
        },
        'fusion_vs_no_fusion': {
            **diff(out_fus, out_nof),
            'spike_match_pct': spike_match_pct(out_fus, out_nof),
        },
    }


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


def cuda_time_stats(fn, n_iter=50, n_repeat=7, n_warmup=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'mean_ms': statistics.mean(samples),
        'stdev_ms': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'n_iter': n_iter,
        'n_repeat': n_repeat,
    }


def peak_memory_one_run(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return {
        'peak_allocated_bytes': peak,
        'peak_allocated_MB': peak / 1024 / 1024,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=64)
    p.add_argument('--C_in', type=int, default=128)
    p.add_argument('--C_out', type=int, default=128)
    p.add_argument('--H', type=int, default=28)
    p.add_argument('--W', type=int, default=28)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--padding', type=int, default=1)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--block_m', type=int, default=BLOCK_M_DEFAULT)
    p.add_argument('--block_n', type=int, default=BLOCK_N_DEFAULT)
    p.add_argument('--block_k', type=int, default=BLOCK_K_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=50)
    p.add_argument('--n_repeat', type=int, default=7)
    p.add_argument('--n_warmup', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/conv_bn_lif_min.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C_in, C_out = args.T, args.B, args.C_in, args.C_out
    H_in, W_in = args.H, args.W
    k_conv, padding, stride = args.k, args.padding, args.stride
    H_out = (H_in + 2 * padding - k_conv) // stride + 1
    W_out = (W_in + 2 * padding - k_conv) // stride + 1

    print(f"{'='*86}")
    print(f"Conv(k={k_conv},s={stride},p={padding}) → BN → LIF: "
          f"fusion vs non-fusion")
    print(f"{'='*86}")
    print(f"input  : T={T} B={B} C_in={C_in} H_in={H_in} W_in={W_in}")
    print(f"output : T={T} B={B} C_out={C_out} H_out={H_out} W_out={W_out}")
    print(f"weight : [{C_out},{C_in},{k_conv},{k_conv}]")
    print(f"BN     : per-channel affine (scale, bias)")
    print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print()

    # Inputs
    x = (torch.randn(T, B, C_in, H_in, W_in, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    w = (torch.randn(C_out, C_in, k_conv, k_conv, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    # BN params: scale near 1, bias near 0 (typical post-init values)
    scale = (1.0 + torch.randn(C_out, device=device, dtype=torch.float32) * 0.1).contiguous()
    bias = (torch.randn(C_out, device=device, dtype=torch.float32) * 0.1).contiguous()

    print("[1/4] Analytic HBM bytes")
    hbm = analytic_hbm_bytes(T, B, C_in, C_out, H_in, W_in, H_out, W_out,
                             k_conv, dtype_bytes=4)
    print(f"  step_in      : {hbm['step_in_MB']:.2f} MB")
    print(f"  step_out     : {hbm['step_out_MB']:.2f} MB")
    print(f"  weight       : {hbm['w_MB']:.3f} MB")
    print(f"  bn_params    : {hbm['bn_params_bytes']} bytes")
    print(f"  no-fusion    : {hbm['no_fusion_hbm_MB']:.2f} MB")
    print(f"  fusion (K=T) : {hbm['fusion_hbm_MB']:.2f} MB")
    print(f"  ratio        : {hbm['ratio_fusion_over_no_fusion']:.4f}")
    print(f"  savings      : {hbm['savings_pct']:.2f}%")
    print()

    print("[2/4] Smoke test")
    try:
        out_nof = run_no_fusion(x, w, scale, bias,
                                args.tau, args.v_th, args.v_reset,
                                BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                                BLOCK_K=args.block_k,
                                k_conv=k_conv, padding=padding, stride=stride)
        torch.cuda.synchronize()
        print(f"  non-fusion OK, output shape: {list(out_nof.shape)}")
    except Exception as e:
        print(f"  non-fusion FAILED: {type(e).__name__}: {e}")
        raise

    try:
        out_fus = run_fusion(x, w, scale, bias,
                             args.tau, args.v_th, args.v_reset,
                             BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                             BLOCK_K=args.block_k,
                             k_conv=k_conv, padding=padding, stride=stride)
        torch.cuda.synchronize()
        print(f"  fusion     OK, output shape: {list(out_fus.shape)}")
    except Exception as e:
        print(f"  fusion FAILED: {type(e).__name__}: {e}")
        raise
    print()

    print("[3/4] Parity check")
    parity = parity_check(x, w, scale, bias,
                          args.tau, args.v_th, args.v_reset,
                          k_conv, padding, stride)
    print(f"  spike_rate (reference) : {parity['spike_rate_reference']:.4f}")
    print(f"  spike_rate (fusion)    : {parity['spike_rate_fusion']:.4f}")
    print(f"  spike_rate (no_fusion) : {parity['spike_rate_no_fusion']:.4f}")
    for name, key in [
        ('fusion    vs reference ', 'fusion_vs_reference'),
        ('no_fusion vs reference ', 'no_fusion_vs_reference'),
        ('fusion    vs no_fusion ', 'fusion_vs_no_fusion'),
    ]:
        d = parity[key]
        tag = ' [BIT-EXACT]' if d['bit_exact'] else ''
        print(f"  {name}: max_diff={d['max_diff']:.2e}, "
              f"spike_match={d['spike_match_pct']:.4f}%{tag}")
    print()

    print("[4/4] Wall-clock")

    def bench_fus():
        _ = run_fusion(x, w, scale, bias,
                       args.tau, args.v_th, args.v_reset,
                       BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                       BLOCK_K=args.block_k,
                       k_conv=k_conv, padding=padding, stride=stride)

    def bench_nof():
        _ = run_no_fusion(x, w, scale, bias,
                          args.tau, args.v_th, args.v_reset,
                          BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                          BLOCK_K=args.block_k,
                          k_conv=k_conv, padding=padding, stride=stride)

    wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
    wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)
    mem_fus = peak_memory_one_run(bench_fus)
    mem_nof = peak_memory_one_run(bench_nof)

    speedup = wall_nof['median_ms'] / wall_fus['median_ms']

    print(f"  fusion    : {wall_fus['median_ms']:.4f} ms "
          f"± {wall_fus['stdev_ms']:.4f}  peak {mem_fus['peak_allocated_MB']:.1f} MB")
    print(f"  no-fusion : {wall_nof['median_ms']:.4f} ms "
          f"± {wall_nof['stdev_ms']:.4f}  peak {mem_nof['peak_allocated_MB']:.1f} MB")
    print(f"  speedup   : {speedup:.4f}×  "
          f"({(speedup-1)*100:+.2f}% {'faster' if speedup>1 else 'slower'})")
    print()

    result = {
        'experiment': 'conv_bn_lif_min',
        'section_ref': '§3.9 chained TSI: Conv → BN → LIF fusion vs non-fusion, K=T',
        'config': {
            'shape': {
                'T': T, 'B': B, 'C_in': C_in, 'C_out': C_out,
                'H_in': H_in, 'W_in': W_in, 'H_out': H_out, 'W_out': W_out,
            },
            'conv': {'k': k_conv, 'stride': stride, 'padding': padding},
            'bn': {'mode': 'inference', 'param_init_scale_std': 0.1,
                   'param_init_bias_std': 0.1},
            'lif': {'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset},
            'triton_tiles': {
                'BLOCK_M': args.block_m,
                'BLOCK_N': args.block_n,
                'BLOCK_K': args.block_k,
            },
            'input_scale': args.input_scale,
            'seed': args.seed,
            'dtype': 'float32',
            'device': torch.cuda.get_device_name(device),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'triton_version': triton.__version__,
        },
        'analytic_hbm': hbm,
        'parity': parity,
        'wall_clock': {
            'fusion': wall_fus,
            'no_fusion': wall_nof,
            'speedup_no_fusion_over_fusion': speedup,
        },
        'peak_memory': {
            'fusion': mem_fus,
            'no_fusion': mem_nof,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == '__main__':
    main()