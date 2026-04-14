"""
Conv(3x3, stride=1, padding=1) → LIF: fusion vs non-fusion, K=T only.

Shape (a) of the Conv→LIF minimal experiment. This is the simplest conv
case: kernel 3x3, stride 1, padding 1, so H_out = H_in and W_out = W_in.
Matches the ResNet-50 stage2 main-path 3x3 conv.

Variable being tested:
  does the intermediate tensor z = Conv(x) go through HBM?

  - fusion:     z computed inside tl.dot accumulator, fed to LIF in registers
  - non-fusion: z written to HBM by conv kernel, read back by LIF kernel

Both versions use Triton-written implicit-GEMM conv. The difference is
whether the conv accumulator is spilled to HBM or piped into the LIF
epilogue.

Analytic HBM bytes at K=T (single chunk, one wasted v_carry write):
    |step_in|  = B·C_in·H·W·4
    |step_out| = B·C_out·H_out·W_out·4     (= |step_in| when stride=1 and C_in=C_out)
    |w|        = C_out·C_in·k·k·4

    non-fusion = T·|step_in| + |w| + 3·T·|step_out|
    fusion     = T·|step_in| + |w| + T·|step_out| + |step_out|

For default shape (T=16, B=64, C_in=C_out=128, H=W=28, k=3, s=1, p=1):
    |step_in|  = |step_out| = 25.09 MB
    |w|        = 0.5625 MB
    no_fusion  = 16·25.09 + 0.56 + 3·16·25.09 = 1606.6 MB
    fusion     = 16·25.09 + 0.56 + 16·25.09 + 25.09 = 828.7 MB
    ratio      = 0.5158   (48.42% savings)

Parity policy:
  - fusion vs non-fusion: Both are Triton implicit-GEMM with the same
                          tl.dot call order. Should be bit-exact.
  - fusion vs torch.conv2d reference: Different runtime, ULP-level
                          differences expected. Allow max_diff < 1e-3 and
                          require spike position match > 99.9%.
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
# Fusion kernel: implicit-GEMM conv + LIF epilogue
# ============================================================
#
# Output tensor layout: spike[T, B, C_out, H_out, W_out]
#
# Grid: (cdiv(B*H_out*W_out, BLOCK_M), cdiv(C_out, BLOCK_N))
#   -> pid_m picks a block of output spatial positions flattened as
#      (b, h_out, w_out), size BLOCK_M
#   -> pid_n picks a block of output channels, size BLOCK_N
#
# Per program, each of the BLOCK_M * BLOCK_N output pixels is computed as
# a tl.dot over the K-reduction axis, where K_total = C_in * k_h * k_w.
# The K axis is tiled with BLOCK_K at a time.
#
# For LIF epilogue, we loop over t=0..T-1. Each step:
#   - recompute the full matmul for that t (shares nothing with previous t)
#   - feed z into LIF, update v, write spike
#
# At K=T (single chunk) we do not need v_carry reads inside the loop;
# we only write v_carry once at the end (the "wasted write" in the formula).

@triton.jit
def _conv_lif_fused_kernel(
    x_ptr, w_ptr, spike_ptr, v_carry_ptr,
    T,
    B, C_in, C_out, H, W,
    tau, v_th, v_reset,
    # Strides as raw integers (in elements, not bytes)
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_CONV: tl.constexpr,          # conv kernel size (3 for shape a)
    PADDING: tl.constexpr,          # padding (1 for shape a)
    STRIDE: tl.constexpr,           # conv stride (1 for shape a)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # M axis: flat (b, h_out, w_out) within one time step
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (B * H * W)   # H_out=H, W_out=W for shape a

    # Decompose M offset into (b, h_out, w_out)
    wo = offs_m % W
    tmp = offs_m // W
    ho = tmp % H
    b = tmp // H

    # N axis: output channel
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out

    # LIF state: [BLOCK_M, BLOCK_N]
    v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    inv_tau = 1.0 / tau

    K_TOTAL = C_in * K_CONV * K_CONV

    for t in range(T):
        # --- Compute z[t, offs_m, offs_n] via implicit GEMM ---
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K_TOTAL, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_TOTAL

            # Decompose K offset into (c_in, kh, kw)
            kw = offs_k % K_CONV
            tmp_k = offs_k // K_CONV
            kh = tmp_k % K_CONV
            ci = tmp_k // K_CONV

            # For each (m, k) pair, compute the input coordinate:
            #   h_in = ho[m] * STRIDE + kh[k] - PADDING
            #   w_in = wo[m] * STRIDE + kw[k] - PADDING
            # and mask out-of-bounds
            h_in = ho[:, None] * STRIDE + kh[None, :] - PADDING  # [M, K]
            w_in = wo[:, None] * STRIDE + kw[None, :] - PADDING  # [M, K]
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            valid = in_bounds & mask_m[:, None] & mask_k[None, :]

            # x offset: t*stride_x_t + b*stride_x_b + ci*stride_x_c
            #           + h_in*stride_x_h + w_in*stride_x_w
            x_off = (t * stride_x_t
                     + b[:, None] * stride_x_b
                     + ci[None, :] * stride_x_c
                     + h_in * stride_x_h
                     + w_in * stride_x_w)
            x_tile = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [M, K]

            # Weight layout: [C_out, C_in, k_h, k_w]
            # w[co, ci, kh, kw] at offset
            #   co*(C_in*K_CONV*K_CONV) + ci*(K_CONV*K_CONV) + kh*K_CONV + kw
            w_off = (offs_n[None, :] * (C_in * K_CONV * K_CONV)
                     + ci[:, None] * (K_CONV * K_CONV)
                     + kh[:, None] * K_CONV
                     + kw[:, None])
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [K, N]

            acc += tl.dot(x_tile, w_tile)

        z = acc  # [BLOCK_M, BLOCK_N], register-resident

        # --- LIF update ---
        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s

        # --- Store spike[t, offs_m, offs_n] ---
        s_off = (t * stride_s_t
                 + b[:, None] * stride_s_b
                 + offs_n[None, :] * stride_s_c
                 + ho[:, None] * stride_s_h
                 + wo[:, None] * stride_s_w)
        s_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(spike_ptr + s_off, s, mask=s_mask)

    # --- v_carry wasted write (K=T case, one single chunk boundary) ---
    v_flat_off = (b[:, None] * (C_out * H * W)
                  + offs_n[None, :] * (H * W)
                  + ho[:, None] * W
                  + wo[:, None])
    v_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(v_carry_ptr + v_flat_off, v, mask=v_mask)


# ============================================================
# Non-fusion kernels
# ============================================================

@triton.jit
def _conv_only_kernel(
    x_ptr, w_ptr, z_ptr,
    T,
    B, C_in, C_out, H, W,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_z_t, stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_CONV: tl.constexpr,
    PADDING: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """
    Same implicit-GEMM conv as inside the fusion kernel, but:
      - Grid is 3D: (T, cdiv(B*H*W, BLOCK_M), cdiv(C_out, BLOCK_N))
      - Writes z to HBM (that's the whole point)
      - No LIF, no v, no v_carry
    """
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (B * H * W)
    wo = offs_m % W
    tmp = offs_m // W
    ho = tmp % H
    b = tmp // H

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
        in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
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
def _lif_only_kernel(
    z_ptr, spike_ptr,
    T, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    """Flat LIF over N_spatial = B*C_out*H_out*W_out elements, T steps."""
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


# ============================================================
# Tile sizes
# ============================================================

BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 64
BLOCK_K_DEFAULT = 32
BLOCK_LIF_DEFAULT = 1024


# ============================================================
# Python wrappers
# ============================================================

def run_fusion(x, w, tau=2.0, v_th=1.0, v_reset=0.0,
               BLOCK_M=BLOCK_M_DEFAULT,
               BLOCK_N=BLOCK_N_DEFAULT,
               BLOCK_K=BLOCK_K_DEFAULT,
               k_conv=3, padding=1, stride=1):
    assert x.is_contiguous() and w.is_contiguous()
    T, B, C_in, H, Wdim = x.shape
    C_out, C_in_w, kh, kw = w.shape
    assert kh == kw == k_conv
    assert C_in_w == C_in

    # For shape (a): stride=1, padding=1, k=3 -> H_out=H, W_out=W
    H_out = (H + 2 * padding - k_conv) // stride + 1
    W_out = (Wdim + 2 * padding - k_conv) // stride + 1

    spike = torch.empty(T, B, C_out, H_out, W_out,
                        device=x.device, dtype=x.dtype)
    v_carry = torch.empty(B * C_out * H_out * W_out,
                          device=x.device, dtype=x.dtype)

    # x strides (in elements)
    sx_t = B * C_in * H * Wdim
    sx_b = C_in * H * Wdim
    sx_c = H * Wdim
    sx_h = Wdim
    sx_w = 1

    # spike strides
    ss_t = B * C_out * H_out * W_out
    ss_b = C_out * H_out * W_out
    ss_c = H_out * W_out
    ss_h = W_out
    ss_w = 1

    grid = (
        triton.cdiv(B * H_out * W_out, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )

    _conv_lif_fused_kernel[grid](
        x, w, spike, v_carry,
        T,
        B, C_in, C_out, H, Wdim,
        tau, v_th, v_reset,
        sx_t, sx_b, sx_c, sx_h, sx_w,
        ss_t, ss_b, ss_c, ss_h, ss_w,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        K_CONV=k_conv, PADDING=padding, STRIDE=stride,
    )
    return spike


def run_no_fusion(x, w, tau=2.0, v_th=1.0, v_reset=0.0,
                  BLOCK_M=BLOCK_M_DEFAULT,
                  BLOCK_N=BLOCK_N_DEFAULT,
                  BLOCK_K=BLOCK_K_DEFAULT,
                  BLOCK_LIF=BLOCK_LIF_DEFAULT,
                  k_conv=3, padding=1, stride=1):
    assert x.is_contiguous() and w.is_contiguous()
    T, B, C_in, H, Wdim = x.shape
    C_out, _, kh, kw = w.shape
    H_out = (H + 2 * padding - k_conv) // stride + 1
    W_out = (Wdim + 2 * padding - k_conv) // stride + 1

    z = torch.empty(T, B, C_out, H_out, W_out,
                    device=x.device, dtype=x.dtype)
    spike = torch.empty_like(z)

    sx_t = B * C_in * H * Wdim
    sx_b = C_in * H * Wdim
    sx_c = H * Wdim
    sx_h = Wdim
    sx_w = 1

    sz_t = B * C_out * H_out * W_out
    sz_b = C_out * H_out * W_out
    sz_c = H_out * W_out
    sz_h = W_out
    sz_w = 1

    # Pass 1: conv
    grid_conv = (
        T,
        triton.cdiv(B * H_out * W_out, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )
    _conv_only_kernel[grid_conv](
        x, w, z,
        T,
        B, C_in, C_out, H, Wdim,
        sx_t, sx_b, sx_c, sx_h, sx_w,
        sz_t, sz_b, sz_c, sz_h, sz_w,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        K_CONV=k_conv, PADDING=padding, STRIDE=stride,
    )

    # Pass 2: LIF over flat (B*C_out*H_out*W_out)
    N_spatial = B * C_out * H_out * W_out
    grid_lif = (triton.cdiv(N_spatial, BLOCK_LIF),)
    _lif_only_kernel[grid_lif](
        z, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        N_spatial,
        BLOCK=BLOCK_LIF,
    )
    return spike


def run_reference(x, w, tau=2.0, v_th=1.0, v_reset=0.0,
                  k_conv=3, padding=1, stride=1):
    """torch.nn.functional.conv2d + manual LIF."""
    T, B, C_in, H, Wdim = x.shape
    C_out = w.shape[0]
    # Reshape to (T*B, C, H, W) and apply conv2d
    x_flat = x.reshape(T * B, C_in, H, Wdim)
    z_flat = torch.nn.functional.conv2d(x_flat, w,
                                        stride=stride, padding=padding)
    H_out, W_out = z_flat.shape[-2], z_flat.shape[-1]
    z = z_flat.reshape(T, B, C_out, H_out, W_out)

    v = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    spikes = torch.empty_like(z)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z[t] - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


# ============================================================
# Analytic HBM
# ============================================================

def analytic_hbm_bytes(T, B, C_in, C_out, H, W, H_out, W_out,
                       k_conv, dtype_bytes=4):
    step_in = B * C_in * H * W * dtype_bytes
    step_out = B * C_out * H_out * W_out * dtype_bytes
    w_bytes = C_out * C_in * k_conv * k_conv * dtype_bytes

    no_fusion = T * step_in + w_bytes + 3 * T * step_out
    # fusion K=T: one wasted v_carry write
    fusion = T * step_in + w_bytes + T * step_out + step_out

    return {
        'step_in_MB': step_in / 1024 / 1024,
        'step_out_MB': step_out / 1024 / 1024,
        'w_MB': w_bytes / 1024 / 1024,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


# ============================================================
# Parity check
# ============================================================

def parity_check(x, w, tau, v_th, v_reset, k_conv, padding, stride):
    out_ref = run_reference(x, w, tau, v_th, v_reset,
                            k_conv=k_conv, padding=padding, stride=stride)
    out_fus = run_fusion(x, w, tau, v_th, v_reset,
                         k_conv=k_conv, padding=padding, stride=stride)
    out_nof = run_no_fusion(x, w, tau, v_th, v_reset,
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

    # Spike-position match (how many spike locations agree)
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


# ============================================================
# Timing / memory
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


def cuda_time_stats(fn, n_iter=100, n_repeat=11, n_warmup=20):
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


# ============================================================
# Main
# ============================================================

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
                   default='./results/conv_lif_min_a.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C_in, C_out, H, W = args.T, args.B, args.C_in, args.C_out, args.H, args.W
    k_conv, padding, stride = args.k, args.padding, args.stride
    H_out = (H + 2 * padding - k_conv) // stride + 1
    W_out = (W + 2 * padding - k_conv) // stride + 1

    print(f"{'='*86}")
    print(f"Conv(k={k_conv}, s={stride}, p={padding}) → LIF (shape a): "
          f"fusion vs non-fusion")
    print(f"{'='*86}")
    print(f"input  : T={T} B={B} C_in={C_in} H={H} W={W}")
    print(f"output : T={T} B={B} C_out={C_out} H_out={H_out} W_out={W_out}")
    print(f"weight : [{C_out},{C_in},{k_conv},{k_conv}]")
    print(f"LIF    : tau={args.tau} v_th={args.v_th} v_reset={args.v_reset}")
    print(f"tiles  : BLOCK_M={args.block_m} BLOCK_N={args.block_n} "
          f"BLOCK_K={args.block_k}")
    print(f"scale  : {args.input_scale}")
    print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print()

    # --- inputs ---
    x = (torch.randn(T, B, C_in, H, W, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    w = (torch.randn(C_out, C_in, k_conv, k_conv, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()

    # --- analytic ---
    print("[1/4] Analytic HBM bytes")
    hbm = analytic_hbm_bytes(T, B, C_in, C_out, H, W, H_out, W_out,
                             k_conv, dtype_bytes=4)
    print(f"  step_in      : {hbm['step_in_MB']:.2f} MB")
    print(f"  step_out     : {hbm['step_out_MB']:.2f} MB")
    print(f"  weight       : {hbm['w_MB']:.3f} MB")
    print(f"  no-fusion    : {hbm['no_fusion_hbm_MB']:.2f} MB")
    print(f"  fusion (K=T) : {hbm['fusion_hbm_MB']:.2f} MB")
    print(f"  ratio        : {hbm['ratio_fusion_over_no_fusion']:.4f}")
    print(f"  savings      : {hbm['savings_pct']:.2f}%")
    print()

    # --- smoke test ---
    print("[2/4] Smoke test")
    try:
        out_nof = run_no_fusion(x, w, args.tau, args.v_th, args.v_reset,
                                BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                                BLOCK_K=args.block_k,
                                k_conv=k_conv, padding=padding, stride=stride)
        torch.cuda.synchronize()
        print(f"  non-fusion OK, output shape: {list(out_nof.shape)}")
    except Exception as e:
        print(f"  non-fusion FAILED: {type(e).__name__}: {e}")
        raise

    try:
        out_fus = run_fusion(x, w, args.tau, args.v_th, args.v_reset,
                             BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                             BLOCK_K=args.block_k,
                             k_conv=k_conv, padding=padding, stride=stride)
        torch.cuda.synchronize()
        print(f"  fusion     OK, output shape: {list(out_fus.shape)}")
    except Exception as e:
        print(f"  fusion FAILED: {type(e).__name__}: {e}")
        raise
    print()

    # --- parity ---
    print("[3/4] Parity check")
    parity = parity_check(x, w, args.tau, args.v_th, args.v_reset,
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

    # --- wall-clock ---
    print("[4/4] Wall-clock")

    def bench_fus():
        _ = run_fusion(x, w, args.tau, args.v_th, args.v_reset,
                       BLOCK_M=args.block_m, BLOCK_N=args.block_n,
                       BLOCK_K=args.block_k,
                       k_conv=k_conv, padding=padding, stride=stride)

    def bench_nof():
        _ = run_no_fusion(x, w, args.tau, args.v_th, args.v_reset,
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

    # --- save ---
    result = {
        'experiment': 'conv_lif_min_shape_a',
        'section_ref': '§3.9 Conv(3x3,s1,p1)→LIF fusion vs non-fusion, K=T',
        'config': {
            'shape': {
                'T': T, 'B': B, 'C_in': C_in, 'C_out': C_out,
                'H': H, 'W': W, 'H_out': H_out, 'W_out': W_out,
            },
            'conv': {'k': k_conv, 'stride': stride, 'padding': padding},
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