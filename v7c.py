"""
v7c: Triton fused Conv+LIF with TimeBlock(K) and StateCarry
===========================================================

Step 3 of §3.9. Extends v7b from single-block (K=T) to multi-block execution.
The fused kernel now takes a persistent v_carry tensor: it loads v at the
start of each block launch, runs K steps, and stores v_final back. The Python
outer loop calls the kernel T/K times, with v_carry bridging block boundaries.

Scope:
  - K ∈ {1, 2, 4, 8, 16}, T_total = 16
  - Parity: all K must produce bit-exact identical spikes (verifies
    Corollary 3.11 at the fused-kernel level)
  - Measurement: allocation count and peak memory cross-check for each K
  - Forward only

Next step (v7d): dedicated HBM traffic analysis + cross-shape sweep.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import gc

device = 'cuda:0'
torch.manual_seed(0)

# Shape
B, C_in, C_out, H, W = 32, 128, 128, 16, 16
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


# ============================================================
# v7a conv (for reference)
# ============================================================

@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, z_ptr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_total = C_in * KS * KS
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total
        c_in_idx = offs_k // (KS * KS)
        kh_kw = offs_k % (KS * KS)
        kh_idx = kh_kw // KS
        kw_idx = kh_kw % KS
        h_in = h_out_idx[:, None] * STRIDE + kh_idx[None, :] - PAD
        w_in = w_out_idx[:, None] * STRIDE + kw_idx[None, :] - PAD
        valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        valid = valid & k_mask[None, :]
        valid = valid & (b_idx[:, None] < B)
        x_offs = (b_idx[:, None] * stride_x_b
                  + c_in_idx[None, :] * stride_x_c
                  + h_in * stride_x_h
                  + w_in * stride_x_w)
        x_tile = tl.load(x_ptr + x_offs, mask=valid, other=0.0)
        w_offs = (offs_n[:, None] * stride_w_o
                  + c_in_idx[None, :] * stride_w_c
                  + kh_idx[None, :] * stride_w_kh
                  + kw_idx[None, :] * stride_w_kw)
        w_mask = (offs_n[:, None] < C_out) & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, tl.trans(w_tile))

    z_offs = (b_idx[:, None] * stride_z_b
              + offs_n[None, :] * stride_z_c
              + h_out_idx[:, None] * stride_z_h
              + w_out_idx[:, None] * stride_z_w)
    z_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
    tl.store(z_ptr + z_offs, acc, mask=z_mask)


def triton_conv2d(x, w):
    B_, C_in_, H_, W_ = x.shape
    C_out_ = w.shape[0]
    H_out_ = (H_ + 2 * PAD - KS) // STRIDE + 1
    W_out_ = (W_ + 2 * PAD - KS) // STRIDE + 1
    z = torch.empty(B_, C_out_, H_out_, W_out_, device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B_ * H_out_ * W_out_, BLOCK_M), triton.cdiv(C_out_, BLOCK_N))
    conv2d_implicit_gemm_kernel[grid](
        x, w, z,
        B_, C_in_, H_, W_,
        C_out_, H_out_, W_out_,
        KS=KS, PAD=PAD, STRIDE=STRIDE,
        stride_x_b=x.stride(0), stride_x_c=x.stride(1),
        stride_x_h=x.stride(2), stride_x_w=x.stride(3),
        stride_w_o=w.stride(0), stride_w_c=w.stride(1),
        stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
        stride_z_b=z.stride(0), stride_z_c=z.stride(1),
        stride_z_h=z.stride(2), stride_z_w=z.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return z


# ============================================================
# v7c fused kernel: one block of K steps, reads and writes v_carry
# ============================================================

@triton.jit
def conv_lif_block_kernel(
    x_ptr,          # [K, B, C_in, H, W] — this block's time steps
    w_ptr,          # [C_out, C_in, KS, KS]
    s_ptr,          # [K, B, C_out, H_out, W_out] — this block's output
    v_carry_ptr,    # [B, C_out, H_out, W_out] — read at start, write at end
    K: tl.constexpr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    stride_v_b, stride_v_c, stride_v_h, stride_v_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused Conv+LIF for K consecutive time steps.
    StateCarry: v is loaded from v_carry_ptr at the start of the kernel and
    written back to the same pointer at the end. Between kernels, v travels
    through HBM (this is the block boundary cost in §3.9's formula).
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

    # Load v from v_carry at the block start
    v_offs = (b_idx[:, None] * stride_v_b
              + offs_n[None, :] * stride_v_c
              + h_out_idx[:, None] * stride_v_h
              + w_out_idx[:, None] * stride_v_w)
    v_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
    v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

    # Loop over K time steps inside this block
    K_total = C_in * KS * KS
    for t in tl.static_range(K):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K_total, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_total
            c_in_idx = offs_k // (KS * KS)
            kh_kw = offs_k % (KS * KS)
            kh_idx = kh_kw // KS
            kw_idx = kh_kw % KS
            h_in = h_out_idx[:, None] * STRIDE + kh_idx[None, :] - PAD
            w_in = w_out_idx[:, None] * STRIDE + kw_idx[None, :] - PAD
            valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            valid = valid & k_mask[None, :]
            valid = valid & (b_idx[:, None] < B)
            x_offs = (t * stride_x_t
                      + b_idx[:, None] * stride_x_b
                      + c_in_idx[None, :] * stride_x_c
                      + h_in * stride_x_h
                      + w_in * stride_x_w)
            x_tile = tl.load(x_ptr + x_offs, mask=valid, other=0.0)
            w_offs = (offs_n[:, None] * stride_w_o
                      + c_in_idx[None, :] * stride_w_c
                      + kh_idx[None, :] * stride_w_kh
                      + kw_idx[None, :] * stride_w_kw)
            w_mask = (offs_n[:, None] < C_out) & k_mask[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
            acc += tl.dot(x_tile, tl.trans(w_tile))

        # LIF update
        v = v + (acc - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        # Write spike to HBM
        s_offs = (t * stride_s_t
                  + b_idx[:, None] * stride_s_b
                  + offs_n[None, :] * stride_s_c
                  + h_out_idx[:, None] * stride_s_h
                  + w_out_idx[:, None] * stride_s_w)
        s_mask = v_mask
        tl.store(s_ptr + s_offs, spike, mask=s_mask)

    # Write v back to v_carry at block end
    tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


def triton_conv_lif_blocked(x_seq, w, K):
    """
    x_seq: [T_total, B, C_in, H, W]
    w:     [C_out, C_in, KS, KS]
    K:     block size (T_total must be divisible by K)
    returns s_seq: [T_total, B, C_out, H_out, W_out]
    """
    T_total = x_seq.shape[0]
    assert T_total % K == 0, f"T_total={T_total} not divisible by K={K}"
    n_blocks = T_total // K

    s_seq = torch.empty(T_total, B, C_out, H_out, W_out,
                        device=device, dtype=torch.float32)
    v_carry = torch.zeros(B, C_out, H_out, W_out,
                          device=device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B * H_out * W_out, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    for block_idx in range(n_blocks):
        block_start = block_idx * K
        x_block = x_seq[block_start:block_start + K].contiguous()
        s_block = s_seq[block_start:block_start + K]

        conv_lif_block_kernel[grid](
            x_block, w, s_block, v_carry,
            K=K,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C_out, H_out=H_out, W_out=W_out,
            KS=KS, PAD=PAD, STRIDE=STRIDE,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_x_t=x_block.stride(0), stride_x_b=x_block.stride(1),
            stride_x_c=x_block.stride(2), stride_x_h=x_block.stride(3),
            stride_x_w=x_block.stride(4),
            stride_w_o=w.stride(0), stride_w_c=w.stride(1),
            stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
            stride_s_t=s_block.stride(0), stride_s_b=s_block.stride(1),
            stride_s_c=s_block.stride(2), stride_s_h=s_block.stride(3),
            stride_s_w=s_block.stride(4),
            stride_v_b=v_carry.stride(0), stride_v_c=v_carry.stride(1),
            stride_v_h=v_carry.stride(2), stride_v_w=v_carry.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    return s_seq


# ============================================================
# Reference: v7a conv + pure-Python LIF (naive schedule)
# ============================================================

def reference_conv_lif(x_seq, w, T_total):
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T_total):
        z_t = triton_conv2d(x_seq[t], w)
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# Measurement helpers
# ============================================================

def measure_peak_and_counts(fn, n_iter=10):
    """Return (peak_bytes, alloc_count_per_iter)."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline_peak = torch.cuda.memory_allocated()

    stats_before = torch.cuda.memory_stats()
    count_before = stats_before['allocation.all.allocated']

    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - baseline_peak
    stats_after = torch.cuda.memory_stats()
    count = (stats_after['allocation.all.allocated'] - count_before) / n_iter

    return peak, count


def main():
    print("=" * 78)
    print("v7c: Triton fused Conv+LIF with TimeBlock(K) + StateCarry")
    print(f"Shape: T_total={T_TOTAL}, B={B}, C={C_in}, H=W={H}")
    print("=" * 78)
    print()

    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(2.0)
    w = conv.weight.contiguous()

    x_seq = torch.randn(T_TOTAL, B, C_in, H, W, device=device)

    # Reference
    with torch.no_grad():
        s_ref = reference_conv_lif(x_seq, w, T_TOTAL)
    ref_rate = s_ref.mean().item()
    print(f"Reference spike rate: {ref_rate:.4f}")
    print()

    # Step 1: Parity across K
    print("Step 1: Parity check (all K must match reference bit-exact)")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<14} {'spike_rate':<12}")
    print('-' * 60)
    all_pass = True
    for K in K_LIST:
        s_tri = triton_conv_lif_blocked(x_seq.contiguous(), w, K)
        exact = torch.equal(s_ref, s_tri)
        max_d = (s_ref - s_tri).abs().max().item()
        n_diff = (s_ref != s_tri).sum().item()
        rate = s_tri.mean().item()
        print(f"{K:<4} {str(exact):<8} {max_d:<12.2e} "
              f"{n_diff:<14,} {rate:<12.4f}")
        if not exact:
            all_pass = False

    if not all_pass:
        print()
        print("FAIL: not all K are bit-exact. Corollary 3.11 implementation broken.")
        return
    print()
    print("PASS: all K produce bit-exact identical spikes")
    print("      → Corollary 3.11 verified at the fused-kernel level")
    print()

    # Step 2: Measurement
    print("Step 2: Peak memory and allocation count")
    print(f"{'K':<4} {'n_launches':<12} {'peak (MB)':<12} "
          f"{'alloc/iter':<14} {'analytic ratio':<16}")
    print('-' * 70)

    # Reference measurement first
    def run_ref():
        with torch.no_grad():
            _ = reference_conv_lif(x_seq, w, T_TOTAL)
    for _ in range(3):
        run_ref()
    peak_ref, count_ref = measure_peak_and_counts(run_ref)
    print(f"{'naive':<4} {T_TOTAL:<12} {peak_ref/1e6:<12.2f} "
          f"{count_ref:<14.1f} {'1.000 (ref)':<16}")

    # Each K
    for K in K_LIST:
        def run_ctf(K=K):
            _ = triton_conv_lif_blocked(x_seq.contiguous(), w, K)
        for _ in range(3):
            run_ctf()
        peak_ctf, count_ctf = measure_peak_and_counts(run_ctf)
        analytic = (1 + 2 / K) / 5
        n_blocks = T_TOTAL // K
        print(f"{K:<4} {n_blocks:<12} {peak_ctf/1e6:<12.2f} "
              f"{count_ctf:<14.1f} {analytic:<16.3f}")

    print()
    print("What to check:")
    print("  - Peak MB: CTF should be substantially lower than naive")
    print("    (naive materializes z_seq; CTF doesn't)")
    print("  - Alloc count: naive is high (each step allocates conv output);")
    print("    CTF is low (one s_seq + one v_carry + per-block x_block slices)")
    print("  - CTF alloc count increases as K shrinks (more block launches)")


if __name__ == '__main__':
    main()