"""
v7b: Triton fused Conv + LIF (single-block, K=T)
================================================

This is step 2 of §3.9's CTF execution plan. We extend v7a's implicit GEMM
conv kernel to fold LIF into the epilogue: after computing z_t in registers,
the kernel immediately updates v in registers and writes only the spike
s_t back to HBM. z_t never touches HBM.

Scope:
  - K = T (single block, no TimeBlock split yet)
  - Forward only (no backward kernel)
  - Fixed shape (middle: B=32, C=128, H=W=16, k=3)
  - Hard-reset LIF with tau=2, v_th=1, v_reset=0

Parity target:
  - v7b output should match a pure-PyTorch "v7a conv + sequential LIF"
    implementation BIT-EXACT, because they share the same GEMM
    accumulation order (both use our own implicit gemm).
  - v7b output will NOT be bit-exact to nn.Conv2d + LIF, for the same
    reason v7a wasn't bit-exact to nn.Conv2d.

Next step (v7c): split T into blocks of size K, add StateCarry.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

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


# ============================================================
# v7a's conv kernel, imported verbatim for reference parity
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
    """v7a conv: used as the gemm-correct reference for v7b parity."""
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
# v7b fused kernel: Conv + LIF in one program, K=T
# ============================================================

@triton.jit
def conv_lif_fused_kernel(
    x_ptr,          # [T, B, C_in, H, W]
    w_ptr,          # [C_out, C_in, KS, KS]
    s_ptr,          # [T, B, C_out, H_out, W_out] — only output to HBM
    T: tl.constexpr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    # x strides: it's [T, B, C_in, H, W] so stride_x_t is the biggest
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Each program computes a [BLOCK_M, BLOCK_N] tile of output neurons,
    and runs the full T-step LIF loop for that tile. z_t stays in registers
    (the `acc` variable) and is immediately consumed to update v. Only
    spikes are written back to HBM.

    CTF mapping:
      - StreamFuse(Conv, LIF): z_t never reaches HBM, it flows directly
        from the gemm accumulator into the LIF charge equation
      - StateCarry(LIF) with K=T: v is a register variable held across
        all T steps, zero HBM traffic for v until the kernel ends
        (and we don't even write v back because v_final isn't needed
        when there's only one block)
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

    # Initialize v in registers — same shape as the output tile
    v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over time steps. T is constexpr so Triton unrolls it.
    for t in tl.static_range(T):
        # ---- Conv for time step t ----
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

            # x[t, b, c_in, h_in, w_in]
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

        # ---- LIF update for time step t ----
        # z_t = acc (in registers, never written to HBM)
        # v = v + (z_t - (v - v_reset)) / tau
        v = v + (acc - (v - v_reset)) / tau
        # spike = v >= v_th
        spike = (v >= v_th).to(tl.float32)
        # hard reset: v = v * (1 - spike) + v_reset * spike
        v = v * (1.0 - spike) + v_reset * spike

        # ---- Write spike to HBM ----
        s_offs = (t * stride_s_t
                  + b_idx[:, None] * stride_s_b
                  + offs_n[None, :] * stride_s_c
                  + h_out_idx[:, None] * stride_s_h
                  + w_out_idx[:, None] * stride_s_w)
        s_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
        tl.store(s_ptr + s_offs, spike, mask=s_mask)


def triton_conv_lif(x_seq, w, T):
    """
    x_seq: [T, B, C_in, H, W]
    w:     [C_out, C_in, KS, KS]
    returns s_seq: [T, B, C_out, H_out, W_out]
    """
    assert x_seq.shape[0] == T
    s_seq = torch.empty(T, B, C_out, H_out, W_out, device=device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B * H_out * W_out, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    conv_lif_fused_kernel[grid](
        x_seq, w, s_seq,
        T=T,
        B=B, C_in=C_in, H=H, W=W,
        C_out=C_out, H_out=H_out, W_out=W_out,
        KS=KS, PAD=PAD, STRIDE=STRIDE,
        tau=TAU, v_th=V_TH, v_reset=V_RESET,
        stride_x_t=x_seq.stride(0), stride_x_b=x_seq.stride(1),
        stride_x_c=x_seq.stride(2), stride_x_h=x_seq.stride(3),
        stride_x_w=x_seq.stride(4),
        stride_w_o=w.stride(0), stride_w_c=w.stride(1),
        stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
        stride_s_t=s_seq.stride(0), stride_s_b=s_seq.stride(1),
        stride_s_c=s_seq.stride(2), stride_s_h=s_seq.stride(3),
        stride_s_w=s_seq.stride(4),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return s_seq


# ============================================================
# Reference: v7a conv + pure-Python LIF
# ============================================================

def reference_conv_lif(x_seq, w, T):
    """
    Uses v7a's triton_conv2d for conv, plain Python loop for LIF.
    This is the bit-exact reference for v7b parity.
    """
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        z_t = triton_conv2d(x_seq[t], w)
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# Parity + spike rate sanity
# ============================================================

def main():
    print("=" * 72)
    print("v7b: Triton fused Conv+LIF (K=T) parity vs v7a conv + py LIF")
    print(f"Shape: T=16, B={B}, C={C_in}, H=W={H}")
    print("=" * 72)
    print()

    T = 16
    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    # Amplify weights so we get a meaningful spike rate
    with torch.no_grad():
        conv.weight.mul_(2.0)
    w = conv.weight.contiguous()

    x_seq = torch.randn(T, B, C_in, H, W, device=device)

    # Reference
    with torch.no_grad():
        s_ref = reference_conv_lif(x_seq, w, T)

    # v7b fused
    s_tri = triton_conv_lif(x_seq.contiguous(), w, T)

    # Parity
    max_abs = (s_ref - s_tri).abs().max().item()
    exact = torch.equal(s_ref, s_tri)
    n_total = s_ref.numel()
    n_diff = (s_ref != s_tri).sum().item()

    print(f"Reference spike rate: {s_ref.mean().item():.4f}")
    print(f"v7b       spike rate: {s_tri.mean().item():.4f}")
    print()
    print(f"Bit-exact match: {exact}")
    print(f"Max abs diff:    {max_abs}")
    print(f"Differing spikes: {n_diff} / {n_total} ({n_diff/n_total*100:.4f}%)")
    print()

    if exact:
        print("PASS: v7b is bit-exact to v7a conv + py LIF")
        print("      (this is the strong parity condition — same gemm accum order)")
    elif max_abs == 0:
        print("PASS: numerical match (no floating-point differences)")
    else:
        print(f"FAIL: spike outputs diverge, max_abs={max_abs}")
        print("      The fused kernel's LIF or v-state update has a bug.")


if __name__ == '__main__':
    main()