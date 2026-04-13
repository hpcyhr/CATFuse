"""
v7a: minimal Triton conv2d forward, parity vs nn.Conv2d
=======================================================

This is step 1 of implementing §3.9's CTF execution plan. Before we can
fuse Conv with LIF, we need our own conv kernel because cuDNN is closed-source
and doesn't accept custom epilogues. This file writes that conv kernel and
verifies it's numerically equivalent to nn.Conv2d.

Design choices:
  - We use the implicit-gemm formulation: conv2d is rewritten as a matmul
    between an im2col-style view of the input and the flattened weight.
    Triton handles this naturally because its core primitive is tl.dot.
  - BLOCK_M / BLOCK_N / BLOCK_K are the usual GEMM tile sizes.
  - For parity we only test one shape (middle: B=32, C_in=C_out=128,
    H=W=16, k=3, padding=1). Generalization comes later.

Scope of this file:
  - Just conv forward, no LIF
  - Just parity vs nn.Conv2d, no performance claims
  - Just one shape, no shape sweep

Next step (v7b): fuse LIF into this kernel's epilogue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

device = 'cuda:0'
torch.manual_seed(0)

# Fixed shape for this minimal test
B, C_in, C_out, H, W = 32, 128, 128, 16, 16
KS = 3      # kernel size
PAD = 1     # padding
STRIDE = 1  # stride
H_out = (H + 2 * PAD - KS) // STRIDE + 1  # = 16
W_out = (W + 2 * PAD - KS) // STRIDE + 1  # = 16


@triton.jit
def conv2d_implicit_gemm_kernel(
    # Pointers
    x_ptr,      # [B, C_in, H, W]
    w_ptr,      # [C_out, C_in, KS, KS]
    z_ptr,      # [B, C_out, H_out, W_out]
    # Problem dimensions
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    # Strides (in elements, not bytes)
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    # Tile sizes
    BLOCK_M: tl.constexpr,  # tile along (B * H_out * W_out) — output spatial*batch
    BLOCK_N: tl.constexpr,  # tile along C_out
    BLOCK_K: tl.constexpr,  # tile along (C_in * KS * KS) — gemm K dim
):
    """
    Implicit GEMM conv2d. We view the problem as a matmul:
        z[B*H_out*W_out, C_out] = im2col(x)[B*H_out*W_out, C_in*KS*KS]
                                @ W_flat[C_in*KS*KS, C_out]
    but we never explicitly build im2col — we compute the mapping from
    (m, k) index pairs back to (b, h_out, w_out, c_in, kh, kw) on the fly
    and load directly from x.
    """
    pid_m = tl.program_id(0)  # which tile of output rows (B*H_out*W_out)
    pid_n = tl.program_id(1)  # which tile of output channels (C_out)

    # Output row indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Output column indices (C_out dim)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m → (b, h_out, w_out)
    # m = b * (H_out * W_out) + h_out * W_out + w_out
    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Inner K loop — over (c_in, kh, kw) flattened
    K_total = C_in * KS * KS
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # Decode k → (c_in, kh, kw)
        c_in_idx = offs_k // (KS * KS)
        kh_kw = offs_k % (KS * KS)
        kh_idx = kh_kw // KS
        kw_idx = kh_kw % KS

        # Compute input spatial location for each (m, k) pair
        # h_in = h_out * STRIDE + kh - PAD
        # w_in = w_out * STRIDE + kw - PAD
        h_in = h_out_idx[:, None] * STRIDE + kh_idx[None, :] - PAD  # [BLOCK_M, BLOCK_K]
        w_in = w_out_idx[:, None] * STRIDE + kw_idx[None, :] - PAD

        # Valid input region mask (skip padding)
        valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        valid = valid & k_mask[None, :]
        valid = valid & (b_idx[:, None] < B)

        # Compute x address for each (m, k)
        x_offs = (b_idx[:, None] * stride_x_b
                  + c_in_idx[None, :] * stride_x_c
                  + h_in * stride_x_h
                  + w_in * stride_x_w)
        x_tile = tl.load(x_ptr + x_offs, mask=valid, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Weight: [C_out, C_in * KS * KS] view, indexed by (offs_n, offs_k)
        # W_flat[n, k] = W[offs_n[n], c_in_idx[k], kh_idx[k], kw_idx[k]]
        w_offs = (offs_n[:, None] * stride_w_o
                  + c_in_idx[None, :] * stride_w_c
                  + kh_idx[None, :] * stride_w_kh
                  + kw_idx[None, :] * stride_w_kw)
        w_mask = (offs_n[:, None] < C_out) & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # GEMM: acc += x_tile @ w_tile.T
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Write output: z[b, n, h_out, w_out]
    # offs_m already decodes to (b, h_out, w_out)
    z_offs = (b_idx[:, None] * stride_z_b
              + offs_n[None, :] * stride_z_c
              + h_out_idx[:, None] * stride_z_h
              + w_out_idx[:, None] * stride_z_w)
    z_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
    tl.store(z_ptr + z_offs, acc, mask=z_mask)


def triton_conv2d(x, w):
    """
    x: [B, C_in, H, W]
    w: [C_out, C_in, KS, KS]
    returns z: [B, C_out, H_out, W_out]
    """
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.is_contiguous() and w.is_contiguous()

    B_, C_in_, H_, W_ = x.shape
    C_out_, C_in_w, KS_, KS2_ = w.shape
    assert C_in_ == C_in_w and KS_ == KS2_

    H_out_ = (H_ + 2 * PAD - KS_) // STRIDE + 1
    W_out_ = (W_ + 2 * PAD - KS_) // STRIDE + 1

    z = torch.empty(B_, C_out_, H_out_, W_out_, device=device, dtype=torch.float32)

    # Tile sizes — conservative choice for V100
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    M_total = B_ * H_out_ * W_out_
    N_total = C_out_
    grid = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(N_total, BLOCK_N))

    conv2d_implicit_gemm_kernel[grid](
        x, w, z,
        B_, C_in_, H_, W_,
        C_out_, H_out_, W_out_,
        KS=KS_, PAD=PAD, STRIDE=STRIDE,
        stride_x_b=x.stride(0), stride_x_c=x.stride(1),
        stride_x_h=x.stride(2), stride_x_w=x.stride(3),
        stride_w_o=w.stride(0), stride_w_c=w.stride(1),
        stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
        stride_z_b=z.stride(0), stride_z_c=z.stride(1),
        stride_z_h=z.stride(2), stride_z_w=z.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return z


def main():
    print("=" * 72)
    print("v7a: Triton conv2d parity vs nn.Conv2d")
    print(f"Shape: B={B}, C_in={C_in}, C_out={C_out}, H=W={H}, KS={KS}, PAD={PAD}")
    print(f"Output: [{B}, {C_out}, {H_out}, {W_out}]")
    print("=" * 72)
    print()

    # Reference: nn.Conv2d
    conv = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    x = torch.randn(B, C_in, H, W, device=device)

    with torch.no_grad():
        z_ref = conv(x)  # [B, C_out, H_out, W_out]

    # Triton
    w = conv.weight.contiguous()  # [C_out, C_in, KS, KS]
    z_tri = triton_conv2d(x, w)

    # Parity
    max_abs = (z_ref - z_tri).abs().max().item()
    max_rel = ((z_ref - z_tri).abs() / (z_ref.abs() + 1e-6)).max().item()
    exact = torch.equal(z_ref, z_tri)

    print(f"z_ref  shape: {tuple(z_ref.shape)}, dtype: {z_ref.dtype}")
    print(f"z_tri  shape: {tuple(z_tri.shape)}, dtype: {z_tri.dtype}")
    print()
    print(f"Exact match:     {exact}")
    print(f"Max abs diff:    {max_abs:.2e}")
    print(f"Max rel diff:    {max_rel:.2e}")
    print(f"z_ref  mean/std: {z_ref.mean().item():.4f} / {z_ref.std().item():.4f}")
    print(f"z_tri  mean/std: {z_tri.mean().item():.4f} / {z_tri.std().item():.4f}")
    print()

    # What counts as pass:
    # - We don't expect bit-exact vs cudnn (different algo, different accum order)
    # - We DO expect float32 numerical equivalence: max_abs < 1e-4 on this shape
    if max_abs < 1e-4:
        print("PASS: Triton conv2d numerically equivalent to nn.Conv2d")
        print("      (bit-exact not expected because cudnn uses a different")
        print("       accumulation order than our implicit gemm)")
    else:
        print(f"FAIL: max_abs {max_abs:.2e} exceeds tolerance 1e-4")
        print("      Check kernel implementation, especially index decoding")
        print("      and mask handling.")


if __name__ == '__main__':
    main()