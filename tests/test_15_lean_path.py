"""
Test 15: Lean direct kernel invocation — bypass sparse_conv2d_forward.

Hypothesis: bypassing the Python overhead (0.17ms) + using fast_prescan (0.17ms)
should bring total from 0.67ms down to ~0.22ms.
"""
import sys, time
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

from catfuse.sparseflow.sparse_conv2d_kernel import (
    sparse_conv2d_forward,
    sparse_conv3x3s1_nhwc_kernel_8x8,
    sparse_conv3x3s1_nhwc_kernel_8x16,
    _select_tile_sizes,
    choose_group_size,
    _build_two_stage_metadata,
)
from catfuse.sparseflow.fast_prescan import fast_spike_prescan_2d

B, C_IN, C_OUT = 8, 64, 64
H, W = 56, 56
KS, S, P = 3, 1, 1
SPARSITY = 0.95

H_OUT = (H + 2*P - KS)//S + 1
W_OUT = (W + 2*P - KS)//S + 1
BH, BW = _select_tile_sizes(H_OUT, W_OUT)
GH = triton.cdiv(H_OUT, BH)
GW = triton.cdiv(W_OUT, BW)
N_TILES = B * GH * GW
GROUP_SIZE_C = choose_group_size(C_IN)
NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)
ALL_ONES = (1 << NUM_GROUPS) - 1
DENSE_K = min(GROUP_SIZE_C * 2, 64)
if DENSE_K < 16: DENSE_K = 16
BLOCK_N = 64

print("=" * 70)
print(f"Lean path: B={B}, C={C_IN}→{C_OUT}, H={H}, sp={SPARSITY*100:.0f}%")
print(f"Tiles: {N_TILES}, block={BH}x{BW}, groups={NUM_GROUPS}")
print("=" * 70)

# Input
x = (torch.rand(B, C_IN, H, W, device=device) > SPARSITY).float()
w = torch.randn(C_OUT, C_IN, KS, KS, device=device)
b = torch.randn(C_OUT, device=device)

# Pre-compute everything once (simulates cached state in nn.Module)
x_f16 = x.half().contiguous()
x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()
w_cl = w.half().permute(0, 2, 3, 1).contiguous()
b_f32 = b.float().contiguous()

# Pre-allocate all buffers
ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
y_buf = torch.empty(B, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
# Dummy tile_ids (not used when launch_all_tiles)
tile_ids_buf = torch.arange(N_TILES, dtype=torch.int32, device=device)

N_WARMUP = 100
N_ITER = 1000

kernel = sparse_conv3x3s1_nhwc_kernel_8x16 if BW == 16 else sparse_conv3x3s1_nhwc_kernel_8x8

def _grid(META):
    return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

# ============================================================
# A: old full path (sparse_conv2d_forward)
# ============================================================
print("\n--- A: sparse_conv2d_forward (original) ---")
for _ in range(N_WARMUP):
    _ = sparse_conv2d_forward(x_f16, w, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, w_cl=w_cl, launch_all_tiles=True)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = sparse_conv2d_forward(x_f16, w, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, w_cl=w_cl, launch_all_tiles=True)
torch.cuda.synchronize()
a_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {a_ms:.4f} ms")

# ============================================================
# B: fast_prescan + original compute kernel (lean Python path)
# ============================================================
print("--- B: fast_prescan + lean kernel launch ---")

def lean_sparse_conv(x_nchw, x_nhwc_precomp, w_cl_precomp, bias_precomp, y_out,
                      ag_mask, tile_class, tile_ids):
    """Minimal Python path: just prescan + kernel launch."""
    # 1. Fast prescan (single Triton launch)
    fast_spike_prescan_2d(
        x_nchw, H_OUT, W_OUT,
        kernel_size=KS, stride=S, padding=P,
        block_h=BH, block_w=BW,
        group_size_c=GROUP_SIZE_C,
        ag_mask_out=ag_mask,
        tile_class_out=tile_class,
    )

    # 2. Direct kernel launch (no Python overhead)
    kernel[_grid](
        x_nhwc_precomp, w_cl_precomp, bias_precomp,
        ag_mask, tile_ids, y_out, B,
        C_IN=C_IN, C_OUT=C_OUT,
        H_IN=H, W_IN=W,
        H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW,
        HAS_BIAS=True,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES,
        DENSE_K=DENSE_K,
        USE_TILE_IDS=False,
    )
    return y_out

for _ in range(N_WARMUP):
    lean_sparse_conv(x, x_nhwc, w_cl, b_f32, y_buf,
                      ag_mask_buf, tile_class_buf, tile_ids_buf)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    lean_sparse_conv(x, x_nhwc, w_cl, b_f32, y_buf,
                      ag_mask_buf, tile_class_buf, tile_ids_buf)
torch.cuda.synchronize()
b_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {b_ms:.4f} ms")

# ============================================================
# C: kernel ONLY (no prescan, pre-computed ag_mask)
# ============================================================
print("--- C: kernel only (pre-computed ag_mask, no prescan) ---")

# Pre-compute ag_mask once
fast_spike_prescan_2d(x, H_OUT, W_OUT, kernel_size=KS, stride=S, padding=P,
                       block_h=BH, block_w=BW, group_size_c=GROUP_SIZE_C,
                       ag_mask_out=ag_mask_buf, tile_class_out=tile_class_buf)
torch.cuda.synchronize()

for _ in range(N_WARMUP):
    kernel[_grid](
        x_nhwc, w_cl, b_f32,
        ag_mask_buf, tile_ids_buf, y_buf, B,
        C_IN=C_IN, C_OUT=C_OUT,
        H_IN=H, W_IN=W,
        H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW,
        HAS_BIAS=True,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES,
        DENSE_K=DENSE_K,
        USE_TILE_IDS=False,
    )
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    kernel[_grid](
        x_nhwc, w_cl, b_f32,
        ag_mask_buf, tile_ids_buf, y_buf, B,
        C_IN=C_IN, C_OUT=C_OUT,
        H_IN=H, W_IN=W,
        H_OUT=H_OUT, W_OUT=W_OUT,
        GH=GH, GW=GW,
        HAS_BIAS=True,
        GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS,
        ALL_ONES_MASK=ALL_ONES,
        DENSE_K=DENSE_K,
        USE_TILE_IDS=False,
    )
torch.cuda.synchronize()
c_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {c_ms:.4f} ms")

# ============================================================
# D: cuDNN baseline
# ============================================================
print("--- D: cuDNN F.conv2d ---")
for _ in range(N_WARMUP):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()
d_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {d_ms:.4f} ms")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"  A) Original sparse_conv2d_forward:     {a_ms:.4f} ms ({a_ms/d_ms:.1f}× vs cuDNN)")
print(f"  B) fast_prescan + lean kernel:         {b_ms:.4f} ms ({b_ms/d_ms:.1f}× vs cuDNN)")
print(f"  C) Kernel only (no prescan):           {c_ms:.4f} ms ({c_ms/d_ms:.1f}× vs cuDNN)")
print(f"  D) cuDNN:                              {d_ms:.4f} ms (baseline)")
print(f"")
print(f"  Savings from lean path:  A→B = {(a_ms-b_ms)/a_ms*100:.0f}% faster")
print(f"  Prescan cost:            B-C = {b_ms-c_ms:.4f} ms")
print(f"  Pure kernel vs cuDNN:    C/D = {c_ms/d_ms:.1f}×")
print(f"{'=' * 70}")
