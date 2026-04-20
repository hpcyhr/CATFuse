"""
Test 10: SparseFlow kernel time breakdown.

Break down the 0.75ms into:
  1. Python overhead (argument prep, buffer alloc)
  2. Prescan kernel launch + execution
  3. Metadata processing (nonzero, popcount etc.)
  4. Sparse compute kernel launch + execution
  5. Triton JIT compilation (first-call penalty)

This tells us WHERE to optimize.
"""
import sys, os, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("SparseFlow kernel time breakdown")
print("=" * 70)

from catfuse.sparseflow.sparse_conv2d_kernel import (
    sparse_conv2d_forward,
    _build_two_stage_metadata,
    _select_tile_sizes,
    choose_group_size,
)

# Test config: B=8, C=64, H=56 (SparseFlow's best case per README)
B, C_IN, C_OUT, H, W = 8, 64, 64, 56, 56
KS, S, P = 3, 1, 1
SPARSITY = 0.95

# Create input
x = (torch.rand(B, C_IN, H, W, device=device) > SPARSITY).float()
conv = nn.Conv2d(C_IN, C_OUT, KS, stride=S, padding=P, bias=True).to(device)
w = conv.weight.detach()
b = conv.bias.detach()
x_f16 = x.half().contiguous()
w_cl = w.half().permute(0, 2, 3, 1).contiguous()

H_OUT = (H + 2*P - KS) // S + 1
W_OUT = (W + 2*P - KS) // S + 1
BH, BW = _select_tile_sizes(H_OUT, W_OUT)
GH = triton.cdiv(H_OUT, BH)
GW = triton.cdiv(W_OUT, BW)
N_TILES = B * GH * GW
GROUP_SIZE_C = choose_group_size(C_IN)
NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE_C)

print(f"\nConfig: B={B}, C={C_IN}→{C_OUT}, H={H}, sparsity={SPARSITY*100:.0f}%")
print(f"Tiles: {N_TILES} ({B}×{GH}×{GW}), block={BH}×{BW}, groups={NUM_GROUPS}")

# Pre-allocate buffers (remove alloc overhead from timing)
ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

N_WARMUP = 50
N_ITER = 500

# ============================================================
# 1. Measure: total sparse_conv2d_forward time
# ============================================================
print(f"\n--- 1. Total sparse_conv2d_forward ---")

# Warmup (includes JIT compilation)
print("  Warming up (JIT compile)...")
for _ in range(N_WARMUP):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P, threshold=1e-6)
torch.cuda.synchronize()

t0 = time.perf_counter()
torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P, threshold=1e-6)
torch.cuda.synchronize()
total_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Total: {total_ms:.4f} ms")

# ============================================================
# 2. Measure: prescan only
# ============================================================
print(f"\n--- 2. Prescan (_build_two_stage_metadata) only ---")
for _ in range(N_WARMUP):
    _build_two_stage_metadata(
        x_f16, B, C_IN, H, W, H_OUT, W_OUT,
        BH, BW, GH, GW,
        KS, S, P, 1e-6,
        ag_mask_buf, tile_class_buf,
    )
torch.cuda.synchronize()

t0 = time.perf_counter()
torch.cuda.synchronize()
for _ in range(N_ITER):
    _build_two_stage_metadata(
        x_f16, B, C_IN, H, W, H_OUT, W_OUT,
        BH, BW, GH, GW,
        KS, S, P, 1e-6,
        ag_mask_buf, tile_class_buf,
    )
torch.cuda.synchronize()
prescan_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Prescan: {prescan_ms:.4f} ms")

# ============================================================
# 3. Measure: cuDNN F.conv2d for comparison
# ============================================================
print(f"\n--- 3. cuDNN F.conv2d baseline ---")
for _ in range(N_WARMUP):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()

t0 = time.perf_counter()
torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()
cudnn_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  cuDNN: {cudnn_ms:.4f} ms")

# ============================================================
# 4. Measure: Python overhead (everything except CUDA kernels)
# ============================================================
print(f"\n--- 4. Python-side overhead estimate ---")
# Time just the Python argument preparation path
# by calling with an impossible shape that hits early return

# A rough proxy: measure the function call overhead with a tiny tensor
x_tiny = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float16)
w_tiny = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float16)
b_tiny = torch.zeros(1, device=device)

# The sparse kernel might not handle 1x1, so let's just time the metadata build
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    # Simulate just the Python-level argument prep
    _ = x_f16.shape
    _ = w_cl.shape
    H_OUT_tmp = (H + 2*P - KS) // S + 1
    W_OUT_tmp = (W + 2*P - KS) // S + 1
    BH_tmp, BW_tmp = _select_tile_sizes(H_OUT_tmp, W_OUT_tmp)
    GH_tmp = triton.cdiv(H_OUT_tmp, BH_tmp)
    GW_tmp = triton.cdiv(W_OUT_tmp, BW_tmp)
    gs = choose_group_size(C_IN)
    ng = triton.cdiv(C_IN, gs)
torch.cuda.synchronize()
python_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Python arg prep: {python_ms:.4f} ms")

# ============================================================
# 5. CUDA events for kernel-level timing
# ============================================================
print(f"\n--- 5. CUDA event timing (prescan kernel only) ---")

from catfuse.sparseflow.prescan import _build_rf_prescan_metadata_impl

x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()

# Warmup
for _ in range(20):
    _build_rf_prescan_metadata_impl(
        x_channels_last=x_nhwc,
        spatial_dims=(H_OUT, W_OUT),
        kernel_dims=(KS, KS),
        stride=S, padding=P,
        block_dims=(BH, BW),
        group_size_c=GROUP_SIZE_C,
        num_groups=NUM_GROUPS,
        threshold=1e-6,
        return_debug_stats=False,
        tile_class_out=tile_class_buf,
        ag_mask_out=ag_mask_buf,
    )
torch.cuda.synchronize()

# CUDA events for precise GPU timing
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

prescan_gpu_times = []
for _ in range(100):
    start_ev.record()
    _build_rf_prescan_metadata_impl(
        x_channels_last=x_nhwc,
        spatial_dims=(H_OUT, W_OUT),
        kernel_dims=(KS, KS),
        stride=S, padding=P,
        block_dims=(BH, BW),
        group_size_c=GROUP_SIZE_C,
        num_groups=NUM_GROUPS,
        threshold=1e-6,
        return_debug_stats=False,
        tile_class_out=tile_class_buf,
        ag_mask_out=ag_mask_buf,
    )
    end_ev.record()
    torch.cuda.synchronize()
    prescan_gpu_times.append(start_ev.elapsed_time(end_ev))

import statistics
prescan_gpu_ms = statistics.median(prescan_gpu_times)
print(f"  Prescan GPU time (CUDA events): {prescan_gpu_ms:.4f} ms")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print(f"BREAKDOWN SUMMARY (B={B}, C={C_IN}, H={H}, sparsity={SPARSITY*100:.0f}%)")
print(f"{'=' * 70}")
print(f"  cuDNN total:            {cudnn_ms:.4f} ms")
print(f"  SF total:               {total_ms:.4f} ms  ({total_ms/cudnn_ms:.1f}× slower)")
print(f"  SF prescan (wall):      {prescan_ms:.4f} ms  ({prescan_ms/total_ms*100:.0f}% of SF total)")
print(f"  SF prescan (GPU):       {prescan_gpu_ms:.4f} ms")
print(f"  SF compute (estimate):  {total_ms - prescan_ms:.4f} ms")
print(f"  Python overhead:        {python_ms:.4f} ms")
print(f"")
print(f"  Prescan / cuDNN ratio:  {prescan_ms/cudnn_ms:.1f}×")
print(f"  → Prescan ALONE is {'slower' if prescan_ms > cudnn_ms else 'faster'} than cuDNN")
print(f"{'=' * 70}")
