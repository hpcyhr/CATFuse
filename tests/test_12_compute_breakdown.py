"""
Test 12: Sparse compute kernel time breakdown.

Break the 0.34ms compute overhead into:
  1. x.half().contiguous()           — dtype conversion
  2. x_f16.permute(0,2,3,1)         — NCHW→NHWC layout change
  3. weight.half().permute(0,2,3,1)  — weight layout change
  4. torch.empty() output alloc      — allocator overhead
  5. actual Triton kernel launch     — the real computation

Hypothesis: steps 1-4 are each a CUDA kernel launch (~5-10μs),
totaling ~0.2ms before we even start computing.
"""
import sys, os, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

from catfuse.sparseflow.sparse_conv2d_kernel import (
    sparse_conv2d_forward,
    _build_two_stage_metadata,
    _select_tile_sizes,
    choose_group_size,
)

B, C_IN, C_OUT, H, W = 8, 64, 64, 56, 56
KS, S, P = 3, 1, 1
SPARSITY = 0.95

print("=" * 70)
print(f"Compute kernel breakdown: B={B}, C={C_IN}→{C_OUT}, H={H}, sp={SPARSITY*100:.0f}%")
print("=" * 70)

# Create input
x = (torch.rand(B, C_IN, H, W, device=device) > SPARSITY).float()
conv = nn.Conv2d(C_IN, C_OUT, KS, stride=S, padding=P, bias=True).to(device)
w = conv.weight.detach()
b = conv.bias.detach()

N_WARMUP = 50
N_ITER = 500

# ============================================================
# Step 1: x.half().contiguous()
# ============================================================
print("\n--- Step 1: x.half().contiguous() ---")
# Warmup
for _ in range(N_WARMUP):
    _ = x.half().contiguous()
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = x.half().contiguous()
torch.cuda.synchronize()
step1_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step1_ms:.4f} ms")

# ============================================================
# Step 2: permute NCHW→NHWC + contiguous
# ============================================================
print("--- Step 2: x_f16.permute(0,2,3,1).contiguous() ---")
x_f16 = x.half().contiguous()
for _ in range(N_WARMUP):
    _ = x_f16.permute(0, 2, 3, 1).contiguous()
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = x_f16.permute(0, 2, 3, 1).contiguous()
torch.cuda.synchronize()
step2_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step2_ms:.4f} ms")

# ============================================================
# Step 3: weight.half().permute(0,2,3,1).contiguous()
# ============================================================
print("--- Step 3: w.half().permute(0,2,3,1).contiguous() ---")
for _ in range(N_WARMUP):
    _ = w.half().permute(0, 2, 3, 1).contiguous()
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = w.half().permute(0, 2, 3, 1).contiguous()
torch.cuda.synchronize()
step3_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step3_ms:.4f} ms")

# ============================================================
# Step 4: output allocation
# ============================================================
print("--- Step 4: torch.empty() output allocation ---")
H_OUT = (H + 2*P - KS) // S + 1
W_OUT = (W + 2*P - KS) // S + 1
for _ in range(N_WARMUP):
    _ = torch.empty(B, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = torch.empty(B, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
torch.cuda.synchronize()
step4_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step4_ms:.4f} ms")

# ============================================================
# Step 5: Full sparse_conv2d_forward (total)
# ============================================================
print("--- Step 5: Full sparse_conv2d_forward ---")
# Pre-compute w_cl to isolate weight permute from kernel time
w_cl = w.half().permute(0, 2, 3, 1).contiguous()

for _ in range(N_WARMUP):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P, threshold=1e-6)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P, threshold=1e-6)
torch.cuda.synchronize()
step5_total_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step5_total_ms:.4f} ms (with pre-computed w_cl)")

# ============================================================
# Step 6: sparse_conv2d_forward with BOTH x_nhwc and w_cl pre-computed
# ============================================================
print("--- Step 6: sparse_conv2d_forward with pre-computed x_nhwc + w_cl ---")
x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()

for _ in range(N_WARMUP):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, x_nhwc=x_nhwc)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, x_nhwc=x_nhwc)
torch.cuda.synchronize()
step6_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step6_ms:.4f} ms (x_nhwc + w_cl pre-computed)")

# ============================================================
# Step 7: sparse_conv2d_forward with launch_all_tiles=True (no nonzero() sync)
# ============================================================
print("--- Step 7: launch_all_tiles=True (avoid nonzero() host sync) ---")
for _ in range(N_WARMUP):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, x_nhwc=x_nhwc, launch_all_tiles=True)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=KS, stride=S, padding=P,
                               threshold=1e-6, x_nhwc=x_nhwc, launch_all_tiles=True)
torch.cuda.synchronize()
step7_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {step7_ms:.4f} ms (launch all tiles, no nonzero sync)")

# ============================================================
# Step 8: cuDNN baseline for comparison
# ============================================================
print("--- Step 8: cuDNN F.conv2d baseline ---")
for _ in range(N_WARMUP):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = F.conv2d(x, w, b, stride=S, padding=P)
torch.cuda.synchronize()
cudnn_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  {cudnn_ms:.4f} ms")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print(f"BREAKDOWN SUMMARY")
print(f"{'=' * 70}")
print(f"  cuDNN total:                    {cudnn_ms:.4f} ms")
print(f"  SF total (original):            {step5_total_ms:.4f} ms  ({step5_total_ms/cudnn_ms:.1f}× vs cuDNN)")
print(f"")
print(f"  Step 1: x.half().contiguous():  {step1_ms:.4f} ms")
print(f"  Step 2: NCHW→NHWC permute:      {step2_ms:.4f} ms")
print(f"  Step 3: weight permute:         {step3_ms:.4f} ms")
print(f"  Step 4: output alloc:           {step4_ms:.4f} ms")
print(f"  Steps 1-4 subtotal:             {step1_ms+step2_ms+step3_ms+step4_ms:.4f} ms")
print(f"")
print(f"  SF w/ pre-computed w_cl:        {step5_total_ms:.4f} ms")
print(f"  SF w/ pre-computed x_nhwc+w_cl: {step6_ms:.4f} ms")
print(f"  SF w/ launch_all_tiles:         {step7_ms:.4f} ms")
print(f"  Pure kernel (step7 - prescan):  {step7_ms - 0.17:.4f} ms  (rough estimate)")
print(f"{'=' * 70}")
