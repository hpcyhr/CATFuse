"""
Test: Old prescan vs new fast Triton prescan.

Compares:
  1. Correctness: both produce identical ag_mask and tile_class
  2. Latency: old (6 PyTorch ops) vs new (1 Triton kernel)

Run: python tests/test_11_fast_prescan.py
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("Old prescan vs new fast Triton prescan")
print("=" * 70)

from catfuse.sparseflow.sparse_conv2d_kernel import (
    _build_two_stage_metadata,
    _select_tile_sizes,
    choose_group_size,
)
from catfuse.sparseflow.fast_prescan import fast_spike_prescan_2d

configs = [
    # (label, B, C, H, sparsity)
    ("B=8  C=64  H=56  sp=95%",  8,  64,  56, 0.95),
    ("B=8  C=64  H=56  sp=99%",  8,  64,  56, 0.99),
    ("B=8  C=128 H=28  sp=95%",  8,  128, 28, 0.95),
    ("B=8  C=128 H=28  sp=99%",  8,  128, 28, 0.99),
    ("B=8  C=256 H=14  sp=95%",  8,  256, 14, 0.95),
    ("B=8  C=512 H=7   sp=95%",  8,  512, 7,  0.95),
    ("B=1  C=64  H=56  sp=95%",  1,  64,  56, 0.95),
    ("B=1  C=128 H=28  sp=95%",  1,  128, 28, 0.95),
    ("B=1  C=64  H=56  sp=0%",   1,  64,  56, 1.00),  # all zero
    ("B=8  C=64  H=56  sp=50%",  8,  64,  56, 0.50),  # dense
]

N_WARMUP = 50
N_ITER = 500

print(f"\n{'Config':<30s} {'Parity':>8s}  {'Old':>8s}  {'New':>8s}  {'Speedup':>8s}")
print("-" * 72)

for label, B, C, H, sparsity in configs:
    W = H
    KS, S, P = 3, 1, 1
    H_OUT = (H + 2*P - KS) // S + 1
    W_OUT = (W + 2*P - KS) // S + 1
    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = B * GH * GW
    GROUP_SIZE_C = choose_group_size(C)
    NUM_GROUPS = triton.cdiv(C, GROUP_SIZE_C)

    # Create spike input
    x_nchw = (torch.rand(B, C, H, W, device=device) > sparsity).float()
    x_f16 = x_nchw.half().contiguous()

    # Pre-allocate buffers
    ag_mask_old = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tile_class_old = torch.empty(N_TILES, dtype=torch.int32, device=device)
    ag_mask_new = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tile_class_new = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # --- Run old prescan ---
    _build_two_stage_metadata(
        x_f16, B, C, H, W, H_OUT, W_OUT,
        BH, BW, GH, GW,
        KS, S, P, 1e-6,
        ag_mask_old, tile_class_old,
    )
    torch.cuda.synchronize()

    # --- Run new prescan ---
    fast_spike_prescan_2d(
        x_nchw, H_OUT, W_OUT,
        kernel_size=KS, stride=S, padding=P,
        block_h=BH, block_w=BW,
        group_size_c=GROUP_SIZE_C,
        ag_mask_out=ag_mask_new,
        tile_class_out=tile_class_new,
    )
    torch.cuda.synchronize()

    # --- Check parity ---
    mask_match = (ag_mask_old[:N_TILES] == ag_mask_new[:N_TILES]).all().item()
    class_match = (tile_class_old[:N_TILES] == tile_class_new[:N_TILES]).all().item()
    parity = "OK" if (mask_match and class_match) else "FAIL"

    if not mask_match:
        mismatches = (ag_mask_old[:N_TILES] != ag_mask_new[:N_TILES]).sum().item()
        # Show first few mismatches for debugging
        diff_idx = (ag_mask_old[:N_TILES] != ag_mask_new[:N_TILES]).nonzero(as_tuple=False).flatten()[:5]
        for idx in diff_idx:
            i = idx.item()
            print(f"  MISMATCH tile {i}: old=0x{ag_mask_old[i].item():08x} new=0x{ag_mask_new[i].item():08x}")

    # --- Benchmark old ---
    for _ in range(N_WARMUP):
        _build_two_stage_metadata(
            x_f16, B, C, H, W, H_OUT, W_OUT,
            BH, BW, GH, GW,
            KS, S, P, 1e-6,
            ag_mask_old, tile_class_old,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(N_ITER):
        _build_two_stage_metadata(
            x_f16, B, C, H, W, H_OUT, W_OUT,
            BH, BW, GH, GW,
            KS, S, P, 1e-6,
            ag_mask_old, tile_class_old,
        )
    torch.cuda.synchronize()
    old_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # --- Benchmark new ---
    for _ in range(N_WARMUP):
        fast_spike_prescan_2d(
            x_nchw, H_OUT, W_OUT,
            kernel_size=KS, stride=S, padding=P,
            block_h=BH, block_w=BW,
            group_size_c=GROUP_SIZE_C,
            ag_mask_out=ag_mask_new,
            tile_class_out=tile_class_new,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(N_ITER):
        fast_spike_prescan_2d(
            x_nchw, H_OUT, W_OUT,
            kernel_size=KS, stride=S, padding=P,
            block_h=BH, block_w=BW,
            group_size_c=GROUP_SIZE_C,
            ag_mask_out=ag_mask_new,
            tile_class_out=tile_class_new,
        )
    torch.cuda.synchronize()
    new_ms = (time.perf_counter() - t0) / N_ITER * 1000

    speedup = old_ms / new_ms if new_ms > 0 else 0
    print(f"{label:<30s} {parity:>8s}  {old_ms:>7.3f}ms {new_ms:>7.3f}ms  {speedup:>7.1f}x")

print("\n" + "=" * 70)
