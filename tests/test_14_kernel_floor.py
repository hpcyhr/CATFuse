"""
Test 14: Triton kernel launch floor measurement.

The sparse compute kernel takes ~0.22ms regardless of sparsity.
Is this Triton launch overhead or actual GPU execution?

Approach: launch a trivial "null" Triton kernel with same grid size,
measure its latency. The difference = actual compute time.
"""
import sys, time
sys.path.insert(0, '.')

import torch
import triton
import triton.language as tl

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

# ============================================================
# Null kernel — same grid, does nothing
# ============================================================
@triton.jit
def _null_kernel(out_ptr, N_val):
    pid = tl.program_id(0)
    if pid >= N_val:
        return
    tl.store(out_ptr + pid, 0)

# ============================================================
# Minimal sparse kernel — same grid, loads ag_mask + zero-tile path only
# ============================================================
@triton.jit
def _minimal_sparse_kernel(
    ag_mask_ptr, out_ptr, N_TILES,
    C_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_cout = tl.program_id(1)
    if pid_tile >= N_TILES:
        return
    ag_mask = tl.load(ag_mask_ptr + pid_tile)
    # Just write ag_mask to output (trivial work)
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + pid_tile * C_OUT + offs_n, ag_mask, mask=offs_n < C_OUT)


print("=" * 70)
print("Triton kernel launch floor measurement")
print("=" * 70)

N_WARMUP = 100
N_ITER = 1000

configs = [
    ("224 programs (B=8, 7×4 tiles)", 224),
    ("56 programs (B=2, 7×4 tiles)", 56),
    ("28 programs (B=1, 7×4 tiles)", 28),
    ("1792 programs (B=8, 14×16 tiles)", 1792),
]

for label, n_progs in configs:
    out = torch.empty(n_progs, dtype=torch.int32, device=device)

    # Warmup
    for _ in range(N_WARMUP):
        _null_kernel[(n_progs,)](out, n_progs)
    torch.cuda.synchronize()

    # Time null kernel
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        _null_kernel[(n_progs,)](out, n_progs)
    torch.cuda.synchronize()
    null_ms = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Null kernel ({label}): {null_ms:.4f} ms")


# ============================================================
# 2D grid launch (matching sparse conv)
# ============================================================
print(f"\n--- 2D grid (matching sparse conv) ---")

C_OUT_vals = [64, 128, 256, 512]
N_TILES = 224
BLOCK_N = 64

ag_mask_buf = torch.zeros(N_TILES, dtype=torch.int32, device=device)

for C_OUT in C_OUT_vals:
    grid_1 = triton.cdiv(C_OUT, BLOCK_N)
    out = torch.empty(N_TILES * C_OUT, dtype=torch.int32, device=device)

    for _ in range(N_WARMUP):
        _minimal_sparse_kernel[(N_TILES, grid_1)](
            ag_mask_buf, out, N_TILES, C_OUT=C_OUT, BLOCK_N=BLOCK_N)
    torch.cuda.synchronize()

    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        _minimal_sparse_kernel[(N_TILES, grid_1)](
            ag_mask_buf, out, N_TILES, C_OUT=C_OUT, BLOCK_N=BLOCK_N)
    torch.cuda.synchronize()
    min_ms = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Minimal sparse kernel (tiles={N_TILES}, C_OUT={C_OUT}, grid=({N_TILES},{grid_1})): {min_ms:.4f} ms")


# ============================================================
# Compare: actual sparse conv kernel at 100% zero tiles
# ============================================================
print(f"\n--- Actual sparse conv at 100% zero sparsity ---")
from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward

for B, C, H in [(8, 64, 56), (8, 128, 28), (8, 256, 14)]:
    x = torch.zeros(B, C, H, H, device=device)  # ALL zeros
    w = torch.randn(C, C, 3, 3, device=device).half()
    w_cl = w.permute(0, 2, 3, 1).contiguous()
    b = torch.randn(C, device=device)
    x_f16 = x.half().contiguous()

    for _ in range(50):
        _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                   threshold=1e-6, launch_all_tiles=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(500):
        _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                   threshold=1e-6, launch_all_tiles=True)
    torch.cuda.synchronize()
    sf_ms = (time.perf_counter() - t0) / 500 * 1000
    print(f"  sparse_conv2d_forward (B={B}, C={C}, H={H}, 100% zero): {sf_ms:.4f} ms")


# ============================================================
# Python function call overhead
# ============================================================
print(f"\n--- Python function call overhead ---")
def _noop():
    pass

t0 = time.perf_counter()
for _ in range(100000):
    _noop()
py_us = (time.perf_counter() - t0) / 100000 * 1e6
print(f"  Python noop call: {py_us:.2f} μs")

# Torch empty
t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(N_ITER):
    _ = torch.empty(1, device=device, dtype=torch.int32)
torch.cuda.synchronize()
alloc_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  torch.empty(1): {alloc_ms:.4f} ms")

print(f"\n{'=' * 70}")
