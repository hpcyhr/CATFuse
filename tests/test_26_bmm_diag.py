"""
Test 26: BMM diagnostic — is autotune the culprit?
"""
import sys, time
sys.path.insert(0, '.')
import torch
device = 'cuda:0'

from catfuse.sparseflow.sparse_bmm_kernel import sparse_bmm_forward

B, M, K, N = 8, 196, 64, 196
x = (torch.rand(B, M, K, device=device) > 0.95).float()
y = torch.randn(B, K, N, device=device)

print("=" * 60)
print(f"BMM diagnostic: B={B}, M={M}, K={K}, N={N}")
print("=" * 60)

# 1. Time individual calls to see if first call is slow (autotune)
print("\n--- Individual call times ---")
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = sparse_bmm_forward(x, y, threshold=1e-6)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    times.append(ms)
    print(f"  Call {i}: {ms:.2f} ms")

print(f"\n  First call:  {times[0]:.2f} ms")
print(f"  Calls 2-10:  {sum(times[1:])/len(times[1:]):.2f} ms avg")

# 2. Check if the early-exit conditions are hit
print(f"\n--- Size checks ---")
print(f"  M*K = {M*K} (threshold 512)")
print(f"  B*M*N = {B*M*N} (threshold 4096)")

# 3. Baseline
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    _ = torch.bmm(x, y)
torch.cuda.synchronize()
base = (time.perf_counter() - t0) / 1000 * 1000
print(f"\n  torch.bmm baseline: {base:.3f} ms")

# 4. Check if prescan is the bottleneck
print("\n--- Prescan vs compute breakdown ---")
import triton
from catfuse.sparseflow.sparse_bmm_kernel import (
    _prescan_bmm_kernel, choose_group_size
)

GROUP_SIZE_C = choose_group_size(K)
NUM_GROUPS = triton.cdiv(K, GROUP_SIZE_C)
ALL_ONES = (1 << NUM_GROUPS) - 1
BM = 64 if M >= 128 else (32 if M >= 32 else 16)
N_TILES_M = triton.cdiv(M, BM)
TOTAL_META = B * N_TILES_M

a_f16 = x.half().contiguous()
ag_mask = torch.empty(TOTAL_META, dtype=torch.int32, device=device)
tile_class = torch.empty(TOTAL_META, dtype=torch.int32, device=device)

# Warmup prescan
for _ in range(10):
    _prescan_bmm_kernel[(TOTAL_META,)](
        a_f16, ag_mask, tile_class,
        B=B, M=M, K=K, N_TILES_M=N_TILES_M,
        BLOCK_M=BM, GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS, ALL_ONES=ALL_ONES,
        THRESHOLD=1e-6,
    )
torch.cuda.synchronize()

t0 = time.perf_counter(); torch.cuda.synchronize()
for _ in range(1000):
    _prescan_bmm_kernel[(TOTAL_META,)](
        a_f16, ag_mask, tile_class,
        B=B, M=M, K=K, N_TILES_M=N_TILES_M,
        BLOCK_M=BM, GROUP_SIZE_C=GROUP_SIZE_C,
        NUM_GROUPS=NUM_GROUPS, ALL_ONES=ALL_ONES,
        THRESHOLD=1e-6,
    )
torch.cuda.synchronize()
prescan_ms = (time.perf_counter() - t0) / 1000 * 1000
print(f"  Prescan only: {prescan_ms:.3f} ms")
print(f"  BM={BM}, N_TILES_M={N_TILES_M}, TOTAL_META={TOTAL_META}")
print(f"  GROUP_SIZE_C={GROUP_SIZE_C}, NUM_GROUPS={NUM_GROUPS}")

print("=" * 60)
