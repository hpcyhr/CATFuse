"""
Test 23: Linear kernel breakdown — same methodology as conv2d test_15.
"""
import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 100
N_ITER = 1000

from catfuse.sparseflow.sparse_linear_kernel import (
    sparse_linear_forward,
    sparse_linear_grouped_kernel,
)

configs = [
    ("B=8  512→1000 sp=95%",    8,  512, 1000, 0.95),
    ("B=8  256→512  sp=95%",    8,  256,  512, 0.95),
    ("B=32 512→1000 sp=95%",   32,  512, 1000, 0.95),
    ("B=64 512→1000 sp=95%",   64,  512, 1000, 0.95),
    ("B=256 512→1000 sp=95%", 256,  512, 1000, 0.95),  # large batch
    ("B=1024 512→512 sp=95%",1024,  512,  512, 0.95),  # very large
]

print("=" * 85)
print(f"{'Config':<30s} {'F.linear':>9s} {'SF full':>9s} {'SF ratio':>9s}")
print("-" * 85)

for label, B, M, N, sp in configs:
    x = (torch.rand(B, M, device=device) > sp).float()
    w = torch.randn(N, M, device=device)
    b = torch.randn(N, device=device)

    # Baseline
    base = 0
    for _ in range(N_WARMUP):
        _ = F.linear(x, w, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        _ = F.linear(x, w, b)
    torch.cuda.synchronize()
    base = (time.perf_counter() - t0) / N_ITER * 1000

    # SF full path
    try:
        for _ in range(N_WARMUP):
            _ = sparse_linear_forward(x.half().contiguous(), w, b, threshold=1e-6,
                                       launch_all_tiles=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            _ = sparse_linear_forward(x.half().contiguous(), w, b, threshold=1e-6,
                                       launch_all_tiles=True)
        torch.cuda.synchronize()
        sf = (time.perf_counter() - t0) / N_ITER * 1000
    except Exception as e:
        sf = float('inf')
        print(f"  {label}: SF error: {e}")

    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{label:<30s} {base:>8.3f}ms {sf:>8.3f}ms {r:>8.2f}x  {win}")

print("=" * 85)
