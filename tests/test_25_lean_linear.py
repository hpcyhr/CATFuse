"""Test 25: Lean linear launcher benchmark."""
import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 100
N_ITER = 1000

from catfuse.sparseflow.sparse_linear_kernel import sparse_linear_forward
from catfuse.sparseflow.lean_linear import lean_sparse_linear, LeanSparseLinearCache

configs = [
    ("B=8  512→1000  sp=95%",    8,  512, 1000, 0.95),
    ("B=8  256→512   sp=95%",    8,  256,  512, 0.95),
    ("B=8  512→10    sp=95%",    8,  512,   10, 0.95),
    ("B=8  256→1024  sp=95%",    8,  256, 1024, 0.95),
    ("B=128 512→1000 sp=95%",  128,  512, 1000, 0.95),
    ("B=8  1024→256  sp=95%",    8, 1024,  256, 0.95),
]

print("=" * 90)
print(f"{'Config':<28s} {'F.linear':>9s} {'SF-old':>9s} {'SF-lean':>9s} {'lean/base':>10s}")
print("-" * 90)

for label, B, M, N, sp in configs:
    x = (torch.rand(B, M, device=device) > sp).float()
    w = torch.randn(N, M, device=device)
    b = torch.randn(N, device=device)
    x_f16 = x.half().contiguous()
    w_t = w.t().half().contiguous()

    def _bench(fn):
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_ITER * 1000

    base = _bench(lambda: F.linear(x, w, b))

    try:
        sf_old = _bench(lambda: sparse_linear_forward(x_f16, w, b, threshold=1e-6, launch_all_tiles=True))
    except:
        sf_old = float('inf')

    cache = LeanSparseLinearCache(w, b)
    sf_lean = _bench(lambda: cache(x_f16))

    r = base / sf_lean if sf_lean > 0 else 0
    win = "lean" if r > 1 else "base"
    print(f"{label:<28s} {base:>8.3f}ms {sf_old:>8.3f}ms {sf_lean:>8.3f}ms {r:>9.2f}x  {win}")

print("=" * 90)
