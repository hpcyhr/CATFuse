"""
Test 22: SparseFlow kernel audit — ALL operators vs baseline.

For each SparseFlow kernel, measure:
  1. Baseline (PyTorch/cuDNN) latency
  2. SparseFlow kernel latency
  3. Ratio and verdict

This identifies which kernels have the same ~0.65ms floor problem.
"""
import sys, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

N_WARMUP = 50
N_ITER = 500

def bench_fn(fn, n_warmup=N_WARMUP, n_iter=N_ITER):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000

print("=" * 85)
print("SparseFlow Kernel Audit: ALL operators vs baseline")
print(f"{'Kernel':<25s} {'Config':<25s} {'Baseline':>9s} {'SF':>9s} {'Ratio':>7s} {'Winner':>7s}")
print("=" * 85)

# ============================================================
# 1. Conv2d (already profiled, include for completeness)
# ============================================================
from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward

for B, C, H, sp in [(8, 64, 56, 0.95), (8, 256, 14, 0.95), (8, 512, 7, 0.95)]:
    x = (torch.rand(B, C, H, H, device=device) > sp).float()
    w = torch.randn(C, C, 3, 3, device=device)
    b = torch.randn(C, device=device)
    w_cl = w.half().permute(0, 2, 3, 1).contiguous()
    x_f16 = x.half().contiguous()

    base = bench_fn(lambda: F.conv2d(x, w, b, padding=1))
    try:
        sf = bench_fn(lambda: sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                                      threshold=1e-6, launch_all_tiles=True))
    except Exception as e:
        sf = float('inf')
    r = base / sf if sf > 0 else 0
    win = "SF" if r > 1 else "cuDNN"
    print(f"{'conv2d 3x3':<25s} {'B=%d C=%d H=%d'%(B,C,H):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 2. Linear
# ============================================================
from catfuse.sparseflow.sparse_linear_kernel import sparse_linear_forward

for B, M, N, sp in [(8, 512, 1000, 0.95), (8, 256, 512, 0.95), (32, 512, 1000, 0.95)]:
    x = (torch.rand(B, M, device=device) > sp).float()
    w = torch.randn(N, M, device=device)
    b = torch.randn(N, device=device)

    base = bench_fn(lambda: F.linear(x, w, b))
    try:
        sf = bench_fn(lambda: sparse_linear_forward(x, w, b, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  linear error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{'linear':<25s} {'B=%d %d→%d'%(B,M,N):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 3. AvgPool2d
# ============================================================
from catfuse.sparseflow.sparse_avgpool2d_kernel import sparse_avgpool2d_forward

for B, C, H, k, sp in [(8, 64, 56, 2, 0.95), (8, 256, 14, 2, 0.95), (8, 512, 7, 7, 0.95)]:
    x = (torch.rand(B, C, H, H, device=device) > sp).float()
    pool = nn.AvgPool2d(k).to(device)

    base = bench_fn(lambda: pool(x))
    try:
        sf = bench_fn(lambda: sparse_avgpool2d_forward(x, kernel_size=k, stride=k, padding=0, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  avgpool error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{'avgpool2d':<25s} {'B=%d C=%d H=%d k=%d'%(B,C,H,k):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 4. MaxPool2d
# ============================================================
from catfuse.sparseflow.sparse_maxpool2d_kernel import sparse_maxpool2d_forward

for B, C, H, k, sp in [(8, 64, 56, 2, 0.95), (8, 256, 14, 2, 0.95)]:
    x = (torch.rand(B, C, H, H, device=device) > sp).float()

    base = bench_fn(lambda: F.max_pool2d(x, k))
    try:
        sf = bench_fn(lambda: sparse_maxpool2d_forward(x, kernel_size=k, stride=k, padding=0, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  maxpool error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{'maxpool2d':<25s} {'B=%d C=%d H=%d k=%d'%(B,C,H,k):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 5. DepthwiseConv2d
# ============================================================
from catfuse.sparseflow.sparse_depthwise_conv2d_kernel import sparse_depthwise_conv2d_forward

for B, C, H, sp in [(8, 64, 56, 0.95), (8, 256, 14, 0.95)]:
    x = (torch.rand(B, C, H, H, device=device) > sp).float()
    w = torch.randn(C, 1, 3, 3, device=device)
    b = torch.randn(C, device=device)

    base = bench_fn(lambda: F.conv2d(x, w, b, padding=1, groups=C))
    try:
        sf = bench_fn(lambda: sparse_depthwise_conv2d_forward(x, w, b, stride=1, padding=1, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  dw_conv error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "cuDNN"
    print(f"{'depthwise_conv2d':<25s} {'B=%d C=%d H=%d'%(B,C,H):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 6. Matmul
# ============================================================
from catfuse.sparseflow.sparse_matmul_kernel import sparse_matmul_forward

for B, M, K, N, sp in [(8, 64, 64, 64, 0.95), (8, 256, 256, 256, 0.95)]:
    x = (torch.rand(B, M, K, device=device) > sp).float()
    w = torch.randn(K, N, device=device)

    base = bench_fn(lambda: torch.matmul(x, w))
    try:
        sf = bench_fn(lambda: sparse_matmul_forward(x, w, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  matmul error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{'matmul':<25s} {'B=%d %dx%dx%d'%(B,M,K,N):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 7. BMM
# ============================================================
from catfuse.sparseflow.sparse_bmm_kernel import sparse_bmm_forward

for B, M, K, N, sp in [(8, 64, 64, 64, 0.95), (8, 196, 64, 196, 0.95)]:
    x = (torch.rand(B, M, K, device=device) > sp).float()
    y = torch.randn(B, K, N, device=device)

    base = bench_fn(lambda: torch.bmm(x, y))
    try:
        sf = bench_fn(lambda: sparse_bmm_forward(x, y, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  bmm error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "torch"
    print(f"{'bmm':<25s} {'B=%d %dx%dx%d'%(B,M,K,N):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# 8. Conv1d
# ============================================================
from catfuse.sparseflow.sparse_conv1d_kernel import sparse_conv1d_forward

for B, C, L, sp in [(8, 64, 1024, 0.95), (8, 256, 256, 0.95)]:
    x = (torch.rand(B, C, L, device=device) > sp).float()
    w = torch.randn(C, C, 3, device=device)
    b = torch.randn(C, device=device)

    base = bench_fn(lambda: F.conv1d(x, w, b, padding=1))
    try:
        sf = bench_fn(lambda: sparse_conv1d_forward(x, w, b, kernel_size=3, stride=1, padding=1, threshold=1e-6))
    except Exception as e:
        sf = float('inf')
        print(f"  conv1d error: {e}")
    r = base / sf if sf > 0 and sf != float('inf') else 0
    win = "SF" if r > 1 else "cuDNN"
    print(f"{'conv1d':<25s} {'B=%d C=%d L=%d'%(B,C,L):<25s} {base:>8.3f}ms {sf:>8.3f}ms {r:>6.2f}x {win:>7s}")

# ============================================================
# Summary: find the common floor
# ============================================================
print("\n" + "=" * 85)
print("AUDIT COMPLETE")
print("=" * 85)
