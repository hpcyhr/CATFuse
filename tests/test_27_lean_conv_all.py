"""
Test 27: Lean conv launchers for all variants.
"""
import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 100
N_ITER = 1000

from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward
from catfuse.sparseflow.lean_conv import lean_conv2d, lean_conv1d, LeanConv2dCache

def bench(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1000

print("=" * 90)
print(f"{'Kernel':<12s} {'Config':<30s} {'Baseline':>9s} {'SF-old':>9s} {'SF-lean':>9s} {'lean/base':>10s}")
print("-" * 90)

SP = 0.95

# ============================================================
# Conv2d 3x3 stride=2
# ============================================================
for B, cin, cout, H in [(8, 64, 128, 56), (8, 128, 256, 28), (8, 256, 512, 14)]:
    x = (torch.rand(B, cin, H, H, device=device) > SP).float()
    w = torch.randn(cout, cin, 3, 3, device=device)
    b = torch.randn(cout, device=device)
    w_cl = w.half().permute(0, 2, 3, 1).contiguous()
    x_f16 = x.half().contiguous()

    base = bench(lambda: F.conv2d(x, w, b, stride=2, padding=1))
    
    try:
        old = bench(lambda: sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=2, padding=1,
                                                    threshold=1e-6, launch_all_tiles=True))
    except:
        old = float('inf')
    
    cache = LeanConv2dCache(w, b, kernel_size=3, stride=2, padding=1)
    try:
        lean = bench(lambda: cache(x))
    except Exception as e:
        lean = float('inf')
        print(f"  lean error: {e}")
    
    r = base / lean if lean > 0 and lean != float('inf') else 0
    win = "lean" if r > 1 else "base"
    print(f"{'conv2d 3x3s2':<12s} {'B=%d %d→%d H=%d'%(B,cin,cout,H):<30s} {base:>8.3f}ms {old:>8.3f}ms {lean:>8.3f}ms {r:>9.2f}x  {win}")

# ============================================================
# Conv2d 1x1 stride=1
# ============================================================
for B, cin, cout, H in [(8, 64, 128, 28), (8, 128, 256, 14), (8, 256, 512, 7)]:
    x = (torch.rand(B, cin, H, H, device=device) > SP).float()
    w = torch.randn(cout, cin, 1, 1, device=device)
    b = torch.randn(cout, device=device)
    w_cl = w.half().permute(0, 2, 3, 1).contiguous()
    x_f16 = x.half().contiguous()

    base = bench(lambda: F.conv2d(x, w, b, stride=1, padding=0))
    
    try:
        old = bench(lambda: sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=1, stride=1, padding=0,
                                                    threshold=1e-6, launch_all_tiles=True))
    except:
        old = float('inf')
    
    cache = LeanConv2dCache(w, b, kernel_size=1, stride=1, padding=0)
    try:
        lean = bench(lambda: cache(x))
    except Exception as e:
        lean = float('inf')
        print(f"  lean error: {e}")
    
    r = base / lean if lean > 0 and lean != float('inf') else 0
    win = "lean" if r > 1 else "base"
    print(f"{'conv2d 1x1':<12s} {'B=%d %d→%d H=%d'%(B,cin,cout,H):<30s} {base:>8.3f}ms {old:>8.3f}ms {lean:>8.3f}ms {r:>9.2f}x  {win}")

# ============================================================
# Conv1d
# ============================================================
for B, C, L in [(8, 64, 1024), (8, 256, 256), (32, 128, 512)]:
    x = (torch.rand(B, C, L, device=device) > SP).float()
    w = torch.randn(C, C, 3, device=device)
    b = torch.randn(C, device=device)

    base = bench(lambda: F.conv1d(x, w, b, padding=1))
    
    try:
        lean = bench(lambda: lean_conv1d(x, w, b, kernel_size=3, stride=1, padding=1))
    except Exception as e:
        lean = float('inf')
        print(f"  conv1d lean error: {e}")
    
    r = base / lean if lean > 0 and lean != float('inf') else 0
    win = "lean" if r > 1 else "base"
    print(f"{'conv1d':<12s} {'B=%d C=%d L=%d'%(B,C,L):<30s} {base:>8.3f}ms {'N/A':>8s} {lean:>8.3f}ms {r:>9.2f}x  {win}")

# ============================================================
# Conv2d 3x3 stride=1 (confirm lean path still works)
# ============================================================
for B, C, H in [(8, 256, 14), (8, 512, 7)]:
    x = (torch.rand(B, C, H, H, device=device) > SP).float()
    w = torch.randn(C, C, 3, 3, device=device)
    b = torch.randn(C, device=device)

    base = bench(lambda: F.conv2d(x, w, b, padding=1))
    
    cache = LeanConv2dCache(w, b, kernel_size=3, stride=1, padding=1)
    lean = bench(lambda: cache(x))
    
    r = base / lean if lean > 0 else 0
    win = "lean" if r > 1 else "base"
    print(f"{'conv2d 3x3s1':<12s} {'B=%d C=%d H=%d'%(B,C,H):<30s} {base:>8.3f}ms {'---':>8s} {lean:>8.3f}ms {r:>9.2f}x  {win}")

print("=" * 90)
