"""
Test 13: Find SparseFlow's crossover point.
At what (B, C, H, sparsity) does SF beat cuDNN?
"""
import sys, time
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

configs = [
    # Large batch
    (32,  64,  56, 0.95), (32,  64,  56, 0.99),
    (32, 128,  28, 0.95), (32, 128,  28, 0.99),
    (32, 256,  14, 0.95), (32, 256,  14, 0.99),
    # Large channels
    (8,  256,  28, 0.95), (8,  256,  28, 0.99),
    (8,  512,  14, 0.95), (8,  512,  14, 0.99),
    (8, 1024,   7, 0.95), (8, 1024,   7, 0.99),
    # Extreme sparsity
    (8,  64,  56, 0.999), (8, 128,  28, 0.999),
    # Very large spatial
    (1,  64, 112, 0.95), (1,  64, 112, 0.99),
    (4,  64, 112, 0.95), (4,  64, 112, 0.99),
]

print(f"{'B':>4s} {'C':>5s} {'H':>4s} {'sp%':>6s}   {'cuDNN':>8s} {'SF':>8s} {'ratio':>7s} {'winner'}")
print("-" * 58)

for B, C, H, sp in configs:
    x = (torch.rand(B, C, H, H, device=device) > sp).float()
    w = torch.randn(C, C, 3, 3, device=device).half()
    w_cl = w.permute(0, 2, 3, 1).contiguous()
    b = torch.randn(C, device=device)

    # cuDNN
    for _ in range(20):
        _ = F.conv2d(x, w.float(), b, padding=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _ = F.conv2d(x, w.float(), b, padding=1)
    torch.cuda.synchronize()
    cudnn = (time.perf_counter() - t0) / 200 * 1000

    # SF
    x_f16 = x.half().contiguous()
    try:
        for _ in range(20):
            _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                       threshold=1e-6, launch_all_tiles=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            _ = sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                       threshold=1e-6, launch_all_tiles=True)
        torch.cuda.synchronize()
        sf = (time.perf_counter() - t0) / 200 * 1000
    except Exception as e:
        sf = float('inf')

    ratio = cudnn / sf if sf > 0 else 0
    win = "SF" if ratio > 1 else "cuDNN"
    print(f"{B:>4d} {C:>5d} {H:>4d} {sp*100:>5.1f}%   {cudnn:>7.3f}ms {sf:>7.3f}ms {ratio:>6.2f}x  {win}")
