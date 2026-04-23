"""Minimal PyTorch kernel to verify ncu can read DRAM counters.
Expected DRAM bytes ~= 2 * tensor_size (one read + one write).
"""
import torch

# Simple memory-bound op: elementwise add on large tensor
# 32 MB tensor -> expect ~64 MB DRAM traffic (read + write)
n = 8 * 1024 * 1024  # 8M fp32 = 32 MB
a = torch.randn(n, device='cuda')
b = torch.randn(n, device='cuda')

# Warmup
for _ in range(3):
    c = a + b
torch.cuda.synchronize()

# Measured region
c = a + b
torch.cuda.synchronize()

print(f"tensor size: {n * 4 / 1e6:.1f} MB")
print(f"expected DRAM traffic ~= {3 * n * 4 / 1e6:.1f} MB (2 reads + 1 write)")
