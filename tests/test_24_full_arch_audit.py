"""
Test 24: Full architecture kernel audit.

Maps kernel usage per architecture and measures each kernel's
standalone performance vs baseline.

Architectures:
  - VGG11-BN:    Conv2d, Linear, MaxPool2d
  - SEW-ResNet18: Conv2d, Linear, AvgPool2d, Add
  - SpikFormer:   Conv2d, Linear, MatMul/BMM, Attention

Kernel priority = (# architectures using it) × (time contribution)
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

def bench(fn, label=""):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1000


print("=" * 90)
print("FULL ARCHITECTURE KERNEL AUDIT")
print("=" * 90)

# ================================================================
# Kernel test matrix: representative shapes from each architecture
# ================================================================
# Format: (kernel, config_label, baseline_fn, sf_fn)

results = []

def test_kernel(kernel_name, config, base_fn, sf_fn):
    base = bench(base_fn)
    try:
        sf = bench(sf_fn)
    except Exception as e:
        sf = float('inf')
        print(f"  ERROR {kernel_name} {config}: {type(e).__name__}: {e}")
    ratio = base / sf if sf > 0 and sf != float('inf') else 0
    results.append((kernel_name, config, base, sf, ratio))
    return base, sf, ratio

SP = 0.95  # 95% sparsity throughout

# ================================================================
# Conv2d 3x3 — used by ALL architectures
# ================================================================
print("\n--- Conv2d 3x3 ---")
from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward

conv2d_shapes = [
    # VGG shapes (large spatial, growing channels)
    ("VGG: 64→64 H=32",     8, 64,   64,  32),
    ("VGG: 128→128 H=16",   8, 128, 128,  16),
    ("VGG: 256→256 H=8",    8, 256, 256,   8),
    ("VGG: 512→512 H=4",    8, 512, 512,   4),
    # ResNet shapes (ImageNet)
    ("RN: 64→64 H=56",      8, 64,   64,  56),
    ("RN: 128→128 H=28",    8, 128, 128,  28),
    ("RN: 256→256 H=14",    8, 256, 256,  14),
    ("RN: 512→512 H=7",     8, 512, 512,   7),
    # SpikFormer patch embed
    ("SF: 3→64 H=128",      8,  3,   64, 128),
]

for label, B, cin, cout, H in conv2d_shapes:
    x = (torch.rand(B, cin, H, H, device=device) > SP).float()
    w = torch.randn(cout, cin, 3, 3, device=device)
    b = torch.randn(cout, device=device)
    w_cl = w.half().permute(0, 2, 3, 1).contiguous()
    x_f16 = x.half().contiguous()

    base, sf, r = test_kernel("conv2d", label,
        lambda: F.conv2d(x, w, b, padding=1),
        lambda: sparse_conv2d_forward(x_f16, w_cl, b, kernel_size=3, stride=1, padding=1,
                                       threshold=1e-6, launch_all_tiles=True))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# Linear — used by ALL architectures (classifier + SpikFormer FFN)
# ================================================================
print("\n--- Linear ---")
from catfuse.sparseflow.sparse_linear_kernel import sparse_linear_forward

linear_shapes = [
    # Classifier layers
    ("cls: 512→10",          8,  512,   10),
    ("cls: 512→100",         8,  512,  100),
    ("cls: 512→1000",        8,  512, 1000),
    # SpikFormer FFN
    ("FFN: 256→1024",        8,  256, 1024),
    ("FFN: 1024→256",        8, 1024,  256),
    # Attention projections
    ("Attn: 256→256",        8,  256,  256),
    # Large batch
    ("cls-lg: 512→1000",   128,  512, 1000),
    ("FFN-lg: 256→1024",   128,  256, 1024),
]

for label, B, M, N in linear_shapes:
    x = (torch.rand(B, M, device=device) > SP).float()
    w = torch.randn(N, M, device=device)
    b = torch.randn(N, device=device)
    x_f16 = x.half().contiguous()

    base, sf, r = test_kernel("linear", label,
        lambda: F.linear(x, w, b),
        lambda: sparse_linear_forward(x_f16, w, b, threshold=1e-6, launch_all_tiles=True))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# MaxPool2d — VGG, some ResNets
# ================================================================
print("\n--- MaxPool2d ---")
from catfuse.sparseflow.sparse_maxpool2d_kernel import sparse_maxpool2d_forward

for label, B, C, H, k in [
    ("VGG: C=64 H=32 k=2",  8, 64,  32, 2),
    ("VGG: C=128 H=16 k=2", 8, 128, 16, 2),
    ("VGG: C=512 H=4 k=2",  8, 512,  4, 2),
    ("RN: C=64 H=112 k=3",  8, 64, 112, 3),
]:
    x = (torch.rand(B, C, H, H, device=device) > SP).float()
    pad = 1 if k == 3 else 0
    base, sf, r = test_kernel("maxpool2d", label,
        lambda: F.max_pool2d(x, k, padding=pad),
        lambda: sparse_maxpool2d_forward(x, kernel_size=k, stride=k, padding=pad, threshold=1e-6))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# AvgPool2d — ResNet, SpikFormer
# ================================================================
print("\n--- AvgPool2d ---")
from catfuse.sparseflow.sparse_avgpool2d_kernel import sparse_avgpool2d_forward

for label, B, C, H, k in [
    ("RN: C=512 H=7 k=7",   8, 512, 7, 7),   # global avg pool
    ("SF: C=256 H=14 k=2",  8, 256, 14, 2),
    ("SF: C=64 H=56 k=2",   8, 64, 56, 2),
]:
    x = (torch.rand(B, C, H, H, device=device) > SP).float()
    base, sf, r = test_kernel("avgpool2d", label,
        lambda: F.avg_pool2d(x, k),
        lambda: sparse_avgpool2d_forward(x, kernel_size=k, stride=k, padding=0, threshold=1e-6))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# BMM — SpikFormer attention (Q*K^T, A*V)
# ================================================================
print("\n--- BMM ---")
from catfuse.sparseflow.sparse_bmm_kernel import sparse_bmm_forward

for label, B, M, K, N in [
    ("Attn QK: 8×196×64",    8, 196, 64, 196),
    ("Attn AV: 8×196×196",   8, 196, 196, 64),
    ("Attn QK-sm: 8×49×64",  8,  49, 64,  49),
    ("large: 32×196×64",    32, 196, 64, 196),
]:
    x = (torch.rand(B, M, K, device=device) > SP).float()
    y = torch.randn(B, K, N, device=device)
    base, sf, r = test_kernel("bmm", label,
        lambda: torch.bmm(x, y),
        lambda: sparse_bmm_forward(x, y, threshold=1e-6))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# Conv1d — some 1D temporal models
# ================================================================
print("\n--- Conv1d ---")
from catfuse.sparseflow.sparse_conv1d_kernel import sparse_conv1d_forward

for label, B, C, L in [
    ("C=64 L=1024",  8,  64, 1024),
    ("C=256 L=256",  8, 256,  256),
]:
    x = (torch.rand(B, C, L, device=device) > SP).float()
    w = torch.randn(C, C, 3, device=device)
    b = torch.randn(C, device=device)
    base, sf, r = test_kernel("conv1d", label,
        lambda: F.conv1d(x, w, b, padding=1),
        lambda: sparse_conv1d_forward(x, w, b, kernel_size=3, stride=1, padding=1, threshold=1e-6))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# DepthwiseConv2d — MobileNet-style, some SpikFormer variants
# ================================================================
print("\n--- DepthwiseConv2d ---")
from catfuse.sparseflow.sparse_depthwise_conv2d_kernel import sparse_depthwise_conv2d_forward

for label, B, C, H in [
    ("C=64 H=56",  8,  64, 56),
    ("C=256 H=14", 8, 256, 14),
    ("C=512 H=7",  8, 512,  7),
]:
    x = (torch.rand(B, C, H, H, device=device) > SP).float()
    w = torch.randn(C, 1, 3, 3, device=device)
    b = torch.randn(C, device=device)
    base, sf, r = test_kernel("dw_conv2d", label,
        lambda: F.conv2d(x, w, b, padding=1, groups=C),
        lambda: sparse_depthwise_conv2d_forward(x, w, b, stride=1, padding=1, threshold=1e-6))
    win = "SF" if r > 1 else "base"
    print(f"  {label:<25s} base={base:.3f}ms SF={sf:.3f}ms ratio={r:.2f}x {win}")

# ================================================================
# SUMMARY: Priority matrix
# ================================================================
print("\n" + "=" * 90)
print("PRIORITY MATRIX")
print("=" * 90)

# Group by kernel type
from collections import defaultdict
by_kernel = defaultdict(list)
for kernel, config, base, sf, ratio in results:
    by_kernel[kernel].append((config, base, sf, ratio))

arch_usage = {
    "conv2d":     ["VGG", "ResNet", "SpikFormer"],
    "linear":     ["VGG", "ResNet", "SpikFormer"],
    "maxpool2d":  ["VGG", "ResNet"],
    "avgpool2d":  ["ResNet", "SpikFormer"],
    "bmm":        ["SpikFormer"],
    "conv1d":     ["temporal"],
    "dw_conv2d":  ["MobileNet", "SpikFormer-v2"],
}

print(f"\n{'Kernel':<15s} {'Archs':<30s} {'Best ratio':>10s} {'Worst ratio':>11s} {'Verdict':>10s}")
print("-" * 80)
for kernel, entries in by_kernel.items():
    ratios = [r for _, _, _, r in entries if r > 0]
    if not ratios:
        best, worst = 0, 0
    else:
        best, worst = max(ratios), min(ratios)
    archs = ", ".join(arch_usage.get(kernel, ["?"]))
    if best >= 1.0:
        verdict = "OK/mixed"
    elif worst > 0.5:
        verdict = "FIXABLE"
    else:
        verdict = "BROKEN"
    print(f"{kernel:<15s} {archs:<30s} {best:>9.2f}x {worst:>10.2f}x {verdict:>10s}")

print("\n" + "=" * 90)
