"""
test_63_diagnose2.py — 第二轮诊断

发现: prescan 的判断粒度是 "tile 内有没有非零",threshold=1e-6 太低.
对于 8x16 tile (128 个位置), tile 完全空的概率 = sparsity^128:
  sparsity=0.95  → 0.1%
  sparsity=0.99  → 27%
  sparsity=0.999 → 88%

所以测 active_ratio 应该在 0.95-0.999 范围才能看出变化.
此外, 测 SF 在不同 sparsity 下的真实 latency, 看会不会随 sparsity 变化.
"""

import sys, time
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F


DEVICE = 'cuda:0'
B = 1
C_in, C_out, H, W = 64, 64, 56, 56

print("=" * 70)
print("[A] active_ratio across EXTREME sparsity range")
print("=" * 70)

# Setup
weight = torch.randn(C_out, C_in, 3, 3, device=DEVICE) * 0.1
bn_scale = torch.ones(C_out, device=DEVICE)
bn_bias = torch.zeros(C_out, device=DEVICE)

# Force fallback OFF
import catfuse.sparseflow.config as sf_config
sf_config.ENABLE_RUNTIME_FALLBACK_POLICY = False
import importlib
import catfuse.sparseflow.fused_conv_bn_lif_kernel as sfk
importlib.reload(sfk)

# Use a much wider sparsity range, focused on extreme high
sparsities = [
    0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0
]

print(f"\n{'sparsity':<12}{'actual_sp':<12}{'active_ratio':<15}{'latency_us':<15}")
print('-' * 60)

# Pre-allocate ag_mask_buf (matches default 8x16 tile)
from triton import cdiv
BH, BW = 8, 16
GH = cdiv(H, BH)
GW = cdiv(W, BW)
N_TILES = B * GH * GW
ag_buf = torch.empty(N_TILES, dtype=torch.int32, device=DEVICE)

for sp in sparsities:
    # Generate input
    if sp >= 1.0:
        x = torch.zeros(B, C_in, H, W, device=DEVICE)
    else:
        keep = 1.0 - sp
        rand = torch.rand(B, C_in, H, W, device=DEVICE)
        x = (rand < keep).float()
    actual_sp = 1.0 - x.mean().item()
    
    v = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
    
    try:
        # warmup
        for _ in range(20):
            result = sfk.sparse_fused_conv_bn_lif_forward(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3,
                ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        
        # measure (50 iter)
        t0 = time.perf_counter()
        for _ in range(50):
            result = sfk.sparse_fused_conv_bn_lif_forward(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3,
                ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) / 50 * 1e6
        
        spike_out, v_out, ms, active_ratio = result
        
        ar_str = f"{active_ratio:.4f}" if active_ratio is not None else 'N/A'
        print(f"{sp:<12.4f}{actual_sp:<12.4f}{ar_str:<15}{elapsed_us:<15.1f}")
    except Exception as e:
        print(f"{sp:<12.4f}{actual_sp:<12.4f}FAIL: {e}")


print()
print("=" * 70)
print("[B] Compare SF vs DK at the SAME sparsity range")
print("=" * 70)

# Pre-fold BN
w_fused = weight * bn_scale.view(-1, 1, 1, 1)
b_fused = bn_bias.clone()

try:
    from catfuse.sparseflow.lif_seq_kernel import lif_sequential
    use_triton = True
except ImportError:
    use_triton = False

def bench_dk(x):
    """DK single-step (matched to SF single-step)"""
    z = F.conv2d(x, w_fused, bias=b_fused, stride=1, padding=1)
    z = z.unsqueeze(0)  # add T dim, T=1 for single step
    v_in = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
    if use_triton:
        s, v_out = lif_sequential(z, v_in, tau=2.0, v_threshold=1.0, v_reset=0.0)
    return s

print(f"\n{'sparsity':<12}{'T_DK_us':<12}{'T_SF_us':<12}{'SF/DK':<10}{'active':<10}")
print('-' * 60)

for sp in sparsities:
    if sp >= 1.0:
        x = torch.zeros(B, C_in, H, W, device=DEVICE)
    else:
        rand = torch.rand(B, C_in, H, W, device=DEVICE)
        x = (rand < (1.0 - sp)).float()
    actual_sp = 1.0 - x.mean().item()
    v = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
    
    # Warmup DK
    for _ in range(20):
        bench_dk(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        bench_dk(x)
    torch.cuda.synchronize()
    t_dk = (time.perf_counter() - t0) / 50 * 1e6
    
    # Warmup SF
    try:
        for _ in range(20):
            sfk.sparse_fused_conv_bn_lif_forward(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3, ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            r = sfk.sparse_fused_conv_bn_lif_forward(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3, ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        t_sf = (time.perf_counter() - t0) / 50 * 1e6
        ar = r[3]
    except Exception as e:
        t_sf = float('nan')
        ar = None
    
    ratio = t_sf / t_dk if not (t_sf != t_sf) else float('nan')
    print(f"{actual_sp:<12.4f}{t_dk:<12.1f}{t_sf:<12.1f}{ratio:<10.2f}{ar if ar is not None else 'N/A':<10}")


print()
print("=" * 70)
print("Verdict:")
print(" [A] If active_ratio drops to <1.0 at sparsity > 0.99: prescan IS working,")
print("     just needs extreme sparsity to trigger.")
print(" [B] If T_SF drops at sparsity > 0.99 (relative to dense):")
print("     SF has real sparsity benefit but only at extreme regime.")
print(" Otherwise SF kernel is essentially dense regardless of input.")