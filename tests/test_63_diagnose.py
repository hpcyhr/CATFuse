"""
test_63_diagnose.py — 诊断 test_63 为什么 SF 时间不变

3 个问题需要回答:
  1. SF 是否每次都走 fallback 路径(dense)?
  2. fallback_ratio 默认值是多少?是否需要调整?
  3. 真正的 SF sparse path 的延迟是多少?对 sparsity 敏感吗?

Strategy:
  - 用 return_avg_active_ratio=True 看每次 active_ratio
  - 看 FALLBACK_RATIO 默认值
  - 强制禁用 fallback (ENABLE_RUNTIME_FALLBACK_POLICY=False) 跑真 SF kernel
"""

import sys, time, statistics
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F


DEVICE = 'cuda:0'
T_TIME = 4
B = 1
N_WARMUP = 10
N_ITER = 50

# ============================================================
# 1. 看默认配置
# ============================================================

print("=" * 70)
print("[1] CONFIG INSPECTION")
print("=" * 70)

from catfuse.sparseflow.config import ENABLE_RUNTIME_FALLBACK_POLICY
from catfuse.sparseflow.sparse_conv2d_kernel import FALLBACK_RATIO

print(f"ENABLE_RUNTIME_FALLBACK_POLICY: {ENABLE_RUNTIME_FALLBACK_POLICY}")
print(f"FALLBACK_RATIO: {FALLBACK_RATIO}")
print()


# ============================================================
# 2. 用 return_avg_active_ratio 看每次 active_ratio
# ============================================================

print("=" * 70)
print("[2] PROBE active_ratio AT DIFFERENT SPARSITIES")
print("=" * 70)

from catfuse.sparseflow.fused_conv_bn_lif_kernel import sparse_fused_conv_bn_lif_forward

# Setup: SEW-RN18 layer1.0.conv1 shape
C_in, C_out, H, W = 64, 64, 56, 56
weight = torch.randn(C_out, C_in, 3, 3, device=DEVICE) * 0.1
bn_scale = torch.ones(C_out, device=DEVICE)
bn_bias = torch.zeros(C_out, device=DEVICE)

print(f"\nShape: C_in={C_in}, C_out={C_out}, H={H}, W={W}")
print(f"\n{'sparsity':<12}{'actual_sp':<12}{'active_ratio':<15}{'time_us':<10}{'note'}")
print("-" * 80)

for sp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    keep = 1.0 - sp
    rand = torch.rand(B, C_in, H, W, device=DEVICE)
    x = (rand < keep).float()
    actual_sp = 1.0 - x.mean().item()
    
    v = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
    
    # Single call with return_avg_active_ratio=True
    try:
        result = sparse_fused_conv_bn_lif_forward(
            x, v, weight,
            bias=None,
            bn_scale=bn_scale,
            bn_bias=bn_bias,
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            kernel_size=3,
            return_avg_active_ratio=True,
        )
        # Result is (sp, vn, ms_or_zero, avg_active_ratio)
        if len(result) == 4:
            spike_out, v_out, ms, active_ratio = result
        else:
            spike_out, v_out, ms = result
            active_ratio = None
        
        # Check if SF went through fallback (zero ms means fallback or skipped)
        note = "FALLBACK (dense)" if ms == 0.0 else f"sparse path (ms={ms:.3f})"
    except Exception as e:
        active_ratio = None
        note = f"FAIL: {e}"
    
    print(f"{sp:<12.3f}{actual_sp:<12.3f}{active_ratio if active_ratio is not None else 'N/A':<15}{'-':<10}{note}")


# ============================================================
# 3. 测真 SF kernel (强制禁用 fallback)
# ============================================================

print()
print("=" * 70)
print("[3] BENCHMARK WITH FALLBACK DISABLED (test true SF kernel)")
print("=" * 70)

# Disable fallback at runtime by monkey-patching
import catfuse.sparseflow.config as sf_config
sf_config.ENABLE_RUNTIME_FALLBACK_POLICY = False
print(f"\nMonkey-patched: ENABLE_RUNTIME_FALLBACK_POLICY = {sf_config.ENABLE_RUNTIME_FALLBACK_POLICY}")

# Need to also reload the module that imported the constant
# (the constant was imported as `from ... import ENABLE_RUNTIME_FALLBACK_POLICY` so
# changing config doesn't affect the already-loaded value in fused_conv_bn_lif_kernel.py)
import importlib
import catfuse.sparseflow.fused_conv_bn_lif_kernel as sf_kernel
importlib.reload(sf_kernel)

from catfuse.sparseflow.fused_conv_bn_lif_kernel import sparse_fused_conv_bn_lif_forward as sf_fwd_no_fallback

print(f"\n{'sparsity':<12}{'actual_sp':<12}{'time_us':<12}{'active_ratio':<15}")
print("-" * 60)

ag_buf = None  # reuse buffer across calls
for sp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    keep = 1.0 - sp
    rand = torch.rand(B, C_in, H, W, device=DEVICE)
    x = (rand < keep).float()
    actual_sp = 1.0 - x.mean().item()
    
    v = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
    
    try:
        # Allocate ag_mask_buf once per shape
        if ag_buf is None:
            from triton import cdiv
            BH, BW = 8, 16  # default tile
            GH = cdiv(H, BH)
            GW = cdiv(W, BW)
            N_TILES = B * GH * GW
            ag_buf = torch.empty(N_TILES, dtype=torch.int32, device=DEVICE)
        
        # Warmup
        for _ in range(N_WARMUP):
            sf_fwd_no_fallback(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3,
                ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        
        # Measure
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            result = sf_fwd_no_fallback(
                x, v, weight, bias=None,
                bn_scale=bn_scale, bn_bias=bn_bias,
                tau=2.0, v_threshold=1.0, v_reset=0.0,
                kernel_size=3,
                ag_mask_buf=ag_buf,
                return_avg_active_ratio=True,
            )
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - t0) / N_ITER * 1e6
        
        spike_out, v_out, ms, active_ratio = result
    except Exception as e:
        elapsed_us = float('nan')
        active_ratio = None
        print(f"{sp:<12.3f}{actual_sp:<12.3f}FAIL: {e}")
        continue
    
    print(f"{sp:<12.3f}{actual_sp:<12.3f}{elapsed_us:<12.1f}{active_ratio if active_ratio is not None else 'N/A':<15}")


print()
print("=" * 70)
print("Analysis:")
print(" - If [2] shows 'FALLBACK (dense)' for all sparsities: prescan threshold is wrong")
print(" - If [3] shows time varies with sparsity: real SF kernel IS sparsity-aware")
print(" - If [3] shows time DOES NOT vary: SF kernel is essentially dense even when prescan triggers")