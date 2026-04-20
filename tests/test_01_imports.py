"""
Test 1: Import smoke test for CATFuse-SF merged codebase.

Run from project root:
    python tests/test_01_imports.py

Expected: all prints show OK, no ImportError.
"""

import sys
import os

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  [OK] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        errors.append(label)


print("=" * 60)
print("Test 1: Import smoke test")
print("=" * 60)

# --- catfuse core ---
print("\n--- catfuse core ---")
check("catfuse",               lambda: __import__("catfuse"))
check("catfuse.patterns",      lambda: __import__("catfuse.patterns"))
check("catfuse.substitute",    lambda: __import__("catfuse.substitute"))
check("catfuse.policy",        lambda: __import__("catfuse.policy"))

# --- catfuse.kernels ---
print("\n--- catfuse.kernels ---")
check("catfuse.kernels",       lambda: __import__("catfuse.kernels"))

# --- catfuse.sparseflow config/utils (no triton needed) ---
print("\n--- catfuse.sparseflow (config/utils) ---")
check("catfuse.sparseflow",            lambda: __import__("catfuse.sparseflow"))
check("catfuse.sparseflow.config",     lambda: __import__("catfuse.sparseflow.config"))
check("catfuse.sparseflow.registry",   lambda: __import__("catfuse.sparseflow.registry"))

# --- catfuse.sparseflow kernels (need triton) ---
print("\n--- catfuse.sparseflow kernels (need triton) ---")
check("catfuse.sparseflow.prescan",                lambda: __import__("catfuse.sparseflow.prescan"))
check("catfuse.sparseflow.sparse_conv2d_kernel",   lambda: __import__("catfuse.sparseflow.sparse_conv2d_kernel"))
check("catfuse.sparseflow.fused_conv_lif_kernel",  lambda: __import__("catfuse.sparseflow.fused_conv_lif_kernel"))
check("catfuse.sparseflow.fused_conv_bn_lif_kernel", lambda: __import__("catfuse.sparseflow.fused_conv_bn_lif_kernel"))

# --- catfuse.sparseflow ops ---
print("\n--- catfuse.sparseflow.ops ---")
check("catfuse.sparseflow.ops.sparse_conv2d",          lambda: __import__("catfuse.sparseflow.ops.sparse_conv2d"))
check("catfuse.sparseflow.ops.sparse_fused_conv_lif",  lambda: __import__("catfuse.sparseflow.ops.sparse_fused_conv_lif"))
check("catfuse.sparseflow.ops.st_fusion_conv_bn_lif",  lambda: __import__("catfuse.sparseflow.ops.st_fusion_conv_bn_lif"))
check("catfuse.sparseflow.ops.static_zero_conv2d",     lambda: __import__("catfuse.sparseflow.ops.static_zero_conv2d"))

# --- policy engine ---
print("\n--- policy engine ---")
def _test_policy():
    from catfuse.policy import get_policy, SpatialBackend
    row = get_policy("Conv3x3_BN_LIF", C_in=64, C_out=64, H=56, W=56)
    assert row.spatial_backend == SpatialBackend.DENSE_KEEP, f"expected DenseKeep, got {row.spatial_backend}"
    row2 = get_policy("Conv3x3_BN_LIF", C_in=512, C_out=512, H=7, W=7)
    assert row2.spatial_backend == SpatialBackend.SPARSE_FLOW, f"expected SparseFlow, got {row2.spatial_backend}"
check("policy lookup (compute-bound → DenseKeep)", _test_policy)

# --- pattern registry ---
print("\n--- pattern registry ---")
def _test_registry():
    from catfuse.patterns import PATTERN_REGISTRY
    assert "PartialFusionConvBNLIF" in PATTERN_REGISTRY
    assert "STFusionConvBNLIF" in PATTERN_REGISTRY, "STFusionConvBNLIF not registered"
check("PATTERN_REGISTRY has STFusionConvBNLIF", _test_registry)

# --- summary ---
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} import(s) broken: {errors}")
    sys.exit(1)
else:
    print("ALL IMPORTS OK")
    sys.exit(0)
