"""
Test 2: STFusionConvBNLIF construction + forward pass.

Run from project root:
    python tests/test_02_forward.py

Tests:
  a) Construct STFusionConvBNLIF from SJ Conv2d + BN + LIFNode
  b) Run forward with random input [T=4, B=2, C=64, H=16, W=16]
  c) Run forward with sparse input (spike rate ~5%) to exercise sparse path
  d) Run forward with all-zero input to exercise StaticZero path
  e) Verify output shape and dtype
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# --- Setup: create SJ-compatible modules ---
torch.manual_seed(42)

C_IN, C_OUT = 64, 128
H, W = 16, 16
T, B = 4, 2
K = 2  # TimeBlock size

# Conv2d
conv = nn.Conv2d(C_IN, C_OUT, 3, stride=1, padding=1, bias=True).to(device)

# BatchNorm2d (set to eval mode with some running stats)
bn = nn.BatchNorm2d(C_OUT).to(device)
bn.eval()
# Simulate trained BN stats
with torch.no_grad():
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.fill_(1.0).add_(torch.rand(C_OUT, device=device) * 0.5)
    bn.weight.fill_(1.0)
    bn.bias.zero_()

# LIF node (mock — just need tau, v_threshold, v_reset attributes)
class MockLIF:
    tau = 2.0
    v_threshold = 1.0
    v_reset = 0.0
    decay_input = True

lif = MockLIF()

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}")
        return result
    except Exception as e:
        import traceback
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        errors.append(label)
        return None


print("=" * 60)
print("Test 2: STFusionConvBNLIF forward pass")
print("=" * 60)

# --- Test a: Construction ---
print("\n--- a) Construction from SJ modules ---")

def _test_construct():
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
    module = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=K)
    module = module.to(device)
    assert module.weight.shape == (C_OUT, C_IN, 3, 3)
    assert module.bn_scale is not None
    assert module.bn_bias_folded is not None
    assert module.K == K
    print(f"       {module}")
    return module

module = check("Construct STFusionConvBNLIF", _test_construct)
if module is None:
    print("\nConstruction failed — skipping remaining tests")
    sys.exit(1)

# --- Test b: Forward with random input ---
print("\n--- b) Forward (random input, spike_rate ~50%) ---")

def _test_forward_random():
    x = (torch.rand(T, B, C_IN, H, W, device=device) > 0.5).float()
    spike_rate = x.mean().item()
    print(f"       Input spike rate: {spike_rate:.3f}")

    # Reset state
    module.v = 0.0
    out = module(x)

    assert out.shape == (T, B, C_OUT, H, W), f"Shape mismatch: {out.shape}"
    assert out.dtype == torch.float32, f"Dtype mismatch: {out.dtype}"
    out_rate = out.mean().item()
    print(f"       Output shape: {out.shape}")
    print(f"       Output spike rate: {out_rate:.4f}")
    assert 0.0 <= out_rate <= 1.0, f"Spike rate out of range: {out_rate}"
    return out

check("Forward random input", _test_forward_random)

# --- Test c: Forward with sparse input ---
print("\n--- c) Forward (sparse input, spike_rate ~5%) ---")

def _test_forward_sparse():
    x = (torch.rand(T, B, C_IN, H, W, device=device) > 0.95).float()
    spike_rate = x.mean().item()
    print(f"       Input spike rate: {spike_rate:.3f}")

    module.v = 0.0
    out = module(x)

    assert out.shape == (T, B, C_OUT, H, W)
    out_rate = out.mean().item()
    print(f"       Output spike rate: {out_rate:.4f}")
    return out

check("Forward sparse input", _test_forward_sparse)

# --- Test d: Forward with all-zero input ---
print("\n--- d) Forward (all-zero input — StaticZero path) ---")

def _test_forward_zero():
    x = torch.zeros(T, B, C_IN, H, W, device=device)

    module.v = 0.0
    out = module(x)

    assert out.shape == (T, B, C_OUT, H, W)
    out_rate = out.mean().item()
    print(f"       Output spike rate: {out_rate:.4f}")
    # Even with zero input, LIF might fire due to bias → BN → threshold crossing
    return out

check("Forward all-zero input", _test_forward_zero)

# --- Test e: StateCarry — v persists across calls ---
print("\n--- e) StateCarry: v persists across forward calls ---")

def _test_state_carry():
    x = (torch.rand(T, B, C_IN, H, W, device=device) > 0.5).float()

    module.v = 0.0
    _ = module(x)
    v_after_first = module.v.clone()

    _ = module(x)
    v_after_second = module.v.clone()

    # v should be different after two calls (state carried)
    diff = (v_after_first - v_after_second).abs().max().item()
    print(f"       v diff between calls: {diff:.6f}")
    assert diff > 0, "v should change between calls (StateCarry)"
    return True

check("StateCarry persists", _test_state_carry)

# --- Summary ---
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("ALL FORWARD TESTS PASSED")
    sys.exit(0)
