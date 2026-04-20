"""
Test 3: Parity verification — STFusionConvBNLIF vs SpikingJelly reference.

Run from project root:
    python tests/test_03_parity.py

Compares STFusionConvBNLIF output against a pure-PyTorch reference
implementation of Conv → BN → LIF, step by step.

For the Triton sparse kernel path, we expect near-parity (spike_match > 99.9%)
due to fp16 compute in the sparse kernel vs fp32 in the reference.
For the PyTorch fallback path, we expect bit-exact spike match.

Tests:
  a) Parity at 50% spike rate (exercises active-tile path)
  b) Parity at 5% spike rate (exercises sparse path with many zero tiles)
  c) Parity at 0% (all-zero input — exercises StaticZero path)
  d) Parity across K sweep (K = 1, 2, 4, T)
"""

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

C_IN, C_OUT = 64, 128
H, W = 14, 14
T, B = 8, 2
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

print(f"Device: {device}")
print(f"Shape: T={T}, B={B}, C_in={C_IN}, C_out={C_OUT}, H={H}, W={W}")


# ============================================================
# Reference implementation: Conv → BN → LIF, step by step
# ============================================================

def reference_conv_bn_lif(x_seq, conv, bn, tau=TAU, v_th=V_TH, v_reset=V_RESET):
    """Pure PyTorch reference: Conv → BN → LIF, sequential over T.

    Args:
        x_seq: [T, B, C_in, H, W]
        conv: nn.Conv2d
        bn: nn.BatchNorm2d (eval mode)

    Returns:
        spike_seq: [T, B, C_out, H_out, W_out]
        v_final: [B, C_out, H_out, W_out]
    """
    T_ = x_seq.shape[0]
    bn.eval()

    spikes = []
    v = None

    for t in range(T_):
        z = conv(x_seq[t])
        z = bn(z)

        if v is None:
            v = torch.zeros_like(z)

        # LIF charge (decay_input=True)
        v = v + (z - (v - v_reset)) / tau

        # Fire
        spike = (v >= v_th).float()

        # Hard reset
        v = v * (1.0 - spike) + v_reset * spike

        spikes.append(spike)

    return torch.stack(spikes, dim=0), v


# ============================================================
# Setup: shared weights
# ============================================================

conv = nn.Conv2d(C_IN, C_OUT, 3, stride=1, padding=1, bias=True).to(device)
bn = nn.BatchNorm2d(C_OUT).to(device)
bn.eval()
with torch.no_grad():
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.fill_(1.0).add_(torch.rand(C_OUT, device=device) * 0.3)
    bn.weight.fill_(1.0)
    bn.bias.zero_()

class MockLIF:
    tau = TAU
    v_threshold = V_TH
    v_reset = V_RESET
    decay_input = True

lif = MockLIF()


# ============================================================
# Build STFusionConvBNLIF with same weights
# ============================================================

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

errors = []

def parity_check(label, x_seq, K=4):
    """Run parity comparison and report results."""
    try:
        # Reference
        with torch.no_grad():
            ref_spikes, ref_v = reference_conv_bn_lif(x_seq, conv, bn)

        # CATFuse-SF
        module = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=K).to(device)
        module.eval()
        with torch.no_grad():
            module.v = 0.0
            sf_spikes = module(x_seq)

        # Compare
        total = ref_spikes.numel()
        match = (ref_spikes == sf_spikes).sum().item()
        spike_match_pct = match / total * 100

        ref_rate = ref_spikes.mean().item()
        sf_rate = sf_spikes.mean().item()
        max_diff = (ref_spikes - sf_spikes).abs().max().item()

        status = "OK" if spike_match_pct > 99.9 else "FAIL"
        print(f"  [{status}] {label}")
        print(f"       K={K}, spike_match={spike_match_pct:.4f}%")
        print(f"       ref_rate={ref_rate:.4f}, sf_rate={sf_rate:.4f}")
        print(f"       max_diff={max_diff:.6f}, mismatches={total - match}/{total}")

        if spike_match_pct <= 99.9:
            errors.append(label)

        return spike_match_pct

    except Exception as e:
        import traceback
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc()
        errors.append(label)
        return 0.0


print("=" * 60)
print("Test 3: Parity verification")
print("=" * 60)

# --- Test a: 50% spike rate ---
print("\n--- a) Parity at ~50% spike rate ---")
x_50 = (torch.rand(T, B, C_IN, H, W, device=device) > 0.5).float()
print(f"       Input spike rate: {x_50.mean().item():.3f}")
parity_check("50% spike rate, K=4", x_50, K=4)

# --- Test b: 5% spike rate ---
print("\n--- b) Parity at ~5% spike rate (sparse) ---")
x_05 = (torch.rand(T, B, C_IN, H, W, device=device) > 0.95).float()
print(f"       Input spike rate: {x_05.mean().item():.3f}")
parity_check("5% spike rate, K=4", x_05, K=4)

# --- Test c: 0% (all-zero) ---
print("\n--- c) Parity at 0% (all-zero input) ---")
x_00 = torch.zeros(T, B, C_IN, H, W, device=device)
parity_check("0% spike rate, K=4", x_00, K=4)

# --- Test d: K sweep ---
print("\n--- d) K sweep at ~5% spike rate ---")
for K_val in [1, 2, 4, T]:
    parity_check(f"5% spike rate, K={K_val}", x_05, K=K_val)

# --- Test e: Fallback parity (force CPU to use PyTorch fallback) ---
print("\n--- e) PyTorch fallback parity (should be higher precision) ---")
x_50_cpu = x_50.cpu()
conv_cpu = conv.cpu()
bn_cpu = bn.cpu()
bn_cpu.eval()

with torch.no_grad():
    ref_cpu, _ = reference_conv_bn_lif(x_50_cpu, conv_cpu, bn_cpu)

module_cpu = STFusionConvBNLIF.from_sj_modules(conv_cpu, bn_cpu, lif, K=4)
module_cpu.eval()
with torch.no_grad():
    module_cpu.v = 0.0
    sf_cpu = module_cpu(x_50_cpu)

total = ref_cpu.numel()
match = (ref_cpu == sf_cpu).sum().item()
pct = match / total * 100
status = "OK" if pct > 99.99 else "FAIL"
print(f"  [{status}] CPU fallback: spike_match={pct:.4f}%, mismatches={total - match}/{total}")
if pct <= 99.99:
    errors.append("CPU fallback parity")

# --- Summary ---
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("ALL PARITY TESTS PASSED")
    sys.exit(0)
