"""
Test 21: StreamFuse kernel parity verification.

Compares STFusionConvBNLIF (StreamFuse path) vs reference
(PyTorch Conv → BN → LIF step-by-step).
"""
import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

print("=" * 70)
print("StreamFuse parity verification")
print("=" * 70)

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

class MockLIF:
    tau = 2.0; v_threshold = 1.0; v_reset = 0.0; decay_input = True

def ref_conv_bn_lif(x_seq, conv, bn, tau=2.0, v_th=1.0, v_reset=0.0):
    T = x_seq.shape[0]
    bn.eval()
    spikes, v = [], None
    for t in range(T):
        z = conv(x_seq[t])
        z = bn(z)
        if v is None:
            v = torch.zeros_like(z)
        v = v + (z - (v - v_reset)) / tau
        spike = (v >= v_th).float()
        v = v * (1.0 - spike) + v_reset * spike
        spikes.append(spike)
    return torch.stack(spikes, 0), v

configs = [
    ("C=256 H=14 T=4 sp=30%",  256, 256, 14, 4, 0.70),
    ("C=256 H=14 T=4 sp=95%",  256, 256, 14, 4, 0.95),
    ("C=512 H=7  T=4 sp=30%",  512, 512, 7,  4, 0.70),
    ("C=512 H=7  T=4 sp=95%",  512, 512, 7,  4, 0.95),
    ("C=256 H=14 T=8 sp=50%",  256, 256, 14, 8, 0.50),
    ("C=256 H=14 T=1 sp=50%",  256, 256, 14, 1, 0.50),
    ("C=256 H=14 T=4 sp=0%",   256, 256, 14, 4, 1.00),  # all zero
    ("C=256 H=14 T=4 sp=100%", 256, 256, 14, 4, 0.00),  # all one
]

B = 2
all_ok = True

for label, cin, cout, H, T, sp in configs:
    conv = nn.Conv2d(cin, cout, 3, padding=1, bias=True).to(device)
    bn = nn.BatchNorm2d(cout).to(device).eval()
    with torch.no_grad():
        bn.running_mean.normal_(0, 0.1)
        bn.running_var.fill_(1.0).add_(torch.rand(cout, device=device) * 0.3)

    x = (torch.rand(T, B, cin, H, H, device=device) > sp).float()

    # Reference
    with torch.no_grad():
        ref_spikes, ref_v = ref_conv_bn_lif(x, conv, bn)

    # StreamFuse
    sf_mod = STFusionConvBNLIF.from_sj_modules(conv, bn, MockLIF(), K=T).to(device).eval()
    sf_mod.v = 0.0
    with torch.no_grad():
        sf_spikes = sf_mod(x)

    match = (ref_spikes == sf_spikes).sum().item()
    total = ref_spikes.numel()
    pct = match / total * 100
    ref_rate = ref_spikes.mean().item()
    sf_rate = sf_spikes.mean().item()

    status = "OK" if pct > 99.0 else "FAIL"
    if pct <= 99.0:
        all_ok = False
    print(f"  [{status}] {label:30s} match={pct:.2f}% ref_rate={ref_rate:.4f} sf_rate={sf_rate:.4f}")

    if pct < 100:
        diff = (ref_spikes != sf_spikes)
        ndiff = diff.sum().item()
        # Show v difference
        if isinstance(sf_mod.v, torch.Tensor) and ref_v is not None:
            v_diff = (sf_mod.v - ref_v).abs().max().item()
            v_mean = (sf_mod.v - ref_v).abs().mean().item()
            print(f"         mismatches={ndiff}/{total} v_max_diff={v_diff:.6f} v_mean_diff={v_mean:.6f}")

print(f"\n{'=' * 70}")
print("ALL PARITY OK" if all_ok else "PARITY FAILURES DETECTED")
print("=" * 70)
