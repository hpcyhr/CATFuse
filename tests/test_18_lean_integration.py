"""
Test 18: Lean path integration — parity + wall-clock.

Verifies:
  a) Lean path is activated for STFusionConvBNLIF
  b) Parity with reference (SJ Conv→BN→LIF)
  c) Wall-clock comparison: old STFusion vs lean STFusion vs SJ baseline
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.substitute import substitute_sf, print_routing_table

device = 'cuda:0'
T, B = 4, 2
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

torch.backends.cudnn.benchmark = False

print("=" * 70)
print(f"Test 18: Lean path integration (T={T}, B={B})")
print("=" * 70)

# ============================================================
# 1. Check lean path availability
# ============================================================
print("\n--- 1. Lean path availability ---")
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import (
    STFusionConvBNLIF, _LEAN_AVAILABLE, _KERNEL_AVAILABLE
)
print(f"  _KERNEL_AVAILABLE = {_KERNEL_AVAILABLE}")
print(f"  _LEAN_AVAILABLE = {_LEAN_AVAILABLE}")

# ============================================================
# 2. Layer-wise parity (lean path vs reference)
# ============================================================
print("\n--- 2. Layer-wise parity ---")

class MockLIF:
    tau = 2.0; v_threshold = 1.0; v_reset = 0.0; decay_input = True

configs = [
    ("C=64  H=32 s=1", 64, 64, 32, 1),   # DenseKeep (cuDNN fallback)
    ("C=128 H=16 s=1", 128, 128, 16, 1),  # DenseKeep
    ("C=256 H=8  s=1", 256, 256, 8, 1),   # SparseFlow (lean path!)
    ("C=512 H=4  s=1", 512, 512, 4, 1),   # SparseFlow (lean path!)
    ("C=64  H=32 s=2", 64, 128, 32, 2),   # stride=2 → fallback
]

for label, cin, cout, H, stride in configs:
    conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True).to(device)
    bn = nn.BatchNorm2d(cout).to(device).eval()
    with torch.no_grad():
        bn.running_mean.normal_(0, 0.1)
        bn.running_var.fill_(1.0).add_(torch.rand(cout, device=device) * 0.3)

    sf_mod = STFusionConvBNLIF.from_sj_modules(conv, bn, MockLIF(), K=4).to(device).eval()

    x_test = (torch.rand(T, B, cin, H, H, device=device) > 0.7).float()

    # Reference
    bn.eval()
    ref_spikes = []
    v_ref = torch.zeros(B, cout, (H + 2 - 3) // stride + 1,
                         (H + 2 - 3) // stride + 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        for t in range(T):
            z = conv(x_test[t])
            z = bn(z)
            v_ref = v_ref + (z - (v_ref - 0.0)) / 2.0
            s = (v_ref >= 1.0).float()
            v_ref = v_ref * (1.0 - s) + 0.0 * s
            ref_spikes.append(s)
    ref_spikes = torch.stack(ref_spikes, dim=0)

    # STFusionConvBNLIF
    sf_mod.v = 0.0
    with torch.no_grad():
        sf_spikes = sf_mod(x_test)

    match = (ref_spikes == sf_spikes).sum().item()
    total = ref_spikes.numel()
    pct = match / total * 100
    lean = sf_mod._lean_ready
    status = "OK" if pct > 99.9 else "FAIL"
    path = "lean" if lean else ("kernel" if stride == 1 else "fallback")
    print(f"  [{status}] {label:18s} match={pct:.2f}% path={path} out_rate={ref_spikes.mean():.4f}")

# ============================================================
# 3. Routing table check
# ============================================================
print("\n--- 3. Policy routing (updated threshold C>=256) ---")
net = sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd.get('model', sd), strict=False)
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

fused_net, stats = substitute_sf(net, T=T, verbose=False)
print_routing_table(stats)

sf_count = sum(1 for _, m in fused_net.named_modules() if 'STFusionConvBNLIF' in type(m).__name__)
pf_count = sum(1 for _, m in fused_net.named_modules() if 'PartialFusionConvBNLIF' in type(m).__name__)
print(f"\n  STFusionConvBNLIF: {sf_count}, PartialFusionConvBNLIF: {pf_count}")

# ============================================================
# 4. E2E wall-clock
# ============================================================
print("\n--- 4. End-to-end wall-clock ---")
x = torch.rand(T, B, 3, 32, 32, device=device)

def bench(model, x_in, label):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x_in)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    med = statistics.median(times)
    print(f"  [{label}] {med:.2f} ms/iter (all={[f'{t:.2f}' for t in times]})")
    return med

results = {}

# SJ baseline
results['SJ'] = bench(net, x, "SJ baseline")

# CATFuse-Dense (all DenseKeep)
import catfuse.policy as pm
_orig = pm.classify_shape_regime
pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
net_dense, _ = substitute_sf(net, T=T)
pm.classify_shape_regime = _orig
net_dense = net_dense.to(device).eval()
functional.set_step_mode(net_dense, 'm')
results['Dense'] = bench(net_dense, x, "CATFuse-Dense")

# CATFuse-SF (lean path)
fused_net = fused_net.to(device).eval()
functional.set_step_mode(fused_net, 'm')
results['SF-lean'] = bench(fused_net, x, "CATFuse-SF-lean")

# Summary
print(f"\n{'=' * 70}")
base = results['SJ']
print(f"  {'Config':<20s} {'ms/iter':>10s} {'Speedup':>10s}")
print("  " + "-" * 42)
for name, ms in results.items():
    print(f"  {name:<20s} {ms:>10.2f} {base/ms:>9.2f}x")
print("=" * 70)
