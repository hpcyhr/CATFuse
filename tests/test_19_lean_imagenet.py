"""
Test 19: Lean path on ImageNet scale (224x224).

At ImageNet scale, deep layers have H=14/7 — SparseFlow's sweet spot.
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.substitute import substitute_sf

device = 'cuda:0'
T, B = 4, 1
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

torch.backends.cudnn.benchmark = False

print("=" * 70)
print(f"Test 19: Lean path ImageNet scale (224x224, T={T}, B={B})")
print("=" * 70)

net = sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

x = torch.rand(T, B, 3, 224, 224, device=device)

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

# CATFuse-Dense
import catfuse.policy as pm
_orig = pm.classify_shape_regime
pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
net_dense, _ = substitute_sf(net, T=T)
pm.classify_shape_regime = _orig
net_dense = net_dense.to(device).eval()
functional.set_step_mode(net_dense, 'm')
results['Dense'] = bench(net_dense, x, "CATFuse-Dense")

# CATFuse-SF-lean
net_sf, stats = substitute_sf(net, T=T)
from catfuse.substitute import print_routing_table
print_routing_table(stats)
net_sf = net_sf.to(device).eval()
functional.set_step_mode(net_sf, 'm')
results['SF-lean'] = bench(net_sf, x, "CATFuse-SF-lean")

# Summary
print(f"\n{'=' * 70}")
base = results['SJ']
print(f"  {'Config':<20s} {'ms/iter':>10s} {'Speedup':>10s}")
print("  " + "-" * 42)
for name, ms in results.items():
    print(f"  {name:<20s} {ms:>10.2f} {base/ms:>9.2f}x")
print("=" * 70)
