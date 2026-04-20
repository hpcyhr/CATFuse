"""
Test 8: ImageNet-scale diagnostic + wall-clock.

224x224 input → feature maps H=112,56,28,14,7
SparseFlow's sweet spot is H>=28.

Since we don't have ImageNet pretrained checkpoint, we:
  1. Use random init SEW-ResNet18 with ImageNet input shape
  2. First run forward to collect actual spike rates at each layer
  3. Then benchmark with the same input

B=1 to simulate latency-sensitive inference.
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.substitute import substitute_sf

device = 'cuda:0'
T, B = 4, 1
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 5

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("=" * 70)
print(f"ImageNet-scale benchmark: 224x224, T={T}, B={B}")
print("=" * 70)

# Build model (ImageNet scale = 1000 classes, input 3x224x224)
net = sew_resnet.sew_resnet18(
    pretrained=False, num_classes=1000, cnf='ADD',
    spiking_neuron=neuron.LIFNode, tau=2.0
)
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

# Use synthetic input that mimics real spike patterns:
# After first conv+bn+lif, spike rate is typically 5-30%
x = torch.rand(T, B, 3, 224, 224, device=device)

# ============================================================
# 1. Profile spike rates
# ============================================================
print("\n--- 1. Spike rate profiling (random init) ---")

spike_info = {}
hooks = []
for name, mod in net.named_modules():
    if isinstance(mod, neuron.LIFNode):
        def make_hook(n):
            def hook_fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    rate = out.mean().item()
                    shape = list(out.shape)
                    h = shape[3] if len(shape) == 5 else shape[2] if len(shape) == 4 else 0
                    spike_info[n] = {'rate': rate, 'shape': shape, 'H': h}
            return hook_fn
        hooks.append(mod.register_forward_hook(make_hook(name)))

functional.reset_net(net)
with torch.no_grad():
    _ = net(x)
for h in hooks:
    h.remove()

print(f"  {'Layer':<35s} {'Shape':<22s} {'H':>4s} {'Spike%':>8s} {'Sparsity%':>10s}")
print("  " + "-" * 82)
for name, info in spike_info.items():
    rate_pct = info['rate'] * 100
    sparsity = (1 - info['rate']) * 100
    shape_s = 'x'.join(str(s) for s in info['shape'])
    print(f"  {name:<35s} {shape_s:<22s} {info['H']:>4d} {rate_pct:>7.2f}% {sparsity:>9.1f}%")

# ============================================================
# 2. Single-layer micro-benchmark at ImageNet shapes
# ============================================================
print("\n--- 2. Single-layer micro-benchmark (ImageNet shapes) ---")

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

class MockLIF:
    tau = 2.0; v_threshold = 1.0; v_reset = 0.0; decay_input = True

configs = [
    ("stem:  C=64,  H=112", 3,   64,  112, 7, 2),  # stem conv7x7 s2
    ("layer1: C=64,  H=56",  64,  64,  56,  3, 1),
    ("layer2: C=128, H=28",  64,  128, 28,  3, 1),  # after stride-2 downsample
    ("layer2: C=128, H=28",  128, 128, 28,  3, 1),
    ("layer3: C=256, H=14",  128, 256, 14,  3, 1),
    ("layer3: C=256, H=14",  256, 256, 14,  3, 1),
    ("layer4: C=512, H=7",   256, 512, 7,   3, 1),
    ("layer4: C=512, H=7",   512, 512, 7,   3, 1),
]

print(f"  {'Config':<28s} {'Sparsity':>8s}  {'cuDNN':>8s}  {'SF':>8s}  {'Ratio':>7s}  {'Winner'}")
print("  " + "-" * 85)

for label, cin, cout, H, ksize, stride in configs:
    if ksize == 7 or stride == 2:
        # SparseFlow kernel only supports 3x3 s=1, skip stem and downsample
        continue
    
    conv = nn.Conv2d(cin, cout, ksize, stride=stride, padding=ksize//2, bias=True).to(device)
    bn = nn.BatchNorm2d(cout).to(device).eval()
    sf_layer = STFusionConvBNLIF.from_sj_modules(conv, bn, MockLIF(), K=2).to(device).eval()

    for sp in [50, 80, 90, 95, 99]:
        thresh = sp / 100.0
        x_test = (torch.rand(2, B, cin, H, H, device=device) > thresh).float()
        actual_sp = (1 - x_test.mean().item()) * 100

        # cuDNN: Conv + BN (no LIF, just the compute part)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                for t in range(2):
                    z = conv(x_test[t])
                    z = bn(z)
        torch.cuda.synchronize()
        cudnn_ms = (time.perf_counter() - t0) / 200 * 1000

        # STFusionConvBNLIF: includes prescan + sparse conv + BN + LIF
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            sf_layer.v = 0.0
            with torch.no_grad():
                _ = sf_layer(x_test)
        torch.cuda.synchronize()
        sf_ms = (time.perf_counter() - t0) / 200 * 1000

        ratio = cudnn_ms / sf_ms if sf_ms > 0 else 0
        winner = "SF" if ratio > 1.0 else "cuDNN"
        print(f"  {label:<28s} {sp:>6d}%  {cudnn_ms:>7.2f}ms {sf_ms:>7.2f}ms  {ratio:>6.2f}x  {winner}")

# ============================================================
# 3. End-to-end wall-clock
# ============================================================
print("\n--- 3. End-to-end wall-clock ---")

def bench(model, x_in, label):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.synchronize()
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

# A: SJ baseline
print("\n  Config A: SJ baseline")
results['SJ'] = bench(net, x, "SJ")

# B: CATFuse-Dense
print("  Config B: CATFuse-Dense")
import catfuse.policy as pm
_orig = pm.classify_shape_regime
pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
net_b, _ = substitute_sf(net, T=T)
pm.classify_shape_regime = _orig
net_b = net_b.to(device).eval()
functional.set_step_mode(net_b, 'm')
results['Dense'] = bench(net_b, x, "Dense")

# C: CATFuse-SF (policy)
print("  Config C: CATFuse-SF")
net_c, _ = substitute_sf(net, T=T)
net_c = net_c.to(device).eval()
functional.set_step_mode(net_c, 'm')
results['SF'] = bench(net_c, x, "SF")

# Summary
print("\n" + "=" * 70)
base = results['SJ']
print(f"  {'Config':<20s} {'ms/iter':>10s} {'Speedup':>10s}")
print("  " + "-" * 42)
for name, ms in results.items():
    print(f"  {name:<20s} {ms:>10.2f} {base/ms:>9.2f}x")
print("=" * 70)
