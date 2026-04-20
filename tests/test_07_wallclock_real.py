"""
Test 7: Wall-clock with realistic spike sparsity.

Uses actual CIFAR-10 test images through the trained SEW-ResNet18
to get real spike patterns, then benchmarks with those.
Also profiles per-layer spike rates to confirm sparsity.
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.substitute import substitute_sf

device = 'cuda:0'
T, B = 4, 2
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 5

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("=" * 70)
print(f"Wall-clock benchmark with REAL CIFAR-10 data")
print(f"T={T}, B={B}, {N_WARMUP} warmup, {N_ITER} timed, {N_REPEAT} repeats")
print("=" * 70)

# Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
try:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)
    # Get one batch
    images, labels = next(iter(testloader))
    # Expand to [T, B, 3, 32, 32] by repeating
    x_real = images.unsqueeze(0).repeat(T, 1, 1, 1, 1).to(device)
    print(f"  Real CIFAR-10 input: {x_real.shape}")
except Exception as e:
    print(f"  CIFAR-10 not available ({e}), using synthetic low-sparsity input")
    x_real = torch.rand(T, B, 3, 32, 32, device=device) * 0.3
    print(f"  Synthetic input: {x_real.shape}")

# Build model
net = sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd.get('model', sd), strict=False)
    print(f"  Loaded: {ckpt}")
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

# Profile spike rates per layer
print("\n--- Layer-wise spike rate profiling ---")
hooks = []
spike_rates = {}

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            rate = output.mean().item()
            sparsity = 1.0 - rate
            spike_rates[name] = {'rate': rate, 'sparsity': sparsity}
    return hook_fn

for name, mod in net.named_modules():
    if isinstance(mod, neuron.LIFNode):
        hooks.append(mod.register_forward_hook(make_hook(name)))

functional.reset_net(net)
with torch.no_grad():
    _ = net(x_real)

for h in hooks:
    h.remove()

print(f"  {'Layer':<40s} {'Spike rate':>12s} {'Sparsity':>12s}")
print("  " + "-" * 66)
for name, info in spike_rates.items():
    print(f"  {name:<40s} {info['rate']:>11.4f}% {info['sparsity']:>11.2f}%")


# Benchmark function
def bench_model(model, x, label):
    model.eval()
    times = []
    for rep in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        torch.cuda.synchronize(device)
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) / N_ITER * 1000)

    med = statistics.median(times)
    print(f"  [{label}] {med:.2f} ms/iter (all={[f'{t:.2f}' for t in times]})")
    return med


results = {}

# A: SJ baseline
print("\n--- SJ baseline ---")
results['SJ'] = bench_model(net, x_real, "SJ baseline")

# B: CATFuse-Dense
print("\n--- CATFuse-Dense ---")
import catfuse.policy as pm
_orig = pm.classify_shape_regime
pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
net_b, _ = substitute_sf(net, T=T)
pm.classify_shape_regime = _orig
net_b = net_b.to(device).eval()
functional.set_step_mode(net_b, 'm')
results['Dense'] = bench_model(net_b, x_real, "CATFuse-Dense")

# C: CATFuse-SF (policy)
print("\n--- CATFuse-SF (policy) ---")
net_c, _ = substitute_sf(net, T=T)
net_c = net_c.to(device).eval()
functional.set_step_mode(net_c, 'm')
results['SF'] = bench_model(net_c, x_real, "CATFuse-SF")

# Summary
print("\n" + "=" * 70)
base = results['SJ']
print(f"  {'Config':<20s} {'ms/iter':>10s} {'Speedup':>10s}")
print("  " + "-" * 42)
for name, ms in results.items():
    print(f"  {name:<20s} {ms:>10.2f} {base/ms:>9.2f}x")
print("=" * 70)
