"""
Test 6: End-to-end wall-clock benchmark — SJ vs CATFuse-SF.

Configs:
  A) SJ baseline:    original SpikingJelly multi-step
  B) CATFuse-Dense:  substitute_sf force all DenseKeep (cuDNN conv + Triton LIF)
  C) CATFuse-SF:     substitute_sf with policy routing (DenseKeep shallow + SparseFlow deep)

Protocol: 20 warmup, 50 timed iters, 5 repeats → report median.
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
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


def bench_model(model, x, label, n_warmup=N_WARMUP, n_iter=N_ITER, n_repeat=N_REPEAT):
    """Benchmark forward latency with proper CUDA sync."""
    model.eval()
    repeat_times = []

    for rep in range(n_repeat):
        # Warmup
        for _ in range(n_warmup):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize(device)

        # Timed
        t0 = time.perf_counter()
        torch.cuda.synchronize(device)
        for _ in range(n_iter):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        ms_per_iter = (t1 - t0) / n_iter * 1000
        repeat_times.append(ms_per_iter)

    median_ms = statistics.median(repeat_times)
    std_ms = statistics.stdev(repeat_times) if len(repeat_times) > 1 else 0
    print(f"  [{label}] {median_ms:.2f} ms/iter (median of {n_repeat}, "
          f"std={std_ms:.2f}, all={[f'{t:.2f}' for t in repeat_times]})")
    return median_ms


print("=" * 70)
print(f"Wall-clock benchmark: SEW-ResNet18, T={T}, B={B}, CIFAR-10 (32x32)")
print(f"Protocol: {N_WARMUP} warmup, {N_ITER} timed, {N_REPEAT} repeats")
print("=" * 70)

# Build base model
net = sew_resnet.sew_resnet18(
    pretrained=False, num_classes=10, cnf='ADD',
    spiking_neuron=neuron.LIFNode, tau=2.0
)
ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd.get('model', sd), strict=False)
    print(f"Loaded: {ckpt}")

x = torch.rand(T, B, 3, 32, 32, device=device)

results = {}

# --- Config A: SJ baseline ---
print("\n--- Config A: SJ baseline ---")
net_a = net.to(device).eval()
functional.set_step_mode(net_a, 'm')
results['SJ baseline'] = bench_model(net_a, x, "SJ baseline")

# --- Config B: CATFuse-Dense (force all DenseKeep) ---
print("\n--- Config B: CATFuse-Dense (all PartialFusionConvBNLIF) ---")
# Use substitute_sf but prevent SparseFlow by temporarily setting policy to return DenseKeep
# Simplest: patch policy to always return DenseKeep
import catfuse.policy as policy_mod
_orig_classify = policy_mod.classify_shape_regime
policy_mod.classify_shape_regime = lambda *a, **kw: "compute_bound"  # force DenseKeep

net_b, stats_b = substitute_sf(net, T=T, verbose=False)
policy_mod.classify_shape_regime = _orig_classify  # restore

sf_b = sum(1 for _, m in net_b.named_modules() if 'STFusionConvBNLIF' in type(m).__name__)
pf_b = sum(1 for _, m in net_b.named_modules() if 'PartialFusionConvBNLIF' in type(m).__name__)
print(f"  Modules: ST={sf_b}, PF={pf_b}")

net_b = net_b.to(device).eval()
functional.set_step_mode(net_b, 'm')
results['CATFuse-Dense'] = bench_model(net_b, x, "CATFuse-Dense")

# --- Config C: CATFuse-SF (policy routing) ---
print("\n--- Config C: CATFuse-SF (policy routing) ---")
net_c, stats_c = substitute_sf(net, T=T, verbose=False)
sf_c = sum(1 for _, m in net_c.named_modules() if 'STFusionConvBNLIF' in type(m).__name__)
pf_c = sum(1 for _, m in net_c.named_modules() if 'PartialFusionConvBNLIF' in type(m).__name__)
print(f"  Modules: ST={sf_c} (SparseFlow), PF={pf_c} (DenseKeep)")

net_c = net_c.to(device).eval()
functional.set_step_mode(net_c, 'm')
results['CATFuse-SF'] = bench_model(net_c, x, "CATFuse-SF")

# --- Config D: CATFuse-SF all sparse ---
print("\n--- Config D: CATFuse-SF-All (force all SparseFlow) ---")
net_d, stats_d = substitute_sf(net, T=T, verbose=False, force_sparse=True)
sf_d = sum(1 for _, m in net_d.named_modules() if 'STFusionConvBNLIF' in type(m).__name__)
print(f"  Modules: ST={sf_d} (all SparseFlow)")

net_d = net_d.to(device).eval()
functional.set_step_mode(net_d, 'm')
results['CATFuse-SF-All'] = bench_model(net_d, x, "CATFuse-SF-All")

# --- Summary ---
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
baseline = results['SJ baseline']
print(f"  {'Config':<20s} {'ms/iter':>10s} {'Speedup':>10s}")
print("  " + "-" * 42)
for name, ms in results.items():
    speedup = baseline / ms if ms > 0 else 0
    print(f"  {name:<20s} {ms:>10.2f} {speedup:>9.2f}x")

print("\n" + "=" * 70)
