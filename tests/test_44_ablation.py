"""Test 44: Ablation — contribution of each CATFuse component."""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

def bench_fn(fn):
    """Benchmark a callable (no args)."""
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER): fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

def make_fn(model, x):
    def fn():
        functional.reset_net(model)
        for m in model.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): return model(x)
    return fn

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")
print("=" * 100)
print("Ablation: Progressive component enablement (T=4, B=2, CIFAR-10)")
print("=" * 100)
print(f"{'Network':<13s} {'single-step':>12s} {'SJ multi':>10s} {'SJ cupy':>10s} {'Dense':>10s} {'Full SF':>10s} {'single/SF':>10s}")
print("-" * 80)

from catfuse.substitute import substitute_sf
import catfuse.policy as pm

configs = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 3, 32),
]

for name, build_fn, C, H in configs:
    x = torch.rand(T, B, C, H, H, device=device)

    # (1) Single-step SJ — worst case, no temporal batching
    net1 = build_fn().to(device).eval()
    functional.set_step_mode(net1, 's')
    def single_fn(net=net1, x=x):
        functional.reset_net(net)
        with torch.no_grad():
            for t in range(T):
                _ = net(x[t])
    t1 = bench_fn(single_fn)

    # (2) SJ multi-step torch — BatchFold for spatial ops
    net2 = build_fn().to(device).eval()
    functional.set_step_mode(net2, 'm')
    t2 = bench_fn(make_fn(net2, x))

    # (3) SJ multi-step cupy — BatchFold + optimized LIF kernel
    net3 = build_fn().to(device).eval()
    functional.set_step_mode(net3, 'm')
    functional.set_backend(net3, 'cupy')
    t3 = bench_fn(make_fn(net3, x))

    # (4) CATFuse Dense only — BN fold + lif_seq_kernel, no SparseFlow
    _orig = pm.classify_shape_regime
    pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
    net4 = build_fn().to(device).eval()
    functional.set_step_mode(net4, 'm')
    net4d, _ = substitute_sf(net4, T=T)
    net4d = net4d.to(device).eval()
    functional.set_step_mode(net4d, 'm')
    pm.classify_shape_regime = _orig
    t4 = bench_fn(make_fn(net4d, x))

    # (5) Full CATFuse-SF — Dense + SparseFlow where policy selects
    net5 = build_fn().to(device).eval()
    functional.set_step_mode(net5, 'm')
    net5f, _ = substitute_sf(net5, T=T)
    net5f = net5f.to(device).eval()
    functional.set_step_mode(net5f, 'm')
    t5 = bench_fn(make_fn(net5f, x))

    print(f"  {name:<11s} {t1:>11.2f} {t2:>9.2f} {t3:>9.2f} {t4:>9.2f} {t5:>9.2f} {t1/t5:>9.2f}×")

    del net1, net2, net3, net4d, net5f, x
    torch.cuda.empty_cache()

print("=" * 100)
