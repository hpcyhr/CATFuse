"""
Test 29: Runtime EGD — dynamic backend selection based on sparsity.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.substitute import substitute_sf

device = 'cuda:0'
T, B = 4, 1
N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

def bench(model, x_in):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            for m in model.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            for m in model.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

net = sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0).to(device).eval()
functional.set_step_mode(net, 'm')

print("=" * 80)
print(f"Runtime EGD benchmark (SEW-ResNet18 ImageNet, T={T}, B={B})")
print(f"{'Input sp%':<12s} {'SJ':>8s} {'Dense':>8s} {'SF+EGD':>8s} {'SJ/Dense':>9s} {'SJ/EGD':>9s}")
print("-" * 80)

for sp_pct in [10, 30, 50, 70, 90, 95, 99]:
    sp = sp_pct / 100.0
    x = (torch.rand(T, B, 3, 224, 224, device=device) > sp).float()

    # SJ baseline
    sj_ms = bench(net, x)

    # Dense only
    import catfuse.policy as pm
    _orig = pm.classify_shape_regime
    pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
    net_d, _ = substitute_sf(net, T=T)
    pm.classify_shape_regime = _orig
    net_d = net_d.to(device).eval()
    functional.set_step_mode(net_d, 'm')
    dense_ms = bench(net_d, x)

    # SF with EGD
    net_sf, _ = substitute_sf(net, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')
    sf_ms = bench(net_sf, x)

    print(f"  sp={sp_pct:2d}%     {sj_ms:>7.2f}ms {dense_ms:>7.2f}ms {sf_ms:>7.2f}ms "
          f" {sj_ms/dense_ms:>8.2f}x {sj_ms/sf_ms:>8.2f}x")

print("=" * 80)
