"""
Test 20: CATFuse-SF-lean at realistic high sparsity.

Random init gives ~50% spike rate. We inject controlled sparsity
to simulate trained SNN behavior (95-99% sparsity in deep layers).
"""
import sys, time, statistics
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
    return statistics.median(times)

net = sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

import catfuse.policy as pm
_orig = pm.classify_shape_regime

print("=" * 70)
print(f"High-sparsity benchmark (224x224, T={T}, B={B})")
print("=" * 70)

for sp_pct in [50, 80, 90, 95, 99]:
    thresh = sp_pct / 100.0
    # Create sparse spike-like input
    x = (torch.rand(T, B, 3, 224, 224, device=device) > thresh).float()
    in_rate = x.mean().item() * 100

    # SJ baseline
    sj_ms = bench(net, x, "SJ")

    # CATFuse-Dense
    pm.classify_shape_regime = lambda *a, **kw: "compute_bound"
    net_d, _ = substitute_sf(net, T=T)
    pm.classify_shape_regime = _orig
    net_d = net_d.to(device).eval()
    functional.set_step_mode(net_d, 'm')
    dense_ms = bench(net_d, x, "Dense")

    # CATFuse-SF
    net_sf, _ = substitute_sf(net, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')
    sf_ms = bench(net_sf, x, "SF")

    print(f"  sp={sp_pct:2d}% (in={in_rate:.1f}%)  SJ={sj_ms:.2f}  Dense={dense_ms:.2f} ({sj_ms/dense_ms:.2f}x)  "
          f"SF={sf_ms:.2f} ({sj_ms/sf_ms:.2f}x)  SF/Dense={dense_ms/sf_ms:.2f}x")

print("=" * 70)
