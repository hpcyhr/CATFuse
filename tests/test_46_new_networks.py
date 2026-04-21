"""Test 46: AlexNet + ZFNet benchmark on V100."""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from catfuse.substitute import substitute_sf

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T = 4
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

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")
print("=" * 90)
print(f"{'Network':<15s} {'SJ-torch':>10s} {'SJ-cupy':>10s} {'CATFuse':>10s} {'vs torch':>10s} {'vs cupy':>10s} {'cov':>5s}")
print("=" * 90)

configs = [
    ("AlexNet-C", lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode,
        tau=2.0, input_size=32), 2, 3, 32),
    ("ZFNet-C", lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode,
        tau=2.0, input_size=32), 2, 3, 32),
    ("AlexNet-I", lambda: SpikingAlexNet(num_classes=1000, spiking_neuron=neuron.LIFNode,
        tau=2.0, input_size=224), 1, 3, 224),
    ("ZFNet-I", lambda: SpikingZFNet(num_classes=1000, spiking_neuron=neuron.LIFNode,
        tau=2.0, input_size=224), 1, 3, 224),
]

for name, build_fn, B, C, H in configs:
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ torch
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_t = bench(net, x)

    # SJ cupy
    net_c = build_fn().to(device).eval()
    functional.set_step_mode(net_c, 'm')
    functional.set_backend(net_c, 'cupy')
    sj_c = bench(net_c, x)

    # CATFuse
    net_sf_base = build_fn().to(device).eval()
    functional.set_step_mode(net_sf_base, 'm')
    net_sf, stats = substitute_sf(net_sf_base, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')
    sf = bench(net_sf, x)

    cov = stats["coverage_pct"]
    print(f"  {name:<13s} {sj_t:>9.2f} {sj_c:>9.2f} {sf:>9.2f} {sj_t/sf:>9.2f}× {sj_c/sf:>9.2f}× {cov:>4.0f}%")

    del net, net_c, net_sf, x
    torch.cuda.empty_cache()

print("=" * 90)
