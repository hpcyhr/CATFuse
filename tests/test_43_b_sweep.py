"""Test 43: B-sweep — CATFuse speedup across batch sizes."""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg
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
print("=" * 100)
print(f"{'Network':<15s} {'B':>3s} {'SJ-torch':>10s} {'SJ-cupy':>10s} {'CATFuse':>10s} {'vs torch':>10s} {'vs cupy':>10s}")
print("=" * 100)

configs = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 3, 32),
]

for name, build_fn, C, H in configs:
    for B in [1, 2, 4, 8]:
        x = torch.rand(T, B, C, H, H, device=device)

        net = build_fn().to(device).eval()
        functional.set_step_mode(net, 'm')
        sj_t = bench(net, x)

        net_c = build_fn().to(device).eval()
        functional.set_step_mode(net_c, 'm')
        functional.set_backend(net_c, 'cupy')
        sj_c = bench(net_c, x)

        net_sf_base = build_fn().to(device).eval()
        functional.set_step_mode(net_sf_base, 'm')
        net_sf, _ = substitute_sf(net_sf_base, T=T)
        net_sf = net_sf.to(device).eval()
        functional.set_step_mode(net_sf, 'm')
        sf = bench(net_sf, x)

        print(f"  {name:<13s} {B:>3d} {sj_t:>9.2f} {sj_c:>9.2f} {sf:>9.2f} {sj_t/sf:>9.2f}× {sj_c/sf:>9.2f}×")

        del net, net_c, net_sf, x
        torch.cuda.empty_cache()
    print()
print("=" * 100)
