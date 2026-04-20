"""Test 41: Extended architectures — deeper ResNet + VGG19."""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
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

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 90)

configs = [
    # Extended ResNet
    ("SEW-RN101", lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN152", lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SpikRN50", lambda: spiking_resnet.spiking_resnet50(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN101", lambda: spiking_resnet.spiking_resnet101(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN152", lambda: spiking_resnet.spiking_resnet152(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    # VGG19
    ("VGG19-BN", lambda: spiking_vgg.spiking_vgg19_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    # ImageNet scale
    ("SEW-RN50-I", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

from catfuse.substitute import substitute_sf

for name, build_fn, B, C, H in configs:
    print(f"\n  {name:<15s}", end="", flush=True)
    try:
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

        # CATFuse-SF
        net_sf_base = build_fn().to(device).eval()
        functional.set_step_mode(net_sf_base, 'm')
        net_sf, stats = substitute_sf(net_sf_base, T=T)
        net_sf = net_sf.to(device).eval()
        functional.set_step_mode(net_sf, 'm')
        
        # Parity
        functional.reset_net(net)
        functional.reset_net(net_sf)
        for m in net_sf.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad():
            ref = net(x)
            sf_out = net_sf(x)
        parity = (ref - sf_out).abs().max().item()
        
        sf = bench(net_sf, x)
        cov = stats["coverage_pct"]

        print(f" torch={sj_t:>7.2f}  cupy={sj_c:>7.2f}  SF={sf:>7.2f}  "
              f"vs_t={sj_t/sf:>5.2f}×  vs_c={sj_c/sf:>5.2f}×  cov={cov:.0f}%  "
              f"par={'✅' if parity < 0.01 else '❌'}")

        del net, net_c, net_sf, x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f" ❌ {e}")
        torch.cuda.empty_cache()

print("\n" + "=" * 90)
