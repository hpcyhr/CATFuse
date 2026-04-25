"""test_53_overhead_ablation.py — Scheduling overhead + Ablation study

对应报告 Table 9 (scheduling overhead) 和 Table 10 (ablation)。
"""
import sys, time, statistics, warnings, copy
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from catfuse.substitute import substitute_sf as substitute
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

# ============================================================
# Part 1: Scheduling Overhead (substitute() execution time)
# ============================================================

print('=== Part 1: Scheduling Overhead (substitute time) ===')
print(f"{'Network':<14} {'Time (ms)':>10}")
print('-'*30)

overhead_configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN101', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('AlexNet', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('ZFNet', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('MobV1', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
]

for name, build_fn in overhead_configs:
    times = []
    for _ in range(5):
        m = build_fn().to(device).eval()
        functional.set_step_mode(m, 'm')
        t0 = time.perf_counter()
        _, stats = substitute(m, T=4)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del m
    print(f'  {name:<12} {sorted(times)[2]:>9.1f}')

# ============================================================
# Part 2: Ablation Study (single → multi → CATFuse)
# ============================================================

print()
print('=== Part 2: Ablation (single → multi → CATFuse) ===')

T, B = 4, 2
N_W, N_I, N_R = 20, 50, 3  # V100: N_R=3, A100: N_R=5


def bench(fn):
    times = []
    for _ in range(N_R):
        for _ in range(N_W): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_I): fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_I * 1000)
    return statistics.median(times)


print(f"{'Network':<12} {'Single':>8} {'Multi':>8} {'CATFuse':>8} {'CF/Single':>10} {'CF/Multi':>9}")
print('-'*60)

ablation_nets = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('AlexNet', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
]

for name, build_fn in ablation_nets:
    x_s = torch.rand(B, 3, 32, 32, device=device)
    x_m = torch.rand(T, B, 3, 32, 32, device=device)

    # Single-step baseline
    m1 = build_fn().to(device).eval()
    functional.set_step_mode(m1, 's')
    ss = bench(lambda: (functional.reset_net(m1), [m1(x_s) for _ in range(T)]))

    # Multi-step baseline
    m2 = build_fn().to(device).eval()
    functional.set_step_mode(m2, 'm')
    ms = bench(lambda: (functional.reset_net(m2), m2(x_m)))

    # CATFuse
    m3 = build_fn().to(device).eval()
    functional.set_step_mode(m3, 'm')
    m3, _ = substitute(m3, T=T)
    m3 = m3.to(device).eval()
    functional.set_step_mode(m3, 'm')
    cf = bench(lambda: (functional.reset_net(m3), m3(x_m)))

    print(f'  {name:<10} {ss:>7.2f} {ms:>7.2f} {cf:>7.2f} {ss/cf:>9.2f}x {ms/cf:>8.2f}x')
    del m1, m2, m3, x_s, x_m
    torch.cuda.empty_cache()

print('='*60)