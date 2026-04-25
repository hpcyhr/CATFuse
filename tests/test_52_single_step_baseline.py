"""test_52_single_step_baseline.py — SJ single-step vs multi-step baseline校准

对应报告 Table 8。用于 Chronos 对标：Chronos 使用 SJ single-step 作为 baseline，
CATFuse 使用 SJ multi-step。本脚本提供换算系数。
"""
import sys, time, statistics, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
N_W, N_I, N_R = 20, 50, 3  # V100: N_R=3, A100: N_R=5


def bench_single_step(model, T, B, C, H):
    model.eval()
    functional.set_step_mode(model, 's')
    x = torch.rand(B, C, H, H, device=device)
    times = []
    for _ in range(N_R):
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad():
                for t in range(T):
                    model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad():
                for t in range(T):
                    model(x)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)/N_I*1000)
    return statistics.median(times)


def bench_multi_step(model, T, B, C, H):
    model.eval()
    functional.set_step_mode(model, 'm')
    x = torch.rand(T, B, C, H, H, device=device)
    times = []
    for _ in range(N_R):
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad(): model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad(): model(x)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)/N_I*1000)
    return statistics.median(times)


gpu = torch.cuda.get_device_name(0)
print(f'SJ single-step vs multi-step baseline (CIFAR 32x32, T={T}, B={B})')
print(f'GPU: {gpu}')
print('='*70)

configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('AlexNet', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('ZFNet', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('MobV1', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
]

for name, build_fn, B, C, H in configs:
    m1 = build_fn().to(device).eval()
    ss = bench_single_step(m1, T, B, C, H)

    m2 = build_fn().to(device).eval()
    ms = bench_multi_step(m2, T, B, C, H)

    print(f'  {name:<12} single={ss:>7.2f}  multi={ms:>7.2f}  s/m={ss/ms:>5.2f}x')
    del m1, m2
    torch.cuda.empty_cache()

print('='*70)