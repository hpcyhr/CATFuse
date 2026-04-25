"""test_51_b_sweep_full.py — Full B-sweep: 11 networks × B∈{1,2,4,8}

对应报告 Table 4a (V100) 和 Table 4b (A100)。
"""
import sys, time, statistics, warnings
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
T = 4
N_W, N_I, N_R = 20, 50, 3  # V100: N_R=3, A100: N_R=5

def bench(model, x):
    model.eval(); times = []
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
print(f'B-Sweep (CIFAR 32x32, T={T}, B in {{1,2,4,8}})')
print(f'GPU: {gpu}')
print('='*80)
print(f"{'Network':<14} {'B':>2}  {'SJ':>7}  {'CF':>7}  {'Spdup':>6}")
print('-'*80)

configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG19-BN', lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN101', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('AlexNet', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('ZFNet', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('MobV1', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
]

B_vals = [1, 2, 4, 8]

for name, build_fn in configs:
    for B in B_vals:
        x = torch.rand(T, B, 3, 32, 32, device=device)

        m = build_fn().to(device).eval()
        functional.set_step_mode(m, 'm')
        sj = bench(m, x)

        m2 = build_fn().to(device).eval()
        functional.set_step_mode(m2, 'm')
        m2, _ = substitute(m2, T=T)
        m2 = m2.to(device).eval()
        functional.set_step_mode(m2, 'm')
        cf = bench(m2, x)

        print(f'  {name:<12} {B:>2}  {sj:>6.2f}  {cf:>6.2f}  {sj/cf:>5.2f}x')

        del m, m2, x
        torch.cuda.empty_cache()
    print()

print('='*80)