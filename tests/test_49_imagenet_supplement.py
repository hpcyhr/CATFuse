"""test_49_imagenet_supplement.py — ImageNet补测（10个缺失网络配置）

对应报告 Table 2 中的补测数据：VGG13/16/19-I, SEW-RN101/152-I, SpikRN18/34/50/101/152-I
在 V100 和 A100 上分别运行。
"""
import sys, time, statistics, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from catfuse.substitute import substitute_sf as substitute

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 1
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

print(f'ImageNet supplement (224x224, B={B}, T={T})')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('='*75)
print(f"{'Network':<15} {'SJ-torch':>9} {'CATFuse':>9} {'Speedup':>8} {'Cov':>5}")
print('-'*75)

configs = [
    ('VGG13-I', lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('VGG16-I', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('VGG19-I', lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('SEW-RN101-I', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN152-I', lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SpikRN18-I', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('SpikRN34-I', lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('SpikRN50-I', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('SpikRN101-I', lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
    ('SpikRN152-I', lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
]

for name, build_fn in configs:
    x = torch.rand(T, B, 3, 224, 224, device=device)
    m = build_fn().to(device).eval()
    functional.set_step_mode(m, 'm')
    sj = bench(m, x)

    m2 = build_fn().to(device).eval()
    functional.set_step_mode(m2, 'm')
    m2, stats = substitute(m2, T=T)
    m2 = m2.to(device).eval()
    functional.set_step_mode(m2, 'm')
    cf = bench(m2, x)

    cov = stats['coverage_pct']
    print(f'  {name:<13} {sj:>8.2f} {cf:>8.2f} {sj/cf:>7.2f}x {cov:>4.0f}%')

    del m, m2, x
    torch.cuda.empty_cache()

print('='*75)