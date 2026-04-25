"""test_54_bladedisc.py — BladeDISC benchmark (docker环境)

对应报告 Table 5 中的 BladeDISC 列。
需要在 bladedisc docker 容器中运行：
  docker run -itd --gpus all -v /data/dagongcheng/yhrtest:/workspace/yhrtest \
    --name bladedisc_test bladedisc/bladedisc:latest-runtime-torch1.7.1-cu110 bash
  docker exec -it bladedisc_test bash
  pip install spikingjelly==0.0.0.0.14
  python /workspace/yhrtest/test_54_bladedisc.py

BladeDISC 使用 PyTorch 1.7.1 + torch_blade 0.2.0。
报告中的加速比使用 snn118 环境 (PyTorch 2.1.0) 的 SJ baseline，不使用 docker 内的 SJ baseline。
仅记录 BladeDISC 的绝对延迟。
"""
import os, sys, time, statistics, warnings
sys.path.insert(0, '/workspace/yhrtest/CATFuse')
warnings.filterwarnings('ignore')

import torch
import torch_blade
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

device = 'cuda:0'
T = 4
N_W, N_I, N_R = 20, 50, 5

print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')


def bench_blade(model, B, C, H):
    model.eval()
    functional.set_step_mode(model, 's')
    functional.reset_net(model)
    x = torch.rand(B, C, H, H, device=device)

    try:
        functional.reset_net(model)
        with torch.no_grad():
            traced = torch.jit.trace(model, x, check_trace=False)
    except Exception as e:
        return None, f'trace:{type(e).__name__}'

    try:
        with torch.no_grad():
            optimized = torch_blade.optimize(traced, allow_tracing=True,
                                              model_inputs=(x,))
    except Exception as e:
        return None, f'optimize:{type(e).__name__}'

    try:
        times = []
        for _ in range(N_R):
            for _ in range(N_W):
                with torch.no_grad():
                    for t in range(T):
                        optimized(x)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(N_I):
                with torch.no_grad():
                    for t in range(T):
                        optimized(x)
            torch.cuda.synchronize(); times.append((time.perf_counter()-t0)/N_I*1000)
        return statistics.median(times), 'OK'
    except Exception as e:
        return None, f'run:{type(e).__name__}: {str(e)[:60]}'


print('='*70)
print(f"{'Network':<15} {'Blade ms':>10} {'Status'}")
print('-'*70)

configs = [
    # CIFAR (B=2, 32x32)
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG13-BN', lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG19-BN', lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN34', lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN101', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN152', lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN34', lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN101', lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN152', lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('AlexNet-C', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('ZFNet-C', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('MobV1-C', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    # ImageNet (B=1, 224x224)
    ('VGG11-I', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SEW-RN18-I', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN50-I', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SpikRN18-I', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SpikRN50-I', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('AlexNet-I', lambda: SpikingAlexNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('ZFNet-I', lambda: SpikingZFNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('MobV1-I', lambda: SpikingMobileNetV1(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
]

for name, build_fn, B, C, H in configs:
    m = build_fn().to(device).eval()
    functional.set_step_mode(m, 's')
    bd_ms, status = bench_blade(m, B, C, H)

    if bd_ms:
        print(f'  {name:<13} {bd_ms:>9.2f}  {status}')
    else:
        print(f'  {name:<13}      FAIL  {status}')

    del m
    torch.cuda.empty_cache()

print('='*70)