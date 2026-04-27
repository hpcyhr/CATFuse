"""test_52b_single_step_baseline_full.py — SJ single-step vs multi-step baseline (full 19 nets).

扩展 test_52 的 8 个 config 到 19 个,与 Table 1 完全对齐。
用于支撑 Table 10 ablation 的全网络版本(BatchFold + StreamFuse+SC 三级分解)。

Protocol 与 test_52 一致:
  - cudnn.benchmark = False
  - T=4, B=2 (CIFAR), 20 warmup + 50 iter × N_R repeats, median
  - V100: N_R=3, A100: N_R=5

输出:
  - stdout: 每行 "name | SJ-single (ms) | SJ-multi (ms) | s/m ratio"
  - 隐式可整理成 Table 8 (校准) + Table 10 (ablation 三级分解)

Run:
  python tests/test_52b_single_step_baseline_full.py
"""
import sys, time, statistics, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

# Project-local model imports (same as test_35/41/46/52)
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

# SpikFormer / QKFormer if present in models/
try:
    from models.spikformer_github import SpikFormer
except Exception:
    SpikFormer = None
try:
    from models.qkformer_github import QKFormer
except Exception:
    QKFormer = None


device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2

# Auto-detect repeat count by GPU name
gpu = torch.cuda.get_device_name(0)
N_R = 5 if 'A100' in gpu else 3
N_W, N_I = 20, 50


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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad():
                for t in range(T):
                    model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_I * 1000)
    return statistics.median(times)


def bench_multi_step(model, T, B, C, H):
    model.eval()
    functional.set_step_mode(model, 'm')
    x = torch.rand(T, B, C, H, H, device=device)
    times = []
    for _ in range(N_R):
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_I * 1000)
    return statistics.median(times)


print(f'SJ single-step vs multi-step baseline — full 19-network (CIFAR 32x32, T={T}, B={B})')
print(f'GPU: {gpu}  | N_R={N_R}  warmup={N_W}  iter={N_I}')
print('=' * 78)

# 19 configs matching Table 1 exactly
configs = [
    # VGG family (4)
    ('VGG11-BN',    lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('VGG13-BN',    lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('VGG16-BN',    lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('VGG19-BN',    lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    # SEW-ResNet family (5)
    ('SEW-RN18',    lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0),  B, 3, 32),
    ('SEW-RN34',    lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0),  B, 3, 32),
    ('SEW-RN50',    lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0),  B, 3, 32),
    ('SEW-RN101',   lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), B, 3, 32),
    ('SEW-RN152',   lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), B, 3, 32),
    # Spiking-ResNet family (5)
    ('SpikRN18',    lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('SpikRN34',    lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('SpikRN50',    lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),     B, 3, 32),
    ('SpikRN101',   lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),    B, 3, 32),
    ('SpikRN152',   lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),    B, 3, 32),
    # Classic CNNs (3)
    ('AlexNet',     lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32),       B, 3, 32),
    ('ZFNet',       lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32),         B, 3, 32),
    ('MobileNet-V1',lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32),   B, 3, 32),
]

# Optionally add Transformer-class (SpikFormer/QKFormer) if available
if SpikFormer is not None:
    configs.append(('SpikFormer', lambda: SpikFormer(num_classes=10, T=T, in_channels=3, image_size=32), B, 3, 32))
if QKFormer is not None:
    configs.append(('QKFormer', lambda: QKFormer(num_classes=10, T=T, in_channels=3, image_size=32), B, 3, 32))

print(f'  {"Network":<14}  {"SJ-single (ms)":>15}  {"SJ-multi (ms)":>15}  {"s/m ratio":>10}')
print(f'  {"-"*14}  {"-"*15}  {"-"*15}  {"-"*10}')

results = []
for name, build_fn, b, c, h in configs:
    try:
        m1 = build_fn().to(device).eval()
        ss = bench_single_step(m1, T, b, c, h)
        del m1
        torch.cuda.empty_cache()

        m2 = build_fn().to(device).eval()
        ms = bench_multi_step(m2, T, b, c, h)
        del m2
        torch.cuda.empty_cache()

        ratio = ss / ms
        print(f'  {name:<14}  {ss:>15.2f}  {ms:>15.2f}  {ratio:>9.2f}x')
        results.append((name, ss, ms, ratio))
    except Exception as e:
        print(f'  {name:<14}  SKIP: {e}')

print('=' * 78)

# Mean ratio
if results:
    mean_ratio = sum(r[3] for r in results) / len(results)
    print(f'  {"Mean s/m ratio":<14}  {"":>15}  {"":>15}  {mean_ratio:>9.2f}x')

# CSV-friendly output for spreadsheet ingestion
print()
print('# CSV format (network, sj_single_ms, sj_multi_ms, single_over_multi)')
for name, ss, ms, ratio in results:
    print(f'{name},{ss:.4f},{ms:.4f},{ratio:.4f}')