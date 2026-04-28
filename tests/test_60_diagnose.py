"""
诊断脚本: 查看 untrained SEW-RN18 在 randn 输入下,每层 LIF 实际产生的输出
判断为什么 sparsity 全是 1.0
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet


DEVICE = 'cuda:0'

torch.manual_seed(42)

# 构造 SEW-RN18
net = sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD',
                              spiking_neuron=neuron.LIFNode, tau=2.0)
net = net.to(DEVICE).eval()
functional.set_step_mode(net, 'm')

# Hook 每层 LIF,打印 input 和 output 的实际数值
spike_data = []

def make_hook(name):
    def hook(module, inputs, output):
        # inputs[0] 是 LIF 的输入张量 (脉冲前电位 z)
        z = inputs[0].detach()
        spike = output.detach()
        spike_data.append({
            'name': name,
            'z_mean': z.float().mean().item(),
            'z_std': z.float().std().item(),
            'z_max': z.float().max().item(),
            'z_min': z.float().min().item(),
            'z_above_th': (z >= 1.0).float().mean().item(),  # 假设 v_th=1
            'spike_mean': spike.float().mean().item(),
            'spike_max': spike.float().max().item(),
            'shape': tuple(spike.shape),
        })
    return hook

handles = []
idx = 0
for name, module in net.named_modules():
    if isinstance(module, neuron.LIFNode):
        h = module.register_forward_hook(make_hook(f"lif_{idx:02d}_{name}"))
        handles.append(h)
        idx += 1

print(f"Attached {idx} hooks")

# 跑一个 input,用 randn (有正有负)
x = torch.randn(4, 1, 3, 224, 224, device=DEVICE)
print(f"\nInput stats:")
print(f"  shape={tuple(x.shape)}")
print(f"  mean={x.mean().item():.4f}, std={x.std().item():.4f}")
print(f"  min={x.min().item():.4f}, max={x.max().item():.4f}")

functional.reset_net(net)
with torch.no_grad():
    _ = net(x)

print(f"\n{'='*100}")
print(f"{'Layer':<35} {'Shape':<25} {'z_mean':>9} {'z_std':>9} {'z_min':>9} {'z_max':>9} {'>1.0':>8} {'spike_avg':>10}")
print('=' * 130)
for d in spike_data:
    print(f"{d['name']:<35} {str(d['shape']):<25} "
          f"{d['z_mean']:>9.4f} {d['z_std']:>9.4f} "
          f"{d['z_min']:>9.4f} {d['z_max']:>9.4f} "
          f"{d['z_above_th']:>7.2%} {d['spike_mean']:>10.4f}")

for h in handles:
    h.remove()