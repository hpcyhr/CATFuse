"""test_57b_diagnose_kernel_names.py — Find actual BN kernel names in profiler.

test_57 reported BN count = 0 for all 17 networks. This is wrong — Table 13c
showed SEW-RN18 has ~120 BN kernels in SJ baseline. The categorization missed
them because the substring keys ('batch_norm', 'batchnorm') didn't match the
actual kernel names PyTorch profiler reports.

This script dumps the top 30 kernels by count for SEW-RN18 SJ multi-step,
showing actual op names. We then fix test_57's KERNEL_GROUPS dict.

Run:
  python tests/test_57b_diagnose_kernel_names.py
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from torch.profiler import profile, ProfilerActivity
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet


device = 'cuda:0'
T, B = 4, 2

m = sew_resnet.sew_resnet18(
    pretrained=False, num_classes=10, cnf='ADD',
    spiking_neuron=neuron.LIFNode, tau=2.0,
).to(device).eval()
functional.set_step_mode(m, 'm')

x = torch.rand(T, B, 3, 32, 32, device=device)

# Warmup
for _ in range(5):
    functional.reset_net(m)
    with torch.no_grad():
        _ = m(x)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(20):
        functional.reset_net(m)
        with torch.no_grad():
            _ = m(x)
torch.cuda.synchronize()

events = prof.key_averages()

# Sort by count desc, dump top 40
sorted_events = sorted(
    [(ev.key, ev.count or 0,
      getattr(ev, 'self_cuda_time_total', 0) or 0,
      getattr(ev, 'cuda_time_total', 0) or 0)
     for ev in events],
    key=lambda x: -x[1],
)

print(f'SEW-RN18 SJ multi-step — top 40 kernels by count (T={T}, B={B}, 20 iter)')
print('=' * 110)
print(f'{"Count":>6} {"self_cuda_us":>12} {"cuda_us":>10}  {"Kernel name"}')
print('-' * 110)
for name, count, self_us, cuda_us in sorted_events[:40]:
    if count == 0:
        continue
    # Highlight BN-related
    flag = ''
    nl = name.lower()
    if any(k in nl for k in ['bn', 'norm', 'batch']):
        flag = ' ← BN-like'
    elif any(k in nl for k in ['conv', 'gemm', 'cutlass']):
        flag = ' ← Conv'
    elif any(k in nl for k in ['heaviside', 'threshold', 'ge', 'gt']):
        flag = ' ← LIF spike'
    elif any(k in nl for k in ['mul', 'add', 'sub']):
        flag = ' ← Elem'
    print(f'{count:>6d} {self_us:>12.1f} {cuda_us:>10.1f}  {name}{flag}')

print()
print('Now identify all BN-related ops by scanning ALL events (not just top 40):')
print('-' * 110)
bn_ops = []
for ev in events:
    nl = (ev.key or '').lower()
    if any(k in nl for k in ['bn', 'norm', 'batch']):
        bn_ops.append((ev.key, ev.count or 0,
                       getattr(ev, 'self_cuda_time_total', 0) or 0))

if bn_ops:
    print(f'  Found {len(bn_ops)} BN-related ops:')
    for name, count, self_us in sorted(bn_ops, key=lambda x: -x[1]):
        print(f'    {count:>6d} count, {self_us:>10.1f} us  {name}')
else:
    print('  NO BN-related ops found in profiler. BN may be fused into elementwise.')