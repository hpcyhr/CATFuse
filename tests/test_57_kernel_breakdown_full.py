"""test_57_kernel_breakdown_full.py — Per-network kernel mechanism breakdown.

Ablation 维度 2 (mechanism-level): 用 torch.profiler 数每个网络在
SJ multi-step vs CATFuse 下的 kernel 分类计数 + 时间,把 442→137 的
kernel 减少分摊到 BN folding / StreamFuse+StateCarry 两个机制层来源。

对应 Table 13 (现仅 SEW-RN18 + VGG11) 扩到全 17 个网络。

Protocol:
  - cudnn.benchmark = False
  - T=4, B=2, CIFAR (32x32)
  - 5 warmup, 20 iter profile
  - kernel 分类按 op name substring (沿用 benchmarks/profiler_multi_network.py)

Output:
  - stdout: 17 行 SJ vs CF kernel/time breakdown
  - 末尾 CSV format

Run:
  python tests/test_57_kernel_breakdown_full.py
"""
import sys, time, copy, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from spikingjelly.activation_based import neuron, functional, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

from catfuse.substitute import substitute_sf as substitute
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1


device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
N_WARMUP = 5
N_ITER = 20


# Kernel name → category (沿用 benchmarks/profiler_multi_network.py 的分类)
KERNEL_GROUPS = {
    'Conv':         ['cudnn::conv', 'implicit_convolve', 'implicit_gemm',
                     'sgemm', 'cutlass', 'winograd', 'volta_', 'scudnn_',
                     'precomputed_convolve'],
    'BN':           ['bn_fw', 'bn_fwd', 'batch_norm', 'batchnorm',
                     'cudnn_batch_norm', 'cudnn::bn'],
    'LIF/Elem':     ['fused_sub_sub_div_add_ge', 'fused_mul_neg_add_mul_add',
                     'heaviside', 'threshold',
                     'unrolled_elementwise_kernel',
                     'vectorized_elementwise_kernel',
                     'cudafunctor_add', 'cudafunctor_mul'],
    'Triton-fused': ['triton_', '_conv_bn_lif', 'partial_fusion',
                     'streamfuse', 'lif_seq', 'sparse_streamfuse',
                     '_bn_lif_state_carry'],
    'MemOps':       ['memcpy', 'memset', 'copy_', 'contiguous',
                     'fillfunctor', 'nchwtonhwc', 'nhwctonchw'],
}


def categorize(name: str) -> str:
    name_lower = name.lower()
    for cat, keys in KERNEL_GROUPS.items():
        if any(k in name_lower for k in keys):
            return cat
    return 'Other'


@torch.no_grad()
def profile_run(net, x, n_warmup=N_WARMUP, n_iter=N_ITER):
    """Profile n_iter forward calls. Return per-category {count, time_us}."""
    net.eval()
    for _ in range(n_warmup):
        functional.reset_net(net)
        _ = net(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_iter):
            functional.reset_net(net)
            _ = net(x)
    torch.cuda.synchronize()

    events = prof.key_averages()
    cat_count = {k: 0 for k in KERNEL_GROUPS}
    cat_count['Other'] = 0
    cat_time = {k: 0.0 for k in KERNEL_GROUPS}
    cat_time['Other'] = 0.0

    for ev in events:
        t_self = getattr(ev, 'self_cuda_time_total', 0) or 0
        if t_self <= 0:
            continue
        cat = categorize(ev.key or '')
        cat_count[cat] += (ev.count or 0)
        cat_time[cat] += t_self / n_iter

    return cat_count, cat_time


def total(d):
    return sum(d.values())


configs = [
    # VGG family
    ('VGG11-BN',     lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG13-BN',     lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG16-BN',     lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('VGG19-BN',     lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    # SEW-ResNet
    ('SEW-RN18',     lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN34',     lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN50',     lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN101',    lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SEW-RN152',    lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    # Spiking-ResNet
    ('SpikRN18',     lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN34',     lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN50',     lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN101',    lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SpikRN152',    lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    # Classic CNN
    ('AlexNet',      lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('ZFNet',        lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
    ('MobileNet-V1', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
]


print(f'Per-network kernel mechanism breakdown — 17 networks (CIFAR 32x32, T={T}, B={B})')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Protocol: {N_WARMUP} warmup + {N_ITER} iter, torch.profiler CUDA activities')
print('=' * 110)
print(f"{'Network':<14} {'Variant':<8} "
      f"{'Conv':>5} {'BN':>5} {'LIF':>5} {'Triton':>7} {'Mem':>5} {'Other':>5} "
      f"{'Total':>6} {'CUDA_us':>9}")
print('-' * 110)

results = []   # (name, sj_cat_count, sj_cat_time, cf_cat_count, cf_cat_time, sj_total, cf_total)

x = torch.rand(T, B, 3, 32, 32, device=device)

for name, build_fn in configs:
    try:
        # SJ multi-step
        m_sj = build_fn().to(device).eval()
        functional.set_step_mode(m_sj, 'm')
        sj_count, sj_time = profile_run(m_sj, x)
        sj_total = total(sj_count)
        sj_cuda_us = total(sj_time)

        print(f'  {name:<14} {"SJ":<8} '
              f'{sj_count["Conv"]:>5d} {sj_count["BN"]:>5d} {sj_count["LIF/Elem"]:>5d} '
              f'{sj_count["Triton-fused"]:>7d} {sj_count["MemOps"]:>5d} {sj_count["Other"]:>5d} '
              f'{sj_total:>6d} {sj_cuda_us:>9.1f}')
        del m_sj
        torch.cuda.empty_cache()

        # CATFuse
        m_cf = build_fn().to(device).eval()
        functional.set_step_mode(m_cf, 'm')
        m_cf, _ = substitute(m_cf, T=T)
        m_cf = m_cf.to(device).eval()
        functional.set_step_mode(m_cf, 'm')
        cf_count, cf_time = profile_run(m_cf, x)
        cf_total = total(cf_count)
        cf_cuda_us = total(cf_time)

        print(f'  {"":14} {"CF":<8} '
              f'{cf_count["Conv"]:>5d} {cf_count["BN"]:>5d} {cf_count["LIF/Elem"]:>5d} '
              f'{cf_count["Triton-fused"]:>7d} {cf_count["MemOps"]:>5d} {cf_count["Other"]:>5d} '
              f'{cf_total:>6d} {cf_cuda_us:>9.1f}')

        # Reductions
        bn_kernel_eliminated = sj_count['BN'] - cf_count['BN']
        elem_kernel_eliminated = sj_count['LIF/Elem'] - cf_count['LIF/Elem']
        bn_time_saved = sj_time['BN'] - cf_time['BN']
        # StreamFuse+StateCarry 节省的时间 = (Elementwise saved) + (Conv saved) - (Triton added)
        elem_time_saved = sj_time['LIF/Elem'] - cf_time['LIF/Elem']
        triton_added = cf_time['Triton-fused'] - sj_time['Triton-fused']
        sf_sc_time_saved = elem_time_saved - triton_added

        print(f'  {"":14} {"Δ":<8} '
              f'{sj_count["Conv"]-cf_count["Conv"]:>5d} '
              f'{bn_kernel_eliminated:>5d} '
              f'{elem_kernel_eliminated:>5d} '
              f'{cf_count["Triton-fused"]-sj_count["Triton-fused"]:>+7d} '
              f'{sj_count["MemOps"]-cf_count["MemOps"]:>5d} '
              f'{sj_count["Other"]-cf_count["Other"]:>+5d} '
              f'{sj_total-cf_total:>6d} {sj_cuda_us-cf_cuda_us:>+9.1f}')

        results.append({
            'name': name,
            'sj_count': sj_count, 'sj_time': sj_time, 'sj_total': sj_total, 'sj_cuda_us': sj_cuda_us,
            'cf_count': cf_count, 'cf_time': cf_time, 'cf_total': cf_total, 'cf_cuda_us': cf_cuda_us,
            'bn_kernel_eliminated': bn_kernel_eliminated,
            'elem_kernel_eliminated': elem_kernel_eliminated,
            'bn_time_saved': bn_time_saved,
            'sf_sc_time_saved': sf_sc_time_saved,
        })
        del m_cf
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  {name:<14}  SKIP: {e}')

print('=' * 110)
print()

# ============================================================
# Mechanism-level breakdown summary
# ============================================================
print('Mechanism-level kernel reduction breakdown')
print('=' * 110)
print(f"{'Network':<14} "
      f"{'SJ tot':>7} {'CF tot':>7} {'Reduce%':>8} "
      f"{'BN-fold':>9} {'SF+SC':>9} "
      f"{'BN_time_us':>11} {'SF+SC_time_us':>15}")
print('-' * 110)

for r in results:
    reduce_pct = 100 * (1 - r['cf_total'] / r['sj_total'])
    print(f"  {r['name']:<14} "
          f"{r['sj_total']:>7d} {r['cf_total']:>7d} {reduce_pct:>7.1f}% "
          f"{r['bn_kernel_eliminated']:>9d} {r['elem_kernel_eliminated']:>9d} "
          f"{r['bn_time_saved']:>11.1f} {r['sf_sc_time_saved']:>15.1f}")

# Mean
if results:
    mean_reduce = sum(100 * (1 - r['cf_total'] / r['sj_total']) for r in results) / len(results)
    mean_bn_kernel = sum(r['bn_kernel_eliminated'] for r in results) / len(results)
    mean_elem_kernel = sum(r['elem_kernel_eliminated'] for r in results) / len(results)
    mean_bn_time = sum(r['bn_time_saved'] for r in results) / len(results)
    mean_sf_sc_time = sum(r['sf_sc_time_saved'] for r in results) / len(results)
    print('-' * 110)
    print(f"  {'Mean':<14} "
          f"{'':>7} {'':>7} {mean_reduce:>7.1f}% "
          f"{mean_bn_kernel:>9.1f} {mean_elem_kernel:>9.1f} "
          f"{mean_bn_time:>11.1f} {mean_sf_sc_time:>15.1f}")

# CSV format
print()
print('# CSV: name,sj_total,cf_total,bn_kernel_eliminated,elem_kernel_eliminated,bn_time_us,sf_sc_time_us')
for r in results:
    print(f"{r['name']},{r['sj_total']},{r['cf_total']},{r['bn_kernel_eliminated']},"
          f"{r['elem_kernel_eliminated']},{r['bn_time_saved']:.2f},{r['sf_sc_time_saved']:.2f}")