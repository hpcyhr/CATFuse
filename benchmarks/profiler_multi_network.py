"""
Multi-network profiler evidence for CATFuse paper §V-G.

Runs torch.profiler on 3 representative networks to show the fusion
mechanism (kernel launch reduction, BN/LIF time reduction) is not
specific to SEW-ResNet-18 cherry-picked.

Networks:
  - SEW-ResNet-18   (BasicBlock, medium depth)
  - SEW-ResNet-50   (Bottleneck, deeper)
  - SpikingVGG-11   (flat VGG, different topology)

Usage:
    /data_priv/dagongcheng/snn118/bin/python profiler_multi_network.py --device cuda:0
"""
import argparse
import copy
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from catfuse_substitute import substitute


def init_bn_random(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.running_mean.data.normal_(0, 0.1)
            m.running_var.data.uniform_(0.5, 1.5)


def build_sew18():
    m = sew_resnet.sew_resnet18(
        spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=1000, cnf='ADD',
        v_threshold=0.1, tau=2.0)
    functional.set_step_mode(m, 'm')
    return m


def build_sew50():
    m = sew_resnet.sew_resnet50(
        spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=1000, cnf='ADD',
        v_threshold=0.1, tau=2.0)
    functional.set_step_mode(m, 'm')
    return m


def build_vgg11():
    m = spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=1000,
        v_threshold=0.1, tau=2.0)
    functional.set_step_mode(m, 'm')
    return m


BUILDERS = {
    'SEW-ResNet-18':  build_sew18,
    'SEW-ResNet-50':  build_sew50,
    'SpikingVGG-11':  build_vgg11,
}


@torch.no_grad()
def profile_one(net, x, label, n_warmup=3, n_iters=10):
    """Profile N iters, return {total_cuda_us, kernel_count, op_breakdown}."""
    net.eval()
    for _ in range(n_warmup):
        _ = net(x).mean(dim=0)
        functional.reset_net(net)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(n_iters):
            _ = net(x).mean(dim=0)
            functional.reset_net(net)
    torch.cuda.synchronize()

    events = prof.key_averages()
    total_kernel_count = 0
    for ev in events:
        if (getattr(ev, 'self_cuda_time_total', 0) or 0) > 0 \
                or (getattr(ev, 'cuda_time_total', 0) or 0) > 0:
            total_kernel_count += (ev.count or 0)

    total_cuda_us = sum(getattr(ev, 'self_cuda_time_total', 0) or 0
                        for ev in events) / n_iters

    groups = {
        'Conv':           ['conv2d', 'cudnn::conv', 'implicit_convolution',
                           'implicit_gemm', 'cutlass'],
        'BN':             ['batch_norm', 'batchnorm', 'cudnn_batch_norm'],
        'LIF/elem':       ['mul', 'add', 'sub', 'div', 'heaviside',
                           'masked_fill', 'where', 'elementwise', 'sigmoid',
                           'threshold', 'ge', 'gt'],
        'Triton-fused':   ['triton_', '_bn_lif_state_carry_kernel',
                           '_conv_bn_lif', 'partial_fusion'],
        'Memcpy/other':   ['memcpy', 'copy_', 'contiguous', 'view', 'reshape',
                           'zero_', 'fill_', 'empty'],
    }

    op_time_us = {k: 0.0 for k in groups}
    op_time_us['uncategorized'] = 0.0
    for ev in events:
        t_self = getattr(ev, 'self_cuda_time_total', 0) or 0
        if t_self <= 0:
            continue
        name = (ev.key or '').lower()
        matched = None
        for group, keys in groups.items():
            if any(k in name for k in keys):
                matched = group
                break
        key = matched or 'uncategorized'
        op_time_us[key] += t_self / n_iters

    return {
        'label': label,
        'total_cuda_us_per_iter': total_cuda_us,
        'kernel_count_per_iter': total_kernel_count / n_iters,
        'op_breakdown_us': op_time_us,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('-T', type=int, default=16)
    p.add_argument('-B', type=int, default=8)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: T={args.T}, B={args.B}, 224x224\n")

    all_results = {}  # network -> {backend -> profile_result}

    for net_name, builder in BUILDERS.items():
        print(f"\n===== {net_name} =====")
        torch.manual_seed(0)
        net_torch = builder().to(device)
        init_bn_random(net_torch)

        net_cupy = copy.deepcopy(net_torch)
        functional.set_backend(net_cupy, 'cupy', instance=neuron.LIFNode)

        net_ctf_base = copy.deepcopy(net_torch)
        net_ctf, cov = substitute(net_ctf_base, verbose=False)
        net_ctf = net_ctf.to(device)

        x = torch.randn(args.T, args.B, 3, 224, 224, device=device)

        results_this = {}
        for net, label in [(net_torch, 'SJ torch'),
                           (net_cupy, 'SJ cupy'),
                           (net_ctf, 'CATFuse')]:
            print(f"  profiling {label} ...", flush=True)
            r = profile_one(net, x, label)
            results_this[label] = r
            print(f"    {r['total_cuda_us_per_iter']/1000:7.2f} ms, "
                  f"{r['kernel_count_per_iter']:6.0f} launches")
        all_results[net_name] = results_this

        del net_torch, net_cupy, net_ctf, net_ctf_base, x
        torch.cuda.empty_cache()

    # Summary — paper-ready table
    print(f"\n\n{'='*120}")
    print(f"=== MULTI-NETWORK PROFILER SUMMARY (for paper Table) ===")
    print(f"{'='*120}")
    print(f"{'Network':<16s} {'Backend':<10s} "
          f"{'CUDA ms':>8s} {'Launches':>9s} {'LaunchR':>8s} "
          f"{'Conv':>7s} {'BN':>6s} {'LIF/el':>8s} {'Triton':>8s} "
          f"{'Memcpy':>8s} {'Uncat':>8s} {'SumChk':>8s}")
    print(f"{'-'*16} {'-'*10} {'-'*8} {'-'*9} {'-'*8} "
          f"{'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for net_name, backends in all_results.items():
        torch_launches = backends['SJ torch']['kernel_count_per_iter']
        for backend in ['SJ torch', 'SJ cupy', 'CATFuse']:
            r = backends[backend]
            red = torch_launches / r['kernel_count_per_iter']
            conv = r['op_breakdown_us'].get('Conv', 0) / 1000
            bn = r['op_breakdown_us'].get('BN', 0) / 1000
            lif = r['op_breakdown_us'].get('LIF/elem', 0) / 1000
            triton = r['op_breakdown_us'].get('Triton-fused', 0) / 1000
            mem = r['op_breakdown_us'].get('Memcpy/other', 0) / 1000
            uncat = r['op_breakdown_us'].get('uncategorized', 0) / 1000
            # Sum check: these buckets should sum to total CUDA time
            sum_all = conv + bn + lif + triton + mem + uncat
            print(f"{net_name:<16s} {backend:<10s} "
                  f"{r['total_cuda_us_per_iter']/1000:>8.2f} "
                  f"{r['kernel_count_per_iter']:>9.0f} "
                  f"{red:>7.2f}x "
                  f"{conv:>7.2f} {bn:>6.2f} {lif:>8.2f} {triton:>8.2f} "
                  f"{mem:>8.2f} {uncat:>8.2f} {sum_all:>8.2f}")
        print()


if __name__ == '__main__':
    main()