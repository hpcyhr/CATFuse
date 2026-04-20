"""
Profiler-based evidence for CATFuse paper §V — kernel launches + op time breakdown.

Uses torch.profiler to measure, for SEW-ResNet-18 at T=16, B=8, on A100:
  (a) total kernel launches per forward pass (SJ torch, SJ cupy, CATFuse)
  (b) time breakdown by operator type (Conv, BN, LIF/elementwise, other)

This directly supports the paper claim "CATFuse's speedup comes from reduced
kernel launches and eliminated BN→LIF HBM round-trip", which until now was
supported only by wall-clock and I/O formula.

Usage:
    /data_priv/dagongcheng/snn118/bin/python profiler_evidence.py --device cuda:0
"""
import argparse
import copy
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

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


def build_model(v_threshold=0.1, tau=2.0):
    m = sew_resnet.sew_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=1000, cnf='ADD',
        v_threshold=v_threshold, tau=tau)
    functional.set_step_mode(m, 'm')
    return m


@torch.no_grad()
def profile_one(net, x, label, n_warmup=3, n_iters=10):
    """Profile N iters, return (total kernel launches per iter, per-op time dict)."""
    net.eval()
    # Warmup
    for _ in range(n_warmup):
        _ = net(x).mean(dim=0)
        functional.reset_net(net)
    torch.cuda.synchronize()

    # Profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(n_iters):
            with record_function(f"fwd_{label}"):
                _ = net(x).mean(dim=0)
                functional.reset_net(net)
    torch.cuda.synchronize()

    # Parse events — PyTorch 2.1 FunctionEventAvg uses cuda_time_total
    events = prof.key_averages()
    # Count kernel launches: sum of 'count' for events that touched GPU
    total_kernel_count = 0
    for ev in events:
        if (getattr(ev, 'self_cuda_time_total', 0) or 0) > 0 \
                or (getattr(ev, 'cuda_time_total', 0) or 0) > 0:
            total_kernel_count += (ev.count or 0)

    # Total CUDA time per iter
    total_cuda_us = sum(getattr(ev, 'self_cuda_time_total', 0) or 0
                        for ev in events) / n_iters

    # Operator-type breakdown: group by keyword in event name
    groups = {
        'Conv':           ['conv2d', 'cudnn::conv', 'implicit_convolution',
                           'implicit_gemm', 'cutlass'],
        'BN':             ['batch_norm', 'batch_norm_elementwise',
                           'batchnorm', 'cudnn_batch_norm'],
        'LIF/elementwise':['mul', 'add', 'sub', 'div', 'heaviside',
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
        op_time_us[key] += t_self / n_iters  # avg per iter

    return {
        'label': label,
        'total_cuda_us_per_iter': total_cuda_us,
        'kernel_count_per_iter': total_kernel_count / n_iters,
        'op_breakdown_us': op_time_us,
        'top10': [(ev.key, ev.count // n_iters,
                   (getattr(ev, 'self_cuda_time_total', 0) or 0) / n_iters)
                  for ev in sorted(events,
                                    key=lambda e: -(getattr(e, 'self_cuda_time_total', 0) or 0))[:10]
                  if (getattr(ev, 'self_cuda_time_total', 0) or 0) > 0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('-T', type=int, default=16)
    p.add_argument('-B', type=int, default=8)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: T={args.T}, B={args.B}, SEW-ResNet-18 @ 224x224\n")

    # Build SJ torch
    torch.manual_seed(0)
    net_torch = build_model().to(device)
    init_bn_random(net_torch)

    # Build SJ cupy (deepcopy + set_backend)
    net_cupy = copy.deepcopy(net_torch)
    functional.set_backend(net_cupy, 'cupy', instance=neuron.LIFNode)

    # Build CATFuse
    net_ctf_base = copy.deepcopy(net_torch)
    net_ctf, cov = substitute(net_ctf_base, verbose=False)
    net_ctf = net_ctf.to(device)
    print(f"CATFuse coverage: {cov.get('coverage_pct', 0):.1f}%\n")

    x = torch.randn(args.T, args.B, 3, 224, 224, device=device)

    # Profile each
    results = []
    for net, label in [(net_torch, 'SJ torch'),
                       (net_cupy, 'SJ cupy'),
                       (net_ctf, 'CATFuse')]:
        print(f"--- Profiling {label} ---")
        r = profile_one(net, x, label)
        results.append(r)
        print(f"  Total CUDA time/iter : {r['total_cuda_us_per_iter']/1000:.2f} ms")
        print(f"  Kernel launches/iter : {r['kernel_count_per_iter']:.0f}")
        print(f"  Op-type breakdown (ms/iter):")
        for k, v in r['op_breakdown_us'].items():
            if v > 1:
                print(f"    {k:20s}: {v/1000:7.3f}")
        print(f"  Top-10 operators by CUDA time:")
        for name, count, t_us in r['top10']:
            print(f"    {name[:50]:50s}  n={count:4.0f}  {t_us/1000:6.2f} ms")
        print()

    # Summary table for paper
    print(f"\n{'='*80}")
    print(f"=== PROFILER SUMMARY (for paper Table) ===")
    print(f"{'='*80}")
    print(f"{'Backend':<12s} {'CUDA ms/iter':>13s} {'Launches/iter':>14s} "
          f"{'Launch reduction':>18s}")
    print(f"{'-'*12} {'-'*13} {'-'*14} {'-'*18}")
    torch_launches = results[0]['kernel_count_per_iter']
    for r in results:
        reduction = torch_launches / r['kernel_count_per_iter']
        print(f"{r['label']:<12s} {r['total_cuda_us_per_iter']/1000:>13.2f} "
              f"{r['kernel_count_per_iter']:>14.0f} "
              f"{reduction:>17.2f}x")

    print(f"\n=== OP-TYPE BREAKDOWN (ms/iter) ===")
    groups_list = ['Conv', 'BN', 'LIF/elementwise', 'Triton-fused',
                   'Memcpy/other', 'uncategorized']
    print(f"{'Backend':<12s}", end='')
    for g in groups_list:
        print(f" {g:>16s}", end='')
    print()
    for r in results:
        print(f"{r['label']:<12s}", end='')
        for g in groups_list:
            v = r['op_breakdown_us'].get(g, 0)
            print(f" {v/1000:>16.2f}", end='')
        print()


if __name__ == '__main__':
    main()