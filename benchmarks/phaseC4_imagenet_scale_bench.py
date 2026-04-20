#!/usr/bin/env python
"""
Phase C-4 — ImageNet-scale (224×224) random-init benchmark.

FULL 14-NETWORK MATRIX (ResNet + VGG families):

  SEW-ResNet:       18, 34, 50, 101, 152
  Spiking-ResNet:   18, 34, 50, 101, 152
  Spiking-VGG_bn:   11, 13, 16, 19

Config:
  - Input: [T, B, 3, 224, 224]
  - v_threshold=0.1 (random init workaround)
  - B=8 default
  - T sweep: 4, 8, 16, 32

Usage:
    # A100 all 14 networks × 4 T values
    /data_priv/dagongcheng/snn118/bin/python phaseC4_imagenet_scale_bench.py --device cuda:0

    # V100 (needs CUDA_VISIBLE_DEVICES for Triton context)
    CUDA_VISIBLE_DEVICES=1 /data_priv/dagongcheng/snn118/bin/python \
        phaseC4_imagenet_scale_bench.py --device cuda:0
"""
import argparse
import copy
import datetime
import gc
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore', message='.*Applied workaround for CuDNN.*')
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger().setLevel(logging.ERROR)
for name in ['root', 'spikingjelly']:
    logging.getLogger(name).setLevel(logging.ERROR)

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from catfuse_substitute import substitute


# ============================================================
# Model builders
# ============================================================
def _build_sew(depth):
    fn = getattr(sew_resnet, f'sew_resnet{depth}')
    def builder(**kw):
        m = fn(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
            detach_reset=True, num_classes=1000, cnf='ADD',
            v_threshold=kw.get('v_threshold', 0.1), tau=kw.get('tau', 2.0))
        functional.set_step_mode(m, 'm')
        return m
    return builder

def _build_spiking(depth):
    fn = getattr(spiking_resnet, f'spiking_resnet{depth}')
    def builder(**kw):
        m = fn(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
            detach_reset=True, num_classes=1000,
            v_threshold=kw.get('v_threshold', 0.1), tau=kw.get('tau', 2.0))
        functional.set_step_mode(m, 'm')
        return m
    return builder

def _build_vgg(depth):
    fn = getattr(spiking_vgg, f'spiking_vgg{depth}_bn')
    def builder(**kw):
        m = fn(
            spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.Sigmoid(),
            detach_reset=True, num_classes=1000,
            v_threshold=kw.get('v_threshold', 0.1), tau=kw.get('tau', 2.0))
        functional.set_step_mode(m, 'm')
        return m
    return builder


MODELS = {
    'sew_resnet18':      _build_sew(18),
    'sew_resnet34':      _build_sew(34),
    'sew_resnet50':      _build_sew(50),
    'sew_resnet101':     _build_sew(101),
    'sew_resnet152':     _build_sew(152),
    'spiking_resnet18':  _build_spiking(18),
    'spiking_resnet34':  _build_spiking(34),
    'spiking_resnet50':  _build_spiking(50),
    'spiking_resnet101': _build_spiking(101),
    'spiking_resnet152': _build_spiking(152),
    'vgg11':             _build_vgg(11),
    'vgg13':             _build_vgg(13),
    'vgg16':             _build_vgg(16),
    'vgg19':             _build_vgg(19),
}


def init_bn_random(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.running_mean.data.normal_(0, 0.1)
            m.running_var.data.uniform_(0.5, 1.5)


def make_input(T, B, device):
    return torch.randn(T, B, 3, 224, 224, device=device)


@torch.no_grad()
def benchmark_wallclock(net, x, n_iters, n_warmup, n_repeats):
    net.eval()
    for _ in range(n_warmup):
        _ = net(x).mean(dim=0)
        functional.reset_net(net)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            _ = net(x).mean(dim=0)
            functional.reset_net(net)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    ts = sorted(times)
    trimmed = ts[1:-1] if len(ts) >= 3 else ts
    mean_ms = sum(trimmed) / len(trimmed)
    std_ms = (sum((t - mean_ms)**2 for t in trimmed) / max(len(trimmed), 1))**0.5
    return mean_ms, std_ms, times


@torch.no_grad()
def check_parity(net_ref, net_ctf, x):
    net_ref.eval(); net_ctf.eval()
    out_ref = net_ref(x); functional.reset_net(net_ref)
    out_ctf = net_ctf(x); functional.reset_net(net_ctf)
    d = (out_ref - out_ctf).abs()
    eq = torch.equal(out_ref, out_ctf)
    return eq, d.max().item(), d.mean().item()


def check_liveness(net, x):
    net.eval()
    hooks, rates = [], []
    def make_hook(name):
        def hook(m, inp, out):
            if out.dtype == torch.float32 and out.max() <= 1.0:
                rates.append((name, out.float().mean().item()))
        return hook
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        net(x)
    functional.reset_net(net)
    for h in hooks:
        h.remove()
    alive = sum(1 for _, r in rates if r > 1e-4)
    return alive, len(rates), rates


def run_one(model_name, builder, T, B, device, args):
    print(f"\n{'='*60}")
    print(f"  {model_name}  T={T}  B={B}  device={device}")
    print(f"{'='*60}", flush=True)

    try:
        torch.manual_seed(args.seed)
        net_ref = builder(v_threshold=args.v_threshold, tau=args.tau).to(device)
        init_bn_random(net_ref)

        net_ctf_base = copy.deepcopy(net_ref)
        net_ctf, cov = substitute(net_ctf_base, verbose=False)
        net_ctf = net_ctf.to(device)

        fused = cov.get('fused_lif_nodes', 0)
        total = cov.get('total_lif_nodes', 0)
        pct = cov.get('coverage_pct', 0)
        patterns = cov.get('patterns_matched', {})
        print(f"  Coverage: {fused}/{total} = {pct:.1f}%  patterns={patterns}", flush=True)

        x = make_input(T, B, device)

        alive, n_lif, _ = check_liveness(net_ref, x)
        print(f"  Liveness: {alive}/{n_lif} LIF alive", flush=True)
        if alive < n_lif * 0.5:
            print(f"  WARNING: <50% LIF alive, data may be degenerate", flush=True)

        eq, max_d, mean_d = check_parity(net_ref, net_ctf, x)
        print(f"  Parity: bit-exact={eq}, max_diff={max_d:.2e}, mean_diff={mean_d:.2e}",
              flush=True)

        t_sj, s_sj, _ = benchmark_wallclock(
            net_ref, x, args.n_iters, args.n_warmup, args.n_repeats)
        print(f"  SJ torch:  {t_sj:8.3f} ± {s_sj:.3f} ms", flush=True)

        t_ctf, s_ctf, _ = benchmark_wallclock(
            net_ctf, x, args.n_iters, args.n_warmup, args.n_repeats)
        speedup = t_sj / t_ctf
        print(f"  CATFuse:   {t_ctf:8.3f} ± {s_ctf:.3f} ms", flush=True)
        print(f"  Speedup:   {speedup:.3f}×", flush=True)

        result = {
            'model': model_name, 'T': T, 'B': B,
            'coverage_pct': pct, 'fused': fused, 'total_lif': total,
            'alive': alive, 'n_lif': n_lif,
            'parity_exact': eq, 'max_diff': max_d,
            'sj_ms': t_sj, 'sj_std': s_sj,
            'ctf_ms': t_ctf, 'ctf_std': s_ctf,
            'speedup': speedup,
        }

        del net_ref, net_ctf, net_ctf_base, x
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM! Skipping {model_name} T={T} B={B}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=list(MODELS.keys()))
    p.add_argument('--T-list', nargs='+', type=int, default=[4, 8, 16, 32])
    p.add_argument('-b', '--batch-size', type=int, default=8)
    p.add_argument('--v-threshold', type=float, default=0.1)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-iters', type=int, default=50)
    p.add_argument('--n-warmup', type=int, default=10)
    p.add_argument('--n-repeats', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)
    hw = torch.cuda.get_device_name(device)

    print(f"=== Phase C-4: ImageNet-Scale Benchmark ({len(args.models)} networks) ===")
    print(f"Hardware: {hw}")
    print(f"Input: [T, {args.batch_size}, 3, 224, 224]")
    print(f"Models: {args.models}")
    print(f"T values: {args.T_list}")
    print(f"v_threshold={args.v_threshold}, tau={args.tau}")
    print(f"Bench: {args.n_iters} iters × {args.n_repeats} repeats, "
          f"{args.n_warmup} warmup")
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    results = []
    for model_name in args.models:
        builder = MODELS.get(model_name)
        if builder is None:
            print(f"Unknown model: {model_name}, skipping")
            continue
        for T in args.T_list:
            r = run_one(model_name, builder, T, args.batch_size, device, args)
            if r is not None:
                results.append(r)

    print(f"\n\n{'='*95}")
    print(f"=== SUMMARY TABLE (paste to chat) ===")
    print(f"{'='*95}")
    print(f"Hardware: {hw}")
    print(f"Input: [T, B={args.batch_size}, 3, 224, 224]")
    print(f"v_threshold={args.v_threshold}, tau={args.tau}, seed={args.seed}")
    print(f"")
    print(f"{'Model':<22s} {'T':>3s} {'Cov%':>5s} {'SJ ms':>9s} {'CTF ms':>9s} "
          f"{'Speedup':>8s} {'Parity':>10s} {'Alive':>7s}")
    print(f"{'-'*22} {'-'*3} {'-'*5} {'-'*9} {'-'*9} {'-'*8} {'-'*10} {'-'*7}")
    for r in results:
        parity_str = 'exact' if r['parity_exact'] else f"{r['max_diff']:.1e}"
        alive_str = f"{r['alive']}/{r['n_lif']}"
        print(f"{r['model']:<22s} {r['T']:>3d} {r['coverage_pct']:>4.0f}% "
              f"{r['sj_ms']:>9.2f} {r['ctf_ms']:>9.2f} "
              f"{r['speedup']:>7.3f}× {parity_str:>10s} {alive_str:>7s}")

    print(f"\nEnd: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()