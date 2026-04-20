#!/usr/bin/env python
"""
Phase C-4 v3 — ImageNet-scale (224x224) benchmark with 3 backends.

Backends:
  1. SJ torch   (reference)
  2. SJ cupy    (SJ's optimized hand-written cuda backend, strongest SNN baseline)
  3. CATFuse    (ours)

14-network matrix × T sweep.
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
def benchmark_wallclock(net, x, n_iters, n_warmup, n_repeats, label):
    try:
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
        return mean_ms, std_ms
    except Exception as ex:
        print(f"  [{label}] benchmark FAILED: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        return None, None


@torch.no_grad()
def check_parity(net_ref, net_other, x):
    net_ref.eval(); net_other.eval()
    out_ref = net_ref(x); functional.reset_net(net_ref)
    out_o = net_other(x); functional.reset_net(net_other)
    d = (out_ref - out_o).abs()
    eq = torch.equal(out_ref, out_o)
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
    return alive, len(rates)


def run_one(model_name, builder, T, B, device, args):
    print(f"\n{'='*60}")
    print(f"  {model_name}  T={T}  B={B}  device={device}")
    print(f"{'='*60}", flush=True)

    try:
        # Build SJ torch reference
        torch.manual_seed(args.seed)
        net_torch = builder(v_threshold=args.v_threshold, tau=args.tau).to(device)
        init_bn_random(net_torch)

        # Build SJ cupy (deepcopy + set_backend)
        net_cupy = copy.deepcopy(net_torch)
        functional.set_backend(net_cupy, 'cupy', instance=neuron.LIFNode)

        # Build CATFuse
        net_ctf_base = copy.deepcopy(net_torch)
        net_ctf, cov = substitute(net_ctf_base, verbose=False)
        net_ctf = net_ctf.to(device)

        fused = cov.get('fused_lif_nodes', 0)
        total = cov.get('total_lif_nodes', 0)
        pct = cov.get('coverage_pct', 0)
        print(f"  Coverage: {fused}/{total} = {pct:.1f}%", flush=True)

        x = make_input(T, B, device)

        alive, n_lif = check_liveness(net_torch, x)
        print(f"  Liveness: {alive}/{n_lif} LIF alive", flush=True)

        eq_ctf, max_ctf, mean_ctf = check_parity(net_torch, net_ctf, x)
        print(f"  Parity (torch vs CTF):  exact={eq_ctf}, max_diff={max_ctf:.2e}",
              flush=True)
        eq_cupy, max_cupy, mean_cupy = check_parity(net_torch, net_cupy, x)
        print(f"  Parity (torch vs cupy): exact={eq_cupy}, max_diff={max_cupy:.2e}",
              flush=True)

        # Benchmark 3 backends
        t_torch, s_torch = benchmark_wallclock(
            net_torch, x, args.n_iters, args.n_warmup, args.n_repeats, 'SJ torch')
        if t_torch is not None:
            print(f"  SJ torch:  {t_torch:8.3f} ± {s_torch:.3f} ms", flush=True)

        t_cupy, s_cupy = benchmark_wallclock(
            net_cupy, x, args.n_iters, args.n_warmup, args.n_repeats, 'SJ cupy')
        if t_cupy is not None:
            print(f"  SJ cupy:   {t_cupy:8.3f} ± {s_cupy:.3f} ms", flush=True)

        t_ctf, s_ctf = benchmark_wallclock(
            net_ctf, x, args.n_iters, args.n_warmup, args.n_repeats, 'CATFuse')
        if t_ctf is not None:
            print(f"  CATFuse:   {t_ctf:8.3f} ± {s_ctf:.3f} ms", flush=True)

        speedup_vs_torch = t_torch / t_ctf if (t_torch and t_ctf) else None
        speedup_vs_cupy  = t_cupy / t_ctf if (t_cupy and t_ctf) else None
        if speedup_vs_torch:
            print(f"  Speedup vs torch: {speedup_vs_torch:.3f}x", flush=True)
        if speedup_vs_cupy:
            print(f"  Speedup vs cupy:  {speedup_vs_cupy:.3f}x", flush=True)

        result = {
            'model': model_name, 'T': T, 'B': B,
            'coverage_pct': pct, 'alive': alive, 'n_lif': n_lif,
            'parity_ctf_exact': eq_ctf, 'max_diff_ctf': max_ctf,
            'parity_cupy_exact': eq_cupy, 'max_diff_cupy': max_cupy,
            'torch_ms': t_torch, 'torch_std': s_torch,
            'cupy_ms': t_cupy, 'cupy_std': s_cupy,
            'ctf_ms': t_ctf, 'ctf_std': s_ctf,
            'sp_vs_torch': speedup_vs_torch,
            'sp_vs_cupy': speedup_vs_cupy,
        }

        del net_torch, net_cupy, net_ctf, net_ctf_base, x
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM!", flush=True)
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

    print(f"=== Phase C-4 v3: ImageNet-Scale Benchmark, 3 backends ===")
    print(f"Hardware: {hw}")
    print(f"Input: [T, {args.batch_size}, 3, 224, 224]")
    print(f"Models ({len(args.models)}): {args.models}")
    print(f"T values: {args.T_list}")
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    results = []
    for model_name in args.models:
        builder = MODELS.get(model_name)
        if builder is None:
            continue
        for T in args.T_list:
            r = run_one(model_name, builder, T, args.batch_size, device, args)
            if r is not None:
                results.append(r)

    # Summary
    print(f"\n\n{'='*110}")
    print(f"=== SUMMARY TABLE (3 backends) ===")
    print(f"{'='*110}")
    print(f"Hardware: {hw}")
    print(f"Input: [T, B={args.batch_size}, 3, 224, 224], v_th={args.v_threshold}, tau={args.tau}")
    print(f"")
    print(f"{'Model':<20s} {'T':>3s} {'Cov%':>5s} "
          f"{'torch ms':>9s} {'cupy ms':>9s} {'CTF ms':>8s} "
          f"{'vs torch':>9s} {'vs cupy':>9s} "
          f"{'P-CTF':>8s} {'P-cupy':>8s}")
    print(f"{'-'*20} {'-'*3} {'-'*5} {'-'*9} {'-'*9} {'-'*8} "
          f"{'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for r in results:
        p_ctf = 'exact' if r['parity_ctf_exact'] else f"{r['max_diff_ctf']:.1e}"
        p_cupy = 'exact' if r['parity_cupy_exact'] else f"{r['max_diff_cupy']:.1e}"
        torch_str = f"{r['torch_ms']:8.2f}" if r['torch_ms'] else "  N/A  "
        cupy_str = f"{r['cupy_ms']:8.2f}" if r['cupy_ms'] else "  N/A  "
        ctf_str = f"{r['ctf_ms']:7.2f}" if r['ctf_ms'] else "  N/A "
        sp_t = f"{r['sp_vs_torch']:7.3f}x" if r['sp_vs_torch'] else "  N/A  "
        sp_c = f"{r['sp_vs_cupy']:7.3f}x" if r['sp_vs_cupy'] else "  N/A  "
        print(f"{r['model']:<20s} {r['T']:>3d} {r['coverage_pct']:>4.0f}% "
              f"{torch_str:>9s} {cupy_str:>9s} {ctf_str:>8s} "
              f"{sp_t:>9s} {sp_c:>9s} {p_ctf:>8s} {p_cupy:>8s}")

    print(f"\nEnd: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()