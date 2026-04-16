"""
Phase 4 Task 4.2 — SEWResNet18 end-to-end evaluation.

Parallels bench_vgg11_e2e.py with three fixes from task 4.1 diagnosis:

  Fix 1: surrogate_function=surrogate.Sigmoid() (not None), so unreplaced
         top-level sn1 doesn't crash on NoneType callable.

  Fix 2: input_scale=1.0 by default (VGG had 0.3 which gave spike rate ~0,
         making parity test degenerate). SEWResNet adds should keep activations
         alive across layers.

  Fix 3: speedup table computed correctly: sj_torch/ctf and sj_cupy/ctf as
         separate values, not via a stale variable.

Baselines:
  (a) CTF substituted model
  (b) SJ torch backend    (parity ground truth + wall-clock baseline)
  (c) SJ cupy backend     (wall-clock baseline, SJ's fastest)

Shape: [T=16, B=32, C=3, H=W=64], num_classes=10.
Protocol: 12 repeats × 100 iter, trimmed mean, 20 warmup. 3 seeds for stability.
"""

# ============================================================
# Suppress noisy warnings BEFORE any other imports
# ============================================================
import warnings
import logging

# Suppress cuDNN nvrtc workaround UserWarning (triggered on every F.conv2d call)
warnings.filterwarnings('ignore', message='.*Applied workaround for CuDNN.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.conv')

# Suppress SJ reset_net warnings about non-MemoryModule CTF patterns.
# These come from logging.warning() calls in SJ's functional.reset_net.
logging.getLogger().setLevel(logging.ERROR)
# More targeted: silence SJ's root logger used by reset_net
for logger_name in ['root', 'spikingjelly']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import copy
import json
import statistics
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.activation_based import functional, neuron, surrogate

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from catfuse_substitute import substitute, print_coverage_report


# ============================================================
# Timing
# ============================================================

def cuda_time_one_shot(fn, n_iter):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def cuda_time_trimmed(fn, n_iter=100, n_repeat=12, n_warmup=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    sorted_s = sorted(samples)
    trimmed = sorted_s[1:-1]
    return {
        'trimmed_mean_ms': statistics.mean(trimmed),
        'trimmed_stdev_ms': statistics.stdev(trimmed),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'all_samples_ms': samples,
    }


# ============================================================
# Model builders
# ============================================================

def build_sew_model(num_classes: int = 10, v_threshold: float = 0.1):
    """
    Build SEWResNet18 with:
      - Sigmoid surrogate (so unreplaced top-level sn1 works in forward)
      - v_threshold lowered from SJ default 1.0 to keep untrained
        random-weight network alive across all layers (Phase 4.2 diagnostic
        found that v_threshold=1.0 + random init gives spike rate=0 past layer1)
    """
    model = sew_resnet.sew_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        num_classes=num_classes,
        cnf='ADD',
        v_threshold=v_threshold,
    )
    functional.set_step_mode(model, 'm')
    return model


def make_all_models_consistent(num_classes: int = 10, seed: int = 0,
                                v_threshold: float = 0.1):
    torch.manual_seed(seed)
    sj_torch = build_sew_model(num_classes, v_threshold=v_threshold)

    for m in sj_torch.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.running_mean.data.normal_(0, 0.1)
            m.running_var.data.uniform_(0.5, 1.5)

    sj_cupy = copy.deepcopy(sj_torch)
    functional.set_backend(sj_cupy, 'cupy')

    ctf_model, coverage = substitute(sj_torch, verbose=False)

    return {
        'sj_torch': sj_torch,
        'sj_cupy': sj_cupy,
        'ctf': ctf_model,
    }, coverage


# ============================================================
# Parity check
# ============================================================

def run_parity(models: dict, x: torch.Tensor) -> dict:
    results = {}

    with torch.no_grad():
        functional.reset_net(models['sj_torch'])
        out_ref = models['sj_torch'](x)
        functional.reset_net(models['sj_cupy'])
        out_cupy = models['sj_cupy'](x)
        functional.reset_net(models['ctf'])
        out_ctf = models['ctf'](x)

    ref_norm = out_ref.abs().max().item() + 1e-9

    def cmp(cand, ref, tag):
        if cand.shape != ref.shape:
            return {
                'tag': tag,
                'shape_match': False,
                'cand_shape': list(cand.shape),
                'ref_shape': list(ref.shape),
            }
        diff = (cand - ref).abs()
        return {
            'tag': tag,
            'shape_match': True,
            'shape': list(cand.shape),
            'max_diff': diff.max().item(),
            'mean_diff': diff.mean().item(),
            'rel_diff_max': diff.max().item() / ref_norm,
            'bit_exact': bool((cand == ref).all().item()),
            'ref_max_abs': ref.abs().max().item(),
            'ref_mean_abs': ref.abs().mean().item(),
        }

    results['sj_cupy_vs_sj_torch'] = cmp(out_cupy, out_ref, 'sj_cupy')
    results['ctf_vs_sj_torch'] = cmp(out_ctf, out_ref, 'ctf')
    return results


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--H', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=12)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--input_scale', type=float, default=1.0)
    p.add_argument('--v_threshold', type=float, default=0.1,
                   help='LIF v_threshold (default 0.1; SJ default 1.0 is too '
                        'high for untrained random-weight networks, causes '
                        'all-zero spike cascade past layer1)')
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f'./results/phase4/sew_resnet18_e2e_v100_seed{args.seed}.json'

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"Phase 4 task 4.2b — SEWResNet18 e2e evaluation (fixed)")
    print(f"  GPU         : {torch.cuda.get_device_name(device)}")
    print(f"  shape       : T={args.T} B={args.B} C=3 H=W={args.H}")
    print(f"  input_scale : {args.input_scale}")
    print(f"  v_threshold : {args.v_threshold}")
    print(f"  seed        : {args.seed}")

    # ----- Build models -----
    print(f"\n{'='*86}")
    print(f"Building models")
    print(f"{'='*86}")
    models, coverage = make_all_models_consistent(
        num_classes=args.num_classes, seed=args.seed,
        v_threshold=args.v_threshold,
    )
    for name, m in models.items():
        models[name] = m.to(device).eval()

    print_coverage_report(coverage, title='CTF coverage (SEWResNet18)')

    # ----- Build input -----
    x = (torch.randn(args.T, args.B, 3, args.H, args.H,
                     device=device, dtype=torch.float32) * args.input_scale).contiguous()
    print(f"\n  input: mean={x.mean().item():.4f} "
          f"std={x.std().item():.4f} "
          f"abs_max={x.abs().max().item():.4f}")

    # ----- Liveness check -----
    print(f"\n{'='*86}")
    print(f"Liveness check (is the network alive under this configuration?)")
    print(f"{'='*86}")

    # Hook each CTFSEWBasicBlock on the CTF model to capture spike rates
    block_outputs = {}
    hooks = []
    def make_hook(name):
        def hook(module, input, output):
            block_outputs[name] = {
                'spike_rate': output.mean().item(),
                'shape': list(output.shape),
            }
        return hook
    for name, module in models['ctf'].named_modules():
        if type(module).__name__ == 'CTFSEWBasicBlock':
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        functional.reset_net(models['ctf'])
        _ = models['ctf'](x)

    for h in hooks:
        h.remove()

    for name, info in block_outputs.items():
        status = 'ALIVE' if info['spike_rate'] > 1e-4 else 'DEAD'
        print(f"  {name:<15} spike_rate={info['spike_rate']:.6f}  "
              f"shape={info['shape']}  {status}")

    n_alive = sum(1 for info in block_outputs.values() if info['spike_rate'] > 1e-4)
    total_blocks = len(block_outputs)
    alive_frac = n_alive / total_blocks if total_blocks > 0 else 0
    print(f"\n  {n_alive}/{total_blocks} blocks alive "
          f"({alive_frac * 100:.1f}%)")

    if alive_frac < 0.8:
        print(f"\n  LIVENESS CHECK FAILED: less than 80% of blocks are alive. "
              f"The network is degenerate; wall-clock numbers will measure "
              f"kernel launches on empty compute, not real inference. "
              f"Aborting bench. Lower v_threshold or larger input_scale.")
        return

    print(f"\n  LIVENESS OK: network is alive through all layers")

    # ----- Parity check -----
    print(f"\n{'='*86}")
    print(f"Parity check (vs SJ torch ground truth)")
    print(f"{'='*86}")
    parity = run_parity(models, x)
    for key, res in parity.items():
        tag = res['tag']
        if not res.get('shape_match', True):
            print(f"  [{tag:<15}] SHAPE MISMATCH")
            continue
        print(f"  [{tag:<15}] shape={res['shape']}  ref_max_abs={res['ref_max_abs']:.6e}")
        print(f"                   max_diff={res['max_diff']:.6e}  "
              f"rel_diff_max={res['rel_diff_max']:.6e}  "
              f"bit_exact={res['bit_exact']}")

    ref_max = parity['ctf_vs_sj_torch']['ref_max_abs']
    ctf_rel_diff = parity['ctf_vs_sj_torch']['rel_diff_max']

    if ref_max < 1e-3:
        print(f"\n  PARITY FAIL: ref_max_abs {ref_max:.2e} < 1e-3 is degenerate. "
              f"Bench output is not meaningful. Aborting wall-clock.")
        return
    elif ctf_rel_diff > 1e-1:
        print(f"\n  PARITY WARNING: CTF rel_diff {ctf_rel_diff:.6e} > 1e-1, "
              f"significant numerical divergence.")
    else:
        print(f"\n  PARITY OK: CTF rel_diff {ctf_rel_diff:.6e}")

    # ----- Wall-clock -----
    print(f"\n{'='*86}")
    print(f"Wall-clock (trimmed mean N={args.n_repeat-2} of {args.n_repeat}, "
          f"{args.n_iter} iter, {args.n_warmup} warmup)")
    print(f"{'='*86}")

    wall_results = {}

    def bench_model(tag, model):
        def fn():
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x)
        stats = cuda_time_trimmed(
            fn, n_iter=args.n_iter,
            n_repeat=args.n_repeat, n_warmup=args.n_warmup,
        )
        wall_results[tag] = stats
        print(f"  {tag:<15} {stats['trimmed_mean_ms']:>10.4f} ms  "
              f"(stdev {stats['trimmed_stdev_ms']:.4f}, "
              f"min {stats['min_ms']:.4f}, max {stats['max_ms']:.4f})")

    print(f"  {'backend':<15} {'trimmed_mean':>10}")
    print(f"  {'-'*66}")

    bench_model('sj_torch', models['sj_torch'])
    bench_model('sj_cupy', models['sj_cupy'])
    bench_model('ctf', models['ctf'])

    # ----- Speedup table -----
    print(f"\n{'='*86}")
    print(f"Speedups (sj_xxx / ctf, >1 means CTF is faster)")
    print(f"{'='*86}")
    ctf_ms = wall_results['ctf']['trimmed_mean_ms']
    sj_torch_ms = wall_results['sj_torch']['trimmed_mean_ms']
    sj_cupy_ms = wall_results['sj_cupy']['trimmed_mean_ms']

    s_torch = sj_torch_ms / ctf_ms
    s_cupy = sj_cupy_ms / ctf_ms
    print(f"  sj_torch / ctf : {s_torch:>7.4f}x  "
          f"(CTF {'faster' if s_torch > 1 else 'slower'} "
          f"by {abs(s_torch - 1) * 100:.1f}%)")
    print(f"  sj_cupy  / ctf : {s_cupy:>7.4f}x  "
          f"(CTF {'faster' if s_cupy > 1 else 'slower'} "
          f"by {abs(s_cupy - 1) * 100:.1f}%)")

    # ----- Save -----
    output = {
        'experiment': 'phase4_task4.2_sew_resnet18_e2e',
        'device': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'coverage': coverage,
        'parity': parity,
        'wall_clock': wall_results,
        'speedup': {
            'sj_torch_over_ctf': s_torch,
            'sj_cupy_over_ctf': s_cupy,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (bool, int, float, str, type(None))):
            return o
        return str(o)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=default)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()