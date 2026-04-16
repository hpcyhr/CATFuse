"""
Phase 1 baseline comparison: 3 compute-bound subgraphs × 3 reference backends
+ our Triton fusion.

Subgraphs:
  (1) Conv 3x3 s=1 p=1 → LIF            [T=16, B=64, C=128, H=W=28]
  (2) Conv 3x3 s=1 p=1 → BN → LIF       [same shape]
  (3) Linear(512→512) → LIF             [T=16, B=32, I=O=512]

Backends under test:
  (a) SpikingJelly multi-step, torch backend           [reference]
  (b) SpikingJelly multi-step, cupy backend
  (c) torch.compile(sj_torch_version, mode='reduce-overhead')
  (d) our Triton fusion (from Phase 0)

Parity policy:
  - SJ torch is the reference (SNN-community ground truth).
  - SJ cupy vs SJ torch: expect max_diff < 1e-3, spike_match > 99.9%.
  - torch.compile vs SJ torch: same tolerance.
  - Triton fusion vs SJ torch: same tolerance.

Wall-clock:
  - Per (subgraph, backend): 12 repeats × 100 inner iters, trimmed mean
    over the middle 10 (drop max + min). Same protocol as Phase 0.
  - 20-iter warmup BEFORE timing to absorb JIT/compile startup.

Usage:
  python benchmarks/sj_baseline.py --gpu 0 --output results/phase1/sj_baseline_v100.json

Output:
  A single JSON with all data for this host (V100 or A100).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

import triton

# SpikingJelly
from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import functional as sj_functional

# Our Triton fusion kernels — import from the existing benchmark files.
# These must be on PYTHONPATH (the `benchmarks/` folder, where this file lives).
from conv_lif_min_v2 import run_fusion as triton_conv_lif_fusion
from conv_bn_lif_min import run_fusion as triton_conv_bn_lif_fusion
from linear_lif_k_sweep_v3 import run_fusion_k as triton_linear_lif_fusion


# ============================================================
# Subgraph factories (SpikingJelly layer modules)
# ============================================================

def make_sj_conv_lif(C_in, C_out, k, stride, padding, tau, v_th, v_reset):
    """Subgraph 1: Conv2d → LIF, multi-step."""
    net = nn.Sequential(
        sj_layer.Conv2d(C_in, C_out, kernel_size=k, stride=stride,
                        padding=padding, bias=False),
        sj_neuron.LIFNode(tau=tau, v_threshold=v_th, v_reset=v_reset,
                          step_mode='m'),
    )
    sj_functional.set_step_mode(net, 'm')
    return net


def make_sj_conv_bn_lif(C_in, C_out, k, stride, padding, tau, v_th, v_reset):
    """Subgraph 2: Conv2d → BN → LIF, multi-step, BN in inference mode."""
    net = nn.Sequential(
        sj_layer.Conv2d(C_in, C_out, kernel_size=k, stride=stride,
                        padding=padding, bias=False),
        sj_layer.BatchNorm2d(C_out),
        sj_neuron.LIFNode(tau=tau, v_threshold=v_th, v_reset=v_reset,
                          step_mode='m'),
    )
    sj_functional.set_step_mode(net, 'm')
    return net


def make_sj_linear_lif(I, O, tau, v_th, v_reset):
    """Subgraph 3: Linear → LIF, multi-step."""
    net = nn.Sequential(
        sj_layer.Linear(I, O, bias=False),
        sj_neuron.LIFNode(tau=tau, v_threshold=v_th, v_reset=v_reset,
                          step_mode='m'),
    )
    sj_functional.set_step_mode(net, 'm')
    return net


# ============================================================
# Weight initialization and sharing
# ============================================================

def copy_conv_weight_from_triton(sj_net, w_triton):
    """Load Triton conv weight into SJ's Conv2d."""
    # sj_net[0] is layer.Conv2d, which wraps an nn.Conv2d; .weight is
    # a Parameter of shape [C_out, C_in, k, k] — same as Triton expects.
    sj_net[0].weight.data.copy_(w_triton)


def copy_linear_weight_from_triton(sj_net, w_triton):
    """Load Triton linear weight into SJ's Linear.

    Triton stores w as [I, O]; PyTorch Linear expects [O, I].
    """
    sj_net[0].weight.data.copy_(w_triton.t())


def set_bn_inference_mode(sj_net, scale, bias, eps=1e-5):
    """Set SJ's BatchNorm2d (layer position [1]) to inference-mode affine
    matching our fusion's (scale, bias).

    Our fusion uses:  z_bn = z * scale[c] + bias[c]
    SJ's BN in eval() computes:
        z_bn = (z - running_mean) / sqrt(running_var + eps) * gamma + beta

    To make SJ match, we set:
        running_mean = 0, running_var = 1 - eps  (so denom ≈ 1)
        gamma = scale, beta = bias
    """
    bn = sj_net[1]
    bn.eval()
    bn.running_mean.data.zero_()
    bn.running_var.data.fill_(1.0 - eps)
    bn.weight.data.copy_(scale)   # gamma
    bn.bias.data.copy_(bias)      # beta


# ============================================================
# Backend runners (one function per backend)
# ============================================================

def run_sj_torch(net, x):
    """Run SJ net in torch backend. Always reset state before forward."""
    sj_functional.reset_net(net)
    return net(x)


def run_sj_cupy(net, x):
    """Switch SJ net to cupy backend and run. Requires cupy installed."""
    sj_functional.reset_net(net)
    return net(x)


def compile_net_reduce_overhead(net):
    """Wrap net in torch.compile with reduce-overhead mode."""
    return torch.compile(net, mode='reduce-overhead', fullgraph=False)


def run_compiled(compiled_net, x):
    sj_functional.reset_net(compiled_net)
    return compiled_net(x)


# ============================================================
# Parity helpers
# ============================================================

def parity_stats(out_a, out_b, tag_a, tag_b):
    d = (out_a - out_b).abs()
    spike_match = (out_a == out_b).float().mean().item() * 100
    return {
        'a': tag_a,
        'b': tag_b,
        'max_diff': d.max().item(),
        'mean_abs_diff': d.mean().item(),
        'spike_match_pct': spike_match,
        'bit_exact': bool((out_a == out_b).all().item()),
        'spike_rate_a': out_a.mean().item(),
        'spike_rate_b': out_b.mean().item(),
    }


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
    """Trimmed mean over n_repeat samples, dropping max + min."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    sorted_s = sorted(samples)
    trimmed = sorted_s[1:-1]  # drop 1 max + 1 min
    return {
        'trimmed_mean_ms': statistics.mean(trimmed),
        'trimmed_stdev_ms': statistics.stdev(trimmed),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'n_iter': n_iter,
        'n_repeat': n_repeat,
        'n_warmup': n_warmup,
        'all_samples_ms': samples,
    }


# ============================================================
# Subgraph configurations
# ============================================================

def get_subgraph_configs(args):
    """Return a list of subgraph configs for this run."""
    return [
        {
            'name': 'conv_lif',
            'T': 16, 'B': 64, 'C_in': 128, 'C_out': 128,
            'H': 28, 'W': 28, 'k': 3, 'stride': 1, 'padding': 1,
            'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset,
            'has_bn': False, 'kind': 'conv',
        },
        {
            'name': 'conv_bn_lif',
            'T': 16, 'B': 64, 'C_in': 128, 'C_out': 128,
            'H': 28, 'W': 28, 'k': 3, 'stride': 1, 'padding': 1,
            'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset,
            'has_bn': True, 'kind': 'conv',
        },
        {
            'name': 'linear_lif',
            'T': 16, 'B': 32, 'I': 512, 'O': 512,
            'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset,
            'has_bn': False, 'kind': 'linear',
        },
    ]


# ============================================================
# One subgraph = one self-contained experiment
# ============================================================

def run_subgraph_experiment(cfg, args, device):
    """Run one subgraph across all 4 backends. Returns a results dict."""
    print(f"\n{'='*86}")
    print(f"Subgraph: {cfg['name']}")
    print(f"{'='*86}")

    torch.manual_seed(args.seed)

    # --- Generate shared weights and inputs ---
    if cfg['kind'] == 'conv':
        T, B, C_in, C_out = cfg['T'], cfg['B'], cfg['C_in'], cfg['C_out']
        H, W = cfg['H'], cfg['W']
        k, s, p = cfg['k'], cfg['stride'], cfg['padding']

        x = (torch.randn(T, B, C_in, H, W, device=device, dtype=torch.float32)
             * args.input_scale).contiguous()
        w = (torch.randn(C_out, C_in, k, k, device=device, dtype=torch.float32)
             * args.input_scale).contiguous()

        if cfg['has_bn']:
            scale = (1.0 + torch.randn(C_out, device=device,
                                       dtype=torch.float32) * 0.1).contiguous()
            bias = (torch.randn(C_out, device=device,
                                dtype=torch.float32) * 0.1).contiguous()
        else:
            scale, bias = None, None

        print(f"  shape: T={T} B={B} C_in={C_in} C_out={C_out} H={H} W={W}")
        print(f"  conv:  k={k} s={s} p={p}  bn={cfg['has_bn']}")
    elif cfg['kind'] == 'linear':
        T, B, I, O = cfg['T'], cfg['B'], cfg['I'], cfg['O']
        x = (torch.randn(T, B, I, device=device, dtype=torch.float32)
             * args.input_scale).contiguous()
        w = (torch.randn(I, O, device=device, dtype=torch.float32)
             * args.input_scale).contiguous()
        scale, bias = None, None
        print(f"  shape: T={T} B={B} I={I} O={O}")
    else:
        raise ValueError(cfg['kind'])

    results = {
        'name': cfg['name'],
        'kind': cfg['kind'],
        'config': cfg,
        'backends': {},
        'parity': {},
    }

    # --- Build SJ net (shared between torch and cupy backends) ---
    if cfg['kind'] == 'conv':
        if cfg['has_bn']:
            sj_net = make_sj_conv_bn_lif(C_in, C_out, k, s, p,
                                          cfg['tau'], cfg['v_th'], cfg['v_reset'])
            copy_conv_weight_from_triton(sj_net, w)
            set_bn_inference_mode(sj_net, scale, bias)
        else:
            sj_net = make_sj_conv_lif(C_in, C_out, k, s, p,
                                       cfg['tau'], cfg['v_th'], cfg['v_reset'])
            copy_conv_weight_from_triton(sj_net, w)
    else:
        sj_net = make_sj_linear_lif(I, O, cfg['tau'], cfg['v_th'], cfg['v_reset'])
        copy_linear_weight_from_triton(sj_net, w)

    sj_net = sj_net.to(device)

    # --- Build callable for each backend ---
    # (a) SJ torch
    sj_functional.set_backend(sj_net, 'torch')
    call_sj_torch = lambda: run_sj_torch(sj_net, x)

    # Run once to get a torch-backend reference output
    with torch.no_grad():
        out_sj_torch = call_sj_torch().detach().clone()
    print(f"  [sj_torch ]  ran OK, output shape: {list(out_sj_torch.shape)}, "
          f"spike_rate: {out_sj_torch.mean().item():.4f}")

    # (d) Triton fusion
    with torch.no_grad():
        if cfg['kind'] == 'conv':
            if cfg['has_bn']:
                out_triton = triton_conv_bn_lif_fusion(
                    x, w, scale, bias,
                    tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
                    k_conv=k, padding=p, stride=s,
                ).detach().clone()
            else:
                out_triton = triton_conv_lif_fusion(
                    x, w,
                    tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
                    k_conv=k, padding=p, stride=s,
                ).detach().clone()
        else:
            out_triton = triton_linear_lif_fusion(
                x, w, K=cfg['T'],
                tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
            ).detach().clone()

    print(f"  [triton   ]  ran OK, output shape: {list(out_triton.shape)}, "
          f"spike_rate: {out_triton.mean().item():.4f}")

    # --- Parity: triton vs sj_torch ---
    triton_parity = parity_stats(out_triton, out_sj_torch, 'triton_fusion', 'sj_torch')
    results['parity']['triton_vs_sj_torch'] = triton_parity
    print(f"  [parity   ]  triton vs sj_torch : "
          f"max_diff={triton_parity['max_diff']:.2e}, "
          f"spike_match={triton_parity['spike_match_pct']:.4f}%")

    # --- (b) SJ cupy ---
    try:
        sj_functional.set_backend(sj_net, 'cupy')
        call_sj_cupy = lambda: run_sj_cupy(sj_net, x)
        with torch.no_grad():
            out_sj_cupy = call_sj_cupy().detach().clone()
        cupy_parity = parity_stats(out_sj_cupy, out_sj_torch, 'sj_cupy', 'sj_torch')
        results['parity']['sj_cupy_vs_sj_torch'] = cupy_parity
        print(f"  [sj_cupy  ]  ran OK, spike_rate: {out_sj_cupy.mean().item():.4f}")
        print(f"  [parity   ]  sj_cupy vs sj_torch: "
              f"max_diff={cupy_parity['max_diff']:.2e}, "
              f"spike_match={cupy_parity['spike_match_pct']:.4f}%")
        sj_cupy_available = True
    except Exception as e:
        print(f"  [sj_cupy  ]  FAILED: {type(e).__name__}: {e}")
        results['parity']['sj_cupy_vs_sj_torch'] = {'error': str(e)}
        sj_cupy_available = False

    # (c) torch.compile — uses SJ net, but must switch back to torch backend
    # because torch.compile can't trace cupy backend.
    try:
        sj_functional.set_backend(sj_net, 'torch')
        compiled_net = compile_net_reduce_overhead(sj_net)
        call_compile = lambda: run_compiled(compiled_net, x)
        with torch.no_grad():
            out_compile = call_compile().detach().clone()
        compile_parity = parity_stats(out_compile, out_sj_torch,
                                      'torch_compile', 'sj_torch')
        results['parity']['torch_compile_vs_sj_torch'] = compile_parity
        print(f"  [compile  ]  ran OK, spike_rate: {out_compile.mean().item():.4f}")
        print(f"  [parity   ]  torch_compile vs sj_torch: "
              f"max_diff={compile_parity['max_diff']:.2e}, "
              f"spike_match={compile_parity['spike_match_pct']:.4f}%")
        compile_available = True
    except Exception as e:
        print(f"  [compile  ]  FAILED: {type(e).__name__}: {e}")
        results['parity']['torch_compile_vs_sj_torch'] = {'error': str(e)}
        compile_available = False

    # --- Wall-clock for each backend ---
    print(f"\n  Wall-clock (trimmed mean, N=12, drop max+min, 100 iter, 20 warmup)")
    print(f"  {'backend':<18} {'trimmed_mean_ms':>18} {'stdev_ms':>12} "
          f"{'min':>8} {'max':>8}")
    print(f"  {'-'*66}")

    def time_and_report(name, fn):
        stats = cuda_time_trimmed(fn, n_iter=args.n_iter,
                                  n_repeat=args.n_repeat, n_warmup=args.n_warmup)
        results['backends'][name] = stats
        print(f"  {name:<18} {stats['trimmed_mean_ms']:>18.4f} "
              f"{stats['trimmed_stdev_ms']:>12.4f} "
              f"{stats['min_ms']:>8.4f} {stats['max_ms']:>8.4f}")

    # Triton fusion
    if cfg['kind'] == 'conv':
        if cfg['has_bn']:
            bench_triton = lambda: triton_conv_bn_lif_fusion(
                x, w, scale, bias,
                tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
                k_conv=k, padding=p, stride=s,
            )
        else:
            bench_triton = lambda: triton_conv_lif_fusion(
                x, w,
                tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
                k_conv=k, padding=p, stride=s,
            )
    else:
        bench_triton = lambda: triton_linear_lif_fusion(
            x, w, K=cfg['T'],
            tau=cfg['tau'], v_th=cfg['v_th'], v_reset=cfg['v_reset'],
        )
    time_and_report('triton_fusion', bench_triton)

    # SJ torch
    sj_functional.set_backend(sj_net, 'torch')
    time_and_report('sj_torch', lambda: run_sj_torch(sj_net, x))

    # SJ cupy
    if sj_cupy_available:
        sj_functional.set_backend(sj_net, 'cupy')
        time_and_report('sj_cupy', lambda: run_sj_cupy(sj_net, x))
    else:
        results['backends']['sj_cupy'] = {'error': 'not_available'}

    # torch.compile (switch back to torch backend)
    if compile_available:
        sj_functional.set_backend(sj_net, 'torch')
        # Use the already-compiled net
        time_and_report('torch_compile', lambda: run_compiled(compiled_net, x))
    else:
        results['backends']['torch_compile'] = {'error': 'not_available'}

    # --- Speedup derivations ---
    # Primary metric: triton_fusion speedup over SJ torch
    if 'trimmed_mean_ms' in results['backends'].get('triton_fusion', {}):
        triton_ms = results['backends']['triton_fusion']['trimmed_mean_ms']
        speedups = {}
        for ref_name in ['sj_torch', 'sj_cupy', 'torch_compile']:
            ref = results['backends'].get(ref_name, {})
            if 'trimmed_mean_ms' in ref:
                speedups[f'triton_over_{ref_name}'] = (
                    ref['trimmed_mean_ms'] / triton_ms
                )
        results['speedups'] = speedups
        print(f"\n  Speedups (triton fusion vs reference):")
        for k_name, v in speedups.items():
            arrow = 'faster' if v > 1 else 'slower'
            print(f"    {k_name:<30}: {v:>7.4f}×  "
                  f"({(v-1)*100:+.1f}% {arrow})")

    return results


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=12)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/phase1/sj_baseline.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"Phase 1: SpikingJelly baseline comparison")
    print(f"  GPU      : {torch.cuda.get_device_name(device)}")
    print(f"  torch    : {torch.__version__}")
    print(f"  triton   : {triton.__version__}")
    import spikingjelly
    print(f"  sj path  : {spikingjelly.__file__}")

    configs = get_subgraph_configs(args)
    all_results = []
    for cfg in configs:
        res = run_subgraph_experiment(cfg, args, device)
        all_results.append(res)

    # Final summary
    print(f"\n{'='*86}")
    print(f"Phase 1 summary (V100 / A100 host, triton_fusion speedups)")
    print(f"{'='*86}")
    print(f"  {'subgraph':<16} {'vs sj_torch':>14} {'vs sj_cupy':>14} "
          f"{'vs torch_compile':>18}")
    print(f"  {'-'*66}")
    for r in all_results:
        sp = r.get('speedups', {})
        a = sp.get('triton_over_sj_torch', float('nan'))
        b = sp.get('triton_over_sj_cupy', float('nan'))
        c = sp.get('triton_over_torch_compile', float('nan'))
        print(f"  {r['name']:<16} {a:>14.4f} {b:>14.4f} {c:>18.4f}")

    output = {
        'experiment': 'phase1_sj_baseline',
        'device': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
        'cuda_version': torch.version.cuda,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'subgraphs': all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()