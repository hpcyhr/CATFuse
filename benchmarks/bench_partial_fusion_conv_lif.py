"""
Phase 2 Task 2.1a wall-clock benchmark — PartialFusionConvLIF vs 3 references.

Backends compared:
  (a) PartialFusionConvLIF (ours, cuDNN conv + Triton LIF/StateCarry, K=T)
  (b) SJ torch backend    (cuDNN conv + torch LIF, 16 kernel launches)
  (c) SJ cupy backend     (cuDNN conv + cupy-fused LIF)
  (d) Phase 0 FusedConvLIF (full Triton conv + LIF, as "no cuDNN delegation" control)

Subgraph: Conv 3x3 s=1 p=1 -> LIF
Shape:    T=16, B=64, C_in=C_out=128, H=W=28

Protocol: 12 repeats x 100 inner iters, trimmed mean (drop max+min, mean of 10),
          20 warmup iters. Same as Phase 1 sj_baseline.py.

Parity reference: SJ torch backend (ground truth).

Expected results (from Path 1 / system framing):
  - (a) PartialFusion should beat (b) SJ torch AND (c) SJ cupy in wall-clock,
    because we save 15 LIF kernel launches + v HBM round-trip by keeping LIF
    state in register across all T time steps
  - (a) should massively beat (d) Phase 0 full Triton (which loses ~5x to cuDNN
    on Conv)
  - Headline: if (a) / (c) < 1.0, CTF partial fusion beats SJ cupy for the
    Conv->LIF compute-bound subgraph -- a new clean positive wall-clock data point
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import functional as sj_functional

from partial_fusion_conv_lif import partial_fusion_conv_lif, analytic_hbm_ratio

# Phase 0 full Triton fusion reference
from conv_lif_min_v2 import run_fusion as full_triton_conv_lif_fusion


# ============================================================
# Timing utility (identical protocol to Phase 1)
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
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=64)
    p.add_argument('--C_in', type=int, default=128)
    p.add_argument('--C_out', type=int, default=128)
    p.add_argument('--H', type=int, default=28)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--padding', type=int, default=1)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=12)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--output', type=str,
                   default='./results/phase2/partial_fusion_conv_lif_v100.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"Phase 2 task 2.1a benchmark: PartialFusionConvLIF vs 3 references")
    print(f"  GPU      : {torch.cuda.get_device_name(device)}")
    print(f"  torch    : {torch.__version__}")
    print(f"  triton   : {triton.__version__}")
    print(f"  shape    : T={args.T} B={args.B} C_in={args.C_in} "
          f"C_out={args.C_out} H=W={args.H}")
    print(f"  conv     : k={args.k} s={args.stride} p={args.padding}")

    # ----- Build input and weight -----
    x = (torch.randn(args.T, args.B, args.C_in, args.H, args.H,
                     device=device, dtype=torch.float32)
         * args.input_scale).contiguous()
    w = (torch.randn(args.C_out, args.C_in, args.k, args.k,
                     device=device, dtype=torch.float32)
         * args.input_scale).contiguous()

    # ----- Build SJ net and copy weight -----
    def make_sj_net(backend: str):
        net = nn.Sequential(
            sj_layer.Conv2d(args.C_in, args.C_out, kernel_size=args.k,
                            stride=args.stride, padding=args.padding, bias=False),
            sj_neuron.LIFNode(tau=args.tau, v_threshold=args.v_th, v_reset=0.0,
                              step_mode='m'),
        )
        sj_functional.set_step_mode(net, 'm')
        net = net.to(device)
        net[0].weight.data.copy_(w)
        sj_functional.set_backend(net, backend)
        return net

    sj_net_torch = make_sj_net('torch')

    # ----- Reference forward and parity -----
    print(f"\n{'='*86}")
    print(f"Parity check (all vs SJ torch ground truth)")
    print(f"{'='*86}")

    with torch.no_grad():
        sj_functional.reset_net(sj_net_torch)
        s_sj_torch = sj_net_torch(x).detach().clone()

    def check_parity(tag, s_candidate, ref):
        flips = (s_candidate != ref).sum().item()
        total = ref.numel()
        match = (1.0 - flips / total) * 100
        status = 'PASS' if match > 99.9 else 'FAIL'
        print(f"  [{tag:<22}] spike_match={match:.6f}%  "
              f"flips={flips}/{total}  {status}")
        return match, flips

    # (a) PartialFusionConvLIF
    with torch.no_grad():
        s_partial = partial_fusion_conv_lif(
            x, w,
            tau=args.tau, v_th=args.v_th, v_reset=0.0,
            stride=args.stride, padding=args.padding,
        ).detach().clone()
    partial_match, partial_flips = check_parity('partial_fusion', s_partial, s_sj_torch)

    # (c) SJ cupy
    try:
        sj_net_cupy = make_sj_net('cupy')
        with torch.no_grad():
            sj_functional.reset_net(sj_net_cupy)
            s_sj_cupy = sj_net_cupy(x).detach().clone()
        cupy_match, _ = check_parity('sj_cupy', s_sj_cupy, s_sj_torch)
        cupy_available = True
    except Exception as e:
        print(f"  [sj_cupy               ] FAILED: {type(e).__name__}: {e}")
        cupy_available = False

    # (d) Phase 0 FusedConvLIF (full Triton)
    try:
        with torch.no_grad():
            s_full_triton = full_triton_conv_lif_fusion(
                x, w,
                tau=args.tau, v_th=args.v_th, v_reset=0.0,
                k_conv=args.k, padding=args.padding, stride=args.stride,
            ).detach().clone()
        full_match, _ = check_parity('full_triton_p0', s_full_triton, s_sj_torch)
        full_triton_available = True
    except Exception as e:
        print(f"  [full_triton_p0        ] FAILED: {type(e).__name__}: {e}")
        full_triton_available = False

    if partial_match <= 99.9:
        print("\nPartial fusion parity FAILED, aborting wall-clock")
        return

    # ----- Wall-clock benchmark -----
    print(f"\n{'='*86}")
    print(f"Wall-clock (trimmed mean N={args.n_repeat-2} of {args.n_repeat}, "
          f"{args.n_iter} inner iters, {args.n_warmup} warmup)")
    print(f"{'='*86}")

    results = {}

    def time_and_report(tag, fn):
        stats = cuda_time_trimmed(
            fn, n_iter=args.n_iter,
            n_repeat=args.n_repeat, n_warmup=args.n_warmup,
        )
        results[tag] = stats
        print(f"  {tag:<22} {stats['trimmed_mean_ms']:>10.4f} ms  "
              f"(stdev {stats['trimmed_stdev_ms']:.4f}, "
              f"min {stats['min_ms']:.4f}, max {stats['max_ms']:.4f})")

    print(f"  {'backend':<22} {'trimmed_mean':>10}")
    print(f"  {'-'*66}")

    # (a) PartialFusionConvLIF
    bench_partial = lambda: partial_fusion_conv_lif(
        x, w,
        tau=args.tau, v_th=args.v_th, v_reset=0.0,
        stride=args.stride, padding=args.padding,
    )
    time_and_report('partial_fusion', bench_partial)

    # (b) SJ torch
    def bench_sj_torch():
        sj_functional.reset_net(sj_net_torch)
        return sj_net_torch(x)
    time_and_report('sj_torch', bench_sj_torch)

    # (c) SJ cupy
    if cupy_available:
        def bench_sj_cupy():
            sj_functional.reset_net(sj_net_cupy)
            return sj_net_cupy(x)
        time_and_report('sj_cupy', bench_sj_cupy)

    # (d) Phase 0 full Triton
    if full_triton_available:
        bench_full = lambda: full_triton_conv_lif_fusion(
            x, w,
            tau=args.tau, v_th=args.v_th, v_reset=0.0,
            k_conv=args.k, padding=args.padding, stride=args.stride,
        )
        time_and_report('full_triton_p0', bench_full)

    # ----- Speedup table -----
    print(f"\n{'='*86}")
    print(f"Speedups (partial_fusion / reference)")
    print(f"{'='*86}")
    partial_ms = results['partial_fusion']['trimmed_mean_ms']
    for ref_tag in ['sj_torch', 'sj_cupy', 'full_triton_p0']:
        if ref_tag in results:
            ref_ms = results[ref_tag]['trimmed_mean_ms']
            speedup = ref_ms / partial_ms
            arrow = 'faster' if speedup > 1 else 'slower'
            delta_pct = (speedup - 1) * 100
            print(f"  partial / {ref_tag:<15}: {speedup:>7.4f}x  "
                  f"({delta_pct:+.1f}% {arrow})")

    # ----- HBM analytic ratios -----
    print(f"\n{'='*86}")
    print(f"HBM analytic ratios vs reference (5T step)")
    print(f"{'='*86}")
    print(f"  partial fusion K=T={args.T}:  "
          f"{analytic_hbm_ratio(args.T):.4f}  (saves "
          f"{(1 - analytic_hbm_ratio(args.T)) * 100:.1f}%)")
    print(f"  partial fusion K=4:          "
          f"{analytic_hbm_ratio(4):.4f}  (saves "
          f"{(1 - analytic_hbm_ratio(4)) * 100:.1f}%)")
    print(f"  full Triton fusion K=4:      "
          f"{(1 + 2/4) / 5:.4f}  (saves "
          f"{(1 - (1 + 2/4) / 5) * 100:.1f}%)")

    # ----- Save JSON -----
    output = {
        'experiment': 'phase2_task2.1a_partial_fusion_conv_lif',
        'device': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'parity': {
            'partial_vs_sj_torch': {
                'spike_match_pct': partial_match,
                'spike_flips': partial_flips,
                'total_spikes': s_sj_torch.numel(),
            },
        },
        'wall_clock': results,
        'hbm_analytic': {
            'partial_K_eq_T': analytic_hbm_ratio(args.T),
            'partial_K_4': analytic_hbm_ratio(4),
            'full_triton_K_4': (1 + 2/4) / 5,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()