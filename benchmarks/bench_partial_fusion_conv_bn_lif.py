"""
Phase 2 Task 2.1b wall-clock benchmark — PartialFusionConvBNLIF vs 3 references.

Backends compared:
  (a) PartialFusionConvBNLIF (ours, cuDNN conv + Triton BN+LIF/StateCarry, K=T)
  (b) SJ torch backend       (cuDNN conv + torch BN + torch LIF, 16 launches)
  (c) SJ cupy backend        (cuDNN conv + torch BN + cupy fused LIF)
  (d) Phase 0 FusedConvBNLIF (full Triton conv + BN + LIF, "no cuDNN delegation" control)

Subgraph: Conv 3x3 s=1 p=1 -> BN -> LIF
Shape:    T=16, B=64, C_in=C_out=128, H=W=28

Protocol: 12 repeats x 100 inner iters, trimmed mean, 20 warmup. Same as task 2.1a.

Parity reference: SJ torch backend.

Expected results (extrapolating from task 2.1a):
  - (a) PartialFusion should beat SJ torch and SJ cupy by even more than
    task 2.1a, because the fused (BN -> LIF) kernel saves an additional BN
    HBM roundtrip beyond what task 2.1a saves
  - (a) should massively beat (d) Phase 0 full Triton ConvBNLIF (which has
    Triton conv as the bottleneck)
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

from catfuse.kernels.partial_fusion_conv_bn_lif_impl import (
    partial_fusion_conv_bn_lif,
    analytic_hbm_ratio_conv_bn_lif,
)

# Phase 0 full Triton ConvBNLIF reference (kept benchmark-internal)
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from conv_bn_lif_min import run_fusion as full_triton_conv_bn_lif_fusion

# ============================================================
# Timing utility (identical to task 2.1a bench)
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
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path; if None, auto-generated with seed')
    args = p.parse_args()

    if args.output is None:
        args.output = f'./results/phase2/partial_fusion_conv_bn_lif_v100_seed{args.seed}.json'

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print(f"Phase 2 task 2.1b benchmark: PartialFusionConvBNLIF vs 3 references")
    print(f"  GPU      : {torch.cuda.get_device_name(device)}")
    print(f"  torch    : {torch.__version__}")
    print(f"  triton   : {triton.__version__}")
    print(f"  shape    : T={args.T} B={args.B} C_in={args.C_in} "
          f"C_out={args.C_out} H=W={args.H}")
    print(f"  conv     : k={args.k} s={args.stride} p={args.padding}, BN, LIF")
    print(f"  seed     : {args.seed}")

    # ----- Build input + weights -----
    x = (torch.randn(args.T, args.B, args.C_in, args.H, args.H,
                     device=device, dtype=torch.float32)
         * args.input_scale).contiguous()
    w_conv = (torch.randn(args.C_out, args.C_in, args.k, args.k,
                          device=device, dtype=torch.float32)
              * args.input_scale).contiguous()
    bn_weight = (1.0 + torch.randn(args.C_out, device=device,
                                    dtype=torch.float32) * 0.1).contiguous()
    bn_bias = (torch.randn(args.C_out, device=device,
                           dtype=torch.float32) * 0.1).contiguous()
    running_mean = (torch.randn(args.C_out, device=device,
                                dtype=torch.float32) * 0.05).contiguous()
    running_var = (1.0 + torch.randn(args.C_out, device=device,
                                      dtype=torch.float32) * 0.05).abs().contiguous()
    eps = 1e-5

    # ----- Build SJ net -----
    def make_sj_net(backend: str):
        net = nn.Sequential(
            sj_layer.Conv2d(args.C_in, args.C_out, kernel_size=args.k,
                            stride=args.stride, padding=args.padding, bias=False),
            sj_layer.BatchNorm2d(args.C_out),
            sj_neuron.LIFNode(tau=args.tau, v_threshold=args.v_th, v_reset=0.0,
                              step_mode='m'),
        )
        sj_functional.set_step_mode(net, 'm')
        net = net.to(device)
        net[0].weight.data.copy_(w_conv)
        bn = net[1]
        bn.eval()
        bn.weight.data.copy_(bn_weight)
        bn.bias.data.copy_(bn_bias)
        bn.running_mean.data.copy_(running_mean)
        bn.running_var.data.copy_(running_var)
        bn.eps = eps
        sj_functional.set_backend(net, backend)
        return net

    sj_net_torch = make_sj_net('torch')

    # ----- Reference forward + parity -----
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

    # (a) PartialFusionConvBNLIF
    with torch.no_grad():
        s_partial = partial_fusion_conv_bn_lif(
            x, w_conv,
            bn_weight, bn_bias, running_mean, running_var, eps=eps,
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

    # (d) Phase 0 full Triton ConvBNLIF
    try:
        with torch.no_grad():
            s_full_triton = full_triton_conv_bn_lif_fusion(
                x, w_conv, bn_weight * torch.rsqrt(running_var + eps),
                bn_bias - running_mean * bn_weight * torch.rsqrt(running_var + eps),
                tau=args.tau, v_th=args.v_th, v_reset=0.0,
                k_conv=args.k, padding=args.padding, stride=args.stride,
            ).detach().clone()
        # Note: Phase 0 conv_bn_lif_min takes (scale, bias) directly, so we
        # precompute fused affine here for the call signature
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

    # (a) PartialFusionConvBNLIF
    bench_partial = lambda: partial_fusion_conv_bn_lif(
        x, w_conv, bn_weight, bn_bias, running_mean, running_var, eps=eps,
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

    # (d) Phase 0 full Triton ConvBNLIF
    if full_triton_available:
        # Precompute fused affine outside the lambda for fairness
        fused_scale_p0 = bn_weight * torch.rsqrt(running_var + eps)
        fused_bias_p0 = bn_bias - running_mean * fused_scale_p0
        bench_full = lambda: full_triton_conv_bn_lif_fusion(
            x, w_conv, fused_scale_p0, fused_bias_p0,
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
    print(f"HBM analytic ratios vs reference (7T step for ConvBNLIF)")
    print(f"{'='*86}")
    print(f"  partial fusion K=T={args.T}:  "
          f"{analytic_hbm_ratio_conv_bn_lif(args.T):.4f}  (saves "
          f"{(1 - analytic_hbm_ratio_conv_bn_lif(args.T)) * 100:.1f}%)")
    print(f"  partial fusion K=4:          "
          f"{analytic_hbm_ratio_conv_bn_lif(4):.4f}  (saves "
          f"{(1 - analytic_hbm_ratio_conv_bn_lif(4)) * 100:.1f}%)")
    print(f"  full Triton K=4 (Phase 0):   0.3439  (saves 65.6%)")

    # ----- Save JSON -----
    output = {
        'experiment': 'phase2_task2.1b_partial_fusion_conv_bn_lif',
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
            'partial_K_eq_T': analytic_hbm_ratio_conv_bn_lif(args.T),
            'partial_K_4': analytic_hbm_ratio_conv_bn_lif(4),
            'full_triton_K_4': 0.3439,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()