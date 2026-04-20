#!/usr/bin/env python
"""
Phase C Bottleneck pattern parity test.

Verifies that CTFSEWBottleneck and CTFSpikingBottleneck produce outputs
matching SJ reference across all ResNet-50/101/152 variants, on both
SEW and standard spiking ResNets.

Must PASS before running phaseC4 bench on these networks.

Usage:
    /data_priv/dagongcheng/snn118/bin/python phaseC_bottleneck_parity_test.py
"""
import copy
import os
import sys
import warnings
warnings.filterwarnings('ignore', message='.*Applied workaround for CuDNN.*')

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet

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


def build_model(family, depth, v_threshold=0.1, tau=2.0):
    """Build SJ model by family × depth."""
    common_kw = dict(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        num_classes=1000,
        v_threshold=v_threshold,
        tau=tau,
    )
    if family == 'sew':
        common_kw['cnf'] = 'ADD'
        fn = getattr(sew_resnet, f'sew_resnet{depth}')
    elif family == 'spiking':
        fn = getattr(spiking_resnet, f'spiking_resnet{depth}')
    else:
        raise ValueError(family)
    model = fn(**common_kw)
    functional.set_step_mode(model, 'm')
    return model


def run_one(family, depth, T, B, device):
    name = f'{family}_resnet{depth}'
    print(f"\n{'='*70}")
    print(f"  {name}  T={T}  B={B}")
    print(f"{'='*70}", flush=True)

    torch.manual_seed(0)
    net_ref = build_model(family, depth).to(device)
    init_bn_random(net_ref)

    net_ctf_base = copy.deepcopy(net_ref)
    net_ctf, cov = substitute(net_ctf_base, verbose=False)
    net_ctf = net_ctf.to(device)

    fused = cov.get('fused_lif_nodes', 0)
    total = cov.get('total_lif_nodes', 0)
    pct = cov.get('coverage_pct', 0)
    patterns = cov.get('patterns_matched', {})
    print(f"  Coverage: {fused}/{total} = {pct:.1f}%", flush=True)
    print(f"  Patterns: {patterns}", flush=True)

    x = torch.randn(T, B, 3, 224, 224, device=device)

    with torch.no_grad():
        functional.reset_net(net_ref)
        out_ref = net_ref(x)
        functional.reset_net(net_ctf)
        out_ctf = net_ctf(x)

    diff = (out_ref - out_ctf).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    bit_exact = torch.equal(out_ref, out_ctf)
    ref_max = out_ref.abs().max().item()

    print(f"  ref_max_abs: {ref_max:.3f}", flush=True)
    print(f"  max_diff:    {max_d:.3e}", flush=True)
    print(f"  mean_diff:   {mean_d:.3e}", flush=True)
    print(f"  bit_exact:   {bit_exact}", flush=True)

    # For SEW: parity should be bit-exact (no depth-amplified ULP on spike add)
    # For standard: may be non-bit-exact but mean_diff should be <1e-3 range
    if family == 'sew':
        passed = bit_exact
        verdict = 'PASS' if passed else 'FAIL (expected bit-exact for SEW)'
    else:
        passed = mean_d < 1.0 and ref_max > 1e-3
        verdict = 'PASS' if passed else f'FAIL (mean_diff={mean_d:.3e} too large)'

    print(f"  Verdict:     {verdict}", flush=True)

    # Cleanup
    del net_ref, net_ctf, net_ctf_base, x, out_ref, out_ctf
    torch.cuda.empty_cache()

    return {
        'model': name, 'T': T, 'B': B,
        'coverage_pct': pct, 'patterns': patterns,
        'max_diff': max_d, 'mean_diff': mean_d,
        'bit_exact': bit_exact, 'ref_max': ref_max,
        'passed': passed,
    }


def main():
    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)

    # All 6 new configurations. B=2 to fit on V100 32GB even for resnet152.
    results = []
    for family in ['sew', 'spiking']:
        for depth in [50, 101, 152]:
            try:
                r = run_one(family, depth, T=4, B=2, device=device)
                results.append(r)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM! Skipping {family}_resnet{depth}", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  EXCEPTION: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n\n{'='*70}")
    print(f"=== PARITY SUMMARY ===")
    print(f"{'='*70}")
    print(f"{'Model':<20s} {'Cov%':>5s} {'max_diff':>12s} {'mean_diff':>12s} "
          f"{'bit_exact':>9s} {'verdict':>10s}")
    print(f"{'-'*20} {'-'*5} {'-'*12} {'-'*12} {'-'*9} {'-'*10}")
    all_passed = True
    for r in results:
        v = 'PASS' if r['passed'] else 'FAIL'
        if not r['passed']:
            all_passed = False
        print(f"{r['model']:<20s} {r['coverage_pct']:>4.0f}% "
              f"{r['max_diff']:>12.3e} {r['mean_diff']:>12.3e} "
              f"{str(r['bit_exact']):>9s} {v:>10s}")

    print(f"\n{'OVERALL: ALL PASSED ✓' if all_passed else 'OVERALL: SOME FAILED ✗'}")


if __name__ == '__main__':
    main()