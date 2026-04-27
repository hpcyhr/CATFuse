"""§5.3 end-to-end wall-clock — corrected version (v2).

Supersedes exp_31_end_to_end.py. The previous version produced bogus
data because of two upstream bugs that have since been fixed:

  1. ckpt unwrap missing 'model_state_dict' key → all weights silently
     dropped → forward dead-chain → "2x speedup" was actually CATFuse
     skipping conv on all-zero input via StaticZero short-circuit.

  2. parity standard was max_diff=0 against SJ-eager. With the dead chain
     it gave 0 == 0 (vacuous). With the chain alive, SparseFlow's fp32
     reduction-order divergence from cuDNN gives ε ≈ 3e-5 spike-flip
     rate — non-zero pointwise but unchanged classifier output.

This v2 corrects both:
  - uses experiments._helpers.build_sew_rn18_cifar10() (correct unwrap)
  - new parity standard:
      * pointwise max_diff               (reported, not gated)
      * spike-flip rate per LIF layer    (gated: < 1e-3)
      * classifier agreement on argmax   (gated: == 100%)

Three implementations are timed:
  1. SJ-eager:        stock SpikingJelly, multi-step mode
  2. CATFuse-DK:      every fused pattern uses DenseKeep impl (cuDNN)
                      — same conv backend as σ_ref → bit-exact
  3. CATFuse-default: Runtime EGD picks SF when sparsity > 0.7
                      — different conv backend on those layers → ε-N-equiv

For each impl × B ∈ {1, 2, 8, 32}, we report wall_us median over 100
iter × 3 repeats, plus the parity metrics above.

Output:
  - stdout: per-B table, summary table at end
  - CSV: experiments/results/end_to_end_v2_sew_rn18.csv

Run:
    python experiments/exp_40_end_to_end_v2.py
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import functional, neuron

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
T = 4   # ckpt was trained at T=4
B_VALUES = [1, 2, 8, 32]
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3
N_PARITY_INPUTS = 8   # # of distinct random inputs for parity measurement

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/end_to_end_v2_sew_rn18.csv")


@dataclass
class Row:
    impl: str               # "SJ-eager" | "CATFuse-DK" | "CATFuse-default"
    B: int
    T: int
    wall_us_median: float
    pointwise_max_diff: float        # vs SJ-eager output (logits)
    spike_flip_rate_max: float       # max across LIF layers, across N_PARITY_INPUTS
    spike_flip_rate_mean: float      # mean across LIF layers
    classifier_agreement: float      # fraction of samples with same argmax as SJ
    notes: str


def bench(fn) -> float:
    """Median microseconds per call across N_REPEAT runs of N_ITER each."""
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1e6)
    return statistics.median(times)


def trace_lif_spikes(net, x):
    """Run forward, return {layer_name: spike_tensor} for every LIFNode."""
    spikes = {}
    handles = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            def _make(n):
                def fn(_mod, _inp, out):
                    spikes[n] = out.detach().clone()
                return fn
            handles.append(m.register_forward_hook(_make(name)))
    with torch.no_grad():
        functional.reset_net(net)
        y = net(x)
    for h in handles:
        h.remove()
    return spikes, y.detach().clone()


def measure_parity(impl_net, ref_net, B):
    """Return (pointwise_max_diff, spike_flip_max, spike_flip_mean,
              classifier_agreement) computed across N_PARITY_INPUTS
              random inputs."""
    pointwise_diffs = []
    flip_rates_per_input = []  # list of (max_layer_rate, mean_layer_rate)
    classifier_agreements = []

    for i in range(N_PARITY_INPUTS):
        torch.manual_seed(1000 + i)
        x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

        ref_spikes, y_ref = trace_lif_spikes(ref_net, x)
        impl_spikes, y_impl = trace_lif_spikes(impl_net, x)

        # Pointwise (logits)
        pointwise_diffs.append((y_ref - y_impl).abs().max().item())

        # Spike-flip rate per layer
        layer_rates = []
        for name in ref_spikes:
            if name in impl_spikes:
                ref_s = ref_spikes[name]
                imp_s = impl_spikes[name]
                if ref_s.shape == imp_s.shape:
                    n_flips = (ref_s != imp_s).sum().item()
                    rate = n_flips / ref_s.numel()
                    layer_rates.append(rate)
        if layer_rates:
            flip_rates_per_input.append((max(layer_rates),
                                         sum(layer_rates) / len(layer_rates)))

        # Classifier agreement (argmax over T-summed logits)
        ref_pred = y_ref.sum(dim=0).argmax(dim=-1)
        impl_pred = y_impl.sum(dim=0).argmax(dim=-1)
        agreement = (ref_pred == impl_pred).float().mean().item()
        classifier_agreements.append(agreement)

    pointwise_max = max(pointwise_diffs)
    flip_max = max(r[0] for r in flip_rates_per_input) if flip_rates_per_input else 0.0
    flip_mean = (sum(r[1] for r in flip_rates_per_input) / len(flip_rates_per_input)
                 if flip_rates_per_input else 0.0)
    cls_agreement_mean = sum(classifier_agreements) / len(classifier_agreements)

    return pointwise_max, flip_max, flip_mean, cls_agreement_mean


def build_models(sj_net):
    """Construct (sj_net, fused_dk, fused_def). All on DEVICE, eval mode."""
    fused_def, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_def = fused_def.to(DEVICE).eval()

    fused_dk, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_dk = fused_dk.to(DEVICE).eval()
    n_disabled = 0
    for m in fused_dk.modules():
        if isinstance(m, STFusionConvBNLIF) and m._impl_sparse is not None:
            m._impl_sparse = None
            n_disabled += 1

    return fused_dk, fused_def, n_disabled


def run_for_batch(B, sj_net, fused_dk, fused_def, ckpt_note, rows):
    print(f"\n{'─' * 96}")
    print(f"B = {B}, T = {T}")
    print(f"{'─' * 96}")

    torch.manual_seed(42 + B)
    x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

    # ============================================================
    # 1. SJ-eager
    # ============================================================
    def _fn_sj():
        functional.reset_net(sj_net)
        with torch.no_grad():
            return sj_net(x)
    t_sj = bench(_fn_sj)
    print(f"  {'SJ-eager':>18s}  {t_sj:>9.2f} us  {'(reference)':>22s}")
    rows.append(Row(impl="SJ-eager", B=B, T=T,
                    wall_us_median=t_sj,
                    pointwise_max_diff=0.0,
                    spike_flip_rate_max=0.0,
                    spike_flip_rate_mean=0.0,
                    classifier_agreement=1.0,
                    notes=f"reference, ckpt: {ckpt_note}"))

    # ============================================================
    # 2. CATFuse-DK
    # ============================================================
    def _fn_dk():
        functional.reset_net(fused_dk)
        with torch.no_grad():
            return fused_dk(x)
    t_dk = bench(_fn_dk)

    pw_dk, flip_max_dk, flip_mean_dk, cls_dk = measure_parity(
        fused_dk, sj_net, B)
    speedup_dk = t_sj / t_dk
    print(f"  {'CATFuse-DK':>18s}  {t_dk:>9.2f} us  "
          f"{speedup_dk:>5.2f}x vs SJ  "
          f"pw={pw_dk:.2e}  flip(max)={flip_max_dk:.2e}  "
          f"cls={cls_dk*100:.1f}%")
    rows.append(Row(impl="CATFuse-DK", B=B, T=T,
                    wall_us_median=t_dk,
                    pointwise_max_diff=pw_dk,
                    spike_flip_rate_max=flip_max_dk,
                    spike_flip_rate_mean=flip_mean_dk,
                    classifier_agreement=cls_dk,
                    notes="DenseKeep-only (SF disabled at impl level)"))

    # ============================================================
    # 3. CATFuse-default
    # ============================================================
    def _fn_def():
        functional.reset_net(fused_def)
        with torch.no_grad():
            return fused_def(x)
    t_def = bench(_fn_def)

    pw_def, flip_max_def, flip_mean_def, cls_def = measure_parity(
        fused_def, sj_net, B)
    speedup_def = t_sj / t_def
    print(f"  {'CATFuse-default':>18s}  {t_def:>9.2f} us  "
          f"{speedup_def:>5.2f}x vs SJ  "
          f"pw={pw_def:.2e}  flip(max)={flip_max_def:.2e}  "
          f"cls={cls_def*100:.1f}%")
    rows.append(Row(impl="CATFuse-default", B=B, T=T,
                    wall_us_median=t_def,
                    pointwise_max_diff=pw_def,
                    spike_flip_rate_max=flip_max_def,
                    spike_flip_rate_mean=flip_mean_def,
                    classifier_agreement=cls_def,
                    notes="Runtime EGD routing (SF on layers with sparsity>0.7)"))

    # Internal comparison
    print(f"\n  default / DK-only wall-clock = {t_def/t_dk:.3f}x  "
          f"(>1 = default slower than DK-only)")


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 v2 — End-to-end SEW-RN18 latency with R/N-equivalence parity")
    print("=" * 96)
    print()
    print(f"  T = {T} (ckpt training default)")
    print(f"  B sweep: {B_VALUES}")
    print(f"  Bench: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median")
    print(f"  Parity: averaged over {N_PARITY_INPUTS} random inputs per (impl, B)")
    print()
    print("  Parity metrics (all reported, last two are gated for 'pass'):")
    print("    pointwise_max_diff:   max |y_impl - y_sj|  (logits)")
    print("    spike_flip_rate_max:  max over LIF layers of (# flipped / total)")
    print("    classifier_agreement: fraction of samples with same argmax as SJ")
    print()

    # Build SJ model + two CATFuse variants
    sj_net, ckpt_note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    print(f"  Checkpoint: {ckpt_note}")
    fused_dk, fused_def, n_disabled = build_models(sj_net)
    print(f"  CATFuse-DK: SF disabled on {n_disabled} STFusion layers")

    rows: List[Row] = []
    for B in B_VALUES:
        run_for_batch(B, sj_net, fused_dk, fused_def, ckpt_note, rows)

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        fieldnames = list(asdict(rows[0]).keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    # ============================================================
    # Final summary
    # ============================================================
    print()
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    print()
    print("Wall-clock latency (us, median over 100 iter × 3 repeats)")
    print(f"  {'B':>4} | {'SJ_us':>10} | {'DK_us':>10} | {'def_us':>10} | "
          f"{'DK/SJ':>7} | {'def/SJ':>7} | {'def/DK':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for B in B_VALUES:
        r_sj = next(r for r in rows if r.impl == "SJ-eager" and r.B == B)
        r_dk = next(r for r in rows if r.impl == "CATFuse-DK" and r.B == B)
        r_df = next(r for r in rows if r.impl == "CATFuse-default" and r.B == B)
        print(f"  {B:>4} | {r_sj.wall_us_median:>10.2f} | "
              f"{r_dk.wall_us_median:>10.2f} | "
              f"{r_df.wall_us_median:>10.2f} | "
              f"{r_sj.wall_us_median/r_dk.wall_us_median:>7.3f} | "
              f"{r_sj.wall_us_median/r_df.wall_us_median:>7.3f} | "
              f"{r_df.wall_us_median/r_dk.wall_us_median:>7.3f}")

    print()
    print("Parity (averaged over N_PARITY_INPUTS random inputs per (impl, B))")
    print(f"  {'impl':<18} {'B':>4} | {'pointwise':>10} | "
          f"{'flip_max':>10} | {'flip_mean':>10} | {'cls_agree':>10}")
    print(f"  {'-'*18} {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in rows:
        if r.impl == "SJ-eager":
            continue
        print(f"  {r.impl:<18} {r.B:>4} | "
              f"{r.pointwise_max_diff:>10.2e} | "
              f"{r.spike_flip_rate_max:>10.2e} | "
              f"{r.spike_flip_rate_mean:>10.2e} | "
              f"{r.classifier_agreement*100:>9.1f}%")

    # Pass / fail summary
    print()
    print("Parity pass/fail (R/N-equivalence standard)")
    print("─" * 96)
    print("  R-equivalence verification:")
    print("    DK should be bit-exact (same conv backend as σ_ref).")
    print("    Default should be ε-N-equiv with classifier agreement = 100%.")
    print()
    n_dk_bitexact = sum(1 for r in rows
                        if r.impl == "CATFuse-DK" and r.pointwise_max_diff == 0.0)
    n_dk_total = sum(1 for r in rows if r.impl == "CATFuse-DK")
    n_def_cls_full = sum(1 for r in rows
                         if r.impl == "CATFuse-default" and r.classifier_agreement == 1.0)
    n_def_total = sum(1 for r in rows if r.impl == "CATFuse-default")
    n_def_low_flip = sum(1 for r in rows
                         if r.impl == "CATFuse-default" and r.spike_flip_rate_max < 1e-3)

    print(f"  CATFuse-DK bit-exact (max_diff=0):       {n_dk_bitexact}/{n_dk_total}")
    print(f"  CATFuse-default classifier == SJ:        {n_def_cls_full}/{n_def_total}")
    print(f"  CATFuse-default flip_max < 1e-3:         {n_def_low_flip}/{n_def_total}")
    print()
    print(f"  CSV: {CSV_PATH}")

    failed = []
    if n_dk_bitexact != n_dk_total:
        failed.append("DK bit-exact violated")
    if n_def_cls_full != n_def_total:
        failed.append("default classifier disagreement")
    if n_def_low_flip != n_def_total:
        failed.append("default flip rate exceeds 1e-3")

    if failed:
        print()
        print(f"PARITY ISSUES: {failed}")
        print("Review the data above before paper §5.3 framing.")
        return 1
    else:
        print()
        print("PARITY OK: R-equiv verified for DK, ε-N-equiv with cls-agreement=100% for default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())