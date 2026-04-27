"""§5.3 end-to-end experiment — SEW-RN18 full-network latency.

Three-way comparison on the deployment forward path of SEW-ResNet18
(CIFAR10 checkpoint, T=4):

  1. SJ-eager:     stock SpikingJelly model, multi-step mode
                   ('m' step_mode), cuDNN conv + SJ's per-step LIF
  2. CATFuse-DK:   `optimize(net, force_sparse=False)` — every fused
                   pattern is DenseKeep (cuDNN conv + Triton lif_sequential)
  3. CATFuse-def:  `optimize(net, use_sparseflow=True)` — default Runtime
                   EGD routing, gives 13 DenseKeep + 7 SparseFlow on RN18

For each of these three, swept across B ∈ {1, 2, 8, 32}:
  - wall-clock latency (median over 100 iter × 3 repeats)
  - max_diff vs SJ-eager output (should be 0 for both CATFuse variants)

This is the §5.3 ablation that answers "what is the net effect of CATFuse
at the network level on V100", separating out the framework-vs-cuDNN
question (DK-only vs SJ-eager) from the routing question (default vs DK-only).

Output:
  - stdout: table per B
  - CSV: experiments/results/end_to_end_sew_rn18.csv

Run:
    cd /path/to/CATFuse
    python experiments/exp_31_end_to_end.py
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, List

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

from catfuse.substitute import substitute_sf
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
T = 4   # checkpoint trained at T=4
B_VALUES = [1, 2, 8, 32]
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3

CSV_PATH = os.path.join(_REPO_ROOT, "experiments/results/end_to_end_sew_rn18.csv")


@dataclass
class Row:
    impl: str               # "SJ-eager" | "CATFuse-DK" | "CATFuse-default"
    B: int
    T: int
    wall_us_median: float
    parity_max_diff: float  # vs SJ-eager output
    notes: str


def build_sj_net():
    """Build SEW-ResNet18 via shared helper (CIFAR10 stem + correct ckpt unwrap)."""
    return build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)


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


def run_for_batch(B: int, sj_net, ckpt_note: str, rows: List[Row]):
    print(f"\n{'─' * 96}")
    print(f"B = {B}, T = {T}")
    print(f"{'─' * 96}")

    torch.manual_seed(42 + B)
    x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

    # ---- 1. SJ eager ----
    def _fn_sj():
        functional.reset_net(sj_net)
        with torch.no_grad():
            return sj_net(x)

    with torch.no_grad():
        functional.reset_net(sj_net)
        y_sj = sj_net(x).detach().clone()
    t_sj = bench(_fn_sj)
    print(f"  {'SJ-eager':>18s}  {t_sj:>9.2f} us  {'(reference)':>22s}")
    rows.append(Row(impl="SJ-eager", B=B, T=T,
                    wall_us_median=t_sj, parity_max_diff=0.0,
                    notes=ckpt_note))

    # ---- 2. CATFuse DenseKeep-only ----
    # substitute_sf(force_sparse=False) goes through the SAME code path as
    # default routing, but every fused pattern is forced through DenseKeep
    # impl. This isolates "framework overhead vs SJ" from "SparseFlow choice".
    # NOTE: force_sparse parameter in substitute_sf inverts: True forces
    # SparseFlow always, False lets policy decide. We need a separate flag
    # to force DenseKeep always — using policy override instead.
    fused_dk, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    # Then walk and force every STFusion's _impl_sparse to None to disable SF
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
    n_disabled = 0
    for m in fused_dk.modules():
        if isinstance(m, STFusionConvBNLIF) and m._impl_sparse is not None:
            m._impl_sparse = None
            n_disabled += 1
    fused_dk = fused_dk.to(DEVICE).eval()
    print(f"  (DK-only: disabled SparseFlow on {n_disabled} STFusion layers; "
          f"_batchfold_forward falls back to DenseKeep per Cor 3.17)")

    def _fn_dk():
        functional.reset_net(fused_dk)
        with torch.no_grad():
            return fused_dk(x)

    with torch.no_grad():
        functional.reset_net(fused_dk)
        y_dk = fused_dk(x).detach()
    parity_dk = (y_dk - y_sj).abs().max().item()
    t_dk = bench(_fn_dk)
    speedup_dk = t_sj / t_dk
    print(f"  {'CATFuse-DK':>18s}  {t_dk:>9.2f} us  "
          f"{speedup_dk:>5.2f}x vs SJ  parity={parity_dk:.2e}")
    rows.append(Row(impl="CATFuse-DK", B=B, T=T,
                    wall_us_median=t_dk, parity_max_diff=parity_dk,
                    notes="force_sparse=False"))

    # ---- 3. CATFuse default routing ----
    # substitute_sf with default behavior — Runtime EGD chooses DK or SF
    # per layer based on policy. On SEW-RN18 this gives 13 DK + 7 SF.
    fused_def, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_def = fused_def.to(DEVICE).eval()

    def _fn_def():
        functional.reset_net(fused_def)
        with torch.no_grad():
            return fused_def(x)

    with torch.no_grad():
        functional.reset_net(fused_def)
        y_def = fused_def(x).detach()
    parity_def = (y_def - y_sj).abs().max().item()
    t_def = bench(_fn_def)
    speedup_def = t_sj / t_def
    print(f"  {'CATFuse-default':>18s}  {t_def:>9.2f} us  "
          f"{speedup_def:>5.2f}x vs SJ  parity={parity_def:.2e}")
    rows.append(Row(impl="CATFuse-default", B=B, T=T,
                    wall_us_median=t_def, parity_max_diff=parity_def,
                    notes="13 DK + 7 SF (default routing)"))

    # ---- Summary line ----
    print(f"\n  Internal CTF comparison: default / DK-only = "
          f"{t_def/t_dk:.3f}x  (>1 means default slower than DK-only)")


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 end-to-end SEW-RN18 latency: SJ-eager vs CATFuse-DK vs CATFuse-default")
    print("=" * 96)
    print()
    print(f"  T = {T} (checkpoint training default)")
    print(f"  B sweep: {B_VALUES}")
    print(f"  Bench: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median reported")
    print(f"  Parity: max_diff vs SJ-eager output (should be 0 for both CATFuse variants)")
    print()
    print("  Notes on the three variants:")
    print("    SJ-eager:        stock SpikingJelly, multi-step mode")
    print("    CATFuse-DK:      substitute_sf with SparseFlow disabled at impl level;")
    print("                     STFusion._batchfold_forward falls back to DenseKeep per Cor 3.17.")
    print("                     STFusion.forward still runs sparsity check (host-device sync per layer).")
    print("    CATFuse-default: full Runtime EGD; STFusion picks SF when sparsity > 0.7.")
    print()

    sj_net, ckpt_note = build_sj_net()
    print(f"  Checkpoint: {ckpt_note}")

    rows: List[Row] = []
    for B in B_VALUES:
        run_for_batch(B, sj_net, ckpt_note, rows)

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        fieldnames = list(asdict(rows[0]).keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    # Final summary
    print()
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    print(f"  {'B':>4} | {'SJ_us':>10} | {'DK_us':>10} | {'def_us':>10} | "
          f"{'DK/SJ':>7} | {'def/SJ':>7} | {'parity_DK':>10} | {'parity_def':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}")
    for B in B_VALUES:
        r_sj = next(r for r in rows if r.impl == "SJ-eager" and r.B == B)
        r_dk = next(r for r in rows if r.impl == "CATFuse-DK" and r.B == B)
        r_df = next(r for r in rows if r.impl == "CATFuse-default" and r.B == B)
        print(f"  {B:>4} | {r_sj.wall_us_median:>10.2f} | {r_dk.wall_us_median:>10.2f} | "
              f"{r_df.wall_us_median:>10.2f} | "
              f"{r_sj.wall_us_median/r_dk.wall_us_median:>7.3f} | "
              f"{r_sj.wall_us_median/r_df.wall_us_median:>7.3f} | "
              f"{r_dk.parity_max_diff:>10.2e} | {r_df.parity_max_diff:>10.2e}")
    print()
    n_total = len(rows)
    n_parity = sum(1 for r in rows if r.parity_max_diff == 0.0)
    print(f"  Bit-exact rows: {n_parity}/{n_total}")
    print(f"  CSV:            {CSV_PATH}")

    if n_parity != n_total:
        print()
        print("CRITICAL: some rows have non-zero parity vs SJ-eager — implementation bug.")
        for r in rows:
            if r.parity_max_diff != 0.0:
                print(f"  {r.impl} B={r.B}: max_diff={r.parity_max_diff:.4e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())