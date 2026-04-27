"""§5 supplementary — single-layer batch size scaling (DK vs SF).

Holds workload fixed (SEW-RN18 layer3-like shape, T=4, K=T=4,
sparsity=85%), sweeps B ∈ {1, 2, 4, 8, 16, 32, 64} and reports DK and
SF wall-clock + per-sample latency.

Purpose: complements exp_40 (network-level batch sweep). At the
single-layer level we expect:
  - DK (cuDNN) to scale near-linearly in B with strong amortization
    of fixed launch overhead at large B (per-sample latency drops)
  - SF (Triton) to also scale but possibly with different per-launch
    amortization; the per-launch overhead measured in exp_43 (~138us)
    becomes a smaller fraction of total at large B

This data localizes the §5.3 network-level "default 0.62× SJ at B=32"
finding: is it because SF gets WORSE at large B, or because SJ-eager
becomes RELATIVELY more efficient?

Run:
    python experiments/exp_47_single_layer_batch_scan.py
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
from spikingjelly.activation_based import (
    functional, neuron, layer as sj_layer
)
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF


DEVICE = "cuda:0"
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 5

# layer3-like shape
CIN = COUT = 256
H = W = 8
T = 4
K = T
SPARSITY = 0.85
B_VALUES = [1, 2, 4, 8, 16, 32, 64]

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/single_layer_batch_scan.csv")


@dataclass
class Row:
    B: int
    dk_wall_us: float
    sf_wall_us: float
    dk_per_sample_us: float       # = dk_wall_us / B
    sf_per_sample_us: float       # = sf_wall_us / B
    sf_over_dk: float
    notes: str


def bench_us(fn) -> float:
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


def main():
    if not torch.cuda.is_available():
        return 1

    print("=" * 96)
    print("§5 supplementary — single-layer batch-size scan (DK vs SF)")
    print("=" * 96)
    print()
    print(f"  Workload: layer3-like (Cin=Cout={CIN}, H=W={H}, T={T}, K={K}, sparsity={SPARSITY:.0%})")
    print(f"  Bench: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median")
    print(f"  B sweep: {B_VALUES}")
    print()

    # Build STFusion (synthetic weights — wall-clock only)
    torch.manual_seed(42)
    conv = sj_layer.Conv2d(CIN, COUT, 3, padding=1, bias=False).to(DEVICE)
    bn = sj_layer.BatchNorm2d(COUT).to(DEVICE)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(DEVICE)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=K).to(DEVICE).eval()
    spec = fused.spec
    params = fused._ensure_params()

    rows: List[Row] = []

    print(f"  {'B':>4} | {'DK_us':>9} | {'SF_us':>9} | "
          f"{'DK/sample_us':>14} | {'SF/sample_us':>14} | "
          f"{'SF/DK':>7}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*14}-+-{'-'*14}-+-{'-'*7}")

    for B in B_VALUES:
        torch.manual_seed(99)
        x = (torch.rand(T, B, CIN, H, W, device=DEVICE) > SPARSITY).float()

        def _fn_dk():
            functional.reset_net(fused)
            with torch.no_grad():
                return fused._impl_dense.forward(x, spec, params, fused.state)
        dk_us = bench_us(_fn_dk)

        def _fn_sf():
            functional.reset_net(fused)
            with torch.no_grad():
                return fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
        sf_us = bench_us(_fn_sf)

        dk_per_sample = dk_us / B
        sf_per_sample = sf_us / B
        ratio = sf_us / dk_us

        print(f"  {B:>4d} | {dk_us:>9.2f} | {sf_us:>9.2f} | "
              f"{dk_per_sample:>14.3f} | {sf_per_sample:>14.3f} | "
              f"{ratio:>7.3f}")

        rows.append(Row(
            B=B,
            dk_wall_us=dk_us, sf_wall_us=sf_us,
            dk_per_sample_us=dk_per_sample, sf_per_sample_us=sf_per_sample,
            sf_over_dk=ratio,
            notes="",
        ))

    # CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        with open(CSV_PATH, "w", newline="") as f:
            fieldnames = list(asdict(rows[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    # Interpretation
    print()
    print("=" * 96)
    print("Interpretation")
    print("=" * 96)
    dk_per_sample_min = min(r.dk_per_sample_us for r in rows)
    dk_per_sample_max = max(r.dk_per_sample_us for r in rows)
    sf_per_sample_min = min(r.sf_per_sample_us for r in rows)
    sf_per_sample_max = max(r.sf_per_sample_us for r in rows)
    print(f"  DK per-sample latency: {dk_per_sample_max:.2f}us (B=1) → "
          f"{dk_per_sample_min:.2f}us (large B)  amortization {dk_per_sample_max/dk_per_sample_min:.1f}×")
    print(f"  SF per-sample latency: {sf_per_sample_max:.2f}us (B=1) → "
          f"{sf_per_sample_min:.2f}us (large B)  amortization {sf_per_sample_max/sf_per_sample_min:.1f}×")
    print()
    ratio_at_B1 = next(r.sf_over_dk for r in rows if r.B == 1)
    ratio_at_max_B = next(r.sf_over_dk for r in rows if r.B == max(B_VALUES))
    print(f"  SF/DK ratio: {ratio_at_B1:.2f}× at B=1 → {ratio_at_max_B:.2f}× at B={max(B_VALUES)}")
    if ratio_at_max_B < ratio_at_B1:
        print(f"  → SF amortizes its launch+setup overhead better at large B than DK,")
        print(f"    closing the ratio gap. But SF stays {ratio_at_max_B:.2f}× slower than DK")
        print(f"    even at the highest B tested.")
    elif ratio_at_max_B > ratio_at_B1:
        print(f"  → DK amortizes better at large B than SF; the gap WIDENS with B.")
    else:
        print(f"  → Both kernels scale similarly with B; the ratio is stable.")

    print(f"  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())