"""§5 supplementary — sparsity scan: SparseFlow vs DenseKeep wall-clock.

Holds workload fixed (SEW-RN18 layer3.0.conv2 shape, T=4, B=2, K=T=4),
sweeps INPUT sparsity ∈ {0%, 25%, 50%, 75%, 85%, 95%, 100%} and reports
DK and SF wall-clock side by side.

Purpose: §5.3 measured default-vs-DK at the network level (1.5-1.9× DK
faster on V100). This experiment localizes WHERE in the sparsity axis
SparseFlow can theoretically beat DenseKeep — and on V100 with our
Triton kernel, whether it ever does. The answer informs §3.10.4
discussion + §6 limitations.

For each sparsity:
  - DK wall_us
  - SF wall_us
  - SF/DK ratio (>1 means SF slower)
  - sparsity-induced compute reduction (SF should benefit more at high sparsity)

Run:
    python experiments/exp_46_sparsity_scan.py
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

# layer3.0.conv2 shape
CIN = COUT = 256
H = W = 8
T = 4
B = 2
K = T   # full-block SF (best case for SF)
SPARSITY_VALUES = [0.0, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95, 0.98, 1.0]

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/sparsity_scan.csv")


@dataclass
class Row:
    sparsity_target: float
    sparsity_actual: float
    dk_wall_us: float
    sf_wall_us: float
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
    print("§5 supplementary — sparsity scan (SF vs DK wall-clock)")
    print("=" * 96)
    print()
    print(f"  Workload: layer3-like (Cin=Cout={CIN}, H=W={H}, B={B}, T={T}, K={K})")
    print(f"  Bench: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median")
    print(f"  Sparsity sweep: {SPARSITY_VALUES}")
    print()

    # Build STFusion (synthetic weights, since we just measure wall-clock)
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

    print(f"  {'sparsity':>10}  {'actual':>9}  "
          f"{'DK_us':>9}  {'SF_us':>9}  {'SF/DK':>8}  {'who_wins':<10}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}")

    for sp_target in SPARSITY_VALUES:
        torch.manual_seed(int(99 + sp_target * 1000))
        # sp_target = 0 means dense (all ones); sp_target = 1 means all zeros
        if sp_target == 0.0:
            x = torch.ones(T, B, CIN, H, W, device=DEVICE)
        elif sp_target == 1.0:
            x = torch.zeros(T, B, CIN, H, W, device=DEVICE)
        else:
            x = (torch.rand(T, B, CIN, H, W, device=DEVICE) > sp_target).float()
        sp_actual = 1.0 - x.count_nonzero().item() / x.numel()

        # DK
        def _fn_dk():
            functional.reset_net(fused)
            with torch.no_grad():
                return fused._impl_dense.forward(x, spec, params, fused.state)
        dk_us = bench_us(_fn_dk)

        # SF
        def _fn_sf():
            functional.reset_net(fused)
            with torch.no_grad():
                return fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
        sf_us = bench_us(_fn_sf)

        ratio = sf_us / dk_us
        if ratio < 1.0:
            wins = "SF wins"
        elif ratio < 1.05:
            wins = "tie"
        else:
            wins = "DK wins"

        notes = ""
        if sp_target == 1.0:
            notes = "all-zero: SF takes StaticZero short-circuit"

        print(f"  {sp_target:>9.0%}  {sp_actual:>9.4f}  "
              f"{dk_us:>9.2f}  {sf_us:>9.2f}  {ratio:>8.3f}  {wins:<10}  {notes}")

        rows.append(Row(
            sparsity_target=sp_target,
            sparsity_actual=sp_actual,
            dk_wall_us=dk_us,
            sf_wall_us=sf_us,
            sf_over_dk=ratio,
            notes=notes,
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
    sf_wins = [r for r in rows if r.sf_over_dk < 1.0]
    if sf_wins:
        print(f"  SF wins at sparsity ≥ {min(r.sparsity_actual for r in sf_wins):.0%}")
        print(f"    Min SF/DK ratio: {min(r.sf_over_dk for r in sf_wins):.3f}")
    else:
        print(f"  SF NEVER beats DK at this layer3 shape on V100.")
        print(f"  Min SF/DK ratio: {min(r.sf_over_dk for r in rows):.3f} at sparsity={[r.sparsity_actual for r in rows if r.sf_over_dk == min(rs.sf_over_dk for rs in rows)][0]:.0%}")
        print()
        print(f"  This is consistent with §3.10.4 / §6 discussion: V100 + open-")
        print(f"  source Triton 2.x is not the ideal target for SF's bit-pack")
        print(f"  + warp-level reduction kernel. Expected to improve on sm_80+ ")
        print(f"  (A100/H100) where Triton is the primary architecture target.")
    print(f"  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())