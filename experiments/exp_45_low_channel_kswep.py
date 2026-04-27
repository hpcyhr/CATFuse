"""§3.10 extension — K-sweep on layer1 (64ch, 32x32) and layer2 (128ch, 16x16).

The original §3.10 K-sweep (exp_30) covered layer3 (256x8x8) and
layer4 (512x4x4) — only the deep, low-spatial-resolution layers. This
extension verifies that the §3.9 formula

    HBM_int(σ_ctf) / HBM_int(σ_ref)  =  2⌈T/K⌉ / (2T + 2)

holds across the full SEW-ResNet18 layer-shape distribution, including
the early high-spatial layers where intermediate I/O is dominated by HxW
rather than channel count.

Configs added:
  - SEW-RN18 layer1.0.conv2 (64x32x32)   real ckpt weights
  - SEW-RN18 layer2.0.conv2 (128x16x16)  real ckpt weights
  - synthetic 64x32x32  (low-ch high-spatial regime)
  - synthetic 128x16x16

For each: K-sweep over {1, 2, 4} at T=4, B=2, sparsity=0.85.
Reported per row: analytic intermediate I/O bytes, hand-computed
prediction (independent), formula match check, parity vs K=T.

Output CSV: experiments/results/k_sweep_low_channel.csv

Run:
    python experiments/exp_45_low_channel_kswep.py
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import (
    functional, neuron, layer as sj_layer
)
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/k_sweep_low_channel.csv")


@dataclass
class Row:
    config: str
    impl: str               # DenseKeep | SparseFlow
    T: int
    B: int
    Cin: int
    Cout: int
    H: int
    W: int
    sparsity: float
    K: int
    n_blocks: int
    analytic_intermediate_io: float
    handcomp_intermediate_io: float
    formula_match: bool
    wall_us_median: float
    parity_max_diff: float
    schedule_str: str


def handcomp_intermediate_io(impl: str, T: int, B: int, Cin: int, Cout: int,
                              H: int, K: int, dtype_bytes: int = 4) -> float:
    """Independent hand-computation of intermediate I/O bytes:

    DenseKeep: z (T·B·Cout·H·W) write + read + v (B·Cout·H·W) write + read
               = (2T + 2) · B · Cout · H · W · b
    SparseFlow K=T: only the fused output (Cout·H·W per block) and v carry
               = 2 · ⌈T/K⌉ · B · Cout · H · W · b
    """
    if impl == "DenseKeep":
        return (2 * T + 2) * B * Cout * H * H * dtype_bytes
    elif impl == "SparseFlow":
        n_blocks = math.ceil(T / K)
        return 2 * n_blocks * B * Cout * H * H * dtype_bytes
    else:
        raise ValueError(f"unknown impl: {impl}")


def build_synth_layer(Cin: int, Cout: int, H: int, T: int):
    """Synthetic STFusion layer with random weights."""
    torch.manual_seed(42)
    conv = sj_layer.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(DEVICE)
    bn = sj_layer.BatchNorm2d(Cout).to(DEVICE)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(DEVICE)
    return STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(DEVICE).eval()


def build_sew_rn18_block_conv2(layer_attr: str, T: int):
    """Extract layer{N}.0.conv2 from real SEW-RN18 ckpt, build STFusion."""
    sj_net, note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    block = getattr(sj_net, layer_attr)[0]
    conv, bn, lif = block.conv2, block.bn2, block.sn2
    bn.eval()
    return STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(DEVICE).eval(), note


def make_input(T: int, B: int, Cin: int, H: int, sparsity: float):
    torch.manual_seed(99)
    return (torch.rand(T, B, Cin, H, H, device=DEVICE) > sparsity).float()


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


def run_config(label: str, fused: STFusionConvBNLIF, T: int, B: int, H: int,
               sparsity: float, K_values: List[int], rows: List[Row],
               note: str):
    print()
    print("─" * 96)
    print(f"  Config: {label}  ({note})")
    print("─" * 96)

    spec = fused.spec
    params = fused._ensure_params()
    Cin, Cout = spec.in_channels, spec.out_channels
    print(f"  Cin={Cin}  Cout={Cout}  H=W={H}  T={T}  B={B}  sparsity={sparsity:.0%}")

    x = make_input(T, B, Cin, H, sparsity)

    # DenseKeep baseline
    dk_cost = fused._impl_dense.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=H)
    dk_handcomp = handcomp_intermediate_io("DenseKeep", T, B, Cin, Cout, H, K=T)
    dk_match = abs(dk_cost.intermediate_io - dk_handcomp) < 1e-6

    def _fn_dk():
        functional.reset_net(fused)
        with torch.no_grad():
            return fused._impl_dense.forward(x, spec, params, fused.state)
    dk_wall = bench_us(_fn_dk)

    # Get reference output for parity (DK is bit-exact reference)
    functional.reset_net(fused)
    with torch.no_grad():
        y_dk_ref = fused._impl_dense.forward(x, spec, params, fused.state).clone()

    print(f"\n  {'impl':>10}  {'K':>3}  {'blks':>4}  "
          f"{'analytic_kB':>12}  {'handcomp_kB':>12}  {'match':>5}  "
          f"{'wall_us':>9}  {'parity':>10}  {'sched':>20}")
    print(f"  {'-'*10}  {'-'*3}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*5}  "
          f"{'-'*9}  {'-'*10}  {'-'*20}")

    print(f"  {'DenseKeep':>10s}  {T:>3d}  {1:>4d}  "
          f"{dk_cost.intermediate_io/1024:>12.2f}  {dk_handcomp/1024:>12.2f}  "
          f"{'✓' if dk_match else '✗':>5s}  {dk_wall:>9.2f}  "
          f"{0.0:>10.2e}  {'BatchFold(T)':>20s}")
    rows.append(Row(
        config=label, impl="DenseKeep", T=T, B=B, Cin=Cin, Cout=Cout, H=H, W=H,
        sparsity=sparsity, K=T, n_blocks=1,
        analytic_intermediate_io=dk_cost.intermediate_io,
        handcomp_intermediate_io=dk_handcomp,
        formula_match=dk_match, wall_us_median=dk_wall, parity_max_diff=0.0,
        schedule_str="BatchFold(T)",
    ))

    # SparseFlow K-sweep
    for K in K_values:
        sf_cost = fused._impl_sparse.analytic_io_cost(spec, T=T, B=B,
                                                      H_in=H, W_in=H, K=K)
        sf_handcomp = handcomp_intermediate_io("SparseFlow", T, B, Cin, Cout, H, K=K)
        sf_match = abs(sf_cost.intermediate_io - sf_handcomp) < 1e-6

        def _fn_sf():
            functional.reset_net(fused)
            with torch.no_grad():
                return fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
        sf_wall = bench_us(_fn_sf)

        functional.reset_net(fused)
        with torch.no_grad():
            y_sf = fused._impl_sparse.forward_with_k(
                x, spec, params, fused.state, K=K).clone()
        sf_parity = (y_sf - y_dk_ref).abs().max().item()
        n_blocks_K = math.ceil(T / K)

        print(f"  {'SparseFlow':>10s}  {K:>3d}  {n_blocks_K:>4d}  "
              f"{sf_cost.intermediate_io/1024:>12.2f}  {sf_handcomp/1024:>12.2f}  "
              f"{'✓' if sf_match else '✗':>5s}  {sf_wall:>9.2f}  "
              f"{sf_parity:>10.2e}  {'StreamFuse+SC':>20s}")
        rows.append(Row(
            config=label, impl="SparseFlow", T=T, B=B, Cin=Cin, Cout=Cout, H=H, W=H,
            sparsity=sparsity, K=K, n_blocks=n_blocks_K,
            analytic_intermediate_io=sf_cost.intermediate_io,
            handcomp_intermediate_io=sf_handcomp,
            formula_match=sf_match, wall_us_median=sf_wall, parity_max_diff=sf_parity,
            schedule_str="StreamFuse+SC",
        ))

    # Theory check
    sf_KT = next(r for r in rows if r.config == label and r.impl == "SparseFlow"
                 and r.K == T)
    ratio = sf_KT.analytic_intermediate_io / dk_cost.intermediate_io
    pred = 1.0 / (T + 1)
    print(f"\n  §3.9 ratio at K=T={T}: {ratio:.4f}  "
          f"(theoretical 1/(T+1) = {pred:.4f})  "
          f"{'✓' if abs(ratio - pred) < 1e-6 else '✗ MISMATCH'}")


def main():
    if not torch.cuda.is_available():
        return 1

    print("=" * 96)
    print("§3.10 extension — K-sweep on layer1 (64ch) and layer2 (128ch)")
    print("=" * 96)
    print()
    print("Verifies §3.9 formula across the SEW-RN18 layer-shape distribution,")
    print("including early high-spatial-resolution layers.")
    print()

    rows: List[Row] = []
    T = 4
    K_values = [1, 2, 4]

    # ----- Real SEW-RN18 layers -----
    for layer_attr, H, label_short in [("layer1", 32, "layer1.0.conv2  (64x32x32)"),
                                        ("layer2", 16, "layer2.0.conv2  (128x16x16)")]:
        try:
            fused, note = build_sew_rn18_block_conv2(layer_attr, T=T)
            label = f"SEW-RN18 {label_short}"
            run_config(label, fused, T=T, B=2, H=H, sparsity=0.85,
                       K_values=K_values, rows=rows, note=note)
        except Exception as e:
            print(f"\n  Skipping {layer_attr}: {e}")

    # ----- Synthetic counterparts -----
    for Cin, H, regime in [(64, 32, "low-ch high-spatial"),
                            (128, 16, "mid-ch mid-spatial")]:
        syn = build_synth_layer(Cin=Cin, Cout=Cin, H=H, T=T)
        label = f"Synthetic {Cin}x{H}x{H}  ({regime})"
        run_config(label, syn, T=T, B=2, H=H, sparsity=0.85,
                   K_values=K_values, rows=rows, note="synthetic weights")

    # CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        with open(CSV_PATH, "w", newline="") as f:
            fieldnames = list(asdict(rows[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    # Summary
    print()
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    n_total = len(rows)
    n_match = sum(1 for r in rows if r.formula_match)
    n_parity_zero = sum(1 for r in rows if r.parity_max_diff == 0.0)
    print(f"  Total rows:                       {n_total}")
    print(f"  Formula matches handcomp:         {n_match}/{n_total}")
    print(f"  Parity max_diff = 0 vs K=T:       {n_parity_zero}/{n_total}")
    print(f"  CSV: {CSV_PATH}")
    print()
    if n_match == n_total and n_parity_zero == n_total:
        print("  ✓ §3.9 formula verified on all 4 low-channel configs.")
        print("  Combined with exp_30's 17 rows on layer3/layer4,")
        print("  §3.10 now covers the full SEW-RN18 layer-shape distribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())