"""§3.10 K-sweep experiment — load-bearing data for paper §5.2.

For a representative Conv→BN→LIF layer (and a sparse spike input), this
script measures across K ∈ [1, T]:

  1. analytic intermediate I/O bytes via Stage 4's
     SparseFlow.analytic_io_cost(spec, T, B, H, W, K)
     — the §3.9 cost-model prediction (no kernel runs).

  2. parity vs K=T reference: max_diff over the entire output tensor.
     §3.13 Lemma 3.14 says this MUST be 0 for every K.

  3. wall-clock latency (median over many iterations) of the deployment
     forward path SparseFlow.forward_with_k. Auxiliary metric — V100
     wall-clock is influenced by Triton conv perf vs cuDNN, which is
     orthogonal to the §3.9 I/O reduction claim.

DenseKeep (the §3.9 baseline) is included as a one-row analytic + wall-clock
reference point — its K is fixed at T by virtue of BatchFold(Conv).

Output:
  - stdout: human-readable tables, one per (config, T) combination
  - CSV:    experiments/results/k_sweep.csv with one row per (config, T, K, impl)

Run:
    cd /path/to/CATFuse
    python experiments/exp_30_k_sweep.py

The CSV is intended as the primary artifact. A separate figure-generation
script can read it later. Keeping the experiment and the plotting decoupled
means data integrity doesn't depend on matplotlib/seaborn versions.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

# Make `catfuse` importable from the repo root regardless of cwd.
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer as sj_layer, functional

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from catfuse.implementations import SparseFlow, DenseKeep


# ---------- Knobs ----------

DEVICE = "cuda:0"
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3
BENCH_DTYPE_BYTES = 4

CSV_PATH = os.path.join(_REPO_ROOT, "experiments/results/k_sweep.csv")


@dataclass
class Row:
    """One CSV row. Schema is locked — figure scripts depend on it."""
    config: str             # human-readable config tag
    impl: str               # "SparseFlow" | "DenseKeep"
    T: int
    B: int
    Cin: int
    Cout: int
    H: int
    W: int
    sparsity: float         # fraction of zero entries in the input tensor
    K: int                  # for DenseKeep this is recorded as T
    num_blocks: int
    # Analytic via Stage 4 IOCost (per-call bytes)
    analytic_x_load: int
    analytic_w_load: int
    analytic_z_io: int
    analytic_v_io: int
    analytic_spike_write: int
    analytic_intermediate_io: int   # z_io + v_io  (the §3.9 reduction target)
    analytic_total: int
    # Hand-computed expectation for cross-check (intermediate_io only —
    # we want the formula AND the implementation to agree)
    handcomp_intermediate_io: int
    formula_match: bool
    # Empirical
    wall_us_median: float
    parity_max_diff: float          # vs K=T reference
    # Schedule decomposition string
    schedule: str


# ---------- Helpers ----------

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


def handcomp_intermediate_io_sf(B: int, T: int, K: int, HWC_out: int) -> int:
    """SparseFlow intermediate_io = z_io + v_io = 0 + 2·ceil(T/K)·B·HWC·4."""
    nb = math.ceil(T / K)
    return 2 * nb * B * HWC_out * BENCH_DTYPE_BYTES


def handcomp_intermediate_io_dk(B: int, T: int, HWC_out: int) -> int:
    """DenseKeep intermediate_io = z_io + v_io = 2T·B·HWC·4 + 2·B·HWC·4."""
    return (2 * T + 2) * B * HWC_out * BENCH_DTYPE_BYTES


def build_synth_layer(Cin: int, Cout: int, H: int, T: int):
    """Build a fresh STFusion layer with synthetic weights for K-sweep
    that doesn't depend on SEW-RN18 checkpoint (used for T=8 case)."""
    torch.manual_seed(42)
    conv = sj_layer.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(DEVICE)
    bn = sj_layer.BatchNorm2d(Cout).to(DEVICE)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(DEVICE)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(DEVICE).eval()
    return fused


def build_sew_rn18_layer4_conv2(T: int):
    """Extract layer4.0.conv2 from the SEW-RN18 CIFAR10 checkpoint as a
    standalone STFusion. Falls back to synthetic weights if checkpoint
    missing.

    Returns (fused_layer, label, source_note).
    """
    label = "SEW-RN18 layer4.0.conv2  (512x4x4)"
    return _build_sew_rn18_block_conv2("layer4", label, T)


def build_sew_rn18_layer3_conv2(T: int):
    """Extract layer3.0.conv2 from the SEW-RN18 CIFAR10 checkpoint."""
    label = "SEW-RN18 layer3.0.conv2  (256x8x8)"
    return _build_sew_rn18_block_conv2("layer3", label, T)


def _build_sew_rn18_block_conv2(layer_attr: str, label: str, T: int):
    """Shared helper: extract <layer_attr>[0].conv2/.bn2/.sn2 as STFusion."""
    note = "synthetic weights"
    try:
        from spikingjelly.activation_based.model import sew_resnet
        from spikingjelly.activation_based import surrogate
        net = sew_resnet.sew_resnet18(
            pretrained=False, cnf="ADD",
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        functional.set_step_mode(net, "m")
        ckpt_path = os.path.join(_REPO_ROOT, "checkpoints",
                                 "sew_resnet18_cifar10_lif_best.pth")
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "net" in state:
                    state = state["net"]
                net.load_state_dict(state, strict=False)
                note = "from sew_resnet18_cifar10_lif_best.pth"
            except Exception:
                pass
        net = net.to(DEVICE).eval()
        block = getattr(net, layer_attr)[0]
        fused = STFusionConvBNLIF.from_sj_modules(
            block.conv2, block.bn2, block.sn2, K=T,
        ).to(DEVICE).eval()
        return fused, label, note
    except Exception as e:
        print(f"  (could not build from SEW-RN18 {layer_attr}: {e}; using synthetic)")
        return None, label, "synthetic (fallback)"


def make_input(T: int, B: int, Cin: int, H: int, sparsity: float):
    """Bernoulli spike input with given sparsity (fraction of zeros)."""
    torch.manual_seed(99)
    return (torch.rand(T, B, Cin, H, H, device=DEVICE) > sparsity).float()


# ---------- Per-config K-sweep ----------

def run_config(label: str, fused, T: int, B: int, H: int, sparsity: float,
               K_values: List[int], rows: List[Row], note: str = ""):
    spec = fused.spec
    params = fused._ensure_params()
    Cin, Cout = spec.in_channels, spec.out_channels
    HWC_out = H * H * Cout  # stride=1, pad=1 ⇒ H_out=H

    print(f"\n{'─' * 96}")
    print(f"{label}  ({note})")
    print(f"  Cin={Cin}  Cout={Cout}  H=W={H}  T={T}  B={B}  sparsity={sparsity:.0%}")
    print(f"{'─' * 96}")

    x = make_input(T, B, Cin, H, sparsity)

    # ---- Reference output (K=T) for parity ----
    functional.reset_net(fused)
    y_ref = fused._impl_sparse.forward_with_k(
        x, spec, params, fused.state, K=T).clone()

    # ---- DenseKeep baseline (one row) ----
    dk_cost = fused._impl_dense.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=H)
    dk_handcomp = handcomp_intermediate_io_dk(B, T, HWC_out)
    dk_match = (dk_cost.intermediate_io == dk_handcomp)

    def _fn_dk():
        functional.reset_net(fused)
        return fused._impl_dense.forward(x, spec, params, fused.state)
    dk_wall = bench(_fn_dk)

    # DenseKeep parity vs SparseFlow K=T (Corollary 3.17 sanity)
    functional.reset_net(fused)
    y_dk = fused._impl_dense.forward(x, spec, params, fused.state)
    dk_parity = (y_dk - y_ref).abs().max().item()

    rows.append(Row(
        config=label, impl="DenseKeep", T=T, B=B,
        Cin=Cin, Cout=Cout, H=H, W=H, sparsity=sparsity,
        K=T, num_blocks=1,
        analytic_x_load=dk_cost.x_load,
        analytic_w_load=dk_cost.w_load,
        analytic_z_io=dk_cost.z_io,
        analytic_v_io=dk_cost.v_io,
        analytic_spike_write=dk_cost.spike_write,
        analytic_intermediate_io=dk_cost.intermediate_io,
        analytic_total=dk_cost.total,
        handcomp_intermediate_io=dk_handcomp,
        formula_match=dk_match,
        wall_us_median=dk_wall,
        parity_max_diff=dk_parity,
        schedule=str(fused._impl_dense.schedule_decomposition(spec, T=T)),
    ))

    # ---- SparseFlow K-sweep ----
    print(f"  {'impl':>10s}  {'K':>3s}  {'#blk':>4s}  "
          f"{'inter_io_KB':>12s}  {'handcomp_KB':>12s}  "
          f"{'match':>5s}  {'wall_us':>9s}  {'parity':>10s}  schedule")
    print(f"  {'─' * 92}")

    print(f"  {'DenseKeep':>10s}  {T:>3d}  {1:>4d}  "
          f"{dk_cost.intermediate_io/1024:>12.2f}  {dk_handcomp/1024:>12.2f}  "
          f"{'✓' if dk_match else '✗':>5s}  {dk_wall:>9.2f}  "
          f"{dk_parity:>10.2e}  baseline")

    for K in K_values:
        sf_cost = fused._impl_sparse.analytic_io_cost(
            spec, T=T, B=B, H_in=H, W_in=H, K=K)
        sf_handcomp = handcomp_intermediate_io_sf(B, T, K, HWC_out)
        sf_match = (sf_cost.intermediate_io == sf_handcomp)

        def _fn_sf():
            functional.reset_net(fused)
            return fused._impl_sparse.forward_with_k(
                x, spec, params, fused.state, K=K)
        sf_wall = bench(_fn_sf)

        functional.reset_net(fused)
        y_sf = fused._impl_sparse.forward_with_k(
            x, spec, params, fused.state, K=K)
        sf_parity = (y_sf - y_ref).abs().max().item()

        rows.append(Row(
            config=label, impl="SparseFlow", T=T, B=B,
            Cin=Cin, Cout=Cout, H=H, W=H, sparsity=sparsity,
            K=K, num_blocks=sf_cost.num_blocks,
            analytic_x_load=sf_cost.x_load,
            analytic_w_load=sf_cost.w_load,
            analytic_z_io=sf_cost.z_io,
            analytic_v_io=sf_cost.v_io,
            analytic_spike_write=sf_cost.spike_write,
            analytic_intermediate_io=sf_cost.intermediate_io,
            analytic_total=sf_cost.total,
            handcomp_intermediate_io=sf_handcomp,
            formula_match=sf_match,
            wall_us_median=sf_wall,
            parity_max_diff=sf_parity,
            schedule=str(fused._impl_sparse.schedule_decomposition(spec, T=T, K=K)),
        ))

        print(f"  {'SparseFlow':>10s}  {K:>3d}  {sf_cost.num_blocks:>4d}  "
              f"{sf_cost.intermediate_io/1024:>12.2f}  {sf_handcomp/1024:>12.2f}  "
              f"{'✓' if sf_match else '✗':>5s}  {sf_wall:>9.2f}  "
              f"{sf_parity:>10.2e}  K={K}")

    # ---- §3.9 ratio summary ----
    sf_KT = next(r for r in rows
                 if r.config == label and r.T == T and r.impl == "SparseFlow"
                 and r.K == T)
    ratio_inter = sf_KT.analytic_intermediate_io / dk_cost.intermediate_io
    ratio_total = sf_KT.analytic_total / dk_cost.total
    pred = 1.0 / (T + 1)
    print(f"\n  §3.9 prediction at K=T={T}:")
    print(f"     SparseFlow.intermediate_io / DenseKeep.intermediate_io = "
          f"{ratio_inter:.4f}  (theoretical 1/(T+1) = {pred:.4f})")
    print(f"     {'✓ matches' if abs(ratio_inter - pred) < 1e-6 else '✗ MISMATCH'}")
    print(f"     SparseFlow.total           / DenseKeep.total           = {ratio_total:.4f}")
    print(f"     (total includes x_load + w_load + spike_write which are NOT")
    print(f"      reduced by StreamFuse — §3.9's claim is ONLY about z_io+v_io)")


# ---------- Main ----------

def main():
    if not torch.cuda.is_available():
        print("This experiment requires CUDA")
        return 1

    print("=" * 96)
    print("§3.10 K-sweep experiment — analytic + parity + wall-clock")
    print("=" * 96)
    print()
    print("This script produces the load-bearing data for paper §5.2.")
    print()
    print("For each config × T × K, we record:")
    print("  - analytic intermediate I/O bytes  (Stage 4 IOCost.intermediate_io)")
    print("  - hand-computed prediction         (independent cross-check)")
    print("  - wall-clock latency               (auxiliary; not the §3.9 metric)")
    print("  - parity vs K=T                    (§3.13 Lemma 3.14, must be 0)")
    print()
    print(f"Bench config: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median reported")
    print(f"CSV output: {CSV_PATH}")

    rows: List[Row] = []

    # ============================================================
    # Config 1: SEW-RN18 layer4.0.conv2 (512×4×4) at T=4, sparse input
    # ============================================================
    fused, label, note = build_sew_rn18_layer4_conv2(T=4)
    if fused is not None:
        # layer4 sees H=4 in CIFAR10 (32→16 stride2 conv1, →16 layer1, →8 layer2,
        # →4 layer3, →4 layer4 since layer4 stride=2 halves layer3's input 8→4).
        # But within an isolated layer test we just pick H=4 and B=2.
        run_config(label=label, fused=fused, T=4, B=2, H=4, sparsity=0.85,
                   K_values=[1, 2, 4], rows=rows, note=note)
    else:
        print("Skipping SEW-RN18 layer4 config")

    # ============================================================
    # Config 1b: SEW-RN18 layer3.0.conv2 (256×8×8) — REAL weights
    # Adding this so we have TWO data points from the real network at
    # different intermediate_io / total ratios. layer3 has a much larger
    # spatial size than layer4, so intermediate_io is a bigger fraction
    # of total — SparseFlow's wall-clock is expected to look better here
    # relative to DenseKeep than in the layer4 case.
    # ============================================================
    fused3, label3, note3 = build_sew_rn18_layer3_conv2(T=4)
    if fused3 is not None:
        run_config(label=label3, fused=fused3, T=4, B=2, H=8, sparsity=0.85,
                   K_values=[1, 2, 4], rows=rows, note=note3)
    else:
        print("Skipping SEW-RN18 layer3 config")

    # ============================================================
    # Config 2: synthetic 256-channel layer at T=4, sparse input
    # (matches layer3 shape regime; provides a 2nd independent data point)
    # ============================================================
    syn = build_synth_layer(Cin=256, Cout=256, H=8, T=4)
    run_config(label="Synthetic 256x8x8  (layer3 regime)",
               fused=syn, T=4, B=2, H=8, sparsity=0.85,
               K_values=[1, 2, 4], rows=rows, note="synthetic weights")

    # ============================================================
    # Config 3: synthetic same shape but T=8 — gives finer K-sweep
    # ============================================================
    syn_t8 = build_synth_layer(Cin=256, Cout=256, H=8, T=8)
    run_config(label="Synthetic 256x8x8  (T=8 fine sweep)",
               fused=syn_t8, T=8, B=2, H=8, sparsity=0.85,
               K_values=[1, 2, 4, 8], rows=rows, note="synthetic weights")

    # ============================================================
    # Write CSV
    # ============================================================
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
    n_total = len(rows)
    n_match = sum(1 for r in rows if r.formula_match)
    n_parity_zero = sum(1 for r in rows if r.parity_max_diff == 0.0)
    print(f"  Total rows:                      {n_total}")
    print(f"  Formula matches handcomp:        {n_match}/{n_total}")
    print(f"  Parity max_diff = 0 vs K=T:      {n_parity_zero}/{n_total}")
    print(f"  CSV written to:                  {CSV_PATH}")

    if n_match != n_total:
        print()
        print("WARNING: some analytic_io_cost values disagree with hand-computed")
        print("         expectations. Investigate before using this data in the paper.")
        for r in rows:
            if not r.formula_match:
                print(f"  MISMATCH: {r.config} T={r.T} K={r.K} impl={r.impl}: "
                      f"formula={r.analytic_intermediate_io}, "
                      f"handcomp={r.handcomp_intermediate_io}")

    if n_parity_zero != n_total:
        print()
        print("CRITICAL: some K-variants produced output that differs from K=T.")
        print("          §3.13 Lemma 3.14 violated — implementation bug, NOT a")
        print("          paper writing fix. Investigate before reporting any data.")
        for r in rows:
            if r.parity_max_diff != 0.0:
                print(f"  PARITY FAIL: {r.config} T={r.T} K={r.K} impl={r.impl}: "
                      f"max_diff={r.parity_max_diff:.4e}")

    return 0 if (n_match == n_total and n_parity_zero == n_total) else 1


if __name__ == "__main__":
    sys.exit(main())