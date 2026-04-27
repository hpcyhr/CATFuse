"""§3.10.3 — Triton kernel launch overhead measurement on V100.

The K-sweep wall-clock data in §3.10 shows that K=1 (T launches per
forward) is consistently slower than K=T (one launch). We attribute
this to per-launch overhead, but in §3.10.3 the number "~50-100us per
launch" was a rough estimate. This experiment provides an anchor by
directly measuring:

  A. Empty Triton kernel launch+sync (pure launch overhead)
  B. Trivial Triton kernel with 1 grid block (launch + minimal compute)
  C. Real SparseFlow streamfuse_conv3x3_bn_lif kernel for one timestep
     of a (B, C, H, W) shape matching SEW-RN18 layer3 dimensions
     (launch + real conv compute, our actual workload's per-launch cost)

These three numbers let us decompose the K-sweep wall-clock into
launch + compute components and check if the SF K=1 vs K=T gap is
explained by launch count.

Output: stdout per-kernel mean wall_us; CSV row per kernel.

Run:
    python experiments/exp_43_launch_overhead.py
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
import triton
import triton.language as tl

DEVICE = "cuda:0"
N_WARMUP = 100
N_ITER = 1000
N_REPEAT = 5

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/launch_overhead.csv")


@dataclass
class Row:
    kernel: str
    launches_per_call: int
    wall_us_median: float
    wall_us_per_launch: float
    description: str


# ============================================================
# Trivial kernels for launch-overhead measurement
# ============================================================

@triton.jit
def empty_kernel(dummy_ptr):
    """Does literally nothing. Pure launch overhead."""
    pass


@triton.jit
def trivial_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    """1 load + 1 store — minimum useful work."""
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x)


def bench_empty(n_launches_per_call: int) -> float:
    """Time `n_launches_per_call` empty kernel launches per call, return median us per call."""
    dummy = torch.zeros(1, device=DEVICE, dtype=torch.float32)
    def _fn():
        for _ in range(n_launches_per_call):
            empty_kernel[(1,)](dummy)
    return _bench(_fn)


def bench_trivial(n_launches_per_call: int) -> float:
    """Time `n_launches_per_call` trivial-kernel launches per call."""
    BLOCK = 16
    x = torch.zeros(BLOCK, device=DEVICE, dtype=torch.float32)
    y = torch.zeros(BLOCK, device=DEVICE, dtype=torch.float32)
    def _fn():
        for _ in range(n_launches_per_call):
            trivial_kernel[(1,)](x, y, BLOCK=BLOCK)
    return _bench(_fn)


def _bench(fn) -> float:
    """Median microseconds per call."""
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
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§3.10.3 — Triton kernel launch overhead measurement on V100")
    print("=" * 96)
    print()
    print(f"  Bench: {N_WARMUP} warmup + {N_ITER} iter × {N_REPEAT} repeats; median")
    print()

    rows: List[Row] = []

    # ============================================================
    # A. Empty kernel
    # ============================================================
    print("Part A — Empty kernel (pure launch overhead)")
    print("─" * 96)
    for n_launches in [1, 4, 8]:
        t = bench_empty(n_launches)
        per_launch = t / n_launches
        print(f"  empty_kernel × {n_launches:>2d} launches per call: "
              f"{t:>8.3f} us total → {per_launch:>7.3f} us per launch")
        rows.append(Row(
            kernel="empty",
            launches_per_call=n_launches,
            wall_us_median=t,
            wall_us_per_launch=per_launch,
            description="empty Triton kernel (pure launch overhead)",
        ))

    # ============================================================
    # B. Trivial kernel
    # ============================================================
    print()
    print("Part B — Trivial kernel (1 load + 1 store, BLOCK=16)")
    print("─" * 96)
    for n_launches in [1, 4, 8]:
        t = bench_trivial(n_launches)
        per_launch = t / n_launches
        print(f"  trivial_kernel × {n_launches:>2d} launches per call: "
              f"{t:>8.3f} us total → {per_launch:>7.3f} us per launch")
        rows.append(Row(
            kernel="trivial",
            launches_per_call=n_launches,
            wall_us_median=t,
            wall_us_per_launch=per_launch,
            description="trivial 1-load-1-store Triton kernel, BLOCK=16",
        ))

    # ============================================================
    # C. Real SparseFlow streamfuse kernel — single timestep of layer3 shape
    # ============================================================
    print()
    print("Part C — Real SparseFlow kernel on layer3 shape (Cin=Cout=256, H=W=8, B=2)")
    print("─" * 96)
    print("  This is the actual production kernel used by §5.3 default routing.")
    print("  We launch it for T_eff=1 timestep on a (B, C, H, W) tile, then")
    print("  for T_eff=4 timesteps on (T*B, C, H, W) — i.e., one block of K=1")
    print("  vs one block of K=4. Difference / 3 = per-extra-launch cost in")
    print("  the real kernel context (not just empty-launch overhead).")
    print()

    try:
        from spikingjelly.activation_based import (
            functional, neuron, layer as sj_layer
        )
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
    except Exception as e:
        print(f"  SKIP: imports failed: {e}")
    else:
        # Build a synthetic layer3-like STFusion
        Cin, Cout, H = 256, 256, 8
        T_full, B = 4, 2
        torch.manual_seed(42)
        conv = sj_layer.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(DEVICE)
        bn = sj_layer.BatchNorm2d(Cout).to(DEVICE)
        bn.running_mean.normal_(0, 0.1)
        bn.running_var.uniform_(0.5, 1.5)
        bn.eval()
        lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                             step_mode="m").to(DEVICE)

        def make_fused(K):
            f = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=K)
            f = f.to(DEVICE).eval()
            # Force SF path to be available
            return f

        spec_T = {1: T_full, 2: T_full, 4: T_full}
        for K in [1, 2, 4]:
            fused = make_fused(K)
            spec = fused.spec
            params = fused._ensure_params()
            # 95% sparse input matching real CIFAR10 distribution
            torch.manual_seed(99)
            x = (torch.rand(T_full, B, Cin, H, H, device=DEVICE) > 0.95).float()
            # Use forward_with_k for explicit K control
            def _fn():
                functional.reset_net(fused)
                return fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
            t = _bench(_fn)
            n_launches = (T_full + K - 1) // K
            per_launch = t / n_launches
            print(f"  SF streamfuse, K={K} (= {n_launches} launches per forward): "
                  f"{t:>8.2f} us total → {per_launch:>8.2f} us per launch")
            rows.append(Row(
                kernel=f"sf_streamfuse_K={K}",
                launches_per_call=n_launches,
                wall_us_median=t,
                wall_us_per_launch=per_launch,
                description=f"SparseFlow Triton kernel on layer3 shape, K={K}",
            ))

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    print(f"  {'kernel':<25} {'launches':>8} {'wall_us':>10} {'per_launch_us':>14}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*14}")
    for r in rows:
        print(f"  {r.kernel:<25} {r.launches_per_call:>8d} "
              f"{r.wall_us_median:>10.2f} {r.wall_us_per_launch:>14.2f}")

    print()
    print("  Interpretation for §3.10.3:")
    empty_4 = next((r for r in rows if r.kernel == "empty" and r.launches_per_call == 4), None)
    if empty_4:
        print(f"    Pure launch overhead per Triton call: ~{empty_4.wall_us_per_launch:.1f} us")
    sf_rows = [r for r in rows if r.kernel.startswith("sf_streamfuse_K=")]
    if len(sf_rows) >= 2:
        sf_K1 = next((r for r in sf_rows if "K=1" in r.kernel), None)
        sf_K4 = next((r for r in sf_rows if "K=4" in r.kernel), None)
        if sf_K1 and sf_K4:
            extra_launch_cost = (sf_K1.wall_us_median - sf_K4.wall_us_median) / 3
            print(f"    SF K=1 vs K=4 wall-clock diff: "
                  f"{sf_K1.wall_us_median - sf_K4.wall_us_median:.1f} us = "
                  f"{extra_launch_cost:.1f} us per extra launch")
            print(f"    (i.e., the per-launch cost in real SF context, "
                  f"includes cache-warm setup)")

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        fieldnames = list(asdict(rows[0]).keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
    print()
    print(f"  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())