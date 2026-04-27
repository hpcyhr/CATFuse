"""§3.10.3 — torch.profiler kernel-count verification.

Cross-check exp_43's launch overhead measurement: if SF K=1 really
launches 4 times and K=4 launches 1 time per forward, the per-launch
overhead computed in exp_43 (via wall-clock diff / launch count diff)
is interpretable. If profiler shows different counts (e.g., K=4 also
launches 4× because we didn't merge correctly), the exp_43 number
has wrong denominator.

This experiment uses torch.profiler.ProfilerActivity.CUDA to count
GPU kernel launches per forward, broken down by kernel name. We focus
on the SparseFlow streamfuse Triton kernel ('sparse_streamfuse_conv3x3_bn_lif').

Measurement: same layer3 shape as exp_43 (Cin=Cout=256, H=W=8, B=2),
T=4, sweep K ∈ {1, 2, 4}. For each K, record:
  - total CUDA kernels launched per forward
  - count of sparse_streamfuse kernel launches
  - count of all other kernels (state init, lif tail, etc.)

Run:
    python experiments/exp_44_profiler_kernel_count.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.profiler import profile, ProfilerActivity

DEVICE = "cuda:0"
T_FULL = 4
B = 2
C = 256
H = W = 8

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/profiler_kernel_count.csv")


@dataclass
class Row:
    K: int
    expected_launches_per_forward: int   # ceil(T/K)
    streamfuse_launches: int
    total_cuda_kernels: int
    other_launches: int
    streamfuse_kernel_name: str
    notes: str


def main():
    if not torch.cuda.is_available():
        return 1

    print("=" * 96)
    print("§3.10.3 — Profiler kernel-count verification (cross-check exp_43)")
    print("=" * 96)
    print()
    print(f"  Workload: layer3-like (Cin=Cout={C}, H=W={H}, B={B}, T={T_FULL})")
    print(f"  Sweep: K ∈ {{1, 2, 4}}")
    print()

    from spikingjelly.activation_based import (
        functional, neuron, layer as sj_layer
    )
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

    rows: List[Row] = []

    for K in [1, 2, 4]:
        torch.manual_seed(42)
        conv = sj_layer.Conv2d(C, C, 3, padding=1, bias=False).to(DEVICE)
        bn = sj_layer.BatchNorm2d(C).to(DEVICE)
        bn.running_mean.normal_(0, 0.1)
        bn.running_var.uniform_(0.5, 1.5)
        bn.eval()
        lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                             step_mode="m").to(DEVICE)
        fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=K)
        fused = fused.to(DEVICE).eval()

        torch.manual_seed(99)
        x = (torch.rand(T_FULL, B, C, H, W, device=DEVICE) > 0.95).float()
        spec = fused.spec
        params = fused._ensure_params()

        # Warmup
        for _ in range(20):
            functional.reset_net(fused)
            with torch.no_grad():
                _ = fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
        torch.cuda.synchronize()

        # Profile a single forward
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            functional.reset_net(fused)
            with torch.no_grad():
                _ = fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
            torch.cuda.synchronize()

        # Extract kernel launches (CUDA events with self_cuda_time_total > 0)
        events = prof.key_averages()
        kernel_counter = Counter()
        for e in events:
            # e.key is the kernel name; e.count is the number of launches
            if e.self_cuda_time_total > 0:
                kernel_counter[e.key] += e.count

        # Find streamfuse-related kernel
        streamfuse_name = None
        streamfuse_count = 0
        for kname, count in kernel_counter.items():
            if "streamfuse" in kname.lower() or "sparse_streamfuse" in kname.lower():
                streamfuse_name = kname
                streamfuse_count = count
                break

        total_kernels = sum(kernel_counter.values())
        other_kernels = total_kernels - streamfuse_count
        expected = (T_FULL + K - 1) // K

        # Print breakdown
        print(f"  K = {K}  (expected SF launches = ⌈T/K⌉ = {expected})")
        print(f"  ─" * 47)
        print(f"  Total CUDA kernel launches: {total_kernels}")
        print(f"  Per-kernel breakdown:")
        for kname, count in sorted(kernel_counter.items(), key=lambda x: -x[1]):
            short = kname[:60] + "..." if len(kname) > 60 else kname
            marker = "  ← SF streamfuse" if kname == streamfuse_name else ""
            print(f"    {count:>5d}  {short}{marker}")
        print()

        match = "match" if streamfuse_count == expected else f"MISMATCH (expected {expected}!)"
        rows.append(Row(
            K=K,
            expected_launches_per_forward=expected,
            streamfuse_launches=streamfuse_count,
            total_cuda_kernels=total_kernels,
            other_launches=other_kernels,
            streamfuse_kernel_name=streamfuse_name or "<not found>",
            notes=match,
        ))

    # ============================================================
    # Summary
    # ============================================================
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    print(f"  {'K':>3} | {'expected':>9} | {'SF actual':>10} | "
          f"{'total kernels':>14} | {'other':>6} | {'verdict':<20}")
    print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*10}-+-{'-'*14}-+-{'-'*6}-+-{'-'*20}")
    all_match = True
    for r in rows:
        verdict = "✓ matches expected" if r.streamfuse_launches == r.expected_launches_per_forward else "✗ MISMATCH"
        if "MISMATCH" in verdict:
            all_match = False
        print(f"  {r.K:>3} | {r.expected_launches_per_forward:>9d} | "
              f"{r.streamfuse_launches:>10d} | {r.total_cuda_kernels:>14d} | "
              f"{r.other_launches:>6d} | {verdict:<20}")

    print()
    print("Interpretation:")
    if all_match:
        print("  ✓ SF kernel launch count matches ⌈T/K⌉ for all K ∈ {1, 2, 4}.")
        print("    The exp_43 per-launch overhead estimate uses the correct denominator.")
        print("    This validates the §3.10 K-sweep wall-clock decomposition into")
        print("    launch + compute components.")
    else:
        print("  ✗ Some K values have unexpected launch counts. Investigate before")
        print("    reporting per-launch overhead numbers.")

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
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())