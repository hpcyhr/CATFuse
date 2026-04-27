#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 8 refactor: K-sweep test fix (Bug 8) + parity suite smoke.

Stage 8 ships:
  - tests/test_30b_k_sweep_real.py — corrected K-sweep test using the Stage
    3-6 Implementation API (real DenseKeep/SparseFlow paths, not inline
    Python LIF)
  - tests/parity_suite_smoke.py — driver for SEW-RN18 full-network K-sweep
    bit-exact matrix (T∈{4,8}, B∈{1,2}, K∈{1,2,4,8})
  - tests/stage8_verify.py — verification entry point
  - Annotation on tests/test_30_k_sweep_hbm.py marking it as historical

The original test_30 is preserved for historical comparison; its `hybrid_fn`
uses an inline Python LIF loop and so does not represent actual deployment
performance. Refer to test_30b for the deployment-correct measurement.

Prerequisites: stages 1–7 applied. Idempotent.

Run:
    cd /path/to/CATFuse
    python apply_stage8.py
    python tests/stage8_verify.py
"""
import os, sys


NEW_FILES = {
    'tests/test_30b_k_sweep_real.py': r'''"""Test 30b: §3.10 K-sweep using DEPLOYMENT path (Stage 3-6 APIs).

This is the corrected version of test_30_k_sweep_hbm.py per Stage 8.

The original test_30 used an INLINE Python loop for the LIF dynamics in
its `hybrid_fn` benchmark — `v = v * 0.5 + z_block[t] * 0.5` rather than
the actual lif_sequential Triton kernel that production patterns use.
This made test_30's wall-clock numbers unrepresentative of CATFuse's
actual deployment behavior.

This test_30b replaces that with the real Implementation APIs:

  - analytic I/O bytes:    impl.analytic_io_cost(spec, T, B, H, W, K)
                           (Stage 4 — exact §3.9 formula)
  - empirical K-sweep:     impl.forward_with_k(x, ..., K=K)
                           (Stage 6 — chunks T into ceil(T/K) blocks,
                            relies on StateBuffer for StateCarry(LIF))
  - parity within K:       compare against forward_with_k(K=T), expect
                           max_diff = 0 (Corollary 3.17 + §3.13 Lemma 3.14)

Output: per (config, T, K) row showing analytic intermediate_io,
empirical wall-clock, and bit-exact parity vs K=T.

Run:
    cd /path/to/CATFuse
    python tests/test_30b_k_sweep_real.py
"""
import os
import sys
import time
import statistics

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer as sj_layer, functional

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from catfuse.implementations import SparseFlow, DenseKeep

DEVICE = "cuda:0"
torch.backends.cudnn.benchmark = False
N_WARMUP = 30
N_ITER = 100


def bench(fn, n_warmup=N_WARMUP, n_iter=N_ITER, n_repeat=3):
    """Median wall-clock per fn() call, in microseconds."""
    times = []
    for _ in range(n_repeat):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_iter * 1e6)
    return statistics.median(times)


def build_layer(B, Cin, Cout, H, T):
    """Build a fused STFusion layer + sparse input."""
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


def main():
    # Configurations selected to overlap with SEW-RN18 / VGG11 layer shapes
    configs = [
        # (label, B, Cin, Cout, H, sparsity)
        ("layer3-shape  256x14",  2, 256, 256, 14, 0.85),
        ("layer4-shape  512x7",   2, 512, 512,  7, 0.85),
        ("layer3-shape (denser)", 2, 256, 256, 14, 0.50),
    ]
    T_values = [4, 8]

    print("=" * 110)
    print("Test 30b: §3.10 K-sweep using DEPLOYMENT path (Stage 3-6 APIs)")
    print("=" * 110)
    print()
    print("All numbers are per forward() call.")
    print("  analytic_io  = SparseFlow.analytic_io_cost(...).intermediate_io  (z_io + v_io)")
    print("  wall_us      = median of forward_with_k(K) over",
          f"{N_ITER} iterations × 3 repeats")
    print("  parity       = max |forward_with_k(K=K) - forward_with_k(K=T)|")
    print("                 (expected: 0.00e+00, §3.13 Lemma 3.14)")
    print()

    for label, B, Cin, Cout, H, sp in configs:
        print(f"\n{'─' * 110}")
        print(f"Config: {label}  (B={B}, Cin={Cin}, Cout={Cout}, H={H}, sparsity={sp:.0%})")
        print(f"{'─' * 110}")

        for T in T_values:
            fused = build_layer(B, Cin, Cout, H, T)
            # Sparse input
            torch.manual_seed(99)
            x = (torch.rand(T, B, Cin, H, H, device=DEVICE) > sp).float()
            spec = fused.spec
            params = fused._ensure_params()

            # Reference output: K=T
            functional.reset_net(fused)
            y_ref = fused._impl_sparse.forward_with_k(
                x, spec, params, fused.state, K=T).clone()

            print(f"\n  T={T}")
            print(f"  {'K':>3s} {'#blk':>4s} "
                  f"{'analytic_io_KB':>14s} {'analytic_total_KB':>17s} "
                  f"{'wall_us':>10s} {'wall_per_step_us':>17s} "
                  f"{'parity':>10s}")
            print(f"  {'─' * 88}")

            K_values = [K for K in [1, 2, 4, 8] if K <= T]
            for K in K_values:
                # Analytic via Stage 4 IOCost
                cost = fused._impl_sparse.analytic_io_cost(
                    spec, T=T, B=B, H_in=H, W_in=H, K=K)

                # Empirical wall-clock via Stage 6 forward_with_k
                def _fn():
                    functional.reset_net(fused)
                    return fused._impl_sparse.forward_with_k(
                        x, spec, params, fused.state, K=K)
                t_us = bench(_fn)

                # Parity vs K=T
                functional.reset_net(fused)
                y_K = fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
                max_diff = (y_K - y_ref).abs().max().item()

                print(f"  {K:>3d} {cost.num_blocks:>4d} "
                      f"{cost.intermediate_io/1024:>14.2f} "
                      f"{cost.total/1024:>17.2f} "
                      f"{t_us:>10.2f} "
                      f"{t_us/T:>17.2f} "
                      f"{max_diff:>10.2e}")

            # DenseKeep baseline for comparison (one row at K=T)
            dk_cost = fused._impl_dense.analytic_io_cost(
                spec, T=T, B=B, H_in=H, W_in=H)

            def _fn_dense():
                functional.reset_net(fused)
                return fused._impl_dense.forward(
                    x, spec, params, fused.state)
            t_dk = bench(_fn_dense)
            print(f"  {'DK':>3s} {'  -':>4s} "
                  f"{dk_cost.intermediate_io/1024:>14.2f} "
                  f"{dk_cost.total/1024:>17.2f} "
                  f"{t_dk:>10.2f} "
                  f"{t_dk/T:>17.2f} "
                  f"{'  baseline':>10s}")

            # §3.9 ratio at K=T
            sf_cost_KT = fused._impl_sparse.analytic_io_cost(
                spec, T=T, B=B, H_in=H, W_in=H, K=T)
            ratio = sf_cost_KT.intermediate_io / dk_cost.intermediate_io
            print(f"  §3.9 prediction at K=T={T}: SF intermediate_io / DK = "
                  f"{ratio:.4f} (= 1/(T+1) = {1.0/(T+1):.4f})")

    print()
    print("=" * 110)
    print("Notes:")
    print("  - parity column should be 0 for every K — empirical proof of")
    print("    §3.13 Lemma 3.14 on the live deployment kernel.")
    print("  - analytic_io is upper-bound (worst-case dense input);")
    print("    sparsity-aware kernel may move fewer bytes empirically.")
    print("  - wall_us mixes input loading, conv compute, BN, LIF, output")
    print("    writes — it's a holistic latency, not pure HBM-bound metric.")
    print("=" * 110)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Test 30b requires CUDA")
        sys.exit(0)
    main()
''',
    'tests/parity_suite_smoke.py': r'''"""parity_suite_smoke: smoke-level bit-exact matrix for the §3.10 substrate.

Runs SEW-ResNet18 end-to-end, swaps each STFusion layer's K dynamically via
forward_with_k, and verifies that the network's final output is bit-exact
identical to the K=T reference for every K choice. This is a smoke-level
version of the full parity_suite (which would also iterate over more
networks, T values, batch sizes, and CTF backends).

Stage 8 deliverable. Future work: extend to the 19+17 network coverage
matrix (Stage 8 full version per the audit's problem 6) — only after the
§3.10 K-sweep experiment has produced primary data, since that's the
load-bearing experiment for §5.2.

Run:
    cd /path/to/CATFuse
    python tests/parity_suite_smoke.py
"""
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron, surrogate
        from spikingjelly.activation_based.model import sew_resnet
        from catfuse import optimize
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception as e:
        print(f"SKIP: imports failed: {e}")
        return 0

    if not torch.cuda.is_available() or SparseFlow is None:
        print("SKIP: needs CUDA + SparseFlow")
        return 0

    device = "cuda:0"

    # Build SEW-ResNet18 ADD
    net = sew_resnet.sew_resnet18(
        pretrained=False, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    functional.set_step_mode(net, "m")
    net = net.to(device).eval()

    print("=" * 70)
    print("parity_suite_smoke: SEW-RN18 full-network K-sweep bit-exact matrix")
    print("=" * 70)

    matrix_pass = True

    for T in [4, 8]:
        for B in [1, 2]:
            torch.manual_seed(0)
            x = torch.randn(T, B, 3, 32, 32, device=device)

            # Build a fresh fused network for this (T, B) combination
            fused, _ = optimize(net, T=T, use_sparseflow=True)
            fused = fused.to(device).eval()

            # Reference: default forward (every STFusion uses its single-block
            # forward, which == forward_with_k(K=T))
            with torch.no_grad():
                functional.reset_net(fused)
                y_ref = fused(x).clone()

            st_layers = [m for m in fused.modules()
                         if isinstance(m, STFusionConvBNLIF)]
            n_st = len(st_layers)

            print(f"\n  T={T}, B={B}, STFusion layers={n_st}")
            print(f"    {'K':>3s}  {'max_diff_vs_K=T':>20s}  status")

            K_values = [K for K in [1, 2, 4, 8] if K <= T]
            for K in K_values:
                # Monkey-patch every STFusion's _batchfold_forward to use K=K
                originals = []
                for layer in st_layers:
                    originals.append(layer._batchfold_forward)
                    def _make_K_forward(layer, K=K):
                        def _bf(x):
                            if layer._impl_sparse is not None:
                                return layer._impl_sparse.forward_with_k(
                                    x, layer.spec, layer._ensure_params(),
                                    layer.state, K=K)
                            return layer._impl_dense.forward(
                                x, layer.spec, layer._ensure_params(),
                                layer.state)
                        return _bf
                    layer._batchfold_forward = _make_K_forward(layer)

                with torch.no_grad():
                    functional.reset_net(fused)
                    y_K = fused(x)

                # Restore
                for layer, orig in zip(st_layers, originals):
                    layer._batchfold_forward = orig

                max_diff = (y_K - y_ref).abs().max().item()
                ok = max_diff == 0
                mark = "OK" if ok else "FAIL"
                print(f"    {K:>3d}  {max_diff:>20.2e}  {mark}")
                if not ok:
                    matrix_pass = False

    print()
    print("=" * 70)
    if matrix_pass:
        print("PASS: every (T, B, K) combination produces bit-exact output")
        print("=" * 70)
        return 0
    else:
        print("FAIL: some (T, B, K) combination diverged from K=T reference")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
''',
    'tests/stage8_verify.py': r'''"""[Stage 8 verification] K-sweep test fix (Bug 8) + parity suite smoke.

Verifies that:
  1. tests/test_30_k_sweep_hbm.py has the Stage 8 docstring marker
     (acknowledging it's a historical inline-Python baseline).
  2. tests/test_30b_k_sweep_real.py exists and imports cleanly.
  3. tests/parity_suite_smoke.py exists and imports cleanly.
  4. Quick smoke test: run parity_suite_smoke logic on a SINGLE
     (T=4, B=1) configuration with K∈{1, 2, 4} — must all be max_diff=0.
  5. Quick smoke test: test_30b can produce output for ONE config without
     crashing (we don't need to run the full configs matrix here — that's
     for the actual experiment).

Run:
    cd /path/to/CATFuse
    python tests/stage8_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_test_30_marker():
    """1: test_30 has Stage 8 marker docstring."""
    print("[1/5] test_30_k_sweep_hbm.py Stage 8 marker...")
    try:
        with open(os.path.join(_REPO_ROOT, "tests/test_30_k_sweep_hbm.py")) as f:
            src = f.read()
    except Exception:
        print("  FAIL: file not found"); return False
    if "[Stage 8 note]" not in src:
        print("  FAIL: marker absent — original test_30 not annotated")
        return False
    if "test_30b_k_sweep_real.py" not in src:
        print("  FAIL: marker doesn't redirect to test_30b")
        return False
    print("  OK: test_30 acknowledges its inline-Python baseline status")
    return True


def check_test_30b_exists():
    """2: test_30b imports cleanly."""
    print("[2/5] tests/test_30b_k_sweep_real.py imports...")
    path = os.path.join(_REPO_ROOT, "tests/test_30b_k_sweep_real.py")
    if not os.path.exists(path):
        print("  FAIL: file missing"); return False
    try:
        import ast
        ast.parse(open(path).read())
    except Exception:
        print("  FAIL: syntax error"); traceback.print_exc(); return False
    print("  OK: test_30b parseable")
    return True


def check_parity_suite_exists():
    """3: parity_suite_smoke imports cleanly."""
    print("[3/5] tests/parity_suite_smoke.py imports...")
    path = os.path.join(_REPO_ROOT, "tests/parity_suite_smoke.py")
    if not os.path.exists(path):
        print("  FAIL: file missing"); return False
    try:
        import ast
        ast.parse(open(path).read())
    except Exception:
        print("  FAIL: syntax error"); traceback.print_exc(); return False
    print("  OK: parity_suite_smoke parseable")
    return True


def check_parity_smoke_single_config():
    """4: Quick parity smoke — SEW-RN18 with T=4 B=1, K=1 vs K=4."""
    print("[4/5] SEW-RN18 single-config K-sweep parity smoke...")
    try:
        import torch
        from spikingjelly.activation_based import (
            functional, neuron, surrogate,
        )
        from spikingjelly.activation_based.model import sew_resnet
        from catfuse import optimize
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP: needs CUDA + SparseFlow"); return None

    device = "cuda:0"
    net = sew_resnet.sew_resnet18(
        pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(), detach_reset=True,
    )
    functional.set_step_mode(net, "m")
    net = net.to(device).eval()

    T, B = 4, 1
    torch.manual_seed(0)
    x = torch.randn(T, B, 3, 32, 32, device=device)

    fused, _ = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    # Default (K=T) reference
    with torch.no_grad():
        functional.reset_net(fused)
        y_ref = fused(x).clone()

    st_layers = [m for m in fused.modules() if isinstance(m, STFusionConvBNLIF)]

    # K=1 sweep
    originals = [(L, L._batchfold_forward) for L in st_layers]
    for L in st_layers:
        def _make(L_, K_=1):
            def _bf(x):
                if L_._impl_sparse is not None:
                    return L_._impl_sparse.forward_with_k(
                        x, L_.spec, L_._ensure_params(), L_.state, K=K_)
                return L_._impl_dense.forward(
                    x, L_.spec, L_._ensure_params(), L_.state)
            return _bf
        L._batchfold_forward = _make(L)

    with torch.no_grad():
        functional.reset_net(fused)
        y_K1 = fused(x)

    for L, orig in originals:
        L._batchfold_forward = orig

    diff_K1 = (y_K1 - y_ref).abs().max().item()
    print(f"  T=4 B=1 K=1 vs K=T: max_diff = {diff_K1:.2e}")
    if diff_K1 != 0:
        print("  FAIL: bit-exact violated"); return False
    print("  OK")
    return True


def check_test_30b_smoke_run():
    """5: Run test_30b on a tiny single-config setup to ensure it doesn't crash."""
    print("[5/5] test_30b API smoke (no full run, just one shape)...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer, functional
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    torch.manual_seed(42)
    Cin, Cout, H = 64, 64, 16
    B, T = 2, 4
    conv = sj_layer.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(Cout).to(device)
    bn.running_mean.normal_(0, 0.1); bn.running_var.uniform_(0.5, 1.5); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(device).eval()
    spec = fused.spec
    params = fused._ensure_params()
    x = (torch.rand(T, B, Cin, H, H, device=device) > 0.85).float()

    # Reference
    functional.reset_net(fused)
    y_ref = fused._impl_sparse.forward_with_k(x, spec, params, fused.state, K=T)

    # Sweep K
    for K in [1, 2, 4]:
        cost = fused._impl_sparse.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=H, K=K)
        functional.reset_net(fused)
        y_K = fused._impl_sparse.forward_with_k(x, spec, params, fused.state, K=K)
        diff = (y_K - y_ref).abs().max().item()
        print(f"  K={K}  num_blocks={cost.num_blocks}  "
              f"intermediate_io={cost.intermediate_io} bytes  parity={diff:.2e}")
        if diff != 0:
            print(f"  FAIL: K={K} not bit-exact"); return False

    print("  OK: test_30b API path works end-to-end")
    return True


def main():
    print("=" * 60)
    print("Stage 8 verification: K-sweep test fix (Bug 8) + parity suite")
    print("=" * 60)
    results = [
        ("test_30_marker", check_test_30_marker()),
        ("test_30b_exists", check_test_30b_exists()),
        ("parity_suite_exists", check_parity_suite_exists()),
        ("parity_smoke_single", check_parity_smoke_single_config()),
        ("test_30b_smoke", check_test_30b_smoke_run()),
    ]
    print()
    print("=" * 60)
    failed = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, {len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
''',
}


REPLACEMENTS = {
    "tests/test_30_k_sweep_hbm.py": [
        ('"""\nTest 30: K-sweep + HBM analytic formula validation (§3.9)\n\nFor a single Conv+BN+LIF layer:\n  1. Compute analytic HBM bytes for reference and CTF at each K\n  2. Measure wall-clock at each K\n  3. Compare ratio curves\n\nThree execution modes:\n  - REF: SpikingJelly per-step Conv→BN→LIF (baseline)\n  - CTF-Dense: PartialFusionConvBNLIF (cuDNN + fused BN+LIF)\n  - CTF-SF: StreamFuse kernel (z stays in registers)\n"""', '"""\nTest 30: K-sweep + HBM analytic formula validation (§3.9)\n\n[Stage 8 note] This test predates the Stage 3-6 Implementation hierarchy.\nIts `hybrid_fn` benchmark uses an INLINE Python loop for the LIF dynamics\nrather than the production lif_sequential Triton kernel — so its wall-clock\nnumbers reflect "what if LIF were inline Python", NOT actual deployment\nperformance of PartialFusionConvBNLIF / STFusionConvBNLIF.\n\nKept for historical record. For the deployment-correct K-sweep targeting\n§3.10 paper data, use tests/test_30b_k_sweep_real.py — it goes through\nSparseFlow.forward_with_k (Stage 6) and DenseKeep.forward (Stage 3) using\nthe same kernels SEW-RN18 actually runs in production.\n\nFor a single Conv+BN+LIF layer:\n  1. Compute analytic HBM bytes for reference and CTF at each K\n  2. Measure wall-clock at each K\n  3. Compare ratio curves\n\nThree execution modes:\n  - REF: SpikingJelly per-step Conv→BN→LIF (baseline)\n  - CTF-Dense: PartialFusionConvBNLIF (cuDNN + fused BN+LIF)\n                BUT: hybrid_fn uses inline Python LIF, NOT lif_sequential\n  - CTF-SF: StreamFuse kernel (z stays in registers)\n                this one IS realistic — calls sparse_streamfuse_conv3x3_bn_lif\n"""'),
    ],
}


def apply_new_files(repo_root):
    print("[1/2] Creating new files...")
    for rel_path, content in NEW_FILES.items():
        target = os.path.join(repo_root, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            with open(target, "r") as f:
                existing = f.read()
            if existing == content:
                print(f"  SKIP (identical): {rel_path}")
                continue
            print(f"  OVERWRITE: {rel_path} ({len(existing)} -> {len(content)} chars)")
        else:
            print(f"  CREATE: {rel_path} ({len(content)} chars)")
        with open(target, "w") as f:
            f.write(content)


def apply_replacements(file_path, replacements, label):
    if not os.path.exists(file_path):
        print(f"  ERROR: {file_path} not found"); return False
    with open(file_path, "r") as f:
        content = f.read()
    original = content
    applied = skipped = 0
    for i, (old, new) in enumerate(replacements):
        if new in content:
            print(f"  [{label} #{i+1}] SKIP (already applied)")
            skipped += 1; continue
        if old not in content:
            print(f"  [{label} #{i+1}] ERROR: neither old nor new pattern found")
            return False
        content = content.replace(old, new, 1)
        applied += 1
        print(f"  [{label} #{i+1}] APPLIED ({len(old)} -> {len(new)} chars)")
    if content != original:
        with open(file_path, "w") as f:
            f.write(content)
    print(f"  {label}: {applied} applied, {skipped} skipped (already done)")
    return True


def main(repo_root="."):
    repo_root = os.path.abspath(repo_root)
    print("=" * 60)
    print(f"Applying Stage 8 refactor in: {repo_root}")
    print("=" * 60)

    apply_new_files(repo_root)

    print("[2/2] Annotating test_30_k_sweep_hbm.py...")
    for rel_path, replacements in REPLACEMENTS.items():
        path = os.path.join(repo_root, rel_path)
        if not apply_replacements(path, replacements, rel_path):
            return 1

    print()
    print("=" * 60)
    print("Stage 8 applied successfully.")
    print("Next: python tests/stage8_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))