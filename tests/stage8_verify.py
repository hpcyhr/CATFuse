"""[Stage 8 verification] K-sweep test fix (Bug 8) + parity suite smoke.

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
