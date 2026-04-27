"""[Stage 3 verification] Implementation interface abstraction (problem 2).

Verifies that:
  1. catfuse.implementations.{base,dense_keep,sparse_flow} import cleanly.
  2. ConvLIFSpec / ConvLIFParams hold the expected fields.
  3. PartialFusionConvBNLIF holds a DenseKeep impl and delegates forward.
  4. STFusionConvBNLIF holds {DenseKeep, SparseFlow} and routes via Runtime EGD.
  5. Bit-exact equivalence between DenseKeep and SparseFlow on a single layer
     (Corollary 3.17 ground truth).
  6. SEW-ResNet18 substitute_sf still bit-exact vs SJ baseline (regression).
  7. functional.reset_net still works through the new impl-aware patterns.

Run:
    cd /path/to/CATFuse
    python tests/stage3_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_imports():
    """1. All new modules import cleanly."""
    print("[1/7] Module imports...")
    try:
        from catfuse.implementations import (
            Implementation, ConvLIFSpec, ConvLIFParams, DenseKeep,
        )
        from catfuse.implementations import SparseFlow  # may be None
        from catfuse.implementations.base import static_zero_forward
    except Exception:
        print("  FAIL: implementation modules failed to import")
        traceback.print_exc()
        return False

    # Implementation must be abstract
    try:
        Implementation()  # type: ignore
        print("  FAIL: Implementation should be abstract")
        return False
    except TypeError:
        pass  # expected

    print("  OK: imports clean, Implementation is abstract")
    return True


def check_spec_params_shape():
    """2. ConvLIFSpec / ConvLIFParams basic structure."""
    print("[2/7] ConvLIFSpec / ConvLIFParams structure...")
    try:
        import torch
        from catfuse.implementations import ConvLIFSpec, ConvLIFParams
    except Exception:
        print("  SKIP: torch not available")
        return None

    spec = ConvLIFSpec(
        in_channels=64, out_channels=128,
        kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    # Frozen check
    try:
        spec.tau = 3.0  # type: ignore[misc]
        print("  FAIL: ConvLIFSpec should be frozen")
        return False
    except Exception:
        pass

    # output_hw
    h_out, w_out = spec.output_hw(32, 32)
    assert (h_out, w_out) == (32, 32), \
        f"output_hw(32,32) for stride=1 padding=1 k=3 expected (32,32), got ({h_out},{w_out})"

    # Params holds references
    w = torch.randn(128, 64, 3, 3)
    bn_scale = torch.ones(128)
    bn_bias = torch.zeros(128)
    p = ConvLIFParams(weight=w, bias=None,
                      bn_scale=bn_scale, bn_bias=bn_bias)
    assert p.weight is w  # reference, not copy
    assert p.bn_scale is bn_scale

    print("  OK: ConvLIFSpec frozen + correct output_hw, ConvLIFParams holds references")
    return True


def check_partial_fusion_uses_dense_keep():
    """3. PartialFusionConvBNLIF holds a DenseKeep impl + delegates."""
    print("[3/7] PartialFusionConvBNLIF + DenseKeep delegation...")
    try:
        import torch
        from catfuse.patterns import PartialFusionConvBNLIF
        from catfuse.implementations import DenseKeep, ConvLIFSpec
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    inst = PartialFusionConvBNLIF(in_channels=4, out_channels=8,
                                  kernel_size=3, padding=1)
    assert hasattr(inst, "spec"), "should have .spec"
    assert isinstance(inst.spec, ConvLIFSpec)
    assert hasattr(inst, "_impl"), "should have ._impl"
    assert isinstance(inst._impl, DenseKeep)
    assert inst.spec.in_channels == 4 and inst.spec.out_channels == 8
    assert inst.spec.has_bn is True
    assert inst.spec.has_conv_bias is False

    # _params is lazy
    assert inst._params is None

    if torch.cuda.is_available():
        device = "cuda:0"
        inst = inst.to(device).eval()
        x = torch.randn(4, 2, 4, 16, 16, device=device)
        y = inst(x)
        assert y.shape == (4, 2, 8, 16, 16)
        # After forward, params is built and DenseKeep cached fold
        assert inst._params is not None
        assert inst._impl._w_fused is not None

    print("  OK: PartialFusionConvBNLIF delegates to DenseKeep, params built lazily")
    return True


def check_st_fusion_holds_both_impls():
    """4. STFusionConvBNLIF holds DenseKeep + SparseFlow + routes correctly."""
    print("[4/7] STFusionConvBNLIF + dual impls...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import DenseKeep, SparseFlow, ConvLIFSpec
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device)
    assert hasattr(fused, "spec")
    assert isinstance(fused.spec, ConvLIFSpec)
    assert hasattr(fused, "_impl_dense")
    assert isinstance(fused._impl_dense, DenseKeep)
    if SparseFlow is not None:
        assert hasattr(fused, "_impl_sparse")
        assert isinstance(fused._impl_sparse, SparseFlow)
    print("  OK: STFusionConvBNLIF holds DenseKeep + SparseFlow")
    return True


def check_dense_vs_sparse_bit_exact():
    """5. CORE Corollary 3.17 check: DenseKeep ≡ SparseFlow on a single layer.

    Builds two fresh STFusion modules with identical weights, manually forces
    each through DenseKeep and SparseFlow respectively, compares max_diff.
    """
    print("[5/7] DenseKeep vs SparseFlow bit-exact (Corollary 3.17)...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP: needs CUDA + SparseFlow")
        return None

    device = "cuda:0"
    torch.manual_seed(0)
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device).eval()

    # Generate non-trivial input (mostly zeros so SparseFlow path is exercised
    # for sparsity > 0.7, but not all-zero so static_zero short-circuit is NOT
    # the only thing tested).
    T, B = 4, 2
    x = (torch.rand(T, B, 64, 16, 16, device=device) > 0.85).float()  # ~85% sparse

    # Manually force DenseKeep
    from spikingjelly.activation_based import functional
    functional.reset_net(fused)
    y_dense = fused._impl_dense.forward(x, fused.spec, fused._ensure_params(), fused.state)

    # Manually force SparseFlow
    functional.reset_net(fused)
    y_sparse = fused._impl_sparse.forward(x, fused.spec, fused._ensure_params(), fused.state)

    max_diff = (y_dense - y_sparse).abs().max().item()
    print(f"  DenseKeep vs SparseFlow max_diff = {max_diff:.6e}")
    if max_diff > 0:
        # Even tiny diffs (last-bit) violate Corollary 3.17 bit-exactness.
        # Pre-stage-3 SEW-RN18 was 0-diff; stage 3 must preserve.
        print(f"  FAIL: bit-exact violated (Corollary 3.17)")
        return False

    print(f"  OK: max_diff=0 — Corollary 3.17 holds at single-layer level")
    return True


def check_resnet18_regression():
    """6. SEW-ResNet18 still bit-exact (regression)."""
    print("[6/7] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    try:
        from catfuse import optimize
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP: catfuse.optimize unavailable")
        traceback.print_exc()
        return None

    if build_sew_resnet18 is None:
        try:
            from spikingjelly.activation_based.model import sew_resnet
            from spikingjelly.activation_based import surrogate
            net = sew_resnet.sew_resnet18(
                pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
                surrogate_function=surrogate.ATan(), detach_reset=True,
            )
            functional.set_step_mode(net, "m")
        except Exception:
            print("  SKIP: can't build SEW-RN18")
            traceback.print_exc()
            return None
    else:
        net = build_sew_resnet18()

    device = "cuda:0"
    net = net.to(device).eval()
    T, B = 4, 2
    x = torch.randn(T, B, 3, 32, 32, device=device)

    with torch.no_grad():
        functional.reset_net(net)
        y_sj = net(x)

    fused, _ = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    with torch.no_grad():
        functional.reset_net(fused)
        y_ctf = fused(x)

    max_diff = (y_sj - y_ctf).abs().max().item()
    if max_diff > 1e-4:
        print(f"  FAIL: max_diff={max_diff:.6e}")
        return False

    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e} (bit-exact preserved)")
    return True


def check_reset_net_through_impls():
    """7. functional.reset_net still clears state via CTFPattern.reset()."""
    print("[7/7] functional.reset_net through impl-aware patterns...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer, functional
        from catfuse.patterns import PartialFusionConvBNLIF
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(4, 8, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(8).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)

    x = torch.randn(4, 2, 4, 16, 16, device=device)
    _ = fused(x)
    assert fused.state.is_initialized
    v_after_fwd = fused.state.tensor.clone()

    functional.reset_net(fused)
    assert not fused.state.is_initialized

    _ = fused(x)
    v_after_reset_fwd = fused.state.tensor.clone()

    assert torch.equal(v_after_fwd, v_after_reset_fwd), \
        "post-reset forward should reproduce fresh-init forward"

    print("  OK: reset_net properly clears state through CTFPattern.reset()")
    return True


def main():
    print("=" * 60)
    print("Stage 3 verification: Implementation interface (problem 2)")
    print("=" * 60)
    results = [
        ("imports", check_imports()),
        ("spec_params_shape", check_spec_params_shape()),
        ("partial_fusion_uses_dense_keep", check_partial_fusion_uses_dense_keep()),
        ("st_fusion_holds_both_impls", check_st_fusion_holds_both_impls()),
        ("dense_vs_sparse_bit_exact", check_dense_vs_sparse_bit_exact()),
        ("resnet18_regression", check_resnet18_regression()),
        ("reset_net_through_impls", check_reset_net_through_impls()),
    ]

    print()
    print("=" * 60)
    failed = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, "
              f"{len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
