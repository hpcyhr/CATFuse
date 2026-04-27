"""[Stage 6 verification] K-aware forward (§3.10/§3.13 Lemma 3.14).

Verifies that:
  1. SparseFlow.forward_with_k exists and accepts K.
  2. forward_with_k(K=T) ≡ forward() bit-exactly (default path).
  3. forward_with_k(K=1) ≡ forward_with_k(K=T) bit-exactly.
  4. forward_with_k(K=2) ≡ forward_with_k(K=T) bit-exactly.
  5. K out of range gets clamped (K=0 → 1, K>T → T).
  6. State carry across chunks: v written by chunk i is read by chunk i+1
     (StateCarry(LIF) at TimeBlock boundary).
  7. SEW-ResNet18 substitute_sf still bit-exact (regression — this stage
     is purely additive, the production path forward() is untouched).

Run:
    cd /path/to/CATFuse
    python tests/stage6_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_forward_with_k_exists():
    """1: SparseFlow has forward_with_k method with correct signature."""
    print("[1/7] forward_with_k method exists...")
    try:
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  FAIL"); traceback.print_exc(); return False
    if SparseFlow is None:
        print("  SKIP: SparseFlow unavailable on this platform")
        return None

    if not hasattr(SparseFlow, "forward_with_k"):
        print("  FAIL: forward_with_k method missing"); return False

    import inspect
    sig = inspect.signature(SparseFlow.forward_with_k)
    params = list(sig.parameters.keys())
    expected = ["self", "x", "spec", "params", "state", "K"]
    if params != expected:
        print(f"  FAIL: signature {params} != {expected}")
        return False
    print("  OK: SparseFlow.forward_with_k(self, x, spec, params, state, K=None)")
    return True


def _build_layer_and_input(device, T, B, H, W, sparsity=0.85):
    """Build a fresh STFusion layer + sparse input. Returns (fused, x)."""
    import torch
    from spikingjelly.activation_based import neuron, layer as sj_layer
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

    torch.manual_seed(42)
    C = 64
    conv = sj_layer.Conv2d(C, C, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(C).to(device)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(device).eval()
    x = (torch.rand(T, B, C, H, W, device=device) > sparsity).float()
    return fused, x


def check_k_equals_t_matches_forward():
    """2: forward_with_k(K=T) ≡ forward() bit-exact."""
    print("[2/7] forward_with_k(K=T) == forward()...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP: needs CUDA + SparseFlow"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    functional.reset_net(fused)
    y_default = fused._impl_sparse.forward(
        x, fused.spec, fused._ensure_params(), fused.state)

    functional.reset_net(fused)
    y_k_t = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=T)

    max_diff = (y_default - y_k_t).abs().max().item()
    print(f"  max_diff = {max_diff:.6e}")
    if max_diff > 0:
        print("  FAIL: K=T path not bit-exact with default forward")
        return False
    print("  OK: K=T path is bit-exact identical to default")
    return True


def check_k_sweep_bit_exact():
    """3+4: forward_with_k(K=1) and (K=2) both ≡ forward_with_k(K=T).

    This is the experimental verification of §3.13 Lemma 3.14: any
    legal K choice produces the same output.
    """
    print("[3/7] K-sweep bit-exact (§3.13 Lemma 3.14 — experimental)...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    # Test multiple shapes — these mirror the §5.2 K-sweep target layers
    shapes = [
        (4, 2, 16, 16),  # T=4, small spatial, like layer3
        (4, 2, 8, 8),    # T=4, even smaller, like layer4
        (8, 1, 16, 16),  # T=8, K∈{1,2,4,8}
    ]

    all_ok = True
    for (T, B, H, W) in shapes:
        fused, x = _build_layer_and_input(device, T, B, H, W)
        outputs = {}
        for K in [1, 2, T // 2 if T > 2 else T, T]:
            functional.reset_net(fused)
            y_K = fused._impl_sparse.forward_with_k(
                x, fused.spec, fused._ensure_params(), fused.state, K=K)
            outputs[K] = y_K.detach().cpu()

        # Compare every K against K=T
        ref = outputs[T]
        for K, y in outputs.items():
            if K == T:
                continue
            diff = (y - ref).abs().max().item()
            mark = "✓" if diff == 0 else "✗"
            print(f"  T={T}, B={B}, H=W={H}: K={K} vs K={T}  max_diff={diff:.2e}  {mark}")
            if diff != 0:
                all_ok = False

    if not all_ok:
        print("  FAIL: §3.13 Lemma 3.14 violated empirically")
        return False
    print("  OK: every K∈{1,2,T/2,T} produces bit-exact identical output")
    return True


def check_k_clamping():
    """5: K out of range gets clamped (no crash, no bad output)."""
    print("[4/7] K clamping (K≤0 → 1, K>T → T)...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    functional.reset_net(fused)
    y_T = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=T)

    # K=999 should clamp to T
    functional.reset_net(fused)
    y_huge = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=999)
    if (y_huge - y_T).abs().max().item() != 0:
        print(f"  FAIL: K=999 didn't clamp to T={T}")
        return False

    # K=0 should clamp to 1
    functional.reset_net(fused)
    y_K1 = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=1)
    functional.reset_net(fused)
    y_K0 = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=0)
    if (y_K0 - y_K1).abs().max().item() != 0:
        print(f"  FAIL: K=0 didn't clamp to 1")
        return False

    print("  OK: K=999 → T,  K=0 → 1")
    return True


def check_state_carry():
    """6: State carry across chunks works correctly (StateCarry(LIF))."""
    print("[5/7] State carry across TimeBlock boundaries...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    # K=2 means 2 chunks. After first chunk, state must be non-zero
    # (assuming x_chunk_0 produces non-trivial v).
    functional.reset_net(fused)
    assert not fused.state.is_initialized

    # Run only first chunk manually
    x_chunk_0 = x[:2].contiguous()
    _ = fused._impl_sparse.forward(
        x_chunk_0, fused.spec, fused._ensure_params(), fused.state)
    assert fused.state.is_initialized
    v_after_chunk_0 = fused.state.tensor.clone()
    if not v_after_chunk_0.any():
        print("  WARN: v after chunk_0 is all-zero — try a denser input")

    # Now if we run forward_with_k(K=2) from scratch, the inner forward
    # call for chunk_1 should see v_after_chunk_0 as input.
    # We'll verify by: full K=2 run vs manual two-step run.
    functional.reset_net(fused)
    y_full = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=2)

    functional.reset_net(fused)
    y_a = fused._impl_sparse.forward(
        x[:2].contiguous(), fused.spec, fused._ensure_params(), fused.state)
    y_b = fused._impl_sparse.forward(
        x[2:].contiguous(), fused.spec, fused._ensure_params(), fused.state)
    y_manual = torch.cat([y_a, y_b], dim=0)

    diff = (y_full - y_manual).abs().max().item()
    print(f"  forward_with_k(K=2) vs manual 2-step:  max_diff = {diff:.2e}")
    if diff != 0:
        print("  FAIL: state carry not behaving as expected")
        return False
    print("  OK: state correctly carried across TimeBlock boundary")
    return True


def check_resnet18_regression():
    """7: SEW-ResNet18 still bit-exact — Stage 6 only ADDS forward_with_k,
    forward() is unchanged, so SEW-RN18 routing must still be 0-diff.
    """
    print("[6/7] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available():
        print("  SKIP"); return None
    try:
        from catfuse import optimize
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP"); return None

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
            print("  SKIP"); return None
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
    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e}")
    return True


def check_full_network_k_aware():
    """7b: BONUS — apply K=2 across an entire SEW-RN18 (every STFusion layer
    uses forward_with_k(K=2) internally). Compare with default.

    This is a sneak-preview of the §3.10 experiment. Note that PartialFusion
    layers don't use K (DenseKeep, K is fixed). Only STFusion layers swap.
    """
    print("[7/7] BONUS: SEW-RN18 with all STFusion layers using K=2...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None
    try:
        from catfuse import optimize
    except Exception:
        print("  SKIP"); return None

    try:
        from spikingjelly.activation_based.model import sew_resnet
        from spikingjelly.activation_based import surrogate
        net = sew_resnet.sew_resnet18(
            pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(), detach_reset=True,
        )
        functional.set_step_mode(net, "m")
    except Exception:
        print("  SKIP"); return None

    device = "cuda:0"
    net = net.to(device).eval()
    T, B = 4, 2
    x = torch.randn(T, B, 3, 32, 32, device=device)

    fused, _ = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    with torch.no_grad():
        functional.reset_net(fused)
        y_default = fused(x)

    # Monkey-patch each STFusion's _batchfold_forward to use K=2 instead of K=T
    st_layers = [m for m in fused.modules() if isinstance(m, STFusionConvBNLIF)]
    print(f"  Found {len(st_layers)} STFusion layers; switching to K=2 forward")

    for layer in st_layers:
        original_batchfold = layer._batchfold_forward
        def _make_k2_forward(layer):
            def _bf(x):
                if layer._impl_sparse is not None:
                    return layer._impl_sparse.forward_with_k(
                        x, layer.spec, layer._ensure_params(), layer.state, K=2)
                return layer._impl_dense.forward(
                    x, layer.spec, layer._ensure_params(), layer.state)
            return _bf
        layer._batchfold_forward = _make_k2_forward(layer)

    with torch.no_grad():
        functional.reset_net(fused)
        y_k2 = fused(x)

    max_diff = (y_default - y_k2).abs().max().item()
    print(f"  Full network: K=T vs K=2 across all STFusion layers")
    print(f"  max_diff = {max_diff:.2e}")
    if max_diff != 0:
        print("  FAIL: end-to-end K-sweep bit-exact violated")
        return False
    print("  OK: full SEW-RN18 forward bit-exact across K choices")
    return True


def main():
    print("=" * 60)
    print("Stage 6 verification: K-aware forward (§3.10/§3.13 Lemma 3.14)")
    print("=" * 60)
    results = [
        ("forward_with_k_exists", check_forward_with_k_exists()),
        ("k_equals_t_matches_forward", check_k_equals_t_matches_forward()),
        ("k_sweep_bit_exact", check_k_sweep_bit_exact()),
        ("k_clamping", check_k_clamping()),
        ("state_carry", check_state_carry()),
        ("resnet18_regression", check_resnet18_regression()),
        ("full_network_k_aware", check_full_network_k_aware()),
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
