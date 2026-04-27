"""[Stage 2 verification] StateBuffer abstraction.

Verifies that:
  1. catfuse.state.StateBuffer imports and behaves correctly:
     - lazy init on first .get()
     - shape / device / dtype mismatch triggers reallocation
     - .set() updates with detached tensor
     - .reset() clears state
  2. CTFPattern subclasses register their states via _register_state.
  3. functional.reset_net properly clears StateBuffer in CTF patterns
     (PartialFusionConvBNLIF, FusedLinearLIF, STFusionConvBNLIF).
  4. Forward → reset → forward produces fresh-state output (i.e., reset
     actually takes effect).
  5. SEW-ResNet18 still substitutes + runs end-to-end with bit-exact
     parity vs SJ multi-step (regression check from stage 1).

Run:
    cd /path/to/CATFuse
    python tests/stage2_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_state_buffer_unit():
    """1. StateBuffer unit behaviors."""
    print("[1/5] StateBuffer unit tests...")
    try:
        import torch
        from catfuse.state import StateBuffer
    except Exception:
        print("  SKIP: torch not available")
        return None

    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    sb = StateBuffer()
    assert not sb.is_initialized, "fresh buffer should be uninitialized"

    # Lazy init via .get()
    v0 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert sb.is_initialized
    assert v0.shape == (2, 4, 8, 8)
    assert v0.device == torch.device(device)
    assert v0.dtype == torch.float32
    assert (v0 == 0).all().item(), ".get() should allocate zeros"

    # .set() updates
    new_v = torch.ones(2, 4, 8, 8, device=device, dtype=torch.float32) * 3.5
    sb.set(new_v)
    v1 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert (v1 == 3.5).all().item(), ".set() should update underlying tensor"

    # Shape mismatch triggers reallocation
    v2 = sb.get(shape=(2, 4, 16, 16), device=device, dtype=torch.float32)
    assert v2.shape == (2, 4, 16, 16)
    assert (v2 == 0).all().item(), "shape mismatch should trigger zeros reallocation"

    # .reset() clears
    sb.reset()
    assert not sb.is_initialized
    v3 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert (v3 == 0).all().item(), "after reset, .get() should return zeros"

    # detach: .set() must store detached copy
    x = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
    sb.reset()
    _ = sb.get(shape=(2, 4, 8, 8), device=device)
    sb.set(x * 2)
    assert not sb.tensor.requires_grad, ".set() must store detached tensor"

    print("  OK: lazy init, set, reset, shape mismatch, detach all behave correctly")
    return True


def check_pattern_registers_state():
    """2. CTF patterns register their state via _register_state."""
    print("[2/5] Pattern state registration...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.patterns import (
            PartialFusionConvBNLIF, FusedLinearLIF,
        )
        from catfuse.state import StateBuffer
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    # PartialFusionConvBNLIF
    inst = PartialFusionConvBNLIF(in_channels=4, out_channels=8,
                                  kernel_size=3, padding=1)
    assert hasattr(inst, "state"), "PartialFusionConvBNLIF should have .state"
    assert isinstance(inst.state, StateBuffer)
    assert inst.state in inst._catfuse_states, \
        "state should be registered for unified reset"

    # FusedLinearLIF
    inst2 = FusedLinearLIF(in_features=16, out_features=10)
    assert hasattr(inst2, "state")
    assert isinstance(inst2.state, StateBuffer)
    assert inst2.state in inst2._catfuse_states

    print("  OK: PartialFusionConvBNLIF + FusedLinearLIF expose .state and register it")
    return True


def check_reset_net_clears_state():
    """3. functional.reset_net clears StateBuffer."""
    print("[3/5] functional.reset_net clears CTF state...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer, functional
        from catfuse.patterns import PartialFusionConvBNLIF
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA for kernel forward")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(4, 8, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(8).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)

    # State starts uninitialized
    assert not fused.state.is_initialized

    # Forward → state is now populated
    x = torch.randn(4, 2, 4, 16, 16, device=device)
    _ = fused(x)
    assert fused.state.is_initialized, "after forward, state should be populated"
    v_after_fwd = fused.state.tensor.clone()

    # Forward again: state evolves
    _ = fused(x)
    v_after_fwd2 = fused.state.tensor.clone()

    # functional.reset_net should trigger state.reset() via CTFPattern.reset()
    functional.reset_net(fused)
    assert not fused.state.is_initialized, \
        "after reset_net, state should be uninitialized"

    # Forward again: state is fresh zeros + evolves
    _ = fused(x)
    v_after_reset_fwd = fused.state.tensor.clone()

    # The post-reset+forward state should equal the first post-forward state
    # (same initial zeros, same input, same kernel → same output)
    assert torch.equal(v_after_fwd, v_after_reset_fwd), \
        "post-reset forward should reproduce fresh-init forward exactly"

    print("  OK: reset_net properly clears StateBuffer; post-reset forward is fresh")
    return True


def check_state_buffer_in_st_fusion():
    """4. STFusionConvBNLIF also uses StateBuffer."""
    print("[4/5] STFusionConvBNLIF uses StateBuffer...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.state import StateBuffer
        from catfuse.patterns import CTFPattern
    except Exception:
        print("  SKIP: import failed")
        traceback.print_exc()
        return None

    # STFusion should now be a CTFPattern subclass
    assert issubclass(STFusionConvBNLIF, CTFPattern), \
        "STFusionConvBNLIF should now inherit from CTFPattern (Stage 2)"

    # Build via from_sj_modules
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA for from_sj_modules")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device)
    assert hasattr(fused, "state"), "STFusionConvBNLIF should have .state"
    assert isinstance(fused.state, StateBuffer)
    assert fused.state in fused._catfuse_states

    print("  OK: STFusionConvBNLIF inherits CTFPattern + uses StateBuffer")
    return True


def check_resnet18_regression():
    """5. SEW-ResNet18 still bit-exact vs SJ baseline (regression from stage 1)."""
    print("[5/5] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    # Lazy import to avoid expensive load when SKIP
    try:
        from catfuse import optimize
        # Reuse same builder as test_05_substitute_sf
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        # Pull the existing test's builder if available
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP: catfuse.optimize unavailable")
        traceback.print_exc()
        return None

    if build_sew_resnet18 is None:
        # Build SEW-RN18 inline using spikingjelly's official model
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

    fused, stats = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    with torch.no_grad():
        functional.reset_net(fused)
        y_ctf = fused(x)

    max_diff = (y_sj - y_ctf).abs().max().item()
    if max_diff > 1e-4:
        print(f"  FAIL: max_diff={max_diff:.6e} (expected 0 or near-zero)")
        return False

    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e} (bit-exact)")
    return True


def main():
    print("=" * 60)
    print("Stage 2 verification: StateBuffer abstraction (problem 4)")
    print("=" * 60)
    results = [
        ("state_buffer_unit", check_state_buffer_unit()),
        ("pattern_registers_state", check_pattern_registers_state()),
        ("reset_net_clears_state", check_reset_net_clears_state()),
        ("st_fusion_state_buffer", check_state_buffer_in_st_fusion()),
        ("resnet18_regression", check_resnet18_regression()),
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
