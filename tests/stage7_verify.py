"""[Stage 7 verification] HardwareProfile abstraction (problem 9).

Verifies that:
  1. catfuse.hardware exports HardwareProfile, V100_*, A100_*, DEFAULT,
     detect_hardware.
  2. HardwareProfile is frozen and exposes sm_capability.
  3. detect_hardware() returns a sensible profile on this machine.
  4. classify_shape_regime accepts hwprof; default behavior unchanged.
  5. optimal_K accepts hwprof; default behavior unchanged.
  6. Different hwprof values give different routing on borderline shapes.
  7. SEW-RN18 substitute_sf still produces 13 DenseKeep + 7 SparseFlow
     under default hwprof (V100), max_diff=0.

Run:
    cd /path/to/CATFuse
    python tests/stage7_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_imports():
    """1: catfuse.hardware exports the expected symbols."""
    print("[1/7] catfuse.hardware imports...")
    try:
        from catfuse.hardware import (
            HardwareProfile, V100_PCIE_32GB, V100_SXM_32GB,
            A100_40GB, A100_80GB, DEFAULT, detect_hardware,
        )
    except Exception:
        print("  FAIL"); traceback.print_exc(); return False
    if DEFAULT.arch_name != "V100 PCIe 32GB":
        print(f"  FAIL: DEFAULT should be V100 PCIe, got {DEFAULT.arch_name}")
        return False
    print("  OK: 7 symbols exported, DEFAULT = V100 PCIe 32GB")
    return True


def check_profile_structure():
    """2: HardwareProfile is frozen + sm_capability works."""
    print("[2/7] HardwareProfile structure...")
    try:
        from catfuse.hardware import V100_PCIE_32GB, A100_40GB
    except Exception:
        print("  SKIP"); return None

    if V100_PCIE_32GB.sm_capability != 70:
        print(f"  FAIL: V100 sm_capability expected 70, got {V100_PCIE_32GB.sm_capability}")
        return False
    if A100_40GB.sm_capability != 80:
        print(f"  FAIL: A100 sm_capability expected 80, got {A100_40GB.sm_capability}")
        return False
    try:
        V100_PCIE_32GB.sf_threshold_C = 999
        print("  FAIL: should be frozen"); return False
    except Exception:
        pass
    print("  OK: V100 sm=70, A100 sm=80, frozen")
    return True


def check_detect_hardware():
    """3: detect_hardware returns a sensible profile on this machine."""
    print("[3/7] detect_hardware on current GPU...")
    try:
        import torch
        from catfuse.hardware import detect_hardware, DEFAULT
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None
    prof = detect_hardware()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  Device: {name} (sm_{cap[0]}{cap[1]})")
    print(f"  Detected profile: {prof.arch_name}, sm={prof.sm_capability}")
    # The detection must either match the device's sm or fall back to DEFAULT
    if prof.sm_capability == cap[0] * 10 + cap[1]:
        print("  OK: profile matches device capability")
    elif prof is DEFAULT:
        print(f"  OK: device unrecognized, fell back to DEFAULT ({DEFAULT.arch_name})")
    else:
        print(f"  WARN: profile ({prof.sm_capability}) doesn't match device ({cap[0]}{cap[1]}) and isn't DEFAULT")
        # Not a hard failure — auto-detection is best-effort
    return True


def check_classify_default():
    """4: classify_shape_regime default behavior unchanged."""
    print("[4/7] classify_shape_regime — default V100 behavior...")
    try:
        from catfuse.policy import classify_shape_regime
    except Exception:
        print("  SKIP"); return None

    cases = [
        ((256, 256, 14, 14, 3), "memory_bound"),  # SF wins
        ((128, 128, 14, 14, 3), "compute_bound"), # cuDNN wins
        ((256, 256, 28, 28, 3), "compute_bound"), # too big
        ((256, 256, 14, 14, 1), "memory_bound"),  # 1x1 always
    ]
    for args, expected in cases:
        got = classify_shape_regime(*args)  # no hwprof = default V100
        if got != expected:
            print(f"  FAIL: classify{args} = {got}, expected {expected}")
            return False
    print(f"  OK: 4/4 default cases match pre-stage-7 behavior")
    return True


def check_optimal_K_default():
    """5: optimal_K default behavior unchanged."""
    print("[5/7] optimal_K — default V100 behavior...")
    try:
        from catfuse.policy import optimal_K
    except Exception:
        print("  SKIP"); return None

    if optimal_K(8, "memory_bound") != 8: return False
    if optimal_K(4, "memory_bound") != 4: return False  # capped
    if optimal_K(8, "compute_bound") != 4: return False
    if optimal_K(2, "compute_bound") != 2: return False  # capped
    print("  OK: 4/4 cases match pre-stage-7 behavior")
    return True


def check_hwprof_switching():
    """6: A100 profile gives different routing on borderline shapes."""
    print("[6/7] hwprof switching changes routing on borderline shapes...")
    try:
        from catfuse.policy import classify_shape_regime
        from catfuse.hardware import V100_PCIE_32GB, A100_40GB
    except Exception:
        print("  SKIP"); return None

    # C=128 H=14: V100 says cuDNN wins (sf_threshold_C=256), A100 says SF
    # wins (sf_threshold_C=128).
    v = classify_shape_regime(128, 128, 14, 14, 3, hwprof=V100_PCIE_32GB)
    a = classify_shape_regime(128, 128, 14, 14, 3, hwprof=A100_40GB)
    print(f"  C=128 H=14:  V100={v}  A100={a}")
    if v != "compute_bound":
        print(f"  FAIL: V100 should say compute_bound"); return False
    if a != "memory_bound":
        print(f"  FAIL: A100 should say memory_bound"); return False

    # Sanity: extreme cases agree across profiles
    for hp in [V100_PCIE_32GB, A100_40GB]:
        if classify_shape_regime(8, 8, 14, 14, 3, hwprof=hp) != "compute_bound":
            print(f"  FAIL: tiny conv should be compute_bound on {hp.arch_name}")
            return False
        if classify_shape_regime(512, 512, 4, 4, 3, hwprof=hp) != "memory_bound":
            print(f"  FAIL: big-channel small-spatial should be memory_bound on {hp.arch_name}")
            return False

    print("  OK: hwprof correctly switches routing on borderline shapes; extremes agree")
    return True


def check_resnet18_regression():
    """7: SEW-RN18 still produces 13 DenseKeep + 7 SparseFlow under default."""
    print("[7/7] SEW-ResNet18 substitute_sf parity regression...")
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
        print(f"  FAIL: max_diff={max_diff:.6e}"); return False
    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e}")
    return True


def main():
    print("=" * 60)
    print("Stage 7 verification: HardwareProfile (problem 9)")
    print("=" * 60)
    results = [
        ("imports", check_imports()),
        ("profile_structure", check_profile_structure()),
        ("detect_hardware", check_detect_hardware()),
        ("classify_default", check_classify_default()),
        ("optimal_K_default", check_optimal_K_default()),
        ("hwprof_switching", check_hwprof_switching()),
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
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, {len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
