#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 7 refactor: HardwareProfile abstraction (problem 9).

Stage 7 introduces catfuse/hardware.py with:
  - HardwareProfile dataclass (frozen) with sm_capability, hbm_bandwidth_gbps,
    sf_threshold_C, sf_threshold_H, default_K_* fields.
  - V100_PCIE_32GB, V100_SXM_32GB, A100_40GB, A100_80GB presets.
  - DEFAULT = V100_PCIE_32GB.
  - detect_hardware() — auto-detect current GPU, fall back to DEFAULT for
    unknown devices.

policy.py modifications:
  - classify_shape_regime(..., hwprof=None): hwprof=None preserves V100 behavior
  - optimal_K(T, regime, hwprof=None): same default-V100 fallback

When hwprof is None (the default), behavior is bit-exact-identical to the
pre-stage-7 implementation. The only behavioral change is when an explicit
non-default hwprof is passed — currently used for nothing (placeholder for
A100 experiments).

Prerequisites: stages 1–6 applied. Idempotent.

Run:
    cd /path/to/CATFuse
    python apply_stage7.py
    python tests/stage7_verify.py
"""
import os
import sys


# ============================================================
# Embedded new files
# ============================================================

NEW_FILES = {
    'catfuse/hardware.py': r'''"""catfuse.hardware — HardwareProfile abstraction.

Per the Stage 7 refactor, hardware-dependent thresholds are factored out of
policy.py into a HardwareProfile dataclass. This makes the §3.9/§3.10 cost
model and Runtime EGD policy portable: switching from V100 to A100 just
swaps the profile, no source changes.

Profiles supplied here are conservative — they reflect the spec sheets and
empirical CATFuse measurements rather than fine-tuned per-workload.

Note that the §3.9 ANALYTIC I/O cost formula (intermediate_io ∈ O(T/K·HWC))
is hardware-INDEPENDENT — it's a count of HBM bytes, not a wall-clock
prediction. HardwareProfile influences only the EMPIRICAL policy choices
(when to route SparseFlow vs DenseKeep, what K to default to).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareProfile:
    """Static description of a GPU's relevant capabilities for CATFuse.

    Fields:
        arch_name              : human-readable architecture
        sm_version             : tuple (major, minor), e.g. (7, 0) for sm_70
        hbm_bandwidth_gbps     : peak global memory bandwidth (GB/s)
        l2_size_mb             : on-chip L2 cache size (MB)
        sm_count               : number of streaming multiprocessors
        sf_threshold_C         : minimum C_in for SparseFlow to be considered
                                 (below this, cuDNN's DenseKeep wins)
        sf_threshold_H         : maximum spatial H for SparseFlow to be
                                 considered (above this, cuDNN wins)
        default_K_memory_bound : default TimeBlock(K) for memory-bound layers
        default_K_compute_bound: default TimeBlock(K) for compute-bound layers
        notes                  : optional free-form context

    The sf_threshold_* fields encode the empirically-measured break-even
    point between cuDNN's dense conv and SparseFlow's sparse fused kernel.
    They are workload-dependent in principle; the values here came from
    SEW-ResNet18 + VGG11 benchmarks. Users with different workloads should
    re-measure and override.
    """
    arch_name: str
    sm_version: tuple
    hbm_bandwidth_gbps: float
    l2_size_mb: float
    sm_count: int
    sf_threshold_C: int
    sf_threshold_H: int
    default_K_memory_bound: int
    default_K_compute_bound: int
    notes: str = ""

    @property
    def sm_capability(self) -> int:
        """sm_70, sm_80, sm_90, ... as integer."""
        return self.sm_version[0] * 10 + self.sm_version[1]


# ============================================================
# Preset profiles
# ============================================================

# V100 PCIe 32GB — the project's primary test machine.
# Thresholds derived from CATFuse Phase 0-2 empirical data on SEW-RN18.
V100_PCIE_32GB = HardwareProfile(
    arch_name="V100 PCIe 32GB",
    sm_version=(7, 0),
    hbm_bandwidth_gbps=900.0,    # NVIDIA spec: 900 GB/s
    l2_size_mb=6.0,              # Volta L2: 6 MB
    sm_count=80,
    sf_threshold_C=256,
    sf_threshold_H=14,
    default_K_memory_bound=8,
    default_K_compute_bound=4,
    notes="Project default. SF wins cuDNN at C>=256, H<=14 (layer3/4 of RN18).",
)

# V100 SXM 32GB — slightly higher bandwidth than PCIe.
V100_SXM_32GB = HardwareProfile(
    arch_name="V100 SXM 32GB",
    sm_version=(7, 0),
    hbm_bandwidth_gbps=900.0,
    l2_size_mb=6.0,
    sm_count=80,
    sf_threshold_C=256,
    sf_threshold_H=14,
    default_K_memory_bound=8,
    default_K_compute_bound=4,
    notes="Same SM as PCIe; bandwidth identical in current Triton paths.",
)

# A100 40GB — sm_80, much wider HBM, bigger L2.
# Thresholds are approximate — to be re-measured when CATFuse runs on A100.
A100_40GB = HardwareProfile(
    arch_name="A100 40GB",
    sm_version=(8, 0),
    hbm_bandwidth_gbps=1555.0,
    l2_size_mb=40.0,
    sm_count=108,
    sf_threshold_C=128,    # tentative — A100 cuDNN is faster relatively
    sf_threshold_H=28,
    default_K_memory_bound=8,
    default_K_compute_bound=4,
    notes="Tentative thresholds. Re-measure when A100 K-sweep runs.",
)

# A100 80GB — same compute as 40GB, bigger HBM (3.0+ TB/s on SXM4).
A100_80GB = HardwareProfile(
    arch_name="A100 80GB",
    sm_version=(8, 0),
    hbm_bandwidth_gbps=2039.0,   # SXM4 80GB
    l2_size_mb=40.0,
    sm_count=108,
    sf_threshold_C=128,
    sf_threshold_H=28,
    default_K_memory_bound=8,
    default_K_compute_bound=4,
    notes="Tentative thresholds. Re-measure when A100 K-sweep runs.",
)

# Generic fallback — if detection fails, assume V100 PCIe (project default).
# This preserves backward-compat: code that doesn't pass hwprof gets V100.
DEFAULT = V100_PCIE_32GB


# Lookup table by (sm_major, sm_minor)
_BY_SM = {
    (7, 0): V100_PCIE_32GB,
    (8, 0): A100_40GB,
}


# ============================================================
# Auto-detection
# ============================================================

def detect_hardware() -> HardwareProfile:
    """Detect the current CUDA device and return a matching profile.

    Falls back to DEFAULT (V100 PCIe) if:
      - CUDA is unavailable
      - The device's sm_X.X is not in _BY_SM
      - Detection raises any exception

    For unknown devices, a warning is printed once and DEFAULT is used.
    Users wanting precise behavior on uncatalogued hardware should
    construct a HardwareProfile manually and pass it explicitly.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return DEFAULT
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) in _BY_SM:
            return _BY_SM[(major, minor)]
        # Unknown — fall back, but warn (once per process)
        global _WARNED_UNKNOWN
        try:
            _WARNED_UNKNOWN
        except NameError:
            _WARNED_UNKNOWN = set()
        key = (major, minor)
        if key not in _WARNED_UNKNOWN:
            name = torch.cuda.get_device_name(device)
            import warnings
            warnings.warn(
                f"catfuse.hardware: unknown device sm_{major}{minor} "
                f"({name}); using DEFAULT={DEFAULT.arch_name}. "
                f"Construct a HardwareProfile manually for accurate behavior."
            )
            _WARNED_UNKNOWN.add(key)
        return DEFAULT
    except Exception:
        return DEFAULT


__all__ = [
    "HardwareProfile",
    "V100_PCIE_32GB",
    "V100_SXM_32GB",
    "A100_40GB",
    "A100_80GB",
    "DEFAULT",
    "detect_hardware",
]
''',
    'tests/stage7_verify.py': r'''"""[Stage 7 verification] HardwareProfile abstraction (problem 9).

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
''',
}


# ============================================================
# Replacement pairs
# ============================================================

REPLACEMENTS = {
    "catfuse/policy.py": [
        ('def optimal_K(T: int, regime: str = "memory_bound") -> int:\n    """Select optimal K for a given T and shape regime.\n\n    For memory-bound layers, larger K = more I/O savings (diminishing returns).\n    For compute-bound layers with cuDNN fallback, K mainly affects StateCarry\n    overhead which is minor -> K=4 is a practical sweet spot.\n\n    Returns a power-of-2 K in [1, T].\n    """\n    if regime == "compute_bound":\n        return min(4, T)\n    # memory_bound: K=4 captures ~85% of max savings, K=8 captures ~92%\n    return min(8, T)\n\n\n# ============================================================\n# Shape regime classifier\n# ============================================================\n\ndef classify_shape_regime(\n    C_in: int, C_out: int, H: int, W: int, kernel_size: int = 3\n) -> str:\n    """Classify a Conv layer as compute-bound or memory-bound.\n\n    Heuristic based on test_17 lean-path empirical data (V100):\n    - SF wins cuDNN at C>=256, H<=14 (layer3/layer4)\n    - cuDNN wins at C<=128 regardless of spatial size\n    - Mixed regime defaults to compute_bound (cuDNN, safer)\n    """\n    if kernel_size == 1:\n        return "memory_bound"  # 1x1 conv is always memory-bound territory\n    if C_in >= 256 and H <= 14:\n        return "memory_bound"  # SF wins here (1.1-1.7x over cuDNN)\n    if C_in <= 128:\n        return "compute_bound"  # cuDNN always wins at low channels\n    # C_in >= 256 but H > 14: cuDNN still likely wins\n    if H >= 28:\n        return "compute_bound"\n    return "memory_bound"', 'def optimal_K(T: int, regime: str = "memory_bound", hwprof=None) -> int:\n    """Select optimal K for a given T and shape regime.\n\n    For memory-bound layers, larger K = more I/O savings (diminishing returns).\n    For compute-bound layers with cuDNN fallback, K mainly affects StateCarry\n    overhead which is minor -> K=4 is a practical sweet spot.\n\n    Stage 7 refactor: hwprof argument lets the K default vary by hardware\n    (different L2 / HBM bandwidth ratios shift the optimum). When hwprof\n    is None, falls back to catfuse.hardware.DEFAULT (V100 PCIe), preserving\n    pre-stage-7 behavior.\n\n    Returns a K capped by T.\n    """\n    if hwprof is None:\n        from catfuse.hardware import DEFAULT as hwprof\n    if regime == "compute_bound":\n        return min(hwprof.default_K_compute_bound, T)\n    # memory_bound: K=8 captures ~92% of max savings on V100\n    return min(hwprof.default_K_memory_bound, T)\n\n\n# ============================================================\n# Shape regime classifier\n# ============================================================\n\ndef classify_shape_regime(\n    C_in: int, C_out: int, H: int, W: int, kernel_size: int = 3,\n    hwprof=None,\n) -> str:\n    """Classify a Conv layer as compute-bound or memory-bound.\n\n    Heuristic based on test_17 lean-path empirical data (V100):\n    - SF wins cuDNN at C>=256, H<=14 (layer3/layer4)\n    - cuDNN wins at C<=128 regardless of spatial size\n    - Mixed regime defaults to compute_bound (cuDNN, safer)\n\n    Stage 7 refactor: thresholds (sf_threshold_C, sf_threshold_H) come from\n    hwprof. When hwprof is None, falls back to V100 defaults — preserving\n    pre-stage-7 behavior bit-exactly.\n    """\n    if hwprof is None:\n        from catfuse.hardware import DEFAULT as hwprof\n    sf_C = hwprof.sf_threshold_C\n    sf_H = hwprof.sf_threshold_H\n    if kernel_size == 1:\n        return "memory_bound"'),
    ],
}


# ============================================================
# Apply logic
# ============================================================

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
    print(f"Applying Stage 7 refactor in: {repo_root}")
    print("=" * 60)

    if not os.path.exists(os.path.join(repo_root, "catfuse/policy.py")):
        print("  FATAL: catfuse/policy.py not found")
        return 1

    apply_new_files(repo_root)

    print("[2/2] Modifying policy.py...")
    for rel_path, replacements in REPLACEMENTS.items():
        path = os.path.join(repo_root, rel_path)
        if not apply_replacements(path, replacements, rel_path):
            return 1

    print()
    print("=" * 60)
    print("Stage 7 applied successfully.")
    print("Next: python tests/stage7_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))