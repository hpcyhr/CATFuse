"""catfuse.hardware — HardwareProfile abstraction.

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
