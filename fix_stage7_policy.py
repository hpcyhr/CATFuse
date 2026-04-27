#!/usr/bin/env python3
"""Fix Stage 7 policy.py corruption.

Server's policy.py was corrupted (likely by a VS Code save that triggered
"Trim Trailing Whitespace" or similar on apply_stage7.py before that script
ran, which truncated the embedded NEW string used as the replacement).

Symptom: classify_shape_regime returns None for almost every input, because
its function body is truncated to just `if kernel_size == 1: return ...`.
SEW-RN18 routing collapses to 20 DenseKeep + 0 SparseFlow.

This script rewrites the affected region of policy.py with the correct
post-Stage-7 version. It is idempotent: if the file is already correct,
it does nothing.

Run on server:
    cd /data/yhr/CATFuse
    python fix_stage7_policy.py
    python tests/stage7_verify.py    # should now PASS 7/7
    python tests/stage6_verify.py    # SEW-RN18 should be back to 13/7
"""
import os, sys

POLICY_PATH = "catfuse/policy.py"


# Marker — first 3 lines of the io_ratio_full_fusion function (unchanged from
# pre-stage-7), used to anchor the rewrite.
ANCHOR_BEFORE = '''def io_ratio_full_fusion(K: int) -> float:
    """I/O ratio for full fusion (Triton Conv+BN+LIF or SparseFlow+BN+LIF).

    Path B: SparseFlow or full Triton dense.
    z stays on-chip (StreamFuse).
    Formula: (1 + 2/K) / 5
    """
    return (1.0 + 2.0 / K) / 5.0'''

# Marker — start of the Default policy table section (unchanged).
ANCHOR_AFTER = '''# ============================================================
# Default policy table
# ============================================================'''


# The CORRECT stage-7 region — between the two anchors. This is what the
# server's policy.py SHOULD have between io_ratio_full_fusion and the
# default policy table section.
CORRECT_REGION = '''def io_ratio_full_fusion(K: int) -> float:
    """I/O ratio for full fusion (Triton Conv+BN+LIF or SparseFlow+BN+LIF).

    Path B: SparseFlow or full Triton dense.
    z stays on-chip (StreamFuse).
    Formula: (1 + 2/K) / 5
    """
    return (1.0 + 2.0 / K) / 5.0


def optimal_K(T: int, regime: str = "memory_bound", hwprof=None) -> int:
    """Select optimal K for a given T and shape regime.

    For memory-bound layers, larger K = more I/O savings (diminishing returns).
    For compute-bound layers with cuDNN fallback, K mainly affects StateCarry
    overhead which is minor -> K=4 is a practical sweet spot.

    Stage 7 refactor: hwprof argument lets the K default vary by hardware
    (different L2 / HBM bandwidth ratios shift the optimum). When hwprof
    is None, falls back to catfuse.hardware.DEFAULT (V100 PCIe), preserving
    pre-stage-7 behavior.

    Returns a K capped by T.
    """
    if hwprof is None:
        from catfuse.hardware import DEFAULT as hwprof
    if regime == "compute_bound":
        return min(hwprof.default_K_compute_bound, T)
    # memory_bound: K=8 captures ~92% of max savings on V100
    return min(hwprof.default_K_memory_bound, T)


# ============================================================
# Shape regime classifier
# ============================================================

def classify_shape_regime(
    C_in: int, C_out: int, H: int, W: int, kernel_size: int = 3,
    hwprof=None,
) -> str:
    """Classify a Conv layer as compute-bound or memory-bound.

    Heuristic based on test_17 lean-path empirical data (V100):
    - SF wins cuDNN at C>=256, H<=14 (layer3/layer4)
    - cuDNN wins at C<=128 regardless of spatial size
    - Mixed regime defaults to compute_bound (cuDNN, safer)

    Stage 7 refactor: thresholds (sf_threshold_C, sf_threshold_H) come from
    hwprof. When hwprof is None, falls back to V100 defaults — preserving
    pre-stage-7 behavior bit-exactly.
    """
    if hwprof is None:
        from catfuse.hardware import DEFAULT as hwprof
    sf_C = hwprof.sf_threshold_C
    sf_H = hwprof.sf_threshold_H
    if kernel_size == 1:
        return "memory_bound"  # 1x1 conv is always memory-bound territory
    if C_in >= sf_C and H <= sf_H:
        return "memory_bound"  # SF wins here (1.1-1.7x over cuDNN on V100)
    if C_in <= sf_C // 2:
        return "compute_bound"  # cuDNN always wins at low channels
    # C_in >= sf_C but H > sf_H: cuDNN still likely wins
    if H >= sf_H * 2:
        return "compute_bound"
    return "memory_bound"


'''


def main():
    if not os.path.exists(POLICY_PATH):
        print(f"FATAL: {POLICY_PATH} not found — run from /data/yhr/CATFuse")
        return 1
    content = open(POLICY_PATH).read()

    # Find anchors
    if ANCHOR_BEFORE not in content:
        print("FATAL: cannot find io_ratio_full_fusion anchor — file structure unexpected")
        return 1
    if ANCHOR_AFTER not in content:
        print("FATAL: cannot find Default policy table anchor")
        return 1

    si = content.index(ANCHOR_BEFORE)
    ei = content.index(ANCHOR_AFTER)
    current_region = content[si:ei]

    # Idempotency: if region already matches CORRECT_REGION, skip
    if current_region == CORRECT_REGION:
        print("OK: policy.py region already correct, no fix needed")
        # Quick sanity check: end-to-end classify call should work
        try:
            from catfuse.policy import classify_shape_regime
            result = classify_shape_regime(256, 256, 14, 14, 3)
            if result == "memory_bound":
                print(f"OK: classify(256,256,14,14,3) = {result!r}")
                return 0
            else:
                print(f"WARN: file looks correct but classify returned {result!r}")
        except Exception as e:
            print(f"WARN: file looks correct but import failed: {e}")
        return 0

    print(f"Detected corruption: current region is {len(current_region)} chars, "
          f"correct is {len(CORRECT_REGION)} chars")
    print(f"Replacing region between io_ratio_full_fusion and "
          f"Default policy table...")

    new_content = content[:si] + CORRECT_REGION + content[ei:]

    # Backup before overwrite
    backup_path = POLICY_PATH + ".bak_pre_fix"
    with open(backup_path, "w") as f:
        f.write(content)
    print(f"Backup saved to {backup_path}")

    with open(POLICY_PATH, "w") as f:
        f.write(new_content)

    # Verify by re-reading
    new_size = len(open(POLICY_PATH).read())
    print(f"Wrote new policy.py: {len(content)} -> {new_size} chars")

    # Quick smoke test
    print()
    print("Smoke test:")
    try:
        # Force re-import in case policy was already imported in this process
        import importlib
        if 'catfuse.policy' in sys.modules:
            del sys.modules['catfuse.policy']
        if 'catfuse.hardware' in sys.modules:
            del sys.modules['catfuse.hardware']
        from catfuse.policy import classify_shape_regime
        from catfuse.hardware import V100_PCIE_32GB, A100_40GB
        cases = [
            ((256, 256, 14, 14, 3), "memory_bound"),
            ((128, 128, 14, 14, 3), "compute_bound"),
            ((256, 256, 8, 8, 3), "memory_bound"),
        ]
        for args, expected in cases:
            got = classify_shape_regime(*args)
            mark = "OK" if got == expected else "FAIL"
            print(f"  {mark}: classify{args} = {got!r}  (expected {expected!r})")
        # Cross-hwprof smoke
        v = classify_shape_regime(128, 128, 14, 14, 3, hwprof=V100_PCIE_32GB)
        a = classify_shape_regime(128, 128, 14, 14, 3, hwprof=A100_40GB)
        print(f"  V100 vs A100 on (128,128,14,14): V100={v!r}  A100={a!r}")
    except Exception as e:
        print(f"  FAIL: smoke test raised {e}")
        return 1

    print()
    print("Fix applied. Next:")
    print("  python tests/stage7_verify.py    # should now 7/7 PASS")
    print("  python tests/stage6_verify.py    # SEW-RN18 routing back to 13/7")
    return 0


if __name__ == "__main__":
    sys.exit(main())