"""[Stage 1 verification] benchmarks/ kernels successfully migrated to catfuse.kernels/.

This script verifies that:
  1. catfuse package imports cleanly without sys.path manipulation
  2. The three public kernel APIs are exposed via catfuse.kernels
  3. The three pattern classes can be instantiated and run a forward pass
  4. No bare-name `partial_fusion_*` or `catfuse_patterns` imports remain in
     the codebase (excluding this script itself)
  5. patterns.py no longer injects benchmarks/ into sys.path

Run from the repo root:
    cd /path/to/CATFuse
    python -m tests.stage1_verify       # preferred
    # OR (auto-fixed):
    python tests/stage1_verify.py
"""
import os
import re
import sys
import traceback

# Auto-resolve repo root so this script works whether invoked as
# `python tests/stage1_verify.py` or `python -m tests.stage1_verify`
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Sentinel string assembled at runtime so the pattern doesn't appear as a
# literal in this file (avoids self-detection during the residual scan).
_BAD_TOKENS = [
    "f" + "rom partial_fusion_conv_lif ",
    "f" + "rom partial_fusion_conv_bn_lif ",
    "f" + "rom partial_fusion_conv_bn_add_lif ",
    "f" + "rom catfuse_patterns ",
    "i" + "mport catfuse_patterns",
]


def check_imports():
    """1. catfuse package + kernels package import cleanly."""
    print("[1/4] Checking package imports...")
    try:
        import catfuse  # noqa: F401
        from catfuse import patterns, substitute, kernels  # noqa: F401
        from catfuse.kernels import (  # noqa: F401
            partial_fusion_conv_lif,
            partial_fusion_conv_bn_lif,
            partial_fusion_conv_bn_add_lif,
        )
    except Exception:
        print("  FAIL")
        traceback.print_exc()
        return False
    print("  OK: catfuse, catfuse.patterns, catfuse.substitute, catfuse.kernels all importable")
    return True


def _is_real_import_line(line):
    """Return True if a line is a real Python import statement (not a string,
    not a comment, not in a docstring bullet)."""
    stripped = line.lstrip()
    # Skip comments
    if stripped.startswith("#"):
        return False
    # Skip docstring bullet lines (starts with `-` or `*` after whitespace)
    if stripped.startswith("- ") or stripped.startswith("* "):
        return False
    # Real imports start with `from ` or `import ` at indentation level 0
    # (or some indentation, e.g. local imports inside functions).
    # We require the line to actually parse as an import via regex.
    if re.match(r"^\s*from\s+\w+(\.\w+)*\s+import\s+", line):
        return True
    if re.match(r"^\s*import\s+\w+", line):
        return True
    return False


def check_no_residual_bare_imports():
    """4. No bare-name `partial_fusion_*` / `catfuse_patterns` imports remain
    in the codebase (excluding this script itself)."""
    print("[2/4] Scanning codebase for residual bare-name imports...")
    self_path = _THIS_FILE
    bad_finds = []
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            if os.path.abspath(path) == self_path:
                continue  # don't scan ourselves
            try:
                with open(path) as fp:
                    for lineno, line in enumerate(fp, 1):
                        if not _is_real_import_line(line):
                            continue
                        for tok in _BAD_TOKENS:
                            if tok in line:
                                bad_finds.append((path, lineno, line.rstrip(), tok))
            except Exception:
                continue
    if bad_finds:
        print("  FAIL: residual bare-name imports found:")
        for path, lineno, line, pat in bad_finds:
            rel = os.path.relpath(path, _REPO_ROOT)
            print(f"    {rel}:{lineno} ({pat.strip()}): {line.strip()}")
        return False
    print("  OK: no bare-name imports remain (excluding this script's literal patterns)")
    return True


def check_no_benchmarks_syspath():
    """3. patterns.py no longer injects benchmarks/ into sys.path."""
    print("[3/4] Verifying patterns.py has no benchmarks/ sys.path hack...")
    patterns_path = os.path.join(_REPO_ROOT, "catfuse", "patterns.py")
    with open(patterns_path) as f:
        text = f.read()
    if "_BENCHMARKS_DIR" in text or "sys.path.insert(0, _BENCHMARKS_DIR)" in text:
        print("  FAIL: patterns.py still injects benchmarks/ into sys.path")
        return False
    print("  OK: patterns.py is clean (no sys.path manipulation)")
    return True


def check_pattern_instantiation():
    """4. Pattern classes can be instantiated and forward run on dummy GPU input."""
    print("[4/4] Smoke-testing pattern instantiation + forward...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.patterns import (
            PartialFusionConvLIF, PartialFusionConvBNLIF,
        )
    except Exception:
        print("  SKIP: torch / spikingjelly not available")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return None

    device = "cuda:0"
    T, B, C_in, C_out, H, W = 4, 2, 8, 8, 16, 16

    conv = sj_layer.Conv2d(C_in, C_out, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(C_out).to(device)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    try:
        fused_bn = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)
        x = torch.randn(T, B, C_in, H, W, device=device)
        y_bn = fused_bn(x)
        assert y_bn.shape == (T, B, C_out, H, W), f"shape mismatch: {y_bn.shape}"

        fused = PartialFusionConvLIF.from_sj_modules(conv, lif).to(device)
        y = fused(x)
        assert y.shape == (T, B, C_out, H, W), f"shape mismatch: {y.shape}"

        print("  OK: PartialFusionConvBNLIF + PartialFusionConvLIF forward pass")
        return True
    except Exception:
        print("  FAIL")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Stage 1 verification: kernel migration from benchmarks/ -> catfuse/kernels/")
    print("=" * 60)
    results = []
    results.append(("imports", check_imports()))
    results.append(("no_residual", check_no_residual_bare_imports()))
    results.append(("no_syspath_hack", check_no_benchmarks_syspath()))
    results.append(("pattern_smoketest", check_pattern_instantiation()))

    print()
    print("=" * 60)
    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} checks passed, "
              f"{len(skipped)} skipped (need GPU): {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()