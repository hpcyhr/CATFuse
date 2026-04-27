"""§5.3 follow-up — locate first parity violation in active spike chain.

The exp_31 results show CATFuse-default and even CATFuse-DK produce
non-zero max_diff vs SJ-eager when the spike chain is real (after the
ckpt unwrap fix). This contradicts the bit-exact guarantee of Corollary
3.17.

Hypothesis: Stage 1-8 SEW-RN18 verify ran on a dead-chain network and
"max_diff = 0" was 0 == 0, masking a real framework bug that only shows
up when LIF actually fires. We need to:

  1. Confirm: SJ-eager vs CATFuse-DK (no SF) on small B has parity 0 or
     non-zero?  If non-zero, the bug is in DenseKeep or partial fusion.
  2. Confirm: SJ-eager vs CATFuse-default on small B parity? If DK is OK
     but default fails, bug is in SparseFlow or routing.
  3. Locate: layer-by-layer hook to find FIRST layer where outputs diverge.

Run this BEFORE any wall-clock experiment; it tells us where to focus
the framework debug.

Run:
    cd /path/to/CATFuse
    python experiments/exp_36_parity_diagnose.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
T, B = 4, 1


def hook_named_modules(net, classes, label_prefix):
    """Install forward hooks on every module of given classes; return
    {label: (input_shape, output)} dict that fills as forward runs."""
    captures = {}
    handles = []
    for name, m in net.named_modules():
        if any(isinstance(m, c) for c in classes):
            label = f"{label_prefix}::{name}"
            def _make(lab):
                def fn(_mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        captures[lab] = out.detach().clone()
                return fn
            handles.append(m.register_forward_hook(_make(label)))
    return captures, handles


def diff_dicts(d1: Dict[str, torch.Tensor], d2: Dict[str, torch.Tensor],
               label1: str, label2: str):
    """Find first differing key between two captured-output dicts.
    Names are matched by stripping the prefix (label1:: / label2::)."""
    keys1 = {k.split("::", 1)[1]: k for k in d1}
    keys2 = {k.split("::", 1)[1]: k for k in d2}
    common = sorted(set(keys1.keys()) & set(keys2.keys()))

    print(f"\n  Layer-by-layer comparison: {label1} vs {label2}")
    print(f"  Common modules: {len(common)}")
    print(f"  {'module':<40} {'shape':<22} {'max_diff':>14}")
    print(f"  {'-'*40} {'-'*22} {'-'*14}")

    first_break = None
    for short in common:
        t1 = d1[keys1[short]]
        t2 = d2[keys2[short]]
        if t1.shape != t2.shape:
            print(f"  {short:<40} (shape mismatch: {t1.shape} vs {t2.shape})")
            if first_break is None:
                first_break = short
            continue
        max_diff = (t1 - t2).abs().max().item()
        marker = ""
        if max_diff > 0:
            marker = "  ← DIFFER"
            if first_break is None:
                first_break = short
        print(f"  {short:<40} {str(tuple(t1.shape)):<22} {max_diff:>14.4e}{marker}")

    print()
    if first_break:
        print(f"  FIRST DIVERGENCE at module: {first_break}")
    else:
        print(f"  No divergence — outputs bit-exact identical")
    return first_break


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 follow-up — locate parity violation in active spike chain")
    print("=" * 96)

    # ============================================================
    # Build three networks: SJ, CATFuse-DK, CATFuse-default
    # ============================================================
    sj_net, note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    print(f"\nCheckpoint: {note}")
    print(f"T={T}  B={B}")

    fused_dk, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_dk = fused_dk.to(DEVICE).eval()
    n_disabled = 0
    for m in fused_dk.modules():
        if isinstance(m, STFusionConvBNLIF) and m._impl_sparse is not None:
            m._impl_sparse = None
            n_disabled += 1
    print(f"DK-only: disabled SparseFlow on {n_disabled} STFusion layers")

    fused_def, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_def = fused_def.to(DEVICE).eval()

    # ============================================================
    # Run forward on identical input, capture LIF outputs
    # ============================================================
    torch.manual_seed(0)
    x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

    # SJ baseline
    sj_caps, sj_handles = hook_named_modules(
        sj_net, [neuron.LIFNode], "sj")
    with torch.no_grad():
        functional.reset_net(sj_net)
        y_sj = sj_net(x).detach().clone()
    for h in sj_handles:
        h.remove()

    # DK
    dk_caps, dk_handles = hook_named_modules(
        fused_dk, [neuron.LIFNode, STFusionConvBNLIF], "dk")
    with torch.no_grad():
        functional.reset_net(fused_dk)
        y_dk = fused_dk(x).detach().clone()
    for h in dk_handles:
        h.remove()

    # Default
    def_caps, def_handles = hook_named_modules(
        fused_def, [neuron.LIFNode, STFusionConvBNLIF], "def")
    with torch.no_grad():
        functional.reset_net(fused_def)
        y_def = fused_def(x).detach().clone()
    for h in def_handles:
        h.remove()

    # Final output diff
    print()
    print("=" * 96)
    print("Final output parity")
    print("=" * 96)
    diff_dk = (y_dk - y_sj).abs().max().item()
    diff_def = (y_def - y_sj).abs().max().item()
    diff_dk_def = (y_dk - y_def).abs().max().item()
    print(f"  CATFuse-DK   vs SJ-eager:        max_diff = {diff_dk:.4e}")
    print(f"  CATFuse-def  vs SJ-eager:        max_diff = {diff_def:.4e}")
    print(f"  CATFuse-def  vs CATFuse-DK:      max_diff = {diff_dk_def:.4e}")

    # ============================================================
    # Compare DK vs SJ first (DK should be bit-exact per §3.6 Cor 3.17)
    # ============================================================
    print()
    print("=" * 96)
    print("Layer-by-layer: SJ-eager vs CATFuse-DK")
    print("=" * 96)
    print("If Cor 3.17 holds (DenseKeep ≡ σ_ref), this should be all zeros.")
    first_break_dk = diff_dicts(sj_caps, dk_caps, "sj", "dk")

    # ============================================================
    # Compare default vs DK
    # ============================================================
    print()
    print("=" * 96)
    print("Layer-by-layer: CATFuse-DK vs CATFuse-default")
    print("=" * 96)
    print("If Cor 3.17 holds (SparseFlow ≡ DenseKeep on bit level), zeros.")
    first_break_def = diff_dicts(dk_caps, def_caps, "dk", "def")

    # ============================================================
    # Conclusion
    # ============================================================
    print()
    print("=" * 96)
    print("Diagnosis")
    print("=" * 96)
    if first_break_dk is None and first_break_def is None:
        print("  All bit-exact — exp_31's parity violations are NOT layer-level")
        print("  drift but possibly a state leak across forwards. Investigate")
        print("  StateBuffer.reset semantics in benchmark loop context.")
    elif first_break_dk is None and first_break_def is not None:
        print(f"  DK is bit-exact vs SJ; default first diverges at:")
        print(f"    {first_break_def}")
        print(f"  → Bug is in SparseFlow path on this layer (or its preceding")
        print(f"    routing decision). Check SparseFlow.forward / _batchfold_forward")
        print(f"    of the diverging layer.")
    elif first_break_dk is not None:
        print(f"  CATFuse-DK already differs from SJ-eager at:")
        print(f"    {first_break_dk}")
        print(f"  → Bug is in DenseKeep impl OR in PartialFusionConvBNLIF")
        print(f"    (depending on which class the diverging module is). This")
        print(f"    is BAD: violates Cor 3.17 at framework level.")

    return 0


if __name__ == "__main__":
    sys.exit(main())