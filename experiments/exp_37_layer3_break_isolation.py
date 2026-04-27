"""§5.3 follow-up — narrow down exactly where layer3.1.conv1 sees different input.

exp_36 showed:
  - SJ vs DK: bit-exact across all LIF layers
  - DK vs default: layer3.0.conv2 OK, layer3.1.conv1 DIFFER (max_diff=1)

This script captures intermediate tensors INSIDE layer3 to find which
exact tensor first differs between DK and default. Possibilities:

  a. layer3.0.sn2 output (LIF spike): differs → bug is in SF kernel of
     layer3.0.conv2 or its sn2 fusion (despite layer3.0.conv2's
     "captured output" matching, that capture is the LIF output of the
     fused layer; if BOTH LIF outputs match, neither tensor differs there.)

  b. layer3.0 block output (after ADD shortcut): differs → bug is in
     SEW-ResNet ADD shortcut handling somewhere in fused model (maybe
     CTFSEWBasicBlock if used, or substitute_sf wiring).

  c. layer3.1.conv1 input == layer3.0 block output: should be same as (b)
     UNLESS something modifies it between blocks (shouldn't).

  d. layer3.1.conv1's STFusion sees the SAME input but produces different
     output: bug is in SF kernel for that layer specifically (state
     leakage? cache miss? routing condition difference?).

This script tries all four.

Run:
    cd /path/to/CATFuse
    python experiments/exp_37_layer3_break_isolation.py
"""
from __future__ import annotations

import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import functional, neuron

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
T, B = 4, 1


def install_full_hooks(net, label):
    """Hook EVERY top-level module in layer3 + layer4 (not just LIF/STFusion)."""
    captures = {}
    handles = []

    # All modules in layer3.* and layer4.*
    for name, m in net.named_modules():
        if not (name.startswith("layer3.") or name.startswith("layer4.")):
            continue
        # Skip the layer3/layer4 itself, just hook children of basic blocks
        # Heuristic: only hook things that are likely to produce useful output
        # — direct attributes of the block (conv, bn, sn, downsample, etc.)
        # Depth = number of dots after "layer3.0."
        parts = name.split(".")
        if len(parts) < 3:  # layer3.0 itself
            continue
        if len(parts) > 4:  # too deep
            continue

        def _make(lab):
            def fn(_mod, _inp, out):
                if isinstance(out, torch.Tensor):
                    captures[lab] = out.detach().clone()
            return fn

        handles.append(m.register_forward_hook(_make(name)))

    return captures, handles


def main():
    if not torch.cuda.is_available():
        return 1

    sj_net, note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    print(f"Checkpoint: {note}")
    print(f"T={T} B={B}")

    fused_dk, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_dk = fused_dk.to(DEVICE).eval()
    for m in fused_dk.modules():
        if isinstance(m, STFusionConvBNLIF) and m._impl_sparse is not None:
            m._impl_sparse = None

    fused_def, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_def = fused_def.to(DEVICE).eval()

    # Same input
    torch.manual_seed(0)
    x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

    # Capture per-module
    dk_caps, dk_handles = install_full_hooks(fused_dk, "dk")
    with torch.no_grad():
        functional.reset_net(fused_dk)
        _ = fused_dk(x)
    for h in dk_handles:
        h.remove()

    def_caps, def_handles = install_full_hooks(fused_def, "def")
    with torch.no_grad():
        functional.reset_net(fused_def)
        _ = fused_def(x)
    for h in def_handles:
        h.remove()

    # Diff
    print()
    print("=" * 96)
    print("All hooked modules in layer3.* and layer4.*  (DK vs default)")
    print("=" * 96)
    print(f"  {'module':<45} {'shape':<22} {'max_diff':>14} {'class'}")
    print(f"  {'-'*45} {'-'*22} {'-'*14} {'-'*30}")

    common = sorted(set(dk_caps.keys()) & set(def_caps.keys()))
    first_break = None
    for name in common:
        t1 = dk_caps[name]
        t2 = def_caps[name]
        if t1.shape != t2.shape:
            print(f"  {name:<45} (shape mismatch)")
            if first_break is None:
                first_break = name
            continue
        d = (t1 - t2).abs().max().item()
        # Get the module class from net
        mod = dict(fused_dk.named_modules()).get(name)
        cls = type(mod).__name__ if mod else "?"
        marker = "  ← DIFFER" if d > 0 else ""
        print(f"  {name:<45} {str(tuple(t1.shape)):<22} {d:>14.4e}  {cls}{marker}")
        if d > 0 and first_break is None:
            first_break = name

    print()
    if first_break:
        print(f"FIRST DIVERGENCE at: {first_break}")
        print(f"  module class: {type(dict(fused_dk.named_modules())[first_break]).__name__}")
        print(f"  Look at the input/output of THIS module to understand the bug")
    else:
        print("No divergence in layer3-4 modules")

    return 0


if __name__ == "__main__":
    sys.exit(main())