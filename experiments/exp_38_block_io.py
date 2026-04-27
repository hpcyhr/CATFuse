"""§5.3 — bug isolation step 2.

exp_37 showed:
  - All children of layer3.0 (conv1/conv2/downsample/sn1/sn2) have
    max_diff = 0 between DK and default.
  - layer3.1.bn1 (Identity, just passes through) has max_diff = 1.

That means: layer3.0 produces correct outputs from ALL children, but
the layer3.1 block sees a DIFFERENT input.

The discrepancy must live in:
  a. The BasicBlock.forward `+` add operation between identity and out
     (compute is deterministic so probably not — but check)
  b. Some side-effect inside an STFusion .forward() call that writes to
     a tensor referenced elsewhere (in-place corruption of input/state)
  c. Something async (Triton kernel writes happening AFTER hook fires)

This script hooks layer3.0 and layer3.1 (the BasicBlocks themselves)
to capture the actual tensor that goes from one block to the next.

Run:
    cd /path/to/CATFuse
    python experiments/exp_38_block_io.py
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


def install_block_hooks(net, label):
    """Hook every BasicBlock-level module (layer3.0, layer3.1, ...)."""
    captures_in = {}
    captures_out = {}
    handles = []

    for name, m in net.named_modules():
        if not (name.startswith("layer3.") or name.startswith("layer4.")):
            continue
        parts = name.split(".")
        # We want depth = 2 (layer3.0, layer3.1, layer4.0, layer4.1 — the
        # BasicBlocks themselves)
        if len(parts) != 2:
            continue

        def _make(lab):
            def fwd_pre(_mod, inp):
                # inp is a tuple of inputs
                if inp and isinstance(inp[0], torch.Tensor):
                    captures_in[lab] = inp[0].detach().clone()
            def fwd(_mod, _inp, out):
                if isinstance(out, torch.Tensor):
                    captures_out[lab] = out.detach().clone()
            return fwd_pre, fwd

        pre, post = _make(name)
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))

    return captures_in, captures_out, handles


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

    torch.manual_seed(0)
    x = torch.randn(T, B, 3, 32, 32, device=DEVICE)

    dk_in, dk_out, h_dk = install_block_hooks(fused_dk, "dk")
    with torch.no_grad():
        functional.reset_net(fused_dk)
        _ = fused_dk(x)
    for h in h_dk:
        h.remove()

    def_in, def_out, h_def = install_block_hooks(fused_def, "def")
    with torch.no_grad():
        functional.reset_net(fused_def)
        _ = fused_def(x)
    for h in h_def:
        h.remove()

    # Compare
    print()
    print("=" * 96)
    print("Per-BasicBlock input/output comparison (DK vs default)")
    print("=" * 96)
    print(f"  {'block':<15} {'INPUT max_diff':>20} {'OUTPUT max_diff':>20}")
    print(f"  {'-'*15} {'-'*20} {'-'*20}")

    common = sorted(set(dk_in.keys()) & set(def_in.keys()))
    first_in = None
    first_out = None
    for name in common:
        i_diff = (dk_in[name] - def_in[name]).abs().max().item()
        o_diff = (dk_out[name] - def_out[name]).abs().max().item()
        in_marker = "  ← INPUT differs" if i_diff > 0 else ""
        out_marker = "  ← OUTPUT differs" if o_diff > 0 else ""
        print(f"  {name:<15} {i_diff:>20.4e} {o_diff:>20.4e}{in_marker}{out_marker}")
        if i_diff > 0 and first_in is None:
            first_in = name
        if o_diff > 0 and first_out is None:
            first_out = name

    print()
    print("=" * 96)
    print("Diagnosis")
    print("=" * 96)
    if first_out is None:
        print("  No block-level output differs. exp_31 parity violation must be")
        print("  outside layer3-4 (avgpool, fc, ...).")
    elif first_in == first_out:
        print(f"  Both INPUT and OUTPUT first differ at {first_out}.")
        print(f"  The block was given different input — bug is BEFORE this block.")
        print(f"  Specifically, the previous block's RETURN tensor differs even")
        print(f"  though all of its children's outputs were identical (per exp_37).")
        print(f"  → STRONG evidence of in-place tensor corruption inside SF kernel:")
        print(f"    SF kernel writes to a buffer that is also the previous block's")
        print(f"    output, OR the addition `identity + out` is somehow non-")
        print(f"    deterministic.")
    elif first_out is not None and (first_in is None or first_in != first_out):
        print(f"  INPUT first differs at: {first_in}")
        print(f"  OUTPUT first differs at: {first_out}")
        print(f"  → Block {first_out} took the same input but produced different")
        print(f"    output. Bug is in this block's STFusion forward.")

    return 0


if __name__ == "__main__":
    sys.exit(main())