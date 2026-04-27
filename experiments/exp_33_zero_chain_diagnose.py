"""§5.3 follow-up — diagnose the all-zero spike chain.

exp_32b suggests every layer past conv1 sees all-zero input. This is
either (a) a hook artifact, or (b) the network really does produce
all-zero spikes after layer1.

This script answers it directly:
  1. Do the same forward as exp_31, capture intermediate spikes by
     reading state.tensor (the LIF's accumulated v) AFTER forward.
  2. Capture the actual logits — if the network's final output
     varies meaningfully across inputs, then somewhere along the
     way there must be non-zero spike activity. If logits are all
     constant zero or constant bias, the chain really is dead.
  3. For sj-eager (no fusion), do the same — does the original
     unfused network produce non-zero late-layer spikes?

This isolates: "is it the network", "is it the fusion", or "is it
the hook".

Run:
    cd /path/to/CATFuse
    python experiments/exp_33_zero_chain_diagnose.py
"""
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from catfuse.patterns import PartialFusionConvBNLIF


DEVICE = "cuda:0"


def build_sj_net():
    net = sew_resnet.sew_resnet18(
        pretrained=False, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    functional.set_step_mode(net, "m")
    ckpt_path = os.path.join(_REPO_ROOT, "checkpoints",
                             "sew_resnet18_cifar10_lif_best.pth")
    note = "synthetic"
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "net" in state:
                state = state["net"]
            net.load_state_dict(state, strict=False)
            note = "loaded ckpt"
        except Exception as e:
            print(f"  Warning: ckpt load failed: {e}")
    return net.to(DEVICE).eval(), note


def make_input(B, T=4, kind="random"):
    """Make CIFAR10-shape input.

    kind="random": pure noise (what exp_32b probably used)
    kind="cifar":  emulate normalized CIFAR data — values ~ N(0,1) but
                   replicated across T (constant frame, common SNN setup)
    kind="repeat_zero": all zeros (sanity check)
    """
    torch.manual_seed(7)
    if kind == "random":
        x = torch.randn(T, B, 3, 32, 32, device=DEVICE)
    elif kind == "cifar":
        # CIFAR10 has approx mean ~0.5, std ~0.25 per channel after
        # normalization. We construct a static frame in [0, 1] and broadcast.
        single = torch.rand(1, B, 3, 32, 32, device=DEVICE)
        x = single.expand(T, B, 3, 32, 32).contiguous()
    elif kind == "repeat_zero":
        x = torch.zeros(T, B, 3, 32, 32, device=DEVICE)
    return x


def measure_intermediate_activity(net, x, label):
    """Run forward; report stats on (a) every PF/ST layer's INPUT
    (captured via forward_pre_hook), and (b) final logits."""
    captured = {}

    def make_hook(name):
        def hook(mod, inputs):
            x = inputs[0]
            # Inputs to fused layers are [T,B,C,H,W]. Compute nnz fraction.
            if x.numel() == 0:
                return
            nnz = x.count_nonzero().item()
            total = x.numel()
            captured[name] = {
                "shape": tuple(x.shape),
                "nnz_frac": nnz / total,
                "max_abs": x.abs().max().item(),
                "mean_abs": x.abs().mean().item(),
                "kind": "ST" if isinstance(mod, STFusionConvBNLIF) else "PF",
            }
        return hook

    handles = []
    for name, mod in net.named_modules():
        if isinstance(mod, (STFusionConvBNLIF, PartialFusionConvBNLIF)):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        functional.reset_net(net)
        y = net(x)

    for h in handles:
        h.remove()

    print(f"\n--- {label} ---")
    print(f"  Input x: shape={tuple(x.shape)}  max_abs={x.abs().max().item():.3f}  "
          f"mean_abs={x.abs().mean().item():.3f}  nnz%={100*x.count_nonzero().item()/x.numel():.1f}%")
    print(f"  Output y: shape={tuple(y.shape)}  max_abs={y.abs().max().item():.4f}  "
          f"unique values count: {y.unique().numel()}")
    print()
    print(f"  {'kind':<5} {'layer':<35} {'shape':<22} {'max_abs':>9} {'|mean|':>9} {'nnz%':>6}")
    for name in sorted(captured.keys()):
        s = captured[name]
        shape_str = "x".join(str(d) for d in s["shape"])
        print(f"  {s['kind']:<5} {name:<35} {shape_str:<22} "
              f"{s['max_abs']:>9.4f} {s['mean_abs']:>9.4f} "
              f"{100*s['nnz_frac']:>5.1f}%")

    return captured, y


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 follow-up — diagnose the all-zero spike chain")
    print("=" * 96)

    sj_net, note = build_sj_net()
    print(f"\n  Checkpoint: {note}")

    fused, _ = substitute_sf(sj_net, T=4, force_sparse=False)
    fused = fused.to(DEVICE).eval()

    # Three input types
    for kind in ["cifar", "random", "repeat_zero"]:
        print()
        print("=" * 96)
        print(f"  Input type: {kind}")
        print("=" * 96)
        x = make_input(B=4, T=4, kind=kind)
        # Note: we use the FUSED model so the hook fires on PF/ST layers
        captured, y = measure_intermediate_activity(
            fused, x, label=f"FUSED forward, input={kind}"
        )

    # Now run sj_net (un-fused) with hooks on every LIF — see if
    # SJ's eager forward also gives all-zero late-layer spikes.
    print()
    print("=" * 96)
    print("  CONTROL: sj-eager forward — same input, watch LIF spike outputs")
    print("=" * 96)
    captured_sj = {}
    sj_handles = []
    for name, mod in sj_net.named_modules():
        if isinstance(mod, neuron.LIFNode):
            def make_hook(n):
                def hook(m, inputs, output):
                    if output is None:
                        return
                    nnz = output.count_nonzero().item()
                    total = output.numel()
                    captured_sj[n] = {
                        "shape": tuple(output.shape),
                        "nnz_frac": nnz / total,
                        "max_abs": output.abs().max().item(),
                    }
                return hook
            sj_handles.append(mod.register_forward_hook(make_hook(name)))

    x = make_input(B=4, T=4, kind="cifar")
    with torch.no_grad():
        functional.reset_net(sj_net)
        y_sj = sj_net(x)
    for h in sj_handles:
        h.remove()

    print(f"\n  SJ output: max_abs={y_sj.abs().max().item():.4f}, "
          f"unique values={y_sj.unique().numel()}")
    print()
    print(f"  {'LIF node':<40} {'shape':<22} {'max_abs':>9} {'nnz%':>6}")
    for name in sorted(captured_sj.keys()):
        s = captured_sj[name]
        shape_str = "x".join(str(d) for d in s["shape"])
        print(f"  {name:<40} {shape_str:<22} "
              f"{s['max_abs']:>9.4f} {100*s['nnz_frac']:>5.1f}%")

    # Diagnosis
    print()
    print("=" * 96)
    print("Diagnosis")
    print("=" * 96)
    sj_late = [s for n, s in captured_sj.items()
               if "layer3" in n or "layer4" in n]
    sj_late_active = [s for s in sj_late if s["nnz_frac"] > 0.001]
    if sj_late_active:
        max_nnz = max(s["nnz_frac"] for s in sj_late_active)
        print(f"  SJ-eager produces NON-ZERO spikes in late layers (max nnz "
              f"fraction = {100*max_nnz:.2f}%).")
        print(f"  → exp_32b's all-zero observation is a HOOK / TIMING ARTIFACT,")
        print(f"     not a true property of the network.")
    else:
        print("  SJ-eager ALSO produces all-zero spikes in late layers.")
        print("  → The all-zero chain is real — the checkpoint or input")
        print("     distribution doesn't activate later layers.")

    return 0


if __name__ == "__main__":
    sys.exit(main())