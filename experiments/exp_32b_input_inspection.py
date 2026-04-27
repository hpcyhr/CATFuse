"""§5.3 follow-up — inspect actual input distributions to STFusion layers.

The previous exp_32 run showed every STFusion layer receives 100% zero
input (sparsity = 1.000) from SEW-RN18 forward. This script verifies
that this is the real network behavior, not an instrumentation bug.

Method: hook each STFusion layer to record actual input tensor stats
(min, max, mean abs, fraction of nonzero, total numel) — NOT just sparsity.
This is informative because:
  - sparsity could be 1.0 because nnz==0 (truly all zeros)
  - or because count_nonzero misses something
  - inspecting min/max/mean tells us if the tensor is truly zero or near-zero

Also report: PartialFusion layers' input stats too — if those have
non-trivial sparsity but STFusion layers are all-zero, that locates
where the network "dies" (becomes all-zero downstream).

Run:
    python experiments/exp_32b_input_inspection.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

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
T = 4
B = 4


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
        except Exception:
            pass
    return net.to(DEVICE).eval(), note


def main():
    if not torch.cuda.is_available():
        return 1

    print("=" * 100)
    print("§5.3 follow-up — Input distribution to fused layers (SEW-RN18)")
    print("=" * 100)
    print()

    sj_net, note = build_sj_net()
    print(f"  Checkpoint: {note}")

    fused, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused = fused.to(DEVICE).eval()

    # Hooks: record (min, max, mean_abs, nnz_fraction, numel) for each
    # PartialFusion + STFusion layer's input.
    stats = defaultdict(lambda: {
        "calls": 0, "min": float("inf"), "max": float("-inf"),
        "abs_mean_sum": 0.0, "nnz_frac_sum": 0.0,
        "numel": 0, "shape": None,
    })

    handles = []

    def make_hook(name, kind):
        def hook(module, inputs, output):
            x = inputs[0]
            s = stats[name]
            s["calls"] += 1
            s["kind"] = kind
            s["min"] = min(s["min"], x.min().item())
            s["max"] = max(s["max"], x.max().item())
            s["abs_mean_sum"] += x.abs().mean().item()
            s["nnz_frac_sum"] += x.count_nonzero().item() / x.numel()
            s["numel"] = x.numel()
            s["shape"] = tuple(x.shape)
        return hook

    for name, mod in fused.named_modules():
        if isinstance(mod, STFusionConvBNLIF):
            handles.append(mod.register_forward_hook(make_hook(name, "ST")))
        elif isinstance(mod, PartialFusionConvBNLIF):
            handles.append(mod.register_forward_hook(make_hook(name, "PF")))

    print(f"  Registered hooks on {len(handles)} fused layers")
    print()

    # Try CIFAR10 if available, else random
    have_cifar = False
    try:
        from torchvision import datasets, transforms
        for path in [os.path.join(_REPO_ROOT, "data"),
                     "/data/yhr/datasets/cifar10",
                     "/root/datasets/cifar10"]:
            if os.path.isdir(path):
                tx = transforms.Compose([transforms.ToTensor()])
                ds = datasets.CIFAR10(root=path, train=False, download=False, transform=tx)
                dl = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=False)
                # Take a few batches
                inputs = []
                for i, (img, _) in enumerate(dl):
                    if i >= 3: break
                    inputs.append(img.to(DEVICE))
                have_cifar = True
                break
    except Exception:
        pass

    if not have_cifar:
        print("  CIFAR10 not found — using random Gaussian inputs")
        inputs = [torch.randn(B, 3, 32, 32, device=DEVICE) for _ in range(3)]
    else:
        print(f"  Using {len(inputs)} CIFAR10 batches (B={B} each)")

    print()
    with torch.no_grad():
        for img in inputs:
            # Stack T copies to make [T, B, 3, 32, 32] (multi-step convention)
            x = img.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            functional.reset_net(fused)
            _ = fused(x)
    torch.cuda.synchronize()

    for h in handles:
        h.remove()

    # Report
    print("=" * 100)
    print("Per-layer input statistics  (avg across forward passes)")
    print("=" * 100)
    header = f"  {'kind':<3}  {'layer':<32}  {'shape':<22}  {'min':>8}  {'max':>8}  {'|x|_mean':>10}  {'nnz%':>6}"
    print(header)
    print(f"  {'-'*3}  {'-'*32}  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")

    # Sort by name (which roughly corresponds to network depth)
    for name in sorted(stats.keys()):
        s = stats[name]
        n = s["calls"]
        if n == 0:
            continue
        shape = "x".join(str(d) for d in s["shape"])
        kind = s["kind"]
        abs_mean_avg = s["abs_mean_sum"] / n
        nnz_pct_avg = s["nnz_frac_sum"] / n * 100
        print(f"  {kind:<3}  {name:<32}  {shape:<22}  "
              f"{s['min']:>8.3f}  {s['max']:>8.3f}  "
              f"{abs_mean_avg:>10.4f}  {nnz_pct_avg:>5.1f}%")

    # Diagnose
    print()
    print("=" * 100)
    print("Diagnosis")
    print("=" * 100)
    st_layers = [(n, s) for n, s in stats.items() if s.get("kind") == "ST" and s["calls"] > 0]
    pf_layers = [(n, s) for n, s in stats.items() if s.get("kind") == "PF" and s["calls"] > 0]

    st_avg_nnz = sum(s["nnz_frac_sum"]/s["calls"] for _, s in st_layers) / max(len(st_layers), 1)
    pf_avg_nnz = sum(s["nnz_frac_sum"]/s["calls"] for _, s in pf_layers) / max(len(pf_layers), 1)

    print(f"  STFusion layers avg nnz fraction: {st_avg_nnz*100:.2f}%")
    print(f"  PartialFusion layers avg nnz fraction: {pf_avg_nnz*100:.2f}%")
    print()

    if st_avg_nnz < 0.001:
        print("  STFusion layers receive ~0% nonzero input.")
        print("  This is consistent with SEW-RN18 producing extremely sparse")
        print("  spike trains in late layers (layer3+layer4) — the spike")
        print("  threshold + small spatial extent makes whole tensors zero.")
        print()
        print("  Implication: on this network + checkpoint + V100, the SF kernel")
        print("  is moot — EGD doesn't pick it because every STFusion call hits")
        print("  the static-zero short-circuit. The 2x end-to-end speedup over")
        print("  SJ-eager comes from:")
        print("    - LIF multi-step kernel (1 launch instead of T)")
        print("    - BN folded into conv weight (1 conv instead of conv+BN)")
        print("    - StaticZero short-circuit (skips conv when input is zero)")
        print("  None of these is SparseFlow.")
    elif st_avg_nnz < 0.30:
        print("  STFusion inputs are sparse (>70% zero on average).")
        print("  EGD's sparsity > 0.7 threshold should fire often. Needs investigation.")
    else:
        print(f"  STFusion inputs are NOT particularly sparse ({st_avg_nnz*100:.1f}% nonzero).")
        print("  EGD correctly routes to DK; this matches the §5.3 latency observation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())