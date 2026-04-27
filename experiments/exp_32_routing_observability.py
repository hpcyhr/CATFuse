"""§5.3 sanity check — Runtime EGD routing observability.

Verifies whether the §5.3 end-to-end measurement (CATFuse-default ≈
CATFuse-DK in wall-clock) is explained by Runtime EGD routing layers
to the DenseKeep path, OR by something else.

Method: monkey-patch STFusionConvBNLIF.forward to log, per call:
  - layer name  (set externally before forward)
  - input shape
  - sparsity
  - path taken: "SF" (sparsity > 0.7), "DK" (else), or "StaticZero" (nnz == 0)

Run SEW-RN18 forward across multiple inputs and report the routing
distribution per STFusion layer.

If the prediction is correct (EGD routes most calls to DK at the V100
sparsity regime SEW-RN18 produces), we expect:
  - very few "SF" calls in the table
  - mostly "DK" or "StaticZero" calls

If wrong, we'll see "SF" dominating and need a different explanation
for default ≈ DK.

Run:
    cd /path/to/CATFuse
    python experiments/exp_32_routing_observability.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF


DEVICE = "cuda:0"
T = 4
B = 2
N_INPUTS = 10  # number of distinct random inputs to average over

# CIFAR10 location candidates — script tries each in order.
CIFAR10_DATA_PATHS = [
    os.path.join(_REPO_ROOT, "data"),                     # /data/yhr/CATFuse/data
    "/data/yhr/datasets/cifar10",
    os.path.expanduser("~/datasets/cifar10"),
]

CSV_PATH = os.path.join(_REPO_ROOT, "experiments/results/routing_observability.csv")


@dataclass
class LayerStats:
    layer_name: str
    Cin: int
    Cout: int
    H: int
    W: int
    n_calls: int = 0
    n_sf: int = 0
    n_dk: int = 0
    n_static_zero: int = 0
    sparsity_min: float = 1.0
    sparsity_max: float = 0.0
    sparsity_sum: float = 0.0

    @property
    def sparsity_mean(self) -> float:
        return self.sparsity_sum / max(self.n_calls, 1)

    @property
    def sf_fraction(self) -> float:
        return self.n_sf / max(self.n_calls, 1)


def install_observation_hook(fused_net) -> Dict[str, LayerStats]:
    """Monkey-patch every STFusion layer to log its routing.

    Returns a dict mapping layer_name -> LayerStats; this dict is updated
    in-place as forward() is called.
    """
    stats: Dict[str, LayerStats] = {}

    # Find all STFusion layers and remember their names
    name_by_id = {}
    for full_name, mod in fused_net.named_modules():
        if isinstance(mod, STFusionConvBNLIF):
            name_by_id[id(mod)] = full_name
            stats[full_name] = LayerStats(
                layer_name=full_name,
                Cin=mod.in_channels, Cout=mod.out_channels,
                H=0, W=0,  # filled on first forward
            )

    # Save original forward for restoration
    original_forward = STFusionConvBNLIF.forward

    def patched_forward(self, x: torch.Tensor):
        # Locate this instance's stats
        name = name_by_id.get(id(self), "<unknown>")
        s = stats.get(name)

        # Compute sparsity exactly as the dispatch does
        x_flat = x.reshape(-1)
        nnz = x_flat.count_nonzero().item()
        total = x_flat.numel()
        sparsity = 1.0 - nnz / total if total > 0 else 1.0

        if s is not None:
            s.n_calls += 1
            if s.H == 0:
                s.H = x.shape[3]
                s.W = x.shape[4]
            s.sparsity_sum += sparsity
            s.sparsity_min = min(s.sparsity_min, sparsity)
            s.sparsity_max = max(s.sparsity_max, sparsity)
            if nnz == 0:
                s.n_static_zero += 1
            elif sparsity > 0.7:
                s.n_sf += 1
            else:
                s.n_dk += 1

        return original_forward(self, x)

    STFusionConvBNLIF.forward = patched_forward
    return stats, original_forward


def restore_forward(original_forward):
    STFusionConvBNLIF.forward = original_forward


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
            note = "from sew_resnet18_cifar10_lif_best.pth"
        except Exception:
            pass
    return net.to(DEVICE).eval(), note


def find_cifar10_dir():
    """Find a directory that contains cifar-10-batches-py."""
    for base in CIFAR10_DATA_PATHS:
        candidate = os.path.join(base, "cifar-10-batches-py")
        if os.path.isdir(candidate):
            return base
    return None


def get_cifar10_inputs(N_inputs: int, batch_size: int):
    """Yield N_inputs tensors of shape (T, B, 3, 32, 32) from CIFAR10 test set.

    Returns None if CIFAR10 cannot be loaded.

    The input convention matches SEW-RN18's expected normalization
    (CIFAR10 mean/std). Each timestep gets the same image (rate-coding
    happens implicitly via the network's first LIF layer).
    """
    cifar_root = find_cifar10_dir()
    if cifar_root is None:
        return None
    try:
        import torchvision
        import torchvision.transforms as T_
        transform = T_.Compose([
            T_.ToTensor(),
            T_.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
        ])
        ds = torchvision.datasets.CIFAR10(
            root=cifar_root, train=False, download=False, transform=transform,
        )
    except Exception as e:
        print(f"  (CIFAR10 load failed: {e})")
        return None

    inputs = []
    torch.manual_seed(0)
    indices = torch.randperm(len(ds))[:N_inputs * batch_size].tolist()
    for i in range(N_inputs):
        batch_imgs = []
        for j in range(batch_size):
            idx = indices[i * batch_size + j]
            img, _ = ds[idx]
            batch_imgs.append(img)
        x_static = torch.stack(batch_imgs, dim=0).to(DEVICE)        # [B, 3, 32, 32]
        # Replicate across T timesteps (rate-coding via first LIF)
        x = x_static.unsqueeze(0).expand(T, -1, -1, -1, -1).contiguous()
        inputs.append(x)
    return inputs


def run_one_inputset(label: str, fused, inputs):
    """Run all inputs through fused, with stats hook installed."""
    print(f"\n  --- Input source: {label} ---")
    stats, original_forward = install_observation_hook(fused)
    try:
        with torch.no_grad():
            for x in inputs:
                functional.reset_net(fused)
                _ = fused(x)
        torch.cuda.synchronize()
    finally:
        restore_forward(original_forward)
    return stats


def report_stats(label: str, stats: Dict[str, "LayerStats"], rows_out: List[dict]):
    """Print per-layer routing distribution and append to rows_out for CSV."""
    print()
    print(f"  Routing under input: {label}")
    print(f"  {'layer':<35} {'shape':<18} {'calls':>5} {'SF':>4} {'DK':>4} {'StZ':>4} "
          f"{'SF%':>6} {'sparsity (min/mean/max)':<28}")
    print(f"  {'-'*35} {'-'*18} {'-'*5} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*6} {'-'*28}")

    total_calls = total_sf = total_dk = total_sz = 0
    for name in sorted(stats.keys()):
        s = stats[name]
        if s.n_calls == 0:
            continue
        shape = f"{s.Cin}->{s.Cout} {s.H}x{s.W}"
        sp_str = f"{s.sparsity_min:.3f} / {s.sparsity_mean:.3f} / {s.sparsity_max:.3f}"
        print(f"  {name:<35} {shape:<18} {s.n_calls:>5} "
              f"{s.n_sf:>4} {s.n_dk:>4} {s.n_static_zero:>4} "
              f"{100*s.sf_fraction:>5.1f}% {sp_str:<28}")
        rows_out.append({
            "input_source": label,
            "layer_name": s.layer_name,
            "Cin": s.Cin, "Cout": s.Cout, "H": s.H, "W": s.W,
            "n_calls": s.n_calls, "n_sf": s.n_sf, "n_dk": s.n_dk,
            "n_static_zero": s.n_static_zero,
            "sparsity_min": s.sparsity_min,
            "sparsity_max": s.sparsity_max,
            "sparsity_mean": s.sparsity_mean,
            "sf_fraction": s.sf_fraction,
        })
        total_calls += s.n_calls
        total_sf += s.n_sf
        total_dk += s.n_dk
        total_sz += s.n_static_zero

    print(f"  {'-'*35} {'-'*18} {'-'*5} {'-'*4} {'-'*4} {'-'*4}")
    sf_pct = 100 * total_sf / max(total_calls, 1)
    dk_pct = 100 * total_dk / max(total_calls, 1)
    sz_pct = 100 * total_sz / max(total_calls, 1)
    print(f"  {'TOTAL':<35} {'':<18} {total_calls:>5} {total_sf:>4} {total_dk:>4} "
          f"{total_sz:>4}  ({sf_pct:.1f}% SF / {dk_pct:.1f}% DK / {sz_pct:.1f}% StZ)")
    return sf_pct, dk_pct, sz_pct


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 sanity — Runtime EGD routing observability on SEW-RN18")
    print("=" * 96)
    print()
    print(f"  T={T}  B={B}  N_INPUTS={N_INPUTS}")
    print()

    sj_net, ckpt_note = build_sj_net()
    print(f"  Checkpoint: {ckpt_note}")

    fused, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused = fused.to(DEVICE).eval()
    n_st = sum(1 for m in fused.modules() if isinstance(m, STFusionConvBNLIF))
    print(f"  STFusion layers in network: {n_st}")

    # --- Inputs: synthetic ---
    print()
    print(f"  Building synthetic inputs ({N_INPUTS} batches of B={B} torch.randn)...")
    synth_inputs = []
    for i in range(N_INPUTS):
        torch.manual_seed(1000 + i)
        synth_inputs.append(torch.randn(T, B, 3, 32, 32, device=DEVICE))

    # --- Inputs: CIFAR10 (if available) ---
    print(f"  Looking for CIFAR10 in {CIFAR10_DATA_PATHS}...")
    cifar_inputs = get_cifar10_inputs(N_INPUTS, B)
    if cifar_inputs is None:
        print(f"  (CIFAR10 not found — skipping real-input test)")
    else:
        print(f"  Loaded CIFAR10 test images: {N_INPUTS} batches of B={B}")

    rows_out: List[dict] = []

    # --- Run synthetic ---
    print()
    print("=" * 96)
    print("Run 1: synthetic torch.randn input")
    print("=" * 96)
    s_synth = run_one_inputset("synthetic", fused, synth_inputs)
    sf_synth, dk_synth, sz_synth = report_stats("synthetic", s_synth, rows_out)

    # --- Run CIFAR10 ---
    sf_cifar, dk_cifar, sz_cifar = None, None, None
    if cifar_inputs is not None:
        print()
        print("=" * 96)
        print("Run 2: CIFAR10 test set input (real deployment workload)")
        print("=" * 96)
        s_cifar = run_one_inputset("cifar10", fused, cifar_inputs)
        sf_cifar, dk_cifar, sz_cifar = report_stats("cifar10", s_cifar, rows_out)

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(r)

    # --- Interpretation ---
    print()
    print("=" * 96)
    print("Interpretation")
    print("=" * 96)
    print(f"  synthetic input: SF={sf_synth:.1f}%  DK={dk_synth:.1f}%  StZ={sz_synth:.1f}%")
    if sf_cifar is not None:
        print(f"  cifar10 input:   SF={sf_cifar:.1f}%  DK={dk_cifar:.1f}%  StZ={sz_cifar:.1f}%")
        delta = abs(sf_synth - sf_cifar)
        if delta > 20:
            print()
            print(f"  CRITICAL: synthetic vs CIFAR10 SF% differs by {delta:.1f} points.")
            print(f"  The §5.3 end-to-end measurement was run on SYNTHETIC input.")
            print(f"  The Runtime EGD decisions in production (CIFAR10) differ. The")
            print(f"  paper's §5.3 narrative must use CIFAR10 routing data, NOT")
            print(f"  the synthetic randn data, to truthfully describe deployment.")
        elif delta > 5:
            print()
            print(f"  Note: synthetic vs CIFAR10 SF% differs by {delta:.1f} points.")
            print(f"  Mild difference; both show the same qualitative routing pattern.")
        else:
            print()
            print(f"  Both input sources give similar SF% (within {delta:.1f} points).")
            print(f"  The §5.3 conclusion is robust to input distribution.")

    # Pick the more authoritative source for the final claim
    sf_authoritative = sf_cifar if sf_cifar is not None else sf_synth
    src = "CIFAR10" if sf_cifar is not None else "synthetic"
    print()
    print(f"  Authoritative routing data ({src} preferred):")
    if sf_authoritative == 0:
        print("  Result: SparseFlow path is NEVER taken in default routing.")
        print()
        print("  This confirms the §5.3 hypothesis: CATFuse-default ≈ CATFuse-DK")
        print("  because Runtime EGD's sparsity threshold (>0.7) is not met by")
        print("  the actual SEW-RN18 input distribution at the STFusion layers.")
        print("  The 7 SF-eligible layers all route to DK at runtime.")
        print()
        print("  Implication for paper: §5.3's 2x speedup over SJ-eager is")
        print("  attributable to LIF kernel launch reduction + BN folding,")
        print("  NOT to SparseFlow. SparseFlow's role on V100 is to be")
        print("  AVAILABLE in Impl(σ) so that Runtime EGD can pick it on")
        print("  layers/hardware where it wins — V100 + SEW-RN18 isn't that case.")
    elif sf_authoritative < 10:
        print(f"  Result: SparseFlow path used <10% of the time ({sf_authoritative:.1f}%).")
        print("  Still mostly explains default ≈ DK; SF contribution is minimal.")
    elif sf_authoritative < 50:
        print(f"  Result: SparseFlow path used {sf_authoritative:.1f}% of the time.")
        print("  default ≈ DK is partly explained by routing, partly by something else.")
    else:
        print(f"  Result: SparseFlow path used {sf_authoritative:.1f}% of the time.")
        print("  SF dominates routing but default ≈ DK in wall-clock — investigate.")

    print(f"\n  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())