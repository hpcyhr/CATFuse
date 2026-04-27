"""§5.3 routing observability — corrected version (v2).

Supersedes exp_32. The previous version used the dead-chain network
(stem mismatch + ckpt unwrap bug), so all sparsity numbers were
artifacts of all-zero spikes. This v2:

  - uses experiments._helpers.build_sew_rn18_cifar10() (correct stem +
    ckpt unwrap)
  - measures routing on CIFAR10 test-set images (real input distribution)
  - reports per-STFusion-layer routing breakdown:
      * n_calls       : total invocations across all batches
      * n_sf          : went to SparseFlow path (sparsity > 0.7 and nnz > 0)
      * n_dk          : went to DenseKeep path (sparsity < 0.7 in lean path)
      * n_static_zero : skipped via StaticZero short-circuit (nnz == 0)
      * sparsity_*    : observed input sparsity statistics

Run:
    cd /path/to/CATFuse
    python experiments/exp_42_routing_observability_v2.py
    # default --data-path /data/yhr/datasets/cifar10
    # --n-batches limits how many batches to scan (default 20 = 2560 samples)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from spikingjelly.activation_based import functional, neuron

from catfuse.substitute import substitute_sf
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from experiments._helpers import build_sew_rn18_cifar10


DEVICE = "cuda:0"
T = 4
DEFAULT_DATA_PATH = "/data/yhr/datasets/cifar10"
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_BATCHES = 20    # 20 * 128 = 2560 samples (~25% of test set)

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/routing_observability_v2.csv")


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


def install_observation_hook(fused_net) -> tuple:
    """Monkey-patch every STFusion's _batchfold_forward to log routing.

    We patch _batchfold_forward instead of forward because the dispatch
    is INSIDE forward — the choice happens between _batchfold_forward
    (sparse path) and _dense_forward (dense path).

    Reading STFusion.forward source: it computes nnz/sparsity, then either
    calls _batchfold_forward (which goes through SF / StaticZero) or
    _dense_forward (DK). We hook the dispatch by wrapping forward itself.
    """
    stats: Dict[str, LayerStats] = {}
    name_by_id = {}

    for full_name, mod in fused_net.named_modules():
        if isinstance(mod, STFusionConvBNLIF):
            name_by_id[id(mod)] = full_name
            stats[full_name] = LayerStats(
                layer_name=full_name,
                Cin=mod.in_channels, Cout=mod.out_channels,
                H=0, W=0,
            )

    original_forward = STFusionConvBNLIF.forward

    def patched_forward(self, x: torch.Tensor):
        name = name_by_id.get(id(self), "<unknown>")
        s = stats.get(name)

        # Match the dispatch logic in STFusionConvBNLIF.forward exactly:
        # - if nnz == 0: StaticZero (path = static_zero)
        # - elif sparsity > 0.7: SF (path = sparseflow)
        # - else: DK (path = dense_forward)
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


def get_test_loader(data_path: str, batch_size: int):
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=False)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def encode(x: torch.Tensor, T: int) -> torch.Tensor:
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES,
                        help=f"# of batches to scan (default {DEFAULT_N_BATCHES})")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1
    if not os.path.exists(args.data_path):
        print(f"FATAL: CIFAR10 not found at {args.data_path}")
        return 1

    print("=" * 96)
    print("§5.3 routing observability v2 — CATFuse-default on CIFAR10 test images")
    print("=" * 96)
    print()
    print(f"  T={T}  batch_size={args.batch_size}  "
          f"n_batches={args.n_batches}  "
          f"(={args.n_batches * args.batch_size} samples)")
    print()

    sj_net, ckpt_note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    print(f"  Checkpoint: {ckpt_note}")

    # Build CATFuse default model
    fused, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused = fused.to(DEVICE).eval()

    n_st = sum(1 for m in fused.modules() if isinstance(m, STFusionConvBNLIF))
    print(f"  STFusion layers in network: {n_st}")
    print()

    test_loader = get_test_loader(args.data_path, args.batch_size)

    # Install hooks
    stats, original_forward = install_observation_hook(fused)

    # Run on first n_batches batches
    print(f"  Running {args.n_batches} batches...")
    with torch.no_grad():
        for batch_idx, (img, _label) in enumerate(test_loader):
            if batch_idx >= args.n_batches:
                break
            img = img.to(DEVICE, non_blocking=True)
            x = encode(img, T)
            functional.reset_net(fused)
            _ = fused(x)
    torch.cuda.synchronize()

    restore_forward(original_forward)

    # Per-layer table
    print()
    print("=" * 96)
    print("Per-layer routing distribution (default CATFuse on CIFAR10 real images)")
    print("=" * 96)
    print(f"  {'layer':<35} {'shape':<18} {'calls':>6} "
          f"{'SF':>5} {'DK':>5} {'StZ':>5} {'SF%':>7} "
          f"{'sparsity (min/mean/max)':<32}")
    print(f"  {'-'*35} {'-'*18} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*7} "
          f"{'-'*32}")

    rows = []
    total_calls = 0
    total_sf = 0
    total_dk = 0
    total_sz = 0
    for name in sorted(stats.keys()):
        s = stats[name]
        if s.n_calls == 0:
            print(f"  {name:<35} (unused)")
            continue
        shape = f"{s.Cin}->{s.Cout} {s.H}x{s.W}"
        sp_str = (f"{s.sparsity_min:.3f} / {s.sparsity_mean:.3f} / "
                  f"{s.sparsity_max:.3f}")
        print(f"  {name:<35} {shape:<18} {s.n_calls:>6} "
              f"{s.n_sf:>5} {s.n_dk:>5} {s.n_static_zero:>5} "
              f"{100*s.sf_fraction:>6.1f}% {sp_str:<32}")
        rows.append(s)
        total_calls += s.n_calls
        total_sf += s.n_sf
        total_dk += s.n_dk
        total_sz += s.n_static_zero

    print(f"  {'-'*35} {'-'*18} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
    sf_pct = 100 * total_sf / max(total_calls, 1)
    dk_pct = 100 * total_dk / max(total_calls, 1)
    sz_pct = 100 * total_sz / max(total_calls, 1)
    print(f"  {'TOTAL':<35} {'':<18} {total_calls:>6} "
          f"{total_sf:>5} {total_dk:>5} {total_sz:>5}  "
          f"({sf_pct:.1f}% SF / {dk_pct:.1f}% DK / {sz_pct:.1f}% StZ)")

    # CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if rows:
        with open(CSV_PATH, "w", newline="") as f:
            fieldnames = ["layer_name", "Cin", "Cout", "H", "W",
                          "n_calls", "n_sf", "n_dk", "n_static_zero",
                          "sparsity_min", "sparsity_max", "sparsity_mean",
                          "sf_fraction"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in rows:
                writer.writerow({
                    "layer_name": s.layer_name,
                    "Cin": s.Cin, "Cout": s.Cout, "H": s.H, "W": s.W,
                    "n_calls": s.n_calls, "n_sf": s.n_sf, "n_dk": s.n_dk,
                    "n_static_zero": s.n_static_zero,
                    "sparsity_min": s.sparsity_min,
                    "sparsity_max": s.sparsity_max,
                    "sparsity_mean": s.sparsity_mean,
                    "sf_fraction": s.sf_fraction,
                })

    # Interpretation
    print()
    print("=" * 96)
    print("Interpretation")
    print("=" * 96)
    if total_sf == 0:
        print("  Result: SparseFlow path is NEVER taken on real CIFAR10 input.")
        print("  This contradicts the §3.10 single-layer K-sweep data, which")
        print("  used 85% sparse synthetic input. Real CIFAR10 → trained ckpt")
        print("  produces lower-sparsity spike patterns at deeper layers,")
        print("  routing those layers to DK path even when STFusionConvBNLIF.")
    elif sf_pct >= 90:
        print(f"  Result: SparseFlow path used {sf_pct:.1f}% of the time —")
        print("  most STFusion layers reliably hit the SF route at >0.7 sparsity.")
        print("  The default-vs-DK wall-clock gap measured in §5.3 IS due to")
        print("  SparseFlow being slower than DenseKeep on V100 in real")
        print("  workload conditions.")
    elif sf_pct >= 50:
        print(f"  Result: SparseFlow path used {sf_pct:.1f}% of the time.")
        print("  Mixed routing — some layers/inputs go SF, others fall back to DK.")
        print("  Default's wall-clock gap vs DK is partial.")
    elif sf_pct < 10:
        print(f"  Result: SparseFlow path used <10% of the time ({sf_pct:.1f}%).")
        print("  Most STFusion layers fall back to DK at runtime. Default ≈ DK")
        print("  in wall-clock — confirmed by §5.3 measurements.")
    else:
        print(f"  Result: SparseFlow path used {sf_pct:.1f}% of the time.")
        print("  Mixed routing in middle range.")

    print(f"  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())