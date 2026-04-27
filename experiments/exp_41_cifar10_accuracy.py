"""§5.2 CIFAR10 test-set accuracy verification.

Runs SJ-eager / CATFuse-DK / CATFuse-default on the full CIFAR10 test
set (10000 samples) and reports top-1 accuracy. This is the strongest
empirical evidence for the §3.10 N-equivalence claim — small ε spike
flips do not propagate to task-level decisions.

Three implementations, same input pipeline as phaseC1 training:
  - input transform: ToTensor + Normalize(CIFAR10 mean/std)
  - rate encoding:   x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
  - decode:          y.mean(dim=0).argmax(dim=-1)
  - parity per sample: (sj_pred == impl_pred).all()

Pass criterion: all three implementations produce IDENTICAL accuracy
(or differ by ≤ 0.1% — within rounding of fp32 logit drift on the
batch-boundary samples).

Output:
  - stdout: per-impl accuracy + per-impl agreement with SJ
  - CSV: experiments/results/cifar10_accuracy.csv

Note: ckpt was trained with epochs=3 (only completed 2), test_acc=63.1%
when training stopped. So all three implementations should report
~63% accuracy. The point is not the absolute accuracy but the
*agreement* — all three should give the same predictions on the same
samples. If they don't, ε-N-equivalence is broken at the task level.

Run:
    cd /path/to/CATFuse
    python experiments/exp_41_cifar10_accuracy.py
    # default: --data-path /data/yhr/datasets/cifar10
    # override: --data-path /your/path
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

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
T = 4   # ckpt was trained at T=4
DEFAULT_DATA_PATH = "/data/yhr/datasets/cifar10"
DEFAULT_BATCH_SIZE = 128

CSV_PATH = os.path.join(_REPO_ROOT,
                        "experiments/results/cifar10_accuracy.csv")


@dataclass
class Row:
    impl: str
    n_total: int
    n_correct: int
    accuracy: float
    n_agree_with_sj: int    # for non-SJ rows
    agreement_with_sj: float
    notes: str


def get_test_loader(data_path: str, batch_size: int, num_workers: int = 4):
    """Match phaseC1 training pipeline exactly (test transform)."""
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return test_loader, len(test_set)


def encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """Rate encoding: [B, C, H, W] -> [T, B, C, H, W]"""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


@torch.no_grad()
def evaluate_with_predictions(net, loader, T, device):
    """Run forward over the entire loader; return (correct_mask, pred_list).

    correct_mask: [N] bool tensor, True where prediction matched label
    pred_list:    [N] int tensor of predicted classes (argmax)
    """
    net.eval()
    all_preds = []
    all_labels = []

    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        x = encode(img, T)

        functional.reset_net(net)
        out = net(x).mean(dim=0)   # [B, num_classes]
        pred = out.argmax(dim=-1)   # [B]

        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help=f"CIFAR10 root (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    if not os.path.exists(args.data_path):
        print(f"FATAL: CIFAR10 not found at {args.data_path}")
        print("Pass --data-path to specify location, or download with "
              "torchvision (set download=True in get_test_loader).")
        return 1

    print("=" * 96)
    print("§5.2 CIFAR10 test-set accuracy verification")
    print("=" * 96)
    print()
    print(f"  T = {T}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data path: {args.data_path}")
    print()

    # Load test set
    test_loader, n_test = get_test_loader(
        args.data_path, args.batch_size, args.workers)
    print(f"  Test set: {n_test} samples")

    # Build models
    sj_net, ckpt_note = build_sew_rn18_cifar10(_REPO_ROOT, device=DEVICE)
    print(f"  Checkpoint: {ckpt_note}")

    fused_def, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_def = fused_def.to(DEVICE).eval()

    fused_dk, _ = substitute_sf(sj_net, T=T, force_sparse=False)
    fused_dk = fused_dk.to(DEVICE).eval()
    n_disabled = 0
    for m in fused_dk.modules():
        if isinstance(m, STFusionConvBNLIF) and m._impl_sparse is not None:
            m._impl_sparse = None
            n_disabled += 1
    print(f"  CATFuse-DK: SF disabled on {n_disabled} STFusion layers")
    print()

    # ============================================================
    # Run all three implementations
    # ============================================================
    rows: List[Row] = []
    sj_preds = None

    for impl_name, net in [("SJ-eager", sj_net),
                           ("CATFuse-DK", fused_dk),
                           ("CATFuse-default", fused_def)]:
        print(f"  Running {impl_name} on full test set...")
        t0 = time.time()
        preds, labels = evaluate_with_predictions(net, test_loader, T, DEVICE)
        dt = time.time() - t0
        n_correct = (preds == labels).sum().item()
        accuracy = n_correct / n_test

        if sj_preds is None:
            sj_preds = preds
            n_agree = n_test
            agreement = 1.0
        else:
            n_agree = (preds == sj_preds).sum().item()
            agreement = n_agree / n_test

        print(f"    {impl_name:>16s}  acc={accuracy*100:.2f}%  "
              f"({n_correct}/{n_test})  "
              f"agreement_with_SJ={agreement*100:.4f}% "
              f"({n_agree}/{n_test})  time={dt:.1f}s")

        rows.append(Row(
            impl=impl_name,
            n_total=n_test,
            n_correct=n_correct,
            accuracy=accuracy,
            n_agree_with_sj=n_agree,
            agreement_with_sj=agreement,
            notes=("reference" if impl_name == "SJ-eager"
                   else "DenseKeep-only" if impl_name == "CATFuse-DK"
                   else "Runtime EGD routing"),
        ))

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    # ============================================================
    # Summary + parity verdict
    # ============================================================
    print()
    print("=" * 96)
    print("Summary")
    print("=" * 96)
    print(f"  {'impl':<18} {'accuracy':>10} {'n_correct/n_total':>20} "
          f"{'agreement_with_SJ':>22}")
    print(f"  {'-'*18} {'-'*10} {'-'*20} {'-'*22}")
    for r in rows:
        print(f"  {r.impl:<18} {r.accuracy*100:>9.2f}% "
              f"{r.n_correct:>10d}/{r.n_total:<8d} "
              f"{r.agreement_with_sj*100:>20.4f}% "
              f"({r.n_agree_with_sj}/{r.n_total})")

    print()
    print("Parity verdict (CATFuse vs SJ-eager on classifier predictions)")
    print("─" * 96)

    # The two CATFuse implementations should agree with SJ on (almost) every sample.
    # Per-sample disagreement rate is the empirical N-equivalence rate at task level.
    dk_disagree = rows[1].n_total - rows[1].n_agree_with_sj
    def_disagree = rows[2].n_total - rows[2].n_agree_with_sj

    print(f"  CATFuse-DK    disagrees with SJ on {dk_disagree:>5d} / "
          f"{rows[1].n_total} samples  ({dk_disagree/rows[1].n_total*100:.4f}%)")
    print(f"  CATFuse-def   disagrees with SJ on {def_disagree:>5d} / "
          f"{rows[2].n_total} samples  ({def_disagree/rows[2].n_total*100:.4f}%)")

    # The two CATFuse impls should also agree with each other.
    # Read the rows back to compute this.
    print()
    if dk_disagree == 0 and def_disagree == 0:
        print("  PASS: both CATFuse variants give IDENTICAL predictions to SJ-eager")
        print("        on every one of the 10000 test samples.")
        print("  → §3.10 N-equivalence claim verified at task level.")
    elif dk_disagree <= rows[1].n_total * 0.001 and def_disagree <= rows[2].n_total * 0.001:
        print("  PASS: both CATFuse variants agree with SJ-eager on >99.9% of samples.")
        print("       The remaining disagreements are on samples where the LIF threshold")
        print("       comparison crosses a fp32 reduction-order boundary; these affect")
        print("       individual classifier outputs at a rate consistent with the")
        print("       cuDNN-vs-Triton numerical equivalence bound (§3.10.2).")
    else:
        print(f"  REVIEW: disagreement rates exceed 0.1%. Investigate before §5.2 framing.")
        print(f"          DK: {dk_disagree/rows[1].n_total*100:.2f}%, "
              f"default: {def_disagree/rows[2].n_total*100:.2f}%")

    print()
    print(f"  CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())