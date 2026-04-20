#!/usr/bin/env python
"""
Phase C-1 — Benchmark trained CIFAR10 SEW-ResNet18 (LIFNode) checkpoint
            across SJ torch / SJ cupy / CATFuse.

Rigorously uses:
  - phaseC1_train_sew_cifar10.build_sew_model (same build as training)
  - catfuse_substitute.substitute(model, verbose=False) -> (ctf_model, coverage)
  - Consistent v_threshold and tau from the checkpoint args

Outputs:
  - Test set accuracy on each backend
  - Wall-clock per inference (CUDA event timing, trimmed mean)
  - Top-1 decision agreement (image-wise)
  - First-batch spike/logit parity
  - Coverage report

Paste stdout into chat for paper §V-G drafting.

Usage:
    cd /data/yhr/CATFuse/
    conda activate snn118
    python phaseC1_bench.py \
        --checkpoint checkpoints/sew_resnet18_cifar10_lif_best.pth \
        --data-path /data/yhr/datasets/cifar10 \
        -b 32 --T 4 \
        --bench-iters 100 --warmup 20 --repeats 5 \
        2>&1 | tee logs/phaseC1_bench_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import copy
import datetime
import logging
import os
import sys
import warnings
from pathlib import Path

# Silence same warnings as §V bench
warnings.filterwarnings('ignore', message='.*Applied workaround for CuDNN.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.conv')
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['root', 'spikingjelly']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

# Add project root to sys.path so we can import catfuse_substitute
_PROJECT_ROOT = '/data/yhr/CATFuse'
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from catfuse_substitute import substitute  # returns (model, coverage_dict)


# ============================================================
# Same build as training script — MUST use layer.Conv2d (SJ multi-step aware)
# ============================================================
def build_sew_model(num_classes: int = 10,
                    v_threshold: float = 1.0,
                    tau: float = 2.0,
                    cifar10_stem: bool = True):
    model = sew_resnet.sew_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        num_classes=num_classes,
        cnf='ADD',
        v_threshold=v_threshold,
        tau=tau,
    )
    if cifar10_stem:
        # Use SJ layer.Conv2d (multi-step aware), not torch.nn.Conv2d
        model.conv1 = layer.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        model.maxpool = nn.Identity()
    functional.set_step_mode(model, 'm')
    return model


def get_test_loader(data_path: str, batch_size: int, num_workers: int = 4):
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=False)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def encode(x: torch.Tensor, T: int) -> torch.Tensor:
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


# ============================================================
# Evaluation helpers
# ============================================================
@torch.no_grad()
def evaluate_accuracy(net, loader, T, device, label='backend'):
    net.eval()
    total, correct = 0, 0
    all_preds = []
    for img, lbl in loader:
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        out = net(encode(img, T)).mean(dim=0)
        functional.reset_net(net)
        pred = out.argmax(1)
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
        all_preds.append(pred.cpu())
    acc = correct / total
    print(f"  [{label:10s}] test accuracy: {100*acc:.4f}% ({correct}/{total})",
          flush=True)
    return acc, torch.cat(all_preds)


@torch.no_grad()
def benchmark_wallclock(net, loader, T, device, label, n_iters, n_warmup, n_repeats):
    """Per-batch wall-clock via CUDA events. Trimmed mean across repeats
    (drop max + min if >= 3 repeats)."""
    net.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device, non_blocking=True)

    # Warmup
    for _ in range(n_warmup):
        _ = net(encode(batch, T)).mean(dim=0)
        functional.reset_net(net)
    torch.cuda.synchronize()

    # Timed repeats
    repeat_times = []
    for r in range(n_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = net(encode(batch, T)).mean(dim=0)
            functional.reset_net(net)
        end.record()
        torch.cuda.synchronize()
        per_iter_ms = start.elapsed_time(end) / n_iters
        repeat_times.append(per_iter_ms)

    rts = sorted(repeat_times)
    trimmed = rts[1:-1] if len(rts) >= 3 else rts
    mean_ms = sum(trimmed) / len(trimmed)
    std_ms = (sum((t - mean_ms) ** 2 for t in trimmed) / len(trimmed)) ** 0.5
    raw_str = '[' + ', '.join(f'{t:.3f}' for t in repeat_times) + ']'
    print(f"  [{label:10s}] wall-clock: {mean_ms:6.3f} ± {std_ms:.4f} ms/iter "
          f"(trim mean, raw={raw_str})", flush=True)
    return mean_ms, std_ms


@torch.no_grad()
def first_batch_parity(net_a, net_b, loader, T, device, name_a, name_b):
    img, _ = next(iter(loader))
    img = img.to(device, non_blocking=True)
    x = encode(img, T)

    out_a = net_a(x)
    functional.reset_net(net_a)
    out_b = net_b(x)
    functional.reset_net(net_b)

    diff = (out_a - out_b).abs()
    bit_exact = torch.equal(out_a, out_b)
    print(f"  [parity {name_a:10s} vs {name_b:10s}] bit-exact: {bit_exact}, "
          f"max_abs_diff: {diff.max().item():.2e}, "
          f"mean_abs_diff: {diff.mean().item():.2e}", flush=True)
    return bit_exact, diff.max().item()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--bench-iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(f"=== Phase C-1 Bench: trained checkpoint validation ===")
    print(f"Args: {vars(args)}")
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    device = torch.device(args.device)
    test_loader = get_test_loader(args.data_path, args.batch_size)
    print(f"Test set: {len(test_loader.dataset)} samples, "
          f"batch={args.batch_size}", flush=True)

    # ---------- Load checkpoint ----------
    print(f"\n--- Loading checkpoint: {args.checkpoint} ---", flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ck_args = ckpt.get('args', {})
    ck_vth = ck_args.get('v_threshold', 1.0)
    ck_tau = ck_args.get('tau', 2.0)
    ck_stem = ck_args.get('cifar10_stem', True)
    ck_T = ck_args.get('T', args.T)
    print(f"  trained at epoch {ckpt.get('epoch', 'N/A')}, "
          f"reported test_acc {100*ckpt.get('test_acc', 0):.2f}%")
    print(f"  checkpoint args: v_th={ck_vth}, tau={ck_tau}, "
          f"cifar10_stem={ck_stem}, T={ck_T}")
    if ck_T != args.T:
        print(f"  WARNING: bench T={args.T} differs from train T={ck_T}")

    # ---------- Build 3 backends ----------
    def load_and_build():
        m = build_sew_model(num_classes=10, v_threshold=ck_vth,
                            tau=ck_tau, cifar10_stem=ck_stem).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        return m

    print(f"\n--- Building 3 backends ---", flush=True)
    net_torch = load_and_build()
    functional.set_backend(net_torch, 'torch', instance=neuron.LIFNode)
    print(f"  [SJ torch  ] built")

    net_cupy = load_and_build()
    functional.set_backend(net_cupy, 'cupy', instance=neuron.LIFNode)
    print(f"  [SJ cupy   ] built")

    net_catfuse_base = load_and_build()
    # Run substitute() on torch backend (same as §V does)
    net_catfuse, coverage = substitute(net_catfuse_base, verbose=False)
    net_catfuse = net_catfuse.to(device)
    print(f"  [CATFuse   ] built, coverage: "
          f"{coverage.get('fused_lif_nodes', '?')}/{coverage.get('total_lif_nodes', '?')} "
          f"= {coverage.get('coverage_pct', 0):.1f}%")
    if 'patterns_matched' in coverage:
        print(f"              patterns: {coverage['patterns_matched']}")

    # ---------- Step 1: Accuracy ----------
    print(f"\n=== Step 1: Test set accuracy ===", flush=True)
    acc_torch, preds_torch = evaluate_accuracy(net_torch, test_loader, args.T,
                                                device, 'SJ torch')
    acc_cupy, preds_cupy = evaluate_accuracy(net_cupy, test_loader, args.T,
                                              device, 'SJ cupy')
    acc_catfuse, preds_catfuse = evaluate_accuracy(net_catfuse, test_loader,
                                                    args.T, device, 'CATFuse')

    agree_tc = (preds_torch == preds_cupy).float().mean().item()
    agree_tf = (preds_torch == preds_catfuse).float().mean().item()
    agree_cf = (preds_cupy == preds_catfuse).float().mean().item()
    print(f"\n  Top-1 decision agreement:")
    print(f"    torch    vs cupy    : {100*agree_tc:.4f}%")
    print(f"    torch    vs CATFuse : {100*agree_tf:.4f}%")
    print(f"    cupy     vs CATFuse : {100*agree_cf:.4f}%")

    # ---------- Step 2: First batch parity ----------
    print(f"\n=== Step 2: First batch parity (logit-level) ===", flush=True)
    first_batch_parity(net_torch, net_catfuse, test_loader, args.T, device,
                       'SJ torch', 'CATFuse')
    first_batch_parity(net_cupy, net_catfuse, test_loader, args.T, device,
                       'SJ cupy', 'CATFuse')
    first_batch_parity(net_torch, net_cupy, test_loader, args.T, device,
                       'SJ torch', 'SJ cupy')

    # ---------- Step 3: Wall-clock ----------
    print(f"\n=== Step 3: Wall-clock "
          f"(B={args.batch_size}, T={args.T}, "
          f"{args.bench_iters} iters × {args.repeats} repeats, "
          f"{args.warmup} warmup) ===", flush=True)
    t_torch, sd_torch = benchmark_wallclock(net_torch, test_loader, args.T,
                                             device, 'SJ torch',
                                             args.bench_iters, args.warmup,
                                             args.repeats)
    t_cupy, sd_cupy = benchmark_wallclock(net_cupy, test_loader, args.T,
                                           device, 'SJ cupy',
                                           args.bench_iters, args.warmup,
                                           args.repeats)
    t_catf, sd_catf = benchmark_wallclock(net_catfuse, test_loader, args.T,
                                           device, 'CATFuse',
                                           args.bench_iters, args.warmup,
                                           args.repeats)

    # ---------- Summary ----------
    n_test = len(test_loader.dataset)
    print(f"\n" + "=" * 72)
    print(f"=== SUMMARY (paste to chat) ===")
    print(f"=" * 72)
    print(f"")
    print(f"Architecture: SEW-ResNet18 LIFNode (tau={ck_tau}, v_th={ck_vth}, "
          f"cnf=ADD)")
    print(f"Training: CIFAR10, BPTT, T={ck_T}, cifar10_stem={ck_stem}")
    print(f"Trained test accuracy: {100*ckpt.get('test_acc', 0):.2f}% "
          f"(from checkpoint) at epoch {ckpt.get('epoch', 'N/A')}")
    print(f"Hardware: V100 (sm_70)")
    print(f"CATFuse coverage: "
          f"{coverage.get('fused_lif_nodes', '?')}/"
          f"{coverage.get('total_lif_nodes', '?')} "
          f"= {coverage.get('coverage_pct', 0):.1f}%")
    print(f"")
    print(f"Backend   | Test acc   | Wall-clock (ms)    | Top-1 agree vs torch")
    print(f"----------|------------|--------------------|---------------------")
    print(f"SJ torch  | {100*acc_torch:6.3f}%   | {t_torch:6.3f} ± {sd_torch:.3f}    | (reference)")
    print(f"SJ cupy   | {100*acc_cupy:6.3f}%   | {t_cupy:6.3f} ± {sd_cupy:.3f}    | {100*agree_tc:.4f}%")
    print(f"CATFuse   | {100*acc_catfuse:6.3f}%   | {t_catf:6.3f} ± {sd_catf:.3f}    | {100*agree_tf:.4f}%")
    print(f"")
    print(f"Speedup CATFuse vs SJ torch: {t_torch/t_catf:.3f}x")
    print(f"Speedup CATFuse vs SJ cupy:  {t_cupy/t_catf:.3f}x")
    print(f"Accuracy delta CATFuse vs SJ torch: "
          f"{(acc_catfuse - acc_torch)*100:+.4f} pp "
          f"({int(round(abs(acc_catfuse - acc_torch) * n_test))} of {n_test} images differ)")
    print(f"")
    print(f"End: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()