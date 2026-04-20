#!/usr/bin/env python
"""
Universal CATFuse benchmark for trained checkpoints.
Supports: SEW-ResNet18, SpikingResNet18, SpikingVGG11_bn.
Auto-selects build function from checkpoint's 'model_name' field,
or from --model flag.

Usage:
    python phaseC_bench_universal.py \
        --checkpoint checkpoints/spiking_resnet18_cifar10_lif_best.pth \
        --data-path data/cifar10 \
        --model spiking_resnet18 \
        -b 32 --T 4 \
        --bench-iters 100 --warmup 20 --repeats 5
"""
import argparse
import copy
import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore', message='.*Applied workaround for CuDNN.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.conv')
logging.getLogger().setLevel(logging.ERROR)
for name in ['root', 'spikingjelly']:
    logging.getLogger(name).setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from catfuse_substitute import substitute


# ============================================================
# Model builders (must match training scripts exactly)
# ============================================================
def build_sew_resnet18(num_classes=10, v_threshold=1.0, tau=2.0, cifar10_stem=True):
    model = sew_resnet.sew_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=num_classes,
        cnf='ADD', v_threshold=v_threshold, tau=tau,
    )
    if cifar10_stem:
        model.conv1 = layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    functional.set_step_mode(model, 'm')
    return model


def build_spiking_resnet18(num_classes=10, v_threshold=1.0, tau=2.0, cifar10_stem=True):
    model = spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=num_classes,
        v_threshold=v_threshold, tau=tau,
    )
    if cifar10_stem:
        model.conv1 = layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    functional.set_step_mode(model, 'm')
    return model


def build_vgg11(num_classes=10, v_threshold=1.0, tau=2.0, **kwargs):
    model = spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True, num_classes=num_classes,
        v_threshold=v_threshold, tau=tau,
    )
    functional.set_step_mode(model, 'm')
    return model


MODEL_BUILDERS = {
    'sew_resnet18': build_sew_resnet18,
    'spiking_resnet18': build_spiking_resnet18,
    'spiking_vgg11_bn': build_vgg11,
    'vgg11': build_vgg11,
}


def get_test_loader(data_path, batch_size, num_workers=4):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.CIFAR10(data_path, train=False, transform=tfm, download=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def encode(x, T):
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


# ============================================================
# Eval helpers
# ============================================================
@torch.no_grad()
def evaluate_accuracy(net, loader, T, device, label='backend'):
    net.eval()
    total, correct = 0, 0
    all_preds = []
    for img, lbl in loader:
        img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        out = net(encode(img, T)).mean(dim=0)
        functional.reset_net(net)
        pred = out.argmax(1)
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
        all_preds.append(pred.cpu())
    acc = correct / total
    print(f"  [{label:10s}] accuracy: {100*acc:.4f}% ({correct}/{total})", flush=True)
    return acc, torch.cat(all_preds)


@torch.no_grad()
def benchmark_wallclock(net, loader, T, device, label, n_iters, n_warmup, n_repeats):
    net.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device, non_blocking=True)
    for _ in range(n_warmup):
        _ = net(encode(batch, T)).mean(dim=0)
        functional.reset_net(net)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            _ = net(encode(batch, T)).mean(dim=0)
            functional.reset_net(net)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    ts = sorted(times)
    trimmed = ts[1:-1] if len(ts) >= 3 else ts
    mean_ms = sum(trimmed) / len(trimmed)
    std_ms = (sum((t - mean_ms)**2 for t in trimmed) / max(len(trimmed), 1))**0.5
    raw = '[' + ', '.join(f'{t:.3f}' for t in times) + ']'
    print(f"  [{label:10s}] wall-clock: {mean_ms:6.3f} ± {std_ms:.4f} ms  raw={raw}", flush=True)
    return mean_ms, std_ms


@torch.no_grad()
def first_batch_parity(net_a, net_b, loader, T, device, na, nb):
    img, _ = next(iter(loader))
    img = img.to(device, non_blocking=True)
    x = encode(img, T)
    oa = net_a(x); functional.reset_net(net_a)
    ob = net_b(x); functional.reset_net(net_b)
    d = (oa - ob).abs()
    eq = torch.equal(oa, ob)
    print(f"  [parity {na:10s} vs {nb:10s}] bit-exact={eq}, "
          f"max_diff={d.max().item():.2e}, mean_diff={d.mean().item():.2e}", flush=True)
    return eq, d.max().item()


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-path', required=True)
    p.add_argument('--model', type=str, default=None,
                   help='Model name override. Auto-detected from checkpoint if omitted.')
    p.add_argument('-b', '--batch-size', type=int, default=32)
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--bench-iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda:0')
    args = p.parse_args()

    print(f"=== Universal CATFuse Bench ===")
    print(f"Args: {vars(args)}")
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    device = torch.device(args.device)
    loader = get_test_loader(args.data_path, args.batch_size)
    n_test = len(loader.dataset)
    print(f"Test: {n_test} samples, batch={args.batch_size}", flush=True)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ck = ckpt.get('args', {})
    model_name = args.model or ckpt.get('model_name', None)
    if model_name is None:
        print("ERROR: checkpoint has no 'model_name' and --model not given.")
        print("Use --model sew_resnet18 / spiking_resnet18 / vgg11")
        sys.exit(1)

    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        print(f"ERROR: unknown model '{model_name}'. Choose from: {list(MODEL_BUILDERS.keys())}")
        sys.exit(1)

    vth = ck.get('v_threshold', 1.0)
    tau = ck.get('tau', 2.0)
    stem = ck.get('cifar10_stem', model_name != 'spiking_vgg11_bn' and model_name != 'vgg11')
    ck_T = ck.get('T', args.T)

    print(f"\n--- Checkpoint: {args.checkpoint} ---")
    print(f"  model={model_name}, epoch={ckpt.get('epoch','?')}, "
          f"acc={100*ckpt.get('test_acc',0):.2f}%")
    print(f"  v_th={vth}, tau={tau}, cifar10_stem={stem}, T={ck_T}", flush=True)

    # --- Build 3 backends ---
    build_kwargs = dict(num_classes=10, v_threshold=vth, tau=tau)
    if model_name not in ('spiking_vgg11_bn', 'vgg11'):
        build_kwargs['cifar10_stem'] = stem

    def load_model():
        m = builder(**build_kwargs).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        return m

    print(f"\n--- Building backends ---", flush=True)
    net_torch = load_model()
    functional.set_backend(net_torch, 'torch', instance=neuron.LIFNode)
    print(f"  [SJ torch  ] OK")

    net_cupy = load_model()
    functional.set_backend(net_cupy, 'cupy', instance=neuron.LIFNode)
    print(f"  [SJ cupy   ] OK")

    net_ctf_base = load_model()
    net_ctf, cov = substitute(net_ctf_base, verbose=False)
    net_ctf = net_ctf.to(device)
    fused = cov.get('fused_lif_nodes', '?')
    total_lif = cov.get('total_lif_nodes', '?')
    pct = cov.get('coverage_pct', 0)
    print(f"  [CATFuse   ] OK, coverage: {fused}/{total_lif} = {pct:.1f}%")
    if 'patterns_matched' in cov:
        print(f"               patterns: {cov['patterns_matched']}")

    # --- Step 1: Accuracy ---
    print(f"\n=== Step 1: Accuracy ===", flush=True)
    a_t, p_t = evaluate_accuracy(net_torch, loader, args.T, device, 'SJ torch')
    a_c, p_c = evaluate_accuracy(net_cupy, loader, args.T, device, 'SJ cupy')
    a_f, p_f = evaluate_accuracy(net_ctf, loader, args.T, device, 'CATFuse')

    ag_tc = (p_t == p_c).float().mean().item()
    ag_tf = (p_t == p_f).float().mean().item()
    print(f"\n  Agreement: torch-cupy {100*ag_tc:.4f}%, torch-CTF {100*ag_tf:.4f}%")

    # --- Step 2: Parity ---
    print(f"\n=== Step 2: First-batch parity ===", flush=True)
    first_batch_parity(net_torch, net_ctf, loader, args.T, device, 'SJ torch', 'CATFuse')
    first_batch_parity(net_torch, net_cupy, loader, args.T, device, 'SJ torch', 'SJ cupy')

    # --- Step 3: Wall-clock ---
    print(f"\n=== Step 3: Wall-clock (B={args.batch_size}, T={args.T}, "
          f"{args.bench_iters}×{args.repeats}, {args.warmup} warmup) ===", flush=True)
    t_t, s_t = benchmark_wallclock(net_torch, loader, args.T, device, 'SJ torch',
                                    args.bench_iters, args.warmup, args.repeats)
    t_c, s_c = benchmark_wallclock(net_cupy, loader, args.T, device, 'SJ cupy',
                                    args.bench_iters, args.warmup, args.repeats)
    t_f, s_f = benchmark_wallclock(net_ctf, loader, args.T, device, 'CATFuse',
                                    args.bench_iters, args.warmup, args.repeats)

    # --- Summary ---
    print(f"\n{'='*72}")
    print(f"=== SUMMARY ===")
    print(f"{'='*72}")
    print(f"Model: {model_name} (v_th={vth}, tau={tau}, T={ck_T})")
    print(f"Trained acc: {100*ckpt.get('test_acc',0):.2f}% @ epoch {ckpt.get('epoch','?')}")
    print(f"Hardware: {torch.cuda.get_device_name(device)}")
    print(f"Coverage: {fused}/{total_lif} = {pct:.1f}%")
    print(f"")
    print(f"Backend   | Test acc   | Wall-clock (ms)    | Top-1 agree vs torch")
    print(f"----------|------------|--------------------|---------------------")
    print(f"SJ torch  | {100*a_t:6.3f}%   | {t_t:6.3f} ± {s_t:.3f}    | (reference)")
    print(f"SJ cupy   | {100*a_c:6.3f}%   | {t_c:6.3f} ± {s_c:.3f}    | {100*ag_tc:.4f}%")
    print(f"CATFuse   | {100*a_f:6.3f}%   | {t_f:6.3f} ± {s_f:.3f}    | {100*ag_tf:.4f}%")
    print(f"")
    print(f"Speedup CATFuse vs SJ torch: {t_t/t_f:.3f}x")
    print(f"Speedup CATFuse vs SJ cupy:  {t_c/t_f:.3f}x")
    delta = (a_f - a_t) * 100
    n_diff = int(round(abs(a_f - a_t) * n_test))
    print(f"Acc delta CTF vs torch: {delta:+.4f} pp ({n_diff}/{n_test} images)")
    print(f"\nEnd: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()