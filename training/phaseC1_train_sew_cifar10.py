#!/usr/bin/env python
"""
Phase C-1 — Train CIFAR10 SEW-ResNet18 with LIFNode.

Rigorously matches the §V bench_sew_resnet18_e2e.py `build_sew_model` style:
  - sew_resnet.sew_resnet18(spiking_neuron=neuron.LIFNode, ...)
  - cnf='ADD'
  - surrogate_function=surrogate.Sigmoid()
  - detach_reset=True
  - functional.set_step_mode(model, 'm')

Deviations from §V (for trained checkpoint):
  - v_threshold=1.0 (SJ default, production; §V used 0.1 as random-init workaround)
  - tau=2.0 (SJ default, explicit now)
  - CIFAR10 stem adaption: conv1 -> 3x3 stride=1, maxpool -> Identity
    (standard practice for CIFAR10 since 32x32 is too small for 7x7 stride 2)
  - Randomly initialized BN replaced with trained BN (normal init + BPTT training)

Usage:
    cd /data/yhr/CATFuse/
    conda activate snn118
    nohup python phaseC1_train_sew_cifar10.py \
        --data-path /data/yhr/datasets/cifar10 \
        --output-dir checkpoints/ \
        --epochs 200 -b 128 --T 4 --lr 0.1 --amp \
        --cifar10-stem \
        > logs/phaseC1_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Expected on V100: ~2-3 min/epoch with T=4 B=128 AMP -> 7-10h total.
Expected accuracy: 88-92% top-1.
"""
import argparse
import copy
import datetime
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet


# ============================================================
# Model construction — rigorously matches §V build_sew_model
# ============================================================
def build_sew_model(num_classes: int = 10,
                    v_threshold: float = 1.0,
                    tau: float = 2.0,
                    cifar10_stem: bool = True):
    """Build SEWResNet18.

    Matches §V bench_sew_resnet18_e2e.py::build_sew_model exactly in:
      - spiking_neuron=LIFNode
      - surrogate=Sigmoid
      - detach_reset=True
      - cnf='ADD'
      - set_step_mode(model, 'm')

    Differs in:
      - v_threshold=1.0 (production default, vs §V's 0.1 random-init workaround)
      - tau=2.0 explicit (vs §V implicit default)
      - optional CIFAR10 stem adaption (new; §V uses ImageNet stem at H=W=64)

    NOTE on stem: SJ sew_resnet18 uses spikingjelly.activation_based.layer.Conv2d
    (NOT torch.nn.Conv2d) internally, which is a multi-step-aware wrapper that
    reshapes [T, B, C, H, W] -> [T*B, C, H, W] for vendor conv call and back.
    The CIFAR10 stem replacement MUST use the same SJ layer wrapper, otherwise
    under set_step_mode('m') the input arrives as 5D [T,B,C,H,W] and crashes.
    """
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
        # Replace maxpool with identity; nn.Identity is step-mode-agnostic
        # (it just returns input unchanged, same for 4D or 5D tensors)
        model.maxpool = nn.Identity()
    functional.set_step_mode(model, 'm')
    return model


# ============================================================
# Dataloaders
# ============================================================
def get_dataloaders(data_path: str, batch_size: int, num_workers: int = 4):
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        data_path, train=True, transform=train_tfm, download=True)
    test_set = torchvision.datasets.CIFAR10(
        data_path, train=False, transform=test_tfm, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=True)
    return train_loader, test_loader


def encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """Direct (rate) encoding: replicate static image across T time steps.
    [B, C, H, W] -> [T, B, C, H, W]
    """
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(net, loader, criterion, optimizer, scaler, T, device, epoch):
    net.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for batch_idx, (img, label) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            x = encode(img, T)
            out = net(x).mean(dim=0)   # average over T -> [B, num_classes]
            loss = criterion(out, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(net)
        total += label.size(0)
        correct += (out.argmax(1) == label).sum().item()
        loss_sum += loss.item() * label.size(0)

    elapsed = time.time() - t0
    print(f"Epoch {epoch:3d} train: loss={loss_sum/total:.4f} "
          f"acc={100*correct/total:.2f}% time={elapsed:.1f}s", flush=True)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(net, loader, T, device):
    net.eval()
    total, correct = 0, 0
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        x = encode(img, T)
        out = net(x).mean(dim=0)
        functional.reset_net(net)
        total += label.size(0)
        correct += (out.argmax(1) == label).sum().item()
    return correct / total


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v-threshold', type=float, default=1.0)
    parser.add_argument('--cifar10-stem', action='store_true',
                        help='Use CIFAR10 stem (3x3 stride=1 conv1 + no maxpool)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=1,
                        help='evaluate on test set every N epochs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Phase C-1 Training: SEW-ResNet18 on CIFAR10 (LIFNode) ===")
    print(f"Args: {vars(args)}")
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(
        args.data_path, args.batch_size, args.workers)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples", flush=True)

    net = build_sew_model(num_classes=10,
                          v_threshold=args.v_threshold,
                          tau=args.tau,
                          cifar10_stem=args.cifar10_stem).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: SEW-ResNet18 LIFNode (tau={args.tau}, "
          f"v_th={args.v_threshold}, cifar10_stem={args.cifar10_stem}), "
          f"T={args.T}, params={n_params/1e6:.2f}M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(net, train_loader, criterion, optimizer,
                        scaler, args.T, device, epoch)
        scheduler.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            test_acc = evaluate(net, test_loader, args.T, device)
            print(f"Epoch {epoch:3d} test: acc={100*test_acc:.2f}% "
                  f"(best={100*best_acc:.2f}%)", flush=True)
            if test_acc > best_acc:
                best_acc = test_acc
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'args': vars(args),
                }
                save_path = Path(args.output_dir) / 'sew_resnet18_cifar10_lif_best.pth'
                torch.save(ckpt, save_path)
                print(f"           -> saved best checkpoint to {save_path}",
                      flush=True)

    # Always save the final epoch too
    final_ckpt = {
        'epoch': args.epochs - 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': evaluate(net, test_loader, args.T, device),
        'args': vars(args),
    }
    torch.save(final_ckpt, Path(args.output_dir) / 'sew_resnet18_cifar10_lif_final.pth')

    print(f"\n=== Training complete ===")
    print(f"Best test acc: {100*best_acc:.2f}%")
    print(f"Final test acc: {100*final_ckpt['test_acc']:.2f}%")
    print(f"End: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()