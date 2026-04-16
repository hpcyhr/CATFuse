#!/usr/bin/env python
"""
Phase B(2) — script 1: Train SEW-ResNet18 on CIFAR10 with SpikingJelly.

Output: checkpoint at /data/yhr/CATFuse/checkpoints/sew_resnet18_cifar10.pth
Expected accuracy: 90-93% top-1 after 200 epochs on V100.
Expected runtime: 6-10 hours on V100 with batch=64, T=4, AMP enabled.

Usage:
    cd /data/yhr/CATFuse/
    conda activate snn118
    nohup python phaseB2_train_sew_cifar10.py \
        --data-path /data/yhr/datasets/cifar10 \
        --output-dir checkpoints/ \
        --epochs 200 -b 64 --T 4 --lr 0.1 --amp \
        > logs/phaseB2_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Notes:
- Uses T=4 (not T=16) because trained checkpoints in the SEW-ResNet repo use T=4.
  This is the production setting for SEW-ResNet18 SNN training.
- Uses cnf='ADD' (the SEW addition mode that the paper uses).
- Uses MultiStepIFNode (not LIF) per the official SEW-ResNet18 paper.
  This is also what 'sew_resnet18' in spikingjelly.activation_based defaults to.
- AMP enabled to halve memory + ~30% speedup.

After training completes, run phaseB2_bench.py for inference benchmarks.
"""
import argparse
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

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet


def get_dataloaders(data_path: str, batch_size: int, num_workers: int = 4):
    """Standard CIFAR10 train/test loaders with augmentation."""
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
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(T: int, num_classes: int = 10):
    """Build SEW-ResNet18 with multi-step IF neurons.

    SEW-ResNet18 is designed for ImageNet (224×224, 1000 classes); we adapt
    its first conv to CIFAR10 (32×32) following standard practice: replace
    7×7 stride-2 conv + maxpool with 3×3 stride-1 conv, no maxpool.
    """
    net = sew_resnet.sew_resnet18(
        pretrained=False,
        cnf='ADD',
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    # Adapt for CIFAR10 input (32×32 instead of 224×224)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    # Replace final classifier
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    # Set multi-step mode
    functional.set_step_mode(net, 'm')
    functional.set_backend(net, 'cupy', instance=neuron.IFNode)
    return net


def encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """Direct encoding: replicate input across T time steps.

    Input:  x of shape [B, C, H, W]
    Output: shape [T, B, C, H, W]
    """
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


def train_one_epoch(net, loader, criterion, optimizer, scaler, T, device, epoch):
    net.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for batch_idx, (img, label) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            x = encode(img, T)            # [T, B, C, H, W]
            out = net(x).mean(dim=0)      # average over T → [B, num_classes]
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
          f"acc={100*correct/total:.2f}% time={elapsed:.1f}s")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Phase B(2) Training: SEW-ResNet18 on CIFAR10 ===")
    print(f"Args: {vars(args)}")
    print(f"Start: {datetime.datetime.now().isoformat()}")

    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(
        args.data_path, args.batch_size, args.workers)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")

    net = build_model(args.T).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: SEW-ResNet18, T={args.T}, params={n_params/1e6:.2f}M")

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
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            test_acc = evaluate(net, test_loader, args.T, device)
            print(f"Epoch {epoch:3d} test: acc={100*test_acc:.2f}% "
                  f"(best={100*best_acc:.2f}%)")
            if test_acc > best_acc:
                best_acc = test_acc
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'args': vars(args),
                }
                save_path = Path(args.output_dir) / 'sew_resnet18_cifar10_best.pth'
                torch.save(ckpt, save_path)
                print(f"           -> saved best checkpoint to {save_path}")

    print(f"\n=== Training complete ===")
    print(f"Best test acc: {100*best_acc:.2f}%")
    print(f"End: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()