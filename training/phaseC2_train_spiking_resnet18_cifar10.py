#!/usr/bin/env python
"""
Phase C-2 — Train SpikingResNet18 on CIFAR10 with LIFNode.

Matches bench_spiking_resnet18_e2e.py::build_spiking_resnet_model:
  - spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, ...)
  - surrogate_function=surrogate.Sigmoid()
  - detach_reset=True
  - functional.set_step_mode(model, 'm')
  - NO cnf parameter (unlike SEW)

CIFAR10 stem adaption (--cifar10-stem):
  - conv1: 7x7 stride 2 -> 3x3 stride 1 (using SJ layer.Conv2d for multi-step)
  - maxpool -> Identity

Usage:
    cd /data/dagongcheng/yhrtest/CATFuse
    nohup python phaseC2_train_spiking_resnet18_cifar10.py \
        --data-path data/cifar10 \
        --output-dir checkpoints/ \
        --epochs 200 -b 128 --T 4 --lr 0.1 --amp \
        --cifar10-stem --device cuda:0 \
        > logs/phaseC2_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet


def build_spiking_resnet_model(num_classes=10, v_threshold=1.0, tau=2.0,
                                cifar10_stem=True):
    model = spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        num_classes=num_classes,
        v_threshold=v_threshold,
        tau=tau,
    )
    if cifar10_stem:
        model.conv1 = layer.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        model.maxpool = nn.Identity()
    functional.set_step_mode(model, 'm')
    return model


def get_dataloaders(data_path, batch_size, num_workers=4):
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


def encode(x, T):
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


def train_one_epoch(net, loader, criterion, optimizer, scaler, T, device, epoch):
    net.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = net(encode(img, T)).mean(dim=0)
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
    print(f"Epoch {epoch:3d} train: loss={loss_sum/total:.4f} "
          f"acc={100*correct/total:.2f}% time={time.time()-t0:.1f}s", flush=True)


@torch.no_grad()
def evaluate(net, loader, T, device):
    net.eval()
    total, correct = 0, 0
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        out = net(encode(img, T)).mean(dim=0)
        functional.reset_net(net)
        total += label.size(0)
        correct += (out.argmax(1) == label).sum().item()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='checkpoints/')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('-b', '--batch-size', type=int, default=128)
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v-threshold', type=float, default=1.0)
    p.add_argument('--cifar10-stem', action='store_true')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Phase C-2: SpikingResNet18 on CIFAR10 (LIFNode) ===")
    print(f"Args: {vars(args)}", flush=True)
    print(f"Start: {datetime.datetime.now().isoformat()}", flush=True)

    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(args.data_path, args.batch_size, args.workers)
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}", flush=True)

    net = build_spiking_resnet_model(
        num_classes=10, v_threshold=args.v_threshold,
        tau=args.tau, cifar10_stem=args.cifar10_stem).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: SpikingResNet18 LIFNode (tau={args.tau}, v_th={args.v_threshold}, "
          f"cifar10_stem={args.cifar10_stem}), T={args.T}, params={n_params/1e6:.2f}M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(net, train_loader, criterion, optimizer, scaler, args.T, device, epoch)
        scheduler.step()
        test_acc = evaluate(net, test_loader, args.T, device)
        print(f"Epoch {epoch:3d} test: acc={100*test_acc:.2f}% (best={100*best_acc:.2f}%)", flush=True)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'test_acc': test_acc,
                'args': vars(args),
                'model_name': 'spiking_resnet18',
            }, Path(args.output_dir) / 'spiking_resnet18_cifar10_lif_best.pth')
            print(f"           -> saved best", flush=True)

    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': net.state_dict(),
        'test_acc': evaluate(net, test_loader, args.T, device),
        'args': vars(args),
        'model_name': 'spiking_resnet18',
    }, Path(args.output_dir) / 'spiking_resnet18_cifar10_lif_final.pth')

    print(f"\n=== Training complete ===")
    print(f"Best test acc: {100*best_acc:.2f}%")
    print(f"End: {datetime.datetime.now().isoformat()}")


if __name__ == '__main__':
    main()