"""Minimal diagnostic: plain PyTorch Conv2d (no SJ), single-step.
If this works under ncu, the problem is specifically in SJ multi-step path.
If this also fails, the problem is generic PyTorch Conv × ncu.
"""
import os, torch, torch.nn as nn
import torch.backends.cudnn as cudnn

MODE = os.environ.get('CTF_MODE', 'A')
cudnn.benchmark = False
cudnn.deterministic = True

B, C, H, W = 8, 64, 8, 8

if MODE == 'A':
    # Plain single-step Conv2d, no SJ at all
    print("MODE A: plain Conv2d single-step")
    conv = nn.Conv2d(C, C, 3, padding=1).cuda().eval()
    x = torch.randn(B, C, H, W, device='cuda')
    for _ in range(3):
        with torch.no_grad(): _ = conv(x)
    torch.cuda.synchronize()
    with torch.no_grad(): y = conv(x)
    torch.cuda.synchronize()
    print(f"output shape: {y.shape}, OK")

elif MODE == 'B':
    # Plain Conv2d with manually expanded T·B batch (simulates SJ BatchFold)
    print("MODE B: plain Conv2d on [T*B, C, H, W]")
    T = 4
    conv = nn.Conv2d(C, C, 3, padding=1).cuda().eval()
    x = torch.randn(T * B, C, H, W, device='cuda')
    for _ in range(3):
        with torch.no_grad(): _ = conv(x)
    torch.cuda.synchronize()
    with torch.no_grad(): y = conv(x)
    torch.cuda.synchronize()
    print(f"output shape: {y.shape}, OK")

elif MODE == 'C':
    # SJ Conv2d single-step (real SJ but no multi-step machinery)
    from spikingjelly.activation_based import layer, functional
    print("MODE C: SJ Conv2d single-step")
    conv = layer.Conv2d(C, C, 3, padding=1, step_mode='s').cuda().eval()
    x = torch.randn(B, C, H, W, device='cuda')
    for _ in range(3):
        functional.reset_net(conv)
        with torch.no_grad(): _ = conv(x)
    torch.cuda.synchronize()
    functional.reset_net(conv)
    with torch.no_grad(): y = conv(x)
    torch.cuda.synchronize()
    print(f"output shape: {y.shape}, OK")

elif MODE == 'D':
    # SJ Conv2d multi-step (uses seq_to_ann_forward -- suspected culprit)
    from spikingjelly.activation_based import layer, functional
    print("MODE D: SJ Conv2d multi-step (seq_to_ann_forward)")
    T = 4
    conv = layer.Conv2d(C, C, 3, padding=1, step_mode='m').cuda().eval()
    x = torch.randn(T, B, C, H, W, device='cuda')
    for _ in range(3):
        functional.reset_net(conv)
        with torch.no_grad(): _ = conv(x)
    torch.cuda.synchronize()
    functional.reset_net(conv)
    with torch.no_grad(): y = conv(x)
    torch.cuda.synchronize()
    print(f"output shape: {y.shape}, OK")
