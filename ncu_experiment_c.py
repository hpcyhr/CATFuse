"""
Experiment C under ncu: measure global store/load bytes of one
SJ-cupy multi-step Conv→BN→LIF forward, with and without CUDA Graph.

Uses torch.cuda.cudart().cudaProfilerStart/Stop to bracket exactly one
forward so ncu profiles only the measured region (not the warmup).
Requires running ncu with --profile-from-start off.

Conv runs on ATen native (cudnn.enabled=False) to work around an
ncu 2022.3 × cuDNN handle initialization incompatibility.

Shape: (T=4, B=8, C=64, H=W=8) to fit ncu replay memory.
"""
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from spikingjelly.activation_based import (
    neuron, layer, surrogate, functional,
)

T, B, C, H, W = 4, 8, 64, 8, 8
DEVICE = 'cuda'
MODE = os.environ.get('CTF_MODE', 'baseline')

cudnn.enabled = False

torch.manual_seed(42)
net = nn.Sequential(
    layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
    layer.BatchNorm2d(C, step_mode='m'),
    neuron.LIFNode(
        step_mode='m', backend='cupy',
        surrogate_function=surrogate.ATan(),
    ),
).to(DEVICE).eval()
for p in net.parameters():
    p.requires_grad_(False)

x = torch.randn(T, B, C, H, W, device=DEVICE)

# Warmup (not profiled, since --profile-from-start off)
for _ in range(5):
    functional.reset_net(net)
    with torch.no_grad():
        _ = net(x)
torch.cuda.synchronize()

if MODE == 'baseline':
    # Bracket exactly one measured forward
    functional.reset_net(net)
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        _ = net(x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("MODE=baseline done (profiled 1 forward)")

elif MODE == 'cudagraph':
    # Capture on side stream (not profiled)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            functional.reset_net(net)
            with torch.no_grad():
                _ = net(x)
    torch.cuda.current_stream().wait_stream(s)

    static_x = x.clone()
    functional.reset_net(net)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = net(static_x)

    # Bracket exactly one replay
    functional.reset_net(net)
    torch.cuda.cudart().cudaProfilerStart()
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("MODE=cudagraph done (profiled 1 replay)")

else:
    sys.exit(f"Unknown MODE: {MODE}")
