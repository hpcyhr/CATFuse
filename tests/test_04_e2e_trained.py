"""
Test 4: E2E parity with trained SEW-ResNet18.
Finds Conv→BN→LIF by attribute names in BasicBlocks.
"""
import sys, os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based.model import sew_resnet
from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

device = 'cuda:0'
T, B = 4, 2

print("=" * 60)
print("Test 4: E2E trained SEW-ResNet18")
print("=" * 60)

# 1. Load
net = sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
ckpt_path = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get('model', state)
    net.load_state_dict(sd, strict=False)
    print(f"  Loaded: {ckpt_path}")
else:
    print(f"  WARNING: no checkpoint, using random init")

net = net.to(device).eval()
functional.set_step_mode(net, 'm')

# 2. Baseline forward
print("\n--- SJ baseline ---")
x = torch.rand(T, B, 3, 32, 32, device=device)
functional.reset_net(net)
with torch.no_grad():
    ref_out = net(x)
print(f"  Output: {ref_out.shape}, spike_rate={ref_out.mean().item():.4f}")

# 3. Find Conv→BN→LIF triples by attribute naming convention
print("\n--- Layer-wise parity ---")

triples = []
for parent_name, parent_mod in net.named_modules():
    # Look for (conv1, bn1, sn1) or (conv2, bn2, sn2) patterns
    for idx in ['1', '2', '3']:
        conv_name, bn_name, sn_name = f'conv{idx}', f'bn{idx}', f'sn{idx}'
        c = getattr(parent_mod, conv_name, None)
        b = getattr(parent_mod, bn_name, None)
        l = getattr(parent_mod, sn_name, None)
        if c is None or b is None or l is None:
            continue
        if not isinstance(c, (nn.Conv2d, layer.Conv2d)):
            continue
        if not isinstance(b, (nn.BatchNorm2d, layer.BatchNorm2d)):
            continue
        if not isinstance(l, neuron.LIFNode):
            continue
        full_name = f"{parent_name}.conv{idx}→bn{idx}→sn{idx}" if parent_name else f"conv{idx}→bn{idx}→sn{idx}"
        triples.append((full_name, c, b, l))

# Also check stem: conv1, bn1, sn1 at top level of net
for idx in ['1']:
    c = getattr(net, f'conv{idx}', None)
    b = getattr(net, f'bn{idx}', None)
    l = getattr(net, f'sn{idx}', None)
    if c and b and l and isinstance(c, (nn.Conv2d, layer.Conv2d)):
        triples.insert(0, (f"stem.conv{idx}→bn{idx}→sn{idx}", c, b, l))

print(f"  Found {len(triples)} Conv→BN→LIF triples")

tested = 0
passed = 0

for name, c, b, l in triples:
    cin = c.in_channels
    cout = c.out_channels
    ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
    stride = c.stride[0] if isinstance(c.stride, tuple) else c.stride
    
    # Pick test spatial size based on channel count (approximate network stage)
    if cin <= 3:
        H_test = 32
    elif cin <= 64:
        H_test = 32
    elif cin <= 128:
        H_test = 16
    elif cin <= 256:
        H_test = 8
    else:
        H_test = 4
    
    # Generate input with ~30% spike rate
    x_test = (torch.rand(T, B, cin, H_test, H_test, device=device) > 0.7).float()
    in_rate = x_test.mean().item()
    
    # Switch to single-step mode for per-timestep testing
    for m in [c, b, l]:
        if hasattr(m, "step_mode"): m.step_mode = "s"
    # Reference: step-by-step
    b.eval()
    functional.reset_net(l)
    ref_spikes = []
    with torch.no_grad():
        for t in range(T):
            z = c(x_test[t])
            z = b(z)
            s = l(z)
            ref_spikes.append(s)
    ref_spikes = torch.stack(ref_spikes, dim=0)
    
    # CATFuse-SF
    sf_mod = STFusionConvBNLIF.from_sj_modules(c, b, l, K=min(4, T)).to(device).eval()
    with torch.no_grad():
        sf_mod.v = 0.0
        sf_spikes = sf_mod(x_test)
    
    total = ref_spikes.numel()
    match_count = (ref_spikes == sf_spikes).sum().item()
    pct = match_count / total * 100
    ref_rate = ref_spikes.mean().item()
    
    status = "OK" if pct > 99.9 else "FAIL"
    print(f"  [{status}] {name:40s} "
          f"({cin:3d}→{cout:3d}, k={ksize}, s={stride}) "
          f"in={in_rate:.2f} out={ref_rate:.4f} match={pct:.2f}%")
    
    tested += 1
    if pct > 99.9:
        passed += 1
    if pct <= 99.9:
        diff_idx = (ref_spikes != sf_spikes).nonzero(as_tuple=False)
        print(f"         mismatches: {total - match_count}/{total}, first 5: {diff_idx[:5].tolist()}")

print(f"\n{'=' * 60}")
print(f"Result: {passed}/{tested} layers passed (>99.9% spike match)")
if passed == tested and tested > 0:
    print("ALL LAYER PARITY PASSED")
else:
    print("SOME LAYERS FAILED" if tested > 0 else "NO LAYERS FOUND")
print("=" * 60)
