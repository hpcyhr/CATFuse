"""Test 5: substitute_sf on SEW-ResNet18."""
import sys, os
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet

device = 'cuda:0'
T, B = 4, 2

print("=" * 60)
print("Test 5: substitute_sf on SEW-ResNet18")
print("=" * 60)

net = sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd.get('model', sd), strict=False)
    print(f"  Loaded: {ckpt}")

net = net.to(device).eval()
functional.set_step_mode(net, 'm')

# Baseline
x = torch.rand(T, B, 3, 32, 32, device=device)
functional.reset_net(net)
with torch.no_grad():
    ref_out = net(x)
print(f"  SJ baseline: {ref_out.shape}, mean={ref_out.mean().item():.4f}")

# Substitute
from catfuse.substitute import substitute_sf, print_routing_table
fused_net, stats = substitute_sf(net, T=T, verbose=True)
print_routing_table(stats)

# Forward fused
fused_net = fused_net.to(device).eval()
functional.set_step_mode(fused_net, 'm')
functional.reset_net(fused_net)
with torch.no_grad():
    fused_out = fused_net(x)
print(f"\n  Fused output: {fused_out.shape}, mean={fused_out.mean().item():.4f}")

# Parity
diff = (ref_out - fused_out).abs()
print(f"  max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")

# Module count
sf_count = sum(1 for _, m in fused_net.named_modules() if 'STFusionConvBNLIF' in type(m).__name__)
pf_count = sum(1 for _, m in fused_net.named_modules() if 'PartialFusionConvBNLIF' in type(m).__name__)
print(f"\n  STFusionConvBNLIF: {sf_count}, PartialFusionConvBNLIF: {pf_count}")

print("\n" + "=" * 60)
print("PASSED" if (sf_count + pf_count > 0 and fused_out.shape == ref_out.shape) else "FAILED")
print("=" * 60)
