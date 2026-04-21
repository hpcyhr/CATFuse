"""Test 45: Peak memory + kernel launch count comparison."""
import sys, time
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg
from catfuse.substitute import substitute_sf

device = 'cuda:0'
T, B = 4, 2

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")
print("=" * 100)
print(f"{'Network':<15s} {'SJ mem(MB)':>12s} {'SF mem(MB)':>12s} {'mem ratio':>10s} {'SJ allocs':>10s} {'SF allocs':>10s} {'alloc ratio':>12s}")
print("=" * 100)

configs = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 3, 32),
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

# Fix: some configs have 4 values, some 5
configs_fixed = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

for name, build_fn, B_cfg, C, H in configs_fixed:
    x = torch.rand(T, B_cfg, C, H, H, device=device)

    # SJ baseline
    net_sj = build_fn().to(device).eval()
    functional.set_step_mode(net_sj, 'm')

    # Warmup
    functional.reset_net(net_sj)
    with torch.no_grad(): _ = net_sj(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    functional.reset_net(net_sj)
    with torch.no_grad(): _ = net_sj(x)
    sj_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    sj_stats = torch.cuda.memory_stats()
    sj_allocs = sj_stats.get('allocation.all.current', 0)
    if sj_allocs == 0:
        sj_allocs = sj_stats.get('num_alloc_retries', 0)
    # Try another way
    sj_alloc_count = sj_stats.get('allocation.all.allocated', 0)

    # CATFuse-SF
    net_sf_base = build_fn().to(device).eval()
    functional.set_step_mode(net_sf_base, 'm')
    net_sf, _ = substitute_sf(net_sf_base, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')

    # Warmup
    functional.reset_net(net_sf)
    for m in net_sf.modules():
        if hasattr(m, 'reset'): m.reset()
    with torch.no_grad(): _ = net_sf(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    functional.reset_net(net_sf)
    for m in net_sf.modules():
        if hasattr(m, 'reset'): m.reset()
    with torch.no_grad(): _ = net_sf(x)
    sf_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    sf_stats = torch.cuda.memory_stats()
    sf_alloc_count = sf_stats.get('allocation.all.allocated', 0)

    mem_ratio = sf_peak / sj_peak if sj_peak > 0 else 0
    alloc_ratio = sf_alloc_count / sj_alloc_count if sj_alloc_count > 0 else 0

    print(f"  {name:<13s} {sj_peak:>11.1f} {sf_peak:>11.1f} {mem_ratio:>9.2f}× {sj_alloc_count:>9d} {sf_alloc_count:>9d} {alloc_ratio:>11.2f}×")

    del net_sj, net_sf, x
    torch.cuda.empty_cache()

print("=" * 100)
