"""
Test 36: SJ cupy backend baseline for all architectures.

SJ cupy is the fastest built-in SJ backend (hand-written CUDA kernels).
Measures SJ-torch, SJ-cupy, and CATFuse-SF for comparison.
"""
import sys, time, statistics
sys.path.insert(0, '.')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

def bench(model, x_in):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            for m in model.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            for m in model.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

def test_arch(name, build_fn, T, B, C, H):
    print(f"  {name:<35s}", end="", flush=True)
    try:
        x = torch.rand(T, B, C, H, H, device=device)

        # SJ torch
        net_t = build_fn().to(device).eval()
        functional.set_step_mode(net_t, 'm')
        sj_torch = bench(net_t, x)

        # SJ cupy
        net_c = build_fn().to(device).eval()
        functional.set_step_mode(net_c, 'm')
        functional.set_backend(net_c, 'cupy')
        try:
            sj_cupy = bench(net_c, x)
        except Exception as e:
            sj_cupy = float('inf')
            print(f" cupy_err={e}", end="")

        # CATFuse-SF
        from catfuse.substitute import substitute_sf
        net_sf_base = build_fn().to(device).eval()
        functional.set_step_mode(net_sf_base, 'm')
        net_sf, stats = substitute_sf(net_sf_base, T=T)
        net_sf = net_sf.to(device).eval()
        functional.set_step_mode(net_sf, 'm')
        sf_ms = bench(net_sf, x)

        vs_torch = sj_torch / sf_ms
        vs_cupy = sj_cupy / sf_ms if sj_cupy != float('inf') else 0

        print(f" torch={sj_torch:>6.2f}  cupy={sj_cupy:>6.2f}  SF={sf_ms:>6.2f}  "
              f"SF/torch={vs_torch:>5.2f}×  SF/cupy={vs_cupy:>5.2f}×")

        del net_t, net_c, net_sf, x
        torch.cuda.empty_cache()
        return (name, sj_torch, sj_cupy, sf_ms, vs_torch, vs_cupy)
    except Exception as e:
        print(f" ❌ {e}")
        torch.cuda.empty_cache()
        return None

T = 4
results = []

print("=" * 110)
print(f"SJ torch vs SJ cupy vs CATFuse-SF (V100, T={T})")
print(f"{'Network':<37s} {'SJ-torch':>8s}  {'SJ-cupy':>8s}  {'SF':>8s}  {'SF/torch':>9s}  {'SF/cupy':>9s}")
print("=" * 110)

# VGG
print("\n--- VGG (CIFAR-10, 32×32, B=2) ---")
for vname, vfn in [
    ("VGG11-BN", spiking_vgg.spiking_vgg11_bn),
    ("VGG13-BN", spiking_vgg.spiking_vgg13_bn),
    ("VGG16-BN", spiking_vgg.spiking_vgg16_bn),
]:
    r = test_arch(vname, lambda fn=vfn: fn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),
                  T=T, B=2, C=3, H=32)
    if r: results.append(r)

# SEW-ResNet
print("\n--- SEW-ResNet (CIFAR-10, 32×32, B=2) ---")
for rname, rfn in [
    ("SEW-ResNet18", sew_resnet.sew_resnet18),
    ("SEW-ResNet34", sew_resnet.sew_resnet34),
    ("SEW-ResNet50", sew_resnet.sew_resnet50),
]:
    r = test_arch(rname, lambda fn=rfn: fn(pretrained=False, num_classes=10, cnf='ADD',
                  spiking_neuron=neuron.LIFNode, tau=2.0),
                  T=T, B=2, C=3, H=32)
    if r: results.append(r)

# SpikingResNet
print("\n--- SpikingResNet (CIFAR-10, 32×32, B=2) ---")
for rname, rfn in [
    ("SpikingResNet18", spiking_resnet.spiking_resnet18),
    ("SpikingResNet34", spiking_resnet.spiking_resnet34),
]:
    r = test_arch(rname, lambda fn=rfn: fn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),
                  T=T, B=2, C=3, H=32)
    if r: results.append(r)

# Transformers
print("\n--- Transformers (CIFAR-10, 32×32, B=2) ---")
from models.spikformer_github import SpikformerGithub
r = test_arch("SpikFormer",
    lambda: SpikformerGithub(img_size=(32,32), in_channels=3, num_classes=10,
                             embed_dim=256, depth=2, num_heads=8, mlp_ratio=4, T=T,
                             spiking_neuron=neuron.LIFNode, v_threshold=1.0),
    T=T, B=2, C=3, H=32)
if r: results.append(r)

from models.qkformer_github import QKFormerGithub
r = test_arch("QKFormer",
    lambda: QKFormerGithub(img_size=(32,32), in_channels=3, num_classes=10,
                           dims=(128, 256), heads=(4, 8), depths=(2, 2),
                           patch_size=4, mlp_ratio=4, T=T,
                           spiking_neuron=neuron.LIFNode, v_threshold=1.0),
    T=T, B=2, C=3, H=32)
if r: results.append(r)

# ImageNet
print("\n--- ImageNet scale (224×224, B=1) ---")
for name, build_fn in [
    ("SEW-ResNet18 (ImageNet)", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ("SEW-ResNet34 (ImageNet)", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ("VGG11-BN (ImageNet)", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
]:
    r = test_arch(name, build_fn, T=T, B=1, C=3, H=224)
    if r: results.append(r)

# Summary
print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)
print(f"{'Network':<35s} {'SJ-torch':>8s} {'SJ-cupy':>8s} {'CATFuse':>8s} {'vs torch':>9s} {'vs cupy':>9s}")
print("-" * 80)
for name, sj_t, sj_c, sf, vt, vc in results:
    cupy_str = f"{sj_c:>7.2f}" if sj_c != float('inf') else "    N/A"
    vc_str = f"{vc:>8.2f}×" if vc > 0 else "     N/A"
    print(f"{name:<35s} {sj_t:>7.2f} {cupy_str} {sf:>7.2f} {vt:>8.2f}× {vc_str}")

if results:
    avg_vt = statistics.mean(vt for _, _, _, _, vt, _ in results)
    cupy_results = [(vc) for _, _, _, _, _, vc in results if vc > 0]
    avg_vc = statistics.mean(cupy_results) if cupy_results else 0
    print(f"\n  Mean speedup vs SJ-torch: {avg_vt:.2f}×")
    if avg_vc > 0:
        print(f"  Mean speedup vs SJ-cupy:  {avg_vc:.2f}×")
print("=" * 110)
