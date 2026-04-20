"""
Test 35: Comprehensive multi-architecture benchmark.

Tests all available SNN architectures with CATFuse-SF.
"""
import sys, os, time, statistics, traceback
sys.path.insert(0, '.')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

def bench(model, x_in, label=""):
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

def test_arch(name, build_fn, T, B, C, H, num_classes):
    print(f"\n  {name:<35s}", end="", flush=True)
    try:
        x = torch.rand(T, B, C, H, H, device=device)

        # SJ baseline
        net = build_fn().to(device).eval()
        functional.set_step_mode(net, 'm')
        lif_count = sum(1 for _, m in net.named_modules() if isinstance(m, neuron.LIFNode))
        sj_ms = bench(net, x)

        # CATFuse-SF
        from catfuse.substitute import substitute_sf
        net_sf, stats = substitute_sf(net, T=T)
        net_sf = net_sf.to(device).eval()
        functional.set_step_mode(net_sf, 'm')

        # Parity
        functional.reset_net(net); functional.reset_net(net_sf)
        for m in net_sf.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad():
            ref = net(x)
            sf = net_sf(x)
        parity = (ref - sf).abs().max().item()

        sf_ms = bench(net_sf, x)
        cov = stats["coverage_pct"]
        fused = stats["fused_lif_nodes"]

        print(f" LIF={lif_count:>3d}  fused={fused:>3d} ({cov:>5.1f}%)  "
              f"SJ={sj_ms:>6.2f}ms  SF={sf_ms:>6.2f}ms  "
              f"speedup={sj_ms/sf_ms:>5.2f}×  parity={'✅' if parity < 0.01 else '❌'}")

        # Cleanup
        del net, net_sf, x
        torch.cuda.empty_cache()
        return (name, lif_count, fused, cov, sj_ms, sf_ms, sj_ms/sf_ms, parity)

    except Exception as e:
        print(f" ❌ FAILED: {e}")
        torch.cuda.empty_cache()
        return None

# ================================================================
T = 4
results = []

print("=" * 120)
print(f"CATFuse-SF Comprehensive Benchmark (V100-SXM2-32GB, T={T})")
print(f"{'Network':<37s} {'LIF':>4s}  {'fused':>5s} {'cov':>7s}  {'SJ':>8s}  {'SF':>8s}  {'speedup':>8s}  {'parity':>7s}")
print("=" * 120)

# --- VGG family (CIFAR-10, 32×32) ---
print("\n--- VGG family (CIFAR-10, 32×32, B=2) ---")
for vgg_name, vgg_fn in [
    ("VGG11-BN", spiking_vgg.spiking_vgg11_bn),
    ("VGG13-BN", spiking_vgg.spiking_vgg13_bn),
    ("VGG16-BN", spiking_vgg.spiking_vgg16_bn),
]:
    r = test_arch(vgg_name, lambda fn=vgg_fn: fn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),
                  T=T, B=2, C=3, H=32, num_classes=10)
    if r: results.append(r)

# --- ResNet family (CIFAR-10, 32×32) ---
print("\n--- SEW-ResNet family (CIFAR-10, 32×32, B=2) ---")
for rn_name, rn_fn in [
    ("SEW-ResNet18", sew_resnet.sew_resnet18),
    ("SEW-ResNet34", sew_resnet.sew_resnet34),
    ("SEW-ResNet50", sew_resnet.sew_resnet50),
]:
    r = test_arch(rn_name, lambda fn=rn_fn: fn(pretrained=False, num_classes=10, cnf='ADD',
                  spiking_neuron=neuron.LIFNode, tau=2.0),
                  T=T, B=2, C=3, H=32, num_classes=10)
    if r: results.append(r)

print("\n--- SpikingResNet family (CIFAR-10, 32×32, B=2) ---")
for rn_name, rn_fn in [
    ("SpikingResNet18", spiking_resnet.spiking_resnet18),
    ("SpikingResNet34", spiking_resnet.spiking_resnet34),
]:
    r = test_arch(rn_name, lambda fn=rn_fn: fn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10),
                  T=T, B=2, C=3, H=32, num_classes=10)
    if r: results.append(r)

# --- Transformer family (CIFAR-10, 32×32) ---
print("\n--- Transformer family (CIFAR-10, 32×32, B=2) ---")

# SpikFormer
from models.spikformer_github import SpikformerGithub
r = test_arch("SpikFormer (d=2, dim=256)",
    lambda: SpikformerGithub(img_size=(32,32), in_channels=3, num_classes=10,
                             embed_dim=256, depth=2, num_heads=8, mlp_ratio=4, T=T,
                             spiking_neuron=neuron.LIFNode, v_threshold=1.0),
    T=T, B=2, C=3, H=32, num_classes=10)
if r: results.append(r)

# QKFormer
try:
    from models.qkformer_github import QKFormerGithub
    r = test_arch("QKFormer (d=2, dim=256)",
        lambda: QKFormerGithub(img_size=(32,32), in_channels=3, num_classes=10,
                               embed_dim=256, depth=2, num_heads=8, mlp_ratio=4, T=T,
                               spiking_neuron=neuron.LIFNode, v_threshold=1.0),
        T=T, B=2, C=3, H=32, num_classes=10)
    if r: results.append(r)
except Exception as e:
    print(f"\n  QKFormer: ❌ build failed: {e}")

# --- ImageNet scale (224×224, B=1) ---
print("\n--- ImageNet scale (224×224, B=1) ---")
for name, build_fn in [
    ("SEW-ResNet18 (ImageNet)", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ("SEW-ResNet34 (ImageNet)", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ("VGG11-BN (ImageNet)", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000)),
]:
    r = test_arch(name, build_fn, T=T, B=1, C=3, H=224, num_classes=1000)
    if r: results.append(r)

# ================================================================
# Summary table
# ================================================================
print("\n" + "=" * 120)
print("SUMMARY TABLE")
print("=" * 120)
print(f"{'Network':<35s} {'LIF':>4s} {'Cov%':>6s} {'SJ(ms)':>8s} {'SF(ms)':>8s} {'Speedup':>8s}")
print("-" * 75)
for name, lif, fused, cov, sj, sf, spdup, par in results:
    print(f"{name:<35s} {lif:>4d} {cov:>5.1f}% {sj:>7.2f} {sf:>7.2f} {spdup:>7.2f}×")
print("=" * 120)
