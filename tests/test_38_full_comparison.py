"""
Test 38: Full comparison — ALL architectures × ALL tools.
"""
import sys, time, statistics, os, gc
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

def bench_torch(model, x_in):
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

def try_compile(model, x_in):
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)
        for _ in range(5):
            functional.reset_net(compiled)
            with torch.no_grad(): _ = compiled(x_in)
        times = []
        for _ in range(N_REPEAT):
            for _ in range(N_WARMUP):
                functional.reset_net(compiled)
                with torch.no_grad(): _ = compiled(x_in)
            torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
            for _ in range(N_ITER):
                functional.reset_net(compiled)
                with torch.no_grad(): _ = compiled(x_in)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_ITER * 1000)
        return statistics.median(times)
    except:
        return None

def try_ort(model, x_in, T, B, C, H):
    try:
        import onnxruntime as ort
        class W(nn.Module):
            def __init__(self, m, T):
                super().__init__()
                self.m = m; self.T = T
            def forward(self, x):
                functional.reset_net(self.m)
                B2 = x.shape[0] // self.T
                return self.m(x.reshape(self.T, B2, x.shape[1], x.shape[2], x.shape[3]))
        w = W(model, T).to(device).eval()
        x_flat = x_in.reshape(T*B, C, H, H)
        p = "/tmp/_snn_ort.onnx"
        with torch.no_grad():
            torch.onnx.export(w, x_flat, p, input_names=['x'], output_names=['y'],
                              opset_version=14, do_constant_folding=True)
        sess = ort.InferenceSession(p, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        xn = x_flat.cpu().numpy()
        for _ in range(N_WARMUP): sess.run(None, {'x': xn})
        times = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            for _ in range(N_ITER): sess.run(None, {'x': xn})
            times.append((time.perf_counter() - t0) / N_ITER * 1000)
        return statistics.median(times)
    except:
        return None

T = 4

configs = [
    # VGG family CIFAR
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG13-BN", lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    # SEW-ResNet CIFAR
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN34", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    # SpikingResNet CIFAR
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN34", lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    # Transformers CIFAR
    ("SpikFormer", None, 2, 3, 32),  # handled separately
    ("QKFormer", None, 2, 3, 32),    # handled separately
    # ImageNet
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ("SEW-RN34-I", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ("VGG11-I", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
]

# Fix transformer builders
from models.spikformer_github import SpikformerGithub
from models.qkformer_github import QKFormerGithub

for i, (name, fn, B, C, H) in enumerate(configs):
    if name == "SpikFormer":
        configs[i] = ("SpikFormer", lambda: SpikformerGithub(
            img_size=(32,32), in_channels=3, num_classes=10, embed_dim=256,
            depth=2, num_heads=8, mlp_ratio=4, T=T,
            spiking_neuron=neuron.LIFNode, v_threshold=1.0), 2, 3, 32)
    elif name == "QKFormer":
        configs[i] = ("QKFormer", lambda: QKFormerGithub(
            img_size=(32,32), in_channels=3, num_classes=10,
            dims=(128,256), heads=(4,8), depths=(2,2),
            patch_size=4, mlp_ratio=4, T=T,
            spiking_neuron=neuron.LIFNode, v_threshold=1.0), 2, 3, 32)

print("=" * 130)
print(f"FULL COMPARISON: V100-SXM2-32GB, T={T}, cudnn.benchmark=False")
print(f"{'Network':<14s} {'SJ-torch':>9s} {'SJ-cupy':>9s} {'compile':>9s} {'ORT':>9s} {'CATFuse':>9s} "
      f"{'vs torch':>9s} {'vs cupy':>9s} {'vs comp':>9s} {'vs ORT':>9s}")
print("=" * 130)

results = []

for name, build_fn, B, C, H in configs:
    print(f"  {name:<12s}", end="", flush=True)
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ torch
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_t = bench_torch(net, x)

    # SJ cupy
    net_c = build_fn().to(device).eval()
    functional.set_step_mode(net_c, 'm')
    try:
        functional.set_backend(net_c, 'cupy')
        sj_c = bench_torch(net_c, x)
    except:
        sj_c = None
    del net_c; torch.cuda.empty_cache()

    # torch.compile
    net_comp = build_fn().to(device).eval()
    functional.set_step_mode(net_comp, 'm')
    comp = try_compile(net_comp, x)
    del net_comp; torch.cuda.empty_cache()

    # ORT
    net_ort = build_fn().to(device).eval()
    functional.set_step_mode(net_ort, 'm')
    ort_ms = try_ort(net_ort, x, T, B, C, H)
    del net_ort; torch.cuda.empty_cache()

    # CATFuse-SF
    from catfuse.substitute import substitute_sf
    net_sf_base = build_fn().to(device).eval()
    functional.set_step_mode(net_sf_base, 'm')
    net_sf, _ = substitute_sf(net_sf_base, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')
    sf = bench_torch(net_sf, x)
    del net_sf, net; torch.cuda.empty_cache()
    gc.collect()

    # Format
    def fmt(v):
        return f"{v:>8.2f}" if v else "      ✗"
    def spd(base, sf_v):
        return f"{base/sf_v:>8.2f}×" if base and sf_v else "       —"

    print(f" {sj_t:>8.2f} {fmt(sj_c)} {fmt(comp)} {fmt(ort_ms)} {sf:>8.2f} "
          f"{sj_t/sf:>8.2f}× {spd(sj_c, sf)} {spd(comp, sf)} {spd(ort_ms, sf)}")

    results.append((name, sj_t, sj_c, comp, ort_ms, sf))

print("\n" + "=" * 130)
# Compute means
sj_ts = [r[1] for r in results]
sfs = [r[5] for r in results]
avg_vt = statistics.mean(r[1]/r[5] for r in results)
cupy_spds = [r[2]/r[5] for r in results if r[2]]
avg_vc = statistics.mean(cupy_spds) if cupy_spds else 0
print(f"  Mean CATFuse vs SJ-torch: {avg_vt:.2f}×")
if avg_vc: print(f"  Mean CATFuse vs SJ-cupy:  {avg_vc:.2f}×")
print(f"  All {len(results)} architectures accelerated: {'✅' if all(r[1]/r[5] > 1 for r in results) else '❌'}")
print("=" * 130)
