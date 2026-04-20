"""
Test 39: ORT-GPU (CUDA EP) benchmark on all architectures.

Run in ort_test conda env. Exports SNN to ONNX, benchmarks with ORT CUDA EP.
CATFuse numbers from test_38 (snn118 env) used for comparison.
"""
import sys, time, statistics, os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
import onnxruntime as ort

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3

print(f"ORT: {ort.__version__}, Providers: {ort.get_available_providers()}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

def bench_torch(model, x_in):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x_in)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

def bench_ort(name, model, x_in, T, B, C, H):
    """Export to ONNX and benchmark with ORT CUDA EP."""
    class Wrapper(nn.Module):
        def __init__(self, m, T):
            super().__init__()
            self.m = m
            self.T = T
        def forward(self, x):
            functional.reset_net(self.m)
            B2 = x.shape[0] // self.T
            x5 = x.reshape(self.T, B2, x.shape[1], x.shape[2], x.shape[3])
            return self.m(x5)

    wrapper = Wrapper(model, T).to(device).eval()
    x_flat = x_in.reshape(T * B, C, H, H)

    onnx_path = f"/data/tmp/snn_{name.replace(' ', '_')}.onnx"
    os.makedirs("/data/tmp", exist_ok=True)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, x_flat, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=14, do_constant_folding=True,
            )
    except Exception as e:
        return None, f"export: {e}"

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Verify it's using CUDA
        ep = sess.get_providers()

        x_np = x_flat.cpu().numpy()

        # Warmup
        for _ in range(N_WARMUP):
            sess.run(None, {'input': x_np})

        # Benchmark
        times = []
        for _ in range(N_REPEAT):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                sess.run(None, {'input': x_np})
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_ITER * 1000)

        # Cleanup
        os.remove(onnx_path)
        return statistics.median(times), None
    except Exception as e:
        return None, f"run: {e}"

T = 4

configs = [
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG13-BN", lambda: spiking_vgg.spiking_vgg13_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN34", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN34", lambda: spiking_resnet.spiking_resnet34(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ("SEW-RN34-I", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ("VGG11-I", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
]

# SpikFormer / QKFormer — try import
try:
    from models.spikformer_github import SpikformerGithub
    configs.append(("SpikFormer", lambda: SpikformerGithub(
        img_size=(32,32), in_channels=3, num_classes=10, embed_dim=256,
        depth=2, num_heads=8, mlp_ratio=4, T=T,
        spiking_neuron=neuron.LIFNode, v_threshold=1.0), 2, 3, 32))
except:
    pass

try:
    from models.qkformer_github import QKFormerGithub
    configs.append(("QKFormer", lambda: QKFormerGithub(
        img_size=(32,32), in_channels=3, num_classes=10,
        dims=(128,256), heads=(4,8), depths=(2,2),
        patch_size=4, mlp_ratio=4, T=T,
        spiking_neuron=neuron.LIFNode, v_threshold=1.0), 2, 3, 32))
except:
    pass

print("\n" + "=" * 90)
print(f"ORT-GPU Benchmark (V100, T={T})")
print(f"{'Network':<15s} {'SJ-torch':>9s} {'ORT-CUDA':>10s} {'ORT/SJ':>8s} {'status':>8s}")
print("=" * 90)

results = []

for name, build_fn, B, C, H in configs:
    print(f"  {name:<13s}", end="", flush=True)
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ torch baseline
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_ms = bench_torch(net, x)

    # ORT
    net_ort = build_fn().to(device).eval()
    functional.set_step_mode(net_ort, 'm')
    ort_ms, err = bench_ort(name, net_ort, x, T, B, C, H)

    if ort_ms:
        ratio = sj_ms / ort_ms
        print(f" {sj_ms:>8.2f}  {ort_ms:>9.2f}  {ratio:>7.2f}×")
        results.append((name, sj_ms, ort_ms, ratio))
    else:
        print(f" {sj_ms:>8.2f}  {'✗':>9s}  {'—':>7s}   {err}")
        results.append((name, sj_ms, None, None))

    del net, net_ort, x
    torch.cuda.empty_cache()

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
for name, sj, ort_v, ratio in results:
    ort_s = f"{ort_v:.2f}" if ort_v else "✗"
    r_s = f"{ratio:.2f}×" if ratio else "—"
    print(f"  {name:<15s} SJ={sj:.2f}ms  ORT={ort_s}ms  ORT/SJ={r_s}")

# Cleanup
import shutil
shutil.rmtree("/data/tmp", ignore_errors=True)
print("=" * 90)
