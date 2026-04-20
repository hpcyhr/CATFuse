"""
Test 37: ONNX Runtime GPU baseline comparison.

Export SNN models to ONNX, run with ORT CUDA EP, compare with CATFuse-SF.
"""
import sys, time, statistics
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

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

def try_ort(name, model, x_in, T, B, C, H):
    """Try to export model to ONNX and run with ORT."""
    import onnxruntime as ort

    # SNN models can't be directly exported because of stateful LIF
    # Workaround: export the fused model (CATFuse replaces LIF with deterministic ops)
    # Actually, SJ models have multi-step mode which is hard to export
    # Try torch.onnx.export with a wrapper

    class SNNWrapper(nn.Module):
        def __init__(self, snn_model, T):
            super().__init__()
            self.model = snn_model
            self.T = T

        def forward(self, x):
            # x: [T*B, C, H, W] — flatten time into batch for ONNX
            functional.reset_net(self.model)
            # Run step by step
            B_eff = x.shape[0] // self.T
            x_5d = x.reshape(self.T, B_eff, x.shape[1], x.shape[2], x.shape[3])
            return self.model(x_5d)

    wrapper = SNNWrapper(model, T).to(device).eval()
    x_flat = x_in.reshape(T * B, C, H, H)

    # Try export
    onnx_path = f"/tmp/snn_{name.replace(' ', '_')}.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, x_flat,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=14,
                do_constant_folding=True,
            )
    except Exception as e:
        return None, f"export failed: {type(e).__name__}: {e}"

    # Run with ORT
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(onnx_path, providers=providers)

        x_np = x_flat.cpu().numpy()

        # Warmup
        for _ in range(N_WARMUP):
            _ = sess.run(None, {'input': x_np})

        # Benchmark
        times = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                _ = sess.run(None, {'input': x_np})
            times.append((time.perf_counter() - t0) / N_ITER * 1000)

        return statistics.median(times), None
    except Exception as e:
        return None, f"ORT run failed: {type(e).__name__}: {e}"

def try_compile(name, model, x_in, T):
    """Try torch.compile on SJ model."""
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)

        # Warmup
        for _ in range(5):
            functional.reset_net(compiled)
            with torch.no_grad(): _ = compiled(x_in)

        times = []
        for _ in range(N_REPEAT):
            for _ in range(N_WARMUP):
                functional.reset_net(compiled)
                with torch.no_grad(): _ = compiled(x_in)
            torch.cuda.synchronize()
            t0 = time.perf_counter(); torch.cuda.synchronize()
            for _ in range(N_ITER):
                functional.reset_net(compiled)
                with torch.no_grad(): _ = compiled(x_in)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_ITER * 1000)
        return statistics.median(times), None
    except Exception as e:
        return None, f"compile failed: {type(e).__name__}"

print("=" * 120)
print("Multi-tool comparison: SJ-torch vs torch.compile vs ORT-CUDA vs CATFuse-SF")
print("=" * 120)

configs = [
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SEW-ResNet18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SEW-ResNet50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SEW-ResNet18 ImgNet", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
]

results = []

for name, build_fn, T, B, C, H in configs:
    print(f"\n--- {name} (T={T}, B={B}, {H}×{H}) ---")
    x = torch.rand(T, B, C, H, H, device=device)

    # 1. SJ-torch baseline
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_ms = bench_torch(net, x)
    print(f"  SJ-torch:    {sj_ms:.2f} ms")

    # 2. torch.compile
    net_c = build_fn().to(device).eval()
    functional.set_step_mode(net_c, 'm')
    comp_ms, comp_err = try_compile(name, net_c, x, T)
    if comp_ms:
        print(f"  torch.compile: {comp_ms:.2f} ms ({sj_ms/comp_ms:.2f}×)")
    else:
        print(f"  torch.compile: ✗ ({comp_err})")
        comp_ms = None

    # 3. ONNX Runtime
    net_o = build_fn().to(device).eval()
    functional.set_step_mode(net_o, 'm')
    ort_ms, ort_err = try_ort(name, net_o, x, T, B, C, H)
    if ort_ms:
        print(f"  ORT-CUDA:    {ort_ms:.2f} ms ({sj_ms/ort_ms:.2f}×)")
    else:
        print(f"  ORT-CUDA:    ✗ ({ort_err})")
        ort_ms = None

    # 4. CATFuse-SF
    from catfuse.substitute import substitute_sf
    net_sf_base = build_fn().to(device).eval()
    functional.set_step_mode(net_sf_base, 'm')
    net_sf, _ = substitute_sf(net_sf_base, T=T)
    net_sf = net_sf.to(device).eval()
    functional.set_step_mode(net_sf, 'm')
    sf_ms = bench_torch(net_sf, x)
    print(f"  CATFuse-SF:  {sf_ms:.2f} ms ({sj_ms/sf_ms:.2f}×)")

    results.append((name, sj_ms, comp_ms, ort_ms, sf_ms))
    del net, net_c, net_o, net_sf, x
    torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)
print(f"{'Network':<25s} {'SJ-torch':>9s} {'compile':>9s} {'ORT-CUDA':>9s} {'CATFuse':>9s} {'vs torch':>9s} {'vs ORT':>9s}")
print("-" * 85)
for name, sj, comp, ort, sf in results:
    comp_s = f"{comp:.2f}" if comp else "✗"
    ort_s = f"{ort:.2f}" if ort else "✗"
    vs_ort = f"{ort/sf:.2f}×" if ort else "—"
    print(f"{name:<25s} {sj:>8.2f} {comp_s:>9s} {ort_s:>9s} {sf:>8.2f} {sj/sf:>8.2f}× {vs_ort:>9s}")
print("=" * 120)
