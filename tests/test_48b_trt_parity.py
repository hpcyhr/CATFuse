"""Test 48b: TensorRT parity check + extended benchmark."""
import sys, time, statistics, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
import numpy as np
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet

device = 'cuda:0'
T, B = 4, 2

def build_trt_engine(model, x, name):
    import tensorrt as trt
    onnx_path = f'/data/tmp/{name}.onnx'
    model.eval()
    functional.reset_net(model)
    try:
        with torch.no_grad():
            torch.onnx.export(model, x, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=13, do_constant_folding=True)
    except Exception as e:
        print(f"  ONNX export FAILED: {e}")
        return None, None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return None, None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return None, None
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    return engine, context

def trt_infer(engine, context, x):
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    out_shape = list(context.get_tensor_shape(out_name))
    output = torch.empty(out_shape, dtype=torch.float32, device=device)
    context.set_tensor_address(in_name, x.contiguous().data_ptr())
    context.set_tensor_address(out_name, output.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")
print()

# ============================================================
# Part 1: Parity check — does TRT produce same output as SJ?
# ============================================================
print("=" * 70)
print("PART 1: Parity check (TRT vs SJ-torch)")
print("=" * 70)

torch.manual_seed(42)
x_fixed = torch.randn(T, B, 3, 32, 32, device=device)

for name, build_fn in [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0)),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
]:
    print(f"\n  {name}:")
    model = build_fn().to(device).eval()
    functional.set_step_mode(model, 'm')

    # SJ reference output
    functional.reset_net(model)
    with torch.no_grad():
        sj_out = model(x_fixed).clone()

    # TRT output
    engine, context = build_trt_engine(model, x_fixed, f"parity_{name}")
    if engine is None:
        print("    SKIP (engine build failed)")
        continue
    trt_out = trt_infer(engine, context, x_fixed)

    # Compare
    if sj_out.shape != trt_out.shape:
        print(f"    Shape mismatch: SJ={list(sj_out.shape)} TRT={list(trt_out.shape)}")
    else:
        max_diff = (sj_out - trt_out).abs().max().item()
        mean_diff = (sj_out - trt_out).abs().mean().item()
        # For spike outputs (0/1), check spike match
        sj_spikes = (sj_out > 0.5).float()
        trt_spikes = (trt_out > 0.5).float()
        spike_match = (sj_spikes == trt_spikes).float().mean().item() * 100
        print(f"    max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        print(f"    spike_match={spike_match:.2f}%")
        print(f"    SJ output range: [{sj_out.min():.4f}, {sj_out.max():.4f}]")
        print(f"    TRT output range: [{trt_out.min():.4f}, {trt_out.max():.4f}]")

    del model, engine, context
    torch.cuda.empty_cache()

# ============================================================
# Part 2: Extended benchmark — more networks
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Extended TensorRT benchmark")
print("=" * 70)

N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

def bench_trt(engine, context, x):
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    out_shape = list(context.get_tensor_shape(out_name))
    output = torch.empty(out_shape, dtype=torch.float32, device=device)
    context.set_tensor_address(in_name, x.contiguous().data_ptr())
    context.set_tensor_address(out_name, output.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            context.execute_async_v3(stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            context.execute_async_v3(stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

def bench_torch(model, x):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

configs = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN34", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN50", lambda: spiking_resnet.spiking_resnet50(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
]

print(f"\n{'Network':<13s} {'SJ-torch':>9s} {'TensorRT':>9s} {'TRT/SJ':>8s}")
print("-" * 43)

for name, build_fn, B_cfg, C, H in configs:
    x = torch.rand(T, B_cfg, C, H, H, device=device)
    model = build_fn().to(device).eval()
    functional.set_step_mode(model, 'm')

    sj = bench_torch(model, x)
    engine, context = build_trt_engine(model, x, name)
    if engine:
        trt_ms = bench_trt(engine, context, x)
        print(f"  {name:<11s} {sj:>8.2f} {trt_ms:>8.2f} {sj/trt_ms:>7.2f}x")
    else:
        print(f"  {name:<11s} {sj:>8.2f}     FAIL     N/A")

    del model, x
    if engine: del engine, context
    torch.cuda.empty_cache()

print("=" * 70)
