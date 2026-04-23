"""Test 48: TensorRT vs SJ-torch vs CATFuse on representative networks."""
import sys, time, statistics, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

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

def try_onnx_trt(model, x, name):
    import tensorrt as trt
    onnx_path = f'/data/tmp/{name}.onnx'

    model.eval()
    functional.reset_net(model)

    # ONNX export
    try:
        with torch.no_grad():
            torch.onnx.export(model, x, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=13, do_constant_folding=True)
        print(f"    ONNX export OK ({os.path.getsize(onnx_path)//1024} KB)")
    except Exception as e:
        print(f"    ONNX export FAILED: {type(e).__name__}: {str(e)[:120]}")
        return None

    # TRT engine build
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"    TRT parse error: {parser.get_error(i)}")
                return None
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print(f"    TRT engine build FAILED")
            return None
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        print(f"    TRT engine OK")
    except Exception as e:
        print(f"    TRT engine FAILED: {type(e).__name__}: {str(e)[:120]}")
        return None

    # TRT inference benchmark
    try:
        in_name = engine.get_tensor_name(0)
        out_name = engine.get_tensor_name(1)
        out_shape = list(context.get_tensor_shape(out_name))
        input_buf = x.contiguous().cuda()
        output_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
        context.set_tensor_address(in_name, input_buf.data_ptr())
        context.set_tensor_address(out_name, output_buf.data_ptr())
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
        print(f"    TRT inference OK: {statistics.median(times):.2f} ms")
        return statistics.median(times)
    except Exception as e:
        print(f"    TRT inference FAILED: {type(e).__name__}: {str(e)[:120]}")
        return None

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")
import tensorrt as trt
print(f"TensorRT: {trt.__version__}")
print("=" * 85)

configs = [
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
]

print(f"{'Network':<13s} {'SJ-torch':>10s} {'TensorRT':>10s} {'TRT/SJ':>8s}")
print("-" * 45)

for name, build_fn, B_cfg, C, H in configs:
    print(f"\n  {name}:")
    x = torch.rand(T, B_cfg, C, H, H, device=device)

    net_sj = build_fn().to(device).eval()
    functional.set_step_mode(net_sj, 'm')
    sj = bench_torch(net_sj, x)

    trt_ms = try_onnx_trt(net_sj, x, name.replace('-','_'))

    trt_str = f"{trt_ms:.2f}" if trt_ms else "FAIL"
    trt_ratio = f"{sj/trt_ms:.2f}x" if trt_ms else "N/A"
    print(f"  Result: SJ={sj:.2f}ms  TRT={trt_str}ms  ratio={trt_ratio}")

    del net_sj, x
    torch.cuda.empty_cache()

print("\n" + "=" * 85)
