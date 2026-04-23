"""Test 48c: TensorRT on all 26 CATFuse configurations."""
import sys, time, statistics, os, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

def bench_torch(model, x):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad(): _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

def try_trt(model, x, name):
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
    except Exception as e:
        return None, f"ONNX fail: {type(e).__name__}"

    # TRT engine
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                return None, "TRT parse fail"
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            return None, "TRT build fail"
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
    except Exception as e:
        return None, f"TRT engine: {type(e).__name__}"

    # Benchmark
    try:
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
            torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
            for _ in range(N_ITER):
                context.execute_async_v3(stream)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_ITER * 1000)
        del engine, context
        return statistics.median(times), "OK"
    except Exception as e:
        return None, f"TRT infer: {type(e).__name__}"

# Models
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

gpu = torch.cuda.get_device_name(0)
import tensorrt as trt
print(f"GPU: {gpu}")
print(f"TensorRT: {trt.__version__}")
print("=" * 95)
print(f"{'Network':<18s} {'T':>2s} {'B':>2s} {'H':>4s} {'SJ-torch':>9s} {'TRT':>9s} {'TRT/SJ':>8s} {'Status':<25s}")
print("-" * 95)

configs = [
    # CIFAR (32x32)
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("VGG13-BN", lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("VGG19-BN", lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SEW-RN34", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SEW-RN101", lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SEW-RN152", lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SpikRN34", lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SpikRN50", lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SpikRN101", lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SpikRN152", lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("AlexNet-C", lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 4, 2, 3, 32),
    ("ZFNet-C", lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 4, 2, 3, 32),
    ("MobV1-C", lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 4, 2, 3, 32),
    # ImageNet (224x224)
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
    ("SEW-RN34-I", lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
    ("SEW-RN50-I", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
    ("VGG11-I", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 4, 1, 3, 224),
    ("AlexNet-I", lambda: SpikingAlexNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 4, 1, 3, 224),
    ("ZFNet-I", lambda: SpikingZFNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 4, 1, 3, 224),
    ("MobV1-I", lambda: SpikingMobileNetV1(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 4, 1, 3, 224),
]

# SpikFormer/QKFormer — skip if import fails
try:
    sys.path.insert(0, '/data/yhr/CATFuse/ctf_models')
    from spikformer import SpikFormer_SJ
    configs.append(("SpikFormer", lambda: SpikFormer_SJ(), 4, 2, 3, 32))
except:
    print("  SpikFormer import skipped")
try:
    from qkformer import QKFormer_SJ
    configs.append(("QKFormer", lambda: QKFormer_SJ(), 4, 2, 3, 32))
except:
    print("  QKFormer import skipped")

n_ok, n_fail = 0, 0
for name, build_fn, T, B, C, H in configs:
    x = torch.rand(T, B, C, H, H, device=device)
    model = build_fn().to(device).eval()
    functional.set_step_mode(model, 'm')

    sj = bench_torch(model, x)
    trt_ms, status = try_trt(model, x, name.replace('-','_').replace(' ','_'))

    if trt_ms is not None:
        ratio = f"{sj/trt_ms:.2f}x"
        trt_str = f"{trt_ms:.2f}"
        n_ok += 1
    else:
        ratio = "N/A"
        trt_str = "FAIL"
        n_fail += 1

    print(f"  {name:<16s} {T:>2d} {B:>2d} {H:>4d} {sj:>8.2f} {trt_str:>8s} {ratio:>7s}  {status:<25s}")

    del model, x
    torch.cuda.empty_cache()

print("=" * 95)
print(f"Summary: {n_ok} OK, {n_fail} FAIL out of {n_ok+n_fail} total")
