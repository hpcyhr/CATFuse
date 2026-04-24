"""TVM full benchmark - all CATFuse networks."""
import os, sys, time, statistics, warnings
import numpy as np
sys.path.insert(0, '.')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import torch
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

T = 4
N_W, N_I, N_R = 20, 50, 5
TMP = '/data/dagongcheng/yhrtest/tmp'
os.makedirs(TMP, exist_ok=True)

print(f'TVM {tvm.__version__}, PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

def bench_torch(model, x_seq):
    model.eval(); times = []
    for _ in range(N_R):
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad(): model(x_seq)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad(): model(x_seq)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)/N_I*1000)
    return statistics.median(times)

def try_tvm(model, B, C, H, name):
    onnx_path = f'{TMP}/{name}_ss.onnx'
    model.eval()
    functional.set_step_mode(model, 's')
    functional.reset_net(model)
    x_single = torch.rand(B, C, H, H, device='cuda:0')
    try:
        with torch.no_grad():
            torch.onnx.export(model, x_single, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=13, do_constant_folding=True)
    except Exception as e:
        return None, f'ONNX:{type(e).__name__}'
    try:
        onnx_model = onnx.load(onnx_path)
        mod = relax_onnx.from_onnx(onnx_model, shape_dict={'input': [B,C,H,H]},
                                    dtype_dict={'input': 'float32'}, keep_params_in_input=False)
        mod, params = relax.frontend.detach_params(mod)
    except Exception as e:
        return None, f'Relax:{type(e).__name__}'
    try:
        target = tvm.target.cuda()
        with target:
            opt_mod = relax.get_pipeline("zero")(mod)
            opt_mod = tvm.tir.transform.DefaultGPUSchedule()(opt_mod)
        ex = tvm.compile(opt_mod, target="cuda")
    except Exception as e:
        return None, f'Compile:{type(e).__name__}'
    try:
        dev = tvm.device("cuda", 0)
        vm = relax.VirtualMachine(ex, dev)
        inp = tvm.nd.array(np.random.randn(B,C,H,H).astype('float32'), dev)
        times = []
        for _ in range(N_R):
            for _ in range(N_W):
                for t in range(T): vm["main"](inp)
            dev.sync(); t0 = time.perf_counter()
            for _ in range(N_I):
                for t in range(T): vm["main"](inp)
            dev.sync(); times.append((time.perf_counter()-t0)/N_I*1000)
        return statistics.median(times), 'OK'
    except Exception as e:
        return None, f'Run:{type(e).__name__}'

print('='*85)
print(f"{'Network':<15} {'SJ':>8} {'TVM':>8} {'TVM/SJ':>8} {'Status'}")
print('-'*85)

configs = [
    # CIFAR
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN34', lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN101', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('AlexNet-C', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('ZFNet-C', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('MobV1-C', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    # ImageNet
    ('SEW-RN18-I', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN50-I', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SpikRN18-I', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('AlexNet-I', lambda: SpikingAlexNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('ZFNet-I', lambda: SpikingZFNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('MobV1-I', lambda: SpikingMobileNetV1(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
]

nok = nfail = 0
for name, build_fn, B, C, H in configs:
    x_seq = torch.rand(T, B, C, H, H, device='cuda:0')
    m = build_fn().to('cuda:0').eval()
    functional.set_step_mode(m, 'm')
    sj = bench_torch(m, x_seq)

    functional.set_step_mode(m, 's')
    tvm_ms, status = try_tvm(m, B, C, H, name.replace('-','_'))

    if tvm_ms:
        print(f'  {name:<13} {sj:>7.2f} {tvm_ms:>7.2f} {sj/tvm_ms:>7.2f}x {status}')
        nok += 1
    else:
        print(f'  {name:<13} {sj:>7.2f}    FAIL     N/A  {status}')
        nfail += 1

    del m, x_seq
    torch.cuda.empty_cache()

print('='*85)
print(f'OK: {nok}, FAIL: {nfail}')
