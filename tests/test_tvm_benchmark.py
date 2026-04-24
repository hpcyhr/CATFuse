"""TVM benchmark on SNN networks using relax API."""
import os, sys, time, statistics, warnings
import numpy as np
sys.path.insert(0, '.')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
warnings.filterwarnings('ignore')

import torch
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
T, B = 4, 2
N_W, N_I, N_R = 20, 50, 5

print(f'TVM {tvm.__version__}, PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

def bench_torch(model, x):
    model.eval(); times = []
    for _ in range(N_R):
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad(): model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad(): model(x)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)/N_I*1000)
    return statistics.median(times)

def try_tvm(model, x, name):
    onnx_path = f'/data/dagongcheng/yhrtest/tmp/{name}_tvm.onnx'
    
    # Step 1: ONNX export
    model.eval(); functional.reset_net(model)
    try:
        with torch.no_grad():
            torch.onnx.export(model, x, onnx_path, input_names=['input'],
                              output_names=['output'], opset_version=13,
                              do_constant_folding=True)
        print(f'    ONNX export OK ({os.path.getsize(onnx_path)//1024} KB)')
    except Exception as e:
        return None, f'ONNX export: {type(e).__name__}: {str(e)[:80]}'

    # Step 2: Import to TVM via relax
    try:
        onnx_model = onnx.load(onnx_path)
        shape_dict = {'input': list(x.shape)}
        dtype_dict = {'input': 'float32'}
        mod = relax_onnx.from_onnx(onnx_model, shape_dict=shape_dict,
                                    dtype_dict=dtype_dict, keep_params_in_input=False)
        mod, params = relax.frontend.detach_params(mod)
        print(f'    Relax import OK')
    except Exception as e:
        return None, f'Relax import: {type(e).__name__}: {str(e)[:80]}'

    # Step 3: Compile (basic pipeline, no autotuning)
    try:
        target = tvm.target.cuda()
        with target:
            optimized_mod = relax.get_pipeline("zero")(mod)
            optimized_mod = tvm.tir.transform.DefaultGPUSchedule()(optimized_mod)
        ex = tvm.compile(optimized_mod, target="cuda")
        print(f'    TVM compile OK')
    except Exception as e:
        return None, f'TVM compile: {type(e).__name__}: {str(e)[:80]}'

    # Step 4: Benchmark with VirtualMachine
    try:
        dev = tvm.device("cuda", 0)
        vm = relax.VirtualMachine(ex, dev)
        input_np = x.cpu().numpy()
        tvm_input = tvm.nd.array(input_np, dev)

        times = []
        for _ in range(N_R):
            for _ in range(N_W):
                vm["main"](tvm_input)
            dev.sync()
            t0 = time.perf_counter()
            for _ in range(N_I):
                vm["main"](tvm_input)
            dev.sync()
            times.append((time.perf_counter()-t0)/N_I*1000)
        return statistics.median(times), 'OK'
    except Exception as e:
        return None, f'TVM run: {type(e).__name__}: {str(e)[:80]}'

print('='*80)
print(f"{'Network':<15} {'SJ':>8} {'TVM':>8} {'TVM/SJ':>8} {'Status'}")
print('-'*80)

configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 32),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 32),
    ('SEW-RN18-I', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 224),
]

for name, build_fn, B_cfg, H in configs:
    print(f'\n  {name}:')
    x = torch.rand(T, B_cfg, 3, H, H, device=device)
    m = build_fn().to(device).eval()
    functional.set_step_mode(m, 'm')

    sj = bench_torch(m, x)
    tvm_ms, status = try_tvm(m, x, name.replace('-','_'))

    if tvm_ms:
        print(f'  {name:<13} {sj:>7.2f} {tvm_ms:>7.2f} {sj/tvm_ms:>7.2f}x {status}')
    else:
        print(f'  {name:<13} {sj:>7.2f}    FAIL     N/A  {status}')

    del m, x
    torch.cuda.empty_cache()

print('\n' + '='*80)
