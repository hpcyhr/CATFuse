"""TVM benchmark - single-step export, Python T-loop (same as their scripts)."""
import os, sys, time, statistics, warnings
import numpy as np
sys.path.insert(0, '.')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

T = 4
N_W, N_I, N_R = 20, 50, 5
TMP = '/data/dagongcheng/yhrtest/tmp'
os.makedirs(TMP, exist_ok=True)

print(f'TVM {tvm.__version__}, PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

def bench_torch(model, x_seq):
    """Benchmark SJ multi-step model."""
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

def try_tvm_singlestep(model, B, C, H, name):
    """Export single-step model, compile with TVM, benchmark with T-loop."""
    onnx_path = f'{TMP}/{name}_ss.onnx'

    # Step 1: Set single-step mode and export
    model.eval()
    functional.set_step_mode(model, 's')
    functional.reset_net(model)
    x_single = torch.rand(B, C, H, H, device='cuda:0')

    try:
        with torch.no_grad():
            torch.onnx.export(model, x_single, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=13, do_constant_folding=True)
        print(f'    ONNX single-step export OK')
    except Exception as e:
        return None, f'ONNX: {type(e).__name__}: {str(e)[:80]}'

    # Step 2: Import to TVM relax
    try:
        onnx_model = onnx.load(onnx_path)
        shape_dict = {'input': [B, C, H, H]}
        dtype_dict = {'input': 'float32'}
        mod = relax_onnx.from_onnx(onnx_model, shape_dict=shape_dict,
                                    dtype_dict=dtype_dict, keep_params_in_input=False)
        mod, params = relax.frontend.detach_params(mod)
        print(f'    Relax import OK')
    except Exception as e:
        return None, f'Relax: {type(e).__name__}: {str(e)[:80]}'

    # Step 3: Compile
    try:
        target = tvm.target.cuda()
        with target:
            opt_mod = relax.get_pipeline("zero")(mod)
            opt_mod = tvm.tir.transform.DefaultGPUSchedule()(opt_mod)
        ex = tvm.compile(opt_mod, target="cuda")
        print(f'    TVM compile OK')
    except Exception as e:
        return None, f'Compile: {type(e).__name__}: {str(e)[:80]}'

    # Step 4: Benchmark with T-loop (same as their scripts)
    try:
        dev = tvm.device("cuda", 0)
        vm = relax.VirtualMachine(ex, dev)
        input_np = np.random.randn(B, C, H, H).astype('float32')
        tvm_input = tvm.nd.array(input_np, dev)

        times = []
        for _ in range(N_R):
            for _ in range(N_W):
                for t in range(T):
                    vm["main"](tvm_input)
            dev.sync()
            t0 = time.perf_counter()
            for _ in range(N_I):
                for t in range(T):
                    vm["main"](tvm_input)
            dev.sync()
            times.append((time.perf_counter()-t0)/N_I*1000)
        return statistics.median(times), 'OK'
    except Exception as e:
        return None, f'Run: {type(e).__name__}: {str(e)[:80]}'

print('='*80)
print(f"{'Network':<15} {'SJ':>8} {'TVM':>8} {'TVM/SJ':>8} {'Status'}")
print('-'*80)

configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18-I', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

for name, build_fn, B, C, H in configs:
    print(f'\n  {name}:')
    x_seq = torch.rand(T, B, C, H, H, device='cuda:0')
    m = build_fn().to('cuda:0').eval()
    functional.set_step_mode(m, 'm')

    sj = bench_torch(m, x_seq)

    # Reset to single-step for TVM export
    functional.set_step_mode(m, 's')
    tvm_ms, status = try_tvm_singlestep(m, B, C, H, name.replace('-','_'))

    if tvm_ms:
        ratio = sj / tvm_ms
        print(f'  {name:<13} {sj:>7.2f} {tvm_ms:>7.2f} {ratio:>7.2f}x {status}')
    else:
        print(f'  {name:<13} {sj:>7.2f}    FAIL     N/A  {status}')

    del m, x_seq
    torch.cuda.empty_cache()

print('\n' + '='*80)
