"""test_55_tvm_spikformer.py — TVM benchmark for SpikFormer/QKFormer

补充 test_tvm_benchmark_full.py 和 test_tvm_autotune_full.py 中缺失的两个网络。
需要在 ln conda 环境中运行（有 TVM 0.22 + relax）。
"""
import sys, os, time, statistics, warnings
import numpy as np
sys.path.insert(0, '/data/dagongcheng/yhrtest/CATFuse')
sys.path.insert(0, '/data/dagongcheng/yhrtest/CATFuse/models')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import torch, onnx, tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional
from spikformer_github import SpikformerGithub
from qkformer_github import QKFormerGithub

T, B = 4, 2
N_W, N_I, N_R = 20, 50, 5
TMP = '/data/dagongcheng/yhrtest/tmp'
device = 'cuda:0'


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
    x = torch.rand(B, C, H, H, device=device)
    try:
        with torch.no_grad():
            torch.onnx.export(model, x, onnx_path, input_names=['input'],
                              output_names=['output'], opset_version=13,
                              do_constant_folding=True)
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
            opt_mod = relax.get_pipeline('zero')(mod)
            opt_mod = tvm.tir.transform.DefaultGPUSchedule()(opt_mod)
        ex = tvm.compile(opt_mod, target='cuda')
    except Exception as e:
        return None, f'Compile:{type(e).__name__}'
    try:
        dev = tvm.device('cuda', 0)
        vm = relax.VirtualMachine(ex, dev)
        inp = tvm.nd.array(np.random.randn(B,C,H,H).astype('float32'), dev)
        times = []
        for _ in range(N_R):
            for _ in range(N_W):
                for t in range(T): vm['main'](inp)
            dev.sync(); t0 = time.perf_counter()
            for _ in range(N_I):
                for t in range(T): vm['main'](inp)
            dev.sync(); times.append((time.perf_counter()-t0)/N_I*1000)
        return statistics.median(times), 'OK'
    except Exception as e:
        return None, f'Run:{type(e).__name__}'


for name, build_fn in [
    ('SpikFormer', lambda: SpikformerGithub(num_classes=10)),
    ('QKFormer', lambda: QKFormerGithub(num_classes=10)),
]:
    print(f'{name}:')
    x_seq = torch.rand(T, B, 3, 32, 32, device=device)
    m = build_fn().to(device).eval()
    functional.set_step_mode(m, 'm')
    sj = bench_torch(m, x_seq)

    functional.set_step_mode(m, 's')
    tvm_ms, status = try_tvm(m, B, 3, 32, name)

    if tvm_ms:
        print(f'  SJ={sj:.2f}  TVM={tvm_ms:.2f}  {sj/tvm_ms:.2f}x  {status}')
    else:
        print(f'  SJ={sj:.2f}  TVM=FAIL  {status}')

    del m, x_seq
    torch.cuda.empty_cache()