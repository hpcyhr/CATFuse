"""TVM full benchmark with MetaSchedule autotuning - ALL CATFuse networks."""
import os, sys, time, statistics, warnings
import numpy as np
sys.path.insert(0, '.')
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import torch
import onnx
import tvm
from tvm import relax, tir
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from models.spiking_alexnet import SpikingAlexNet
from models.spiking_zfnet import SpikingZFNet
from models.spiking_mobilenet import SpikingMobileNetV1

T = 4
N_W, N_I, N_R = 20, 50, 5
TMP = '/data/dagongcheng/yhrtest/tmp'
TUNE_DIR = '/data/dagongcheng/yhrtest/tvm_tune_logs'
os.makedirs(TMP, exist_ok=True)
os.makedirs(TUNE_DIR, exist_ok=True)

TUNE_TRIALS = 1000

print(f'TVM {tvm.__version__}, PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'AutoTuning trials: {TUNE_TRIALS}')
print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')

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

def try_tvm_tuned(model, B, C, H, name):
    onnx_path = f'{TMP}/{name}_ss.onnx'
    work_dir = f'{TUNE_DIR}/{name}'
    os.makedirs(work_dir, exist_ok=True)

    model.eval()
    functional.set_step_mode(model, 's')
    functional.reset_net(model)
    x_single = torch.rand(B, C, H, H, device='cuda:0')

    # Step 1: ONNX export
    try:
        with torch.no_grad():
            torch.onnx.export(model, x_single, onnx_path,
                              input_names=['input'], output_names=['output'],
                              opset_version=13, do_constant_folding=True)
        print(f'    ONNX export OK')
    except Exception as e:
        return None, None, f'ONNX:{type(e).__name__}'

    # Step 2: Relax import
    try:
        onnx_model = onnx.load(onnx_path)
        mod = relax_onnx.from_onnx(onnx_model, shape_dict={'input': [B,C,H,H]},
                                    dtype_dict={'input': 'float32'}, keep_params_in_input=False)
        mod, params = relax.frontend.detach_params(mod)
        print(f'    Relax import OK')
    except Exception as e:
        return None, None, f'Relax:{type(e).__name__}'

    # Step 3: Compile with autotuning
    target = tvm.target.cuda()
    tune_time = None
    try:
        print(f'    MetaSchedule tuning ({TUNE_TRIALS} trials)...')
        tune_start = time.perf_counter()
        with target:
            optimized_mod = tvm.ir.transform.Sequential([
                relax.get_pipeline("zero"),
                relax.transform.MetaScheduleTuneTIR(
                    work_dir=work_dir,
                    max_trials_global=TUNE_TRIALS,
                ),
                relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir),
                tvm.tir.transform.DefaultGPUSchedule(),
            ])(mod)
        tune_time = time.perf_counter() - tune_start
        print(f'    Tuning done in {tune_time:.1f}s')
        ex = tvm.compile(optimized_mod, target="cuda")
    except Exception as e:
        print(f'    MetaSchedule failed: {e}')
        print(f'    Fallback to zero pipeline...')
        try:
            tune_start = time.perf_counter()
            with target:
                optimized_mod = relax.get_pipeline("zero")(mod)
                optimized_mod = tvm.tir.transform.DefaultGPUSchedule()(optimized_mod)
            tune_time = time.perf_counter() - tune_start
            ex = tvm.compile(optimized_mod, target="cuda")
            print(f'    Fallback OK ({tune_time:.1f}s)')
        except Exception as e2:
            return None, tune_time, f'Compile:{type(e2).__name__}'

    # Step 4: Benchmark
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
        return statistics.median(times), tune_time, 'OK'
    except Exception as e:
        return None, tune_time, f'Run:{type(e).__name__}'

print('='*105)
print(f"{'Network':<15} {'SJ':>7} {'TVM-tune':>9} {'TVM/SJ':>7} {'Tune(s)':>8} {'Status'}")
print('-'*105)

configs = [
    # ===== CIFAR (32x32, B=2) =====
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG13-BN', lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG16-BN', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('VGG19-BN', lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN34', lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN50', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN101', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SEW-RN152', lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=10, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN34', lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN50', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN101', lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('SpikRN152', lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ('AlexNet-C', lambda: SpikingAlexNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('ZFNet-C', lambda: SpikingZFNet(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    ('MobV1-C', lambda: SpikingMobileNetV1(num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32), 2, 3, 32),
    # ===== ImageNet (224x224, B=1) =====
    ('VGG11-I', lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('VGG13-I', lambda: spiking_vgg.spiking_vgg13_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('VGG16-I', lambda: spiking_vgg.spiking_vgg16_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('VGG19-I', lambda: spiking_vgg.spiking_vgg19_bn(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SEW-RN18-I', lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN34-I', lambda: sew_resnet.sew_resnet34(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN50-I', lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN101-I', lambda: sew_resnet.sew_resnet101(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SEW-RN152-I', lambda: sew_resnet.sew_resnet152(pretrained=False, num_classes=1000, cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
    ('SpikRN18-I', lambda: spiking_resnet.spiking_resnet18(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SpikRN34-I', lambda: spiking_resnet.spiking_resnet34(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SpikRN50-I', lambda: spiking_resnet.spiking_resnet50(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SpikRN101-I', lambda: spiking_resnet.spiking_resnet101(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('SpikRN152-I', lambda: spiking_resnet.spiking_resnet152(spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=1000), 1, 3, 224),
    ('AlexNet-I', lambda: SpikingAlexNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('ZFNet-I', lambda: SpikingZFNet(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
    ('MobV1-I', lambda: SpikingMobileNetV1(num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=224), 1, 3, 224),
]

# SpikFormer/QKFormer (CIFAR only, need special import)
try:
    sys.path.insert(0, 'models')
    from spikformer import SpikFormer
    from qkformer import QKFormer
    configs.insert(17, ('SpikFormer', lambda: SpikFormer(num_classes=10), 2, 3, 32))
    configs.insert(18, ('QKFormer', lambda: QKFormer(num_classes=10), 2, 3, 32))
    print('SpikFormer/QKFormer loaded')
except Exception as e:
    print(f'SpikFormer/QKFormer not available: {e}')

nok = nfail = 0
total_tune = 0.0

for name, build_fn, B, C, H in configs:
    print(f'\n[{time.strftime("%H:%M:%S")}] {name}:')
    x_seq = torch.rand(T, B, C, H, H, device='cuda:0')
    m = build_fn().to('cuda:0').eval()
    functional.set_step_mode(m, 'm')
    sj = bench_torch(m, x_seq)

    functional.set_step_mode(m, 's')
    tvm_ms, tune_s, status = try_tvm_tuned(m, B, C, H, name.replace('-','_'))

    tune_str = f'{tune_s:.0f}' if tune_s else 'N/A'
    if tune_s: total_tune += tune_s

    if tvm_ms:
        print(f'  {name:<13} {sj:>6.2f} {tvm_ms:>8.2f} {sj/tvm_ms:>6.2f}x {tune_str:>7}s {status}')
        nok += 1
    else:
        print(f'  {name:<13} {sj:>6.2f}     FAIL    N/A {tune_str:>7}s {status}')
        nfail += 1

    del m, x_seq
    torch.cuda.empty_cache()

print('\n' + '='*105)
print(f'Done. OK: {nok}, FAIL: {nfail}, Total tuning time: {total_tune/3600:.1f} hours')
print(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
