"""
Test 40: TorchScript baseline.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP, N_ITER, N_REPEAT = 20, 50, 3

def bench(fn):
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER): fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

print("=" * 90)
print("TorchScript + torch.jit.optimize_for_inference baseline")
print("=" * 90)

T = 4
configs = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("VGG11-BN", lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

for name, build_fn, B, C, H in configs:
    print(f"\n  {name:<15s}", end="", flush=True)
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ baseline
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_ms = bench(lambda: (functional.reset_net(net), net(x)))

    # TorchScript trace
    try:
        functional.reset_net(net)
        with torch.no_grad():
            traced = torch.jit.trace(net, x)
        traced = torch.jit.optimize_for_inference(traced)

        # Warmup
        for _ in range(20):
            with torch.no_grad(): _ = traced(x)

        ts_ms = bench(lambda: traced(x))
        print(f" SJ={sj_ms:.2f}ms  TorchScript={ts_ms:.2f}ms  ratio={sj_ms/ts_ms:.2f}×")
    except Exception as e:
        print(f" SJ={sj_ms:.2f}ms  TorchScript=✗ ({type(e).__name__})")

    # TorchScript script (alternative)
    try:
        scripted = torch.jit.script(net)
        scripted = torch.jit.optimize_for_inference(scripted)
        for _ in range(20):
            functional.reset_net(scripted)
            with torch.no_grad(): _ = scripted(x)
        scr_ms = bench(lambda: (functional.reset_net(scripted), scripted(x)))
        print(f"                 jit.script={scr_ms:.2f}ms  ratio={sj_ms/scr_ms:.2f}×")
    except Exception as e:
        print(f"                 jit.script=✗ ({type(e).__name__})")

print("\n" + "=" * 90)
