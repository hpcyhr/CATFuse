"""
Test 33: CUDA Graph for end-to-end CATFuse inference.

Captures the entire fused model forward as a CUDA graph,
eliminating per-layer kernel launch overhead.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg
from catfuse.substitute import substitute, substitute_sf

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 100
N_REPEAT = 3

def bench_normal(model, x_in):
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

def bench_cuda_graph(model, x_static):
    """Capture and replay model forward as CUDA Graph."""
    # Warmup & ensure all lazy init done
    for _ in range(5):
        functional.reset_net(model)
        for m in model.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): _ = model(x_static)
    torch.cuda.synchronize()

    # We need to handle reset_net inside the graph
    # Since reset clears v states, we do it outside the graph
    # and only capture the forward pass

    # Create static output buffer
    functional.reset_net(model)
    for m in model.modules():
        if hasattr(m, 'reset'): m.reset()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out_static = model(x_static)
    torch.cuda.synchronize()

    # Benchmark replay
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times), out_static

# ================================================================
# Test 1: SEW-ResNet18 ImageNet
# ================================================================
print("=" * 80)
print("CUDA Graph acceleration")
print("=" * 80)

for arch_name, build_fn, T, B, C, H, num_classes in [
    ("SEW-RN18 ImgNet", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000, cnf='ADD',
        spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224, 1000),
    ("SEW-RN18 CIFAR", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
        spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32, 10),
    ("VGG11 CIFAR", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode, tau=2.0,
        num_classes=10), 4, 2, 3, 32, 10),
]:
    print(f"\n--- {arch_name} (T={T}, B={B}, {H}×{H}) ---")
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ baseline
    sj_ms = bench_normal(net, x)

    # Dense
    net_d, _ = substitute(net)
    net_d = net_d.to(device).eval()
    functional.set_step_mode(net_d, 'm')
    dense_ms = bench_normal(net_d, x)

    # Dense + CUDA Graph
    try:
        # Fresh model for graph capture
        net_dg, _ = substitute(net)
        net_dg = net_dg.to(device).eval()
        functional.set_step_mode(net_dg, 'm')
        x_static = x.clone()  # static input for graph
        graph_ms, _ = bench_cuda_graph(net_dg, x_static)
    except Exception as e:
        graph_ms = float('inf')
        print(f"  CUDA Graph failed: {e}")

    print(f"  SJ:           {sj_ms:.2f} ms")
    print(f"  Dense:        {dense_ms:.2f} ms ({sj_ms/dense_ms:.2f}×)")
    if graph_ms != float('inf'):
        print(f"  Dense+Graph:  {graph_ms:.2f} ms ({sj_ms/graph_ms:.2f}×)")
        print(f"  Graph vs Dense: {dense_ms/graph_ms:.2f}×")

print("\n" + "=" * 80)
