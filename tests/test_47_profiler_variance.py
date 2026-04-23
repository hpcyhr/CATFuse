"""
Test 47: Kernel launch count (torch.profiler) + latency with std.
"""
import sys, time, statistics, math
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from catfuse.substitute import substitute_sf

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
T, B = 4, 2
N_WARMUP, N_ITER = 20, 50
N_REPEAT = 5  # 5 runs for std

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")

# ============================================================
# Part 1: Kernel launch count via torch.profiler
# ============================================================
print("\n" + "=" * 90)
print("PART 1: Kernel launch count (torch.profiler)")
print("=" * 90)

def count_kernels(model, x):
    """Run one forward under profiler and count CUDA kernel launches."""
    functional.reset_net(model)
    for m in model.modules():
        if hasattr(m, 'reset'): m.reset()

    # Warmup outside profiler
    for _ in range(5):
        functional.reset_net(model)
        for m in model.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): _ = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        functional.reset_net(model)
        for m in model.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): _ = model(x)
        torch.cuda.synchronize()

    # Count unique CUDA kernel launches
    events = prof.key_averages()
    total_cuda_calls = 0
    kernel_details = []
    for evt in events:
        if evt.device_type == torch.autograd.DeviceType.CUDA or evt.cuda_time_total > 0:
            total_cuda_calls += evt.count
            if evt.count > 0:
                kernel_details.append((evt.key, evt.count, evt.cuda_time_total / 1000))

    return total_cuda_calls, kernel_details

configs_profiler = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 3, 32),
]

print(f"\n{'Network':<15s} {'SJ kernels':>12s} {'CF kernels':>12s} {'reduction':>12s}")
print("-" * 55)

for name, build_fn, C, H in configs_profiler:
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ
    net_sj = build_fn().to(device).eval()
    functional.set_step_mode(net_sj, 'm')
    sj_count, sj_details = count_kernels(net_sj, x)

    # CATFuse
    net_cf = build_fn().to(device).eval()
    functional.set_step_mode(net_cf, 'm')
    net_cf, _ = substitute_sf(net_cf, T=T)
    net_cf = net_cf.to(device).eval()
    functional.set_step_mode(net_cf, 'm')
    cf_count, cf_details = count_kernels(net_cf, x)

    ratio = cf_count / sj_count if sj_count > 0 else 0
    print(f"  {name:<13s} {sj_count:>11d} {cf_count:>11d} {ratio:>11.2f}x")

    # Print top-10 kernels for SJ
    print(f"    SJ top kernels:")
    sj_details.sort(key=lambda x: -x[1])
    for kname, cnt, ms in sj_details[:8]:
        short = kname[:50]
        print(f"      {short:<50s} x{cnt:>4d}  {ms:>7.2f}ms")

    print(f"    CATFuse top kernels:")
    cf_details.sort(key=lambda x: -x[1])
    for kname, cnt, ms in cf_details[:8]:
        short = kname[:50]
        print(f"      {short:<50s} x{cnt:>4d}  {ms:>7.2f}ms")
    print()

    del net_sj, net_cf, x
    torch.cuda.empty_cache()


# ============================================================
# Part 2: Latency with std (5 repeats)
# ============================================================
print("\n" + "=" * 90)
print("PART 2: End-to-end latency with variance (N=5 repeats)")
print("=" * 90)

def bench_with_stats(model, x_in):
    model.eval()
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
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    mn = min(times)
    mx = max(times)
    return med, std, mn, mx

configs_var = [
    ("SEW-RN18", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("SEW-RN50", lambda: sew_resnet.sew_resnet50(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 2, 3, 32),
    ("VGG16-BN", lambda: spiking_vgg.spiking_vgg16_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SpikRN18", lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10), 2, 3, 32),
    ("SEW-RN18-I", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 1, 3, 224),
]

print(f"\n{'Network':<15s} {'SJ med±std':>16s} {'CF med±std':>16s} {'speedup':>9s} {'SJ cv%':>7s} {'CF cv%':>7s}")
print("-" * 75)

for name, build_fn, B_cfg, C, H in configs_var:
    x = torch.rand(T, B_cfg, C, H, H, device=device)

    # SJ torch
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_med, sj_std, sj_min, sj_max = bench_with_stats(net, x)

    # CATFuse
    net_cf = build_fn().to(device).eval()
    functional.set_step_mode(net_cf, 'm')
    net_cf, _ = substitute_sf(net_cf, T=T)
    net_cf = net_cf.to(device).eval()
    functional.set_step_mode(net_cf, 'm')
    cf_med, cf_std, cf_min, cf_max = bench_with_stats(net_cf, x)

    speedup = sj_med / cf_med
    sj_cv = 100 * sj_std / sj_med if sj_med > 0 else 0
    cf_cv = 100 * cf_std / cf_med if cf_med > 0 else 0

    print(f"  {name:<13s} {sj_med:>7.2f}±{sj_std:>5.3f}ms {cf_med:>7.2f}±{cf_std:>5.3f}ms {speedup:>8.2f}x {sj_cv:>5.1f}% {cf_cv:>5.1f}%")

    del net, net_cf, x
    torch.cuda.empty_cache()

print("=" * 90)
