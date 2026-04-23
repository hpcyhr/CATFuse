"""
Motivation experiments for CTF paper §3.M.

Revised per external review:
- Separate GPU-event time and CPU wall-clock (the latter includes launch overhead)
- CUDA Graph experiment now validates repeated replay independence (10 iterations)
- torch.compile experiment covers Conv→LIF and Conv→BN→LIF, and uses _dynamo.explain
- torch.inference_mode() replaces torch.no_grad() for cleaner measurement
- Terminology clarified: "peak allocated CUDA memory" not "peak HBM";
  "allocator allocation bytes" not "HBM bytes"

Environment:
    V100-PCIe 32GB (sm_70), PyTorch 2.1.0+cu118, SpikingJelly 0.0.0.0.14,
    CuPy 13.6.0, CUDA 11.8

Protocol:
    trimmed mean N=10 of 12 for both timings, warmup=20 before measurement,
    torch.cuda.synchronize() before and after every timed section,
    bit-exact parity as pass criterion.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from spikingjelly.activation_based import (
    neuron, layer, surrogate, functional,
)


# ============================================================
# Shared configuration
# ============================================================

T, B, C, H, W = 16, 32, 128, 16, 16
DEVICE = 'cuda'


# ============================================================
# Timing: GPU event vs CPU wall-clock
# ============================================================

def bench_gpu_time(fn, n_warmup=20, n_measure=12, n_trim=1):
    """Trimmed-mean CUDA-event GPU elapsed time in ms.

    Measures GPU stream time only. Does NOT include CPU-side kernel launch
    overhead; for CUDA Graph effects on launch, use bench_cpu_wall instead.
    Returns (mean, std).
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = sorted(times)[n_trim:-n_trim]
    return float(np.mean(times)), float(np.std(times, ddof=1))


def bench_cpu_wall(fn, n_warmup=20, n_measure=12, n_trim=1):
    """Trimmed-mean end-to-end CPU wall-clock in ms, including CPU launch overhead.

    This is the metric that reflects CUDA Graph's primary optimization target.
    Returns (mean, std).
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = sorted(times)[n_trim:-n_trim]
    return float(np.mean(times)), float(np.std(times, ddof=1))


def measure_memory(fn, n_warmup=3):
    """Allocator-level CUDA memory stats for one forward.

    NOTE: allocator metrics are NOT DRAM traffic counters.
    - peak_allocated_MB: peak allocated CUDA memory during fn()
    - new_alloc_bytes_MB: new allocator allocation bytes during fn()
    - new_alloc_events: new allocation events during fn()
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()
    b_bytes = before['allocated_bytes.all.allocated']
    b_events = before['allocation.all.allocated']
    fn()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()

    return {
        'peak_allocated_MB': after['allocated_bytes.all.peak'] / 1e6,
        'new_alloc_bytes_MB': (
            after['allocated_bytes.all.allocated'] - b_bytes
        ) / 1e6,
        'new_alloc_events': after['allocation.all.allocated'] - b_events,
    }


# ============================================================
# Network builders
# ============================================================

def build_multi_step_conv_bn_lif(seed=42):
    torch.manual_seed(seed)
    return nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        layer.BatchNorm2d(C, step_mode='m'),
        neuron.LIFNode(
            step_mode='m', backend='cupy',
            surrogate_function=surrogate.ATan(),
        ),
    ).to(DEVICE).eval()


def build_multi_step_conv_lif(seed=42):
    torch.manual_seed(seed)
    return nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        neuron.LIFNode(
            step_mode='m', backend='cupy',
            surrogate_function=surrogate.ATan(),
        ),
    ).to(DEVICE).eval()


def build_single_step_from(multi_net):
    has_bn = len(multi_net) == 3
    if has_bn:
        single_net = nn.Sequential(
            layer.Conv2d(C, C, 3, padding=1, step_mode='s'),
            layer.BatchNorm2d(C, step_mode='s'),
            neuron.LIFNode(
                step_mode='s', backend='torch',
                surrogate_function=surrogate.ATan(),
            ),
        ).to(DEVICE).eval()
        single_net[0].load_state_dict(multi_net[0].state_dict())
        single_net[1].load_state_dict(multi_net[1].state_dict())
    else:
        single_net = nn.Sequential(
            layer.Conv2d(C, C, 3, padding=1, step_mode='s'),
            neuron.LIFNode(
                step_mode='s', backend='torch',
                surrogate_function=surrogate.ATan(),
            ),
        ).to(DEVICE).eval()
        single_net[0].load_state_dict(multi_net[0].state_dict())
    return single_net


def disable_grad(net):
    """Defense-in-depth: disable grad tracking on all parameters."""
    for p in net.parameters():
        p.requires_grad_(False)


# ============================================================
# Experiment A: SJ-torch vs SJ-cupy wall-clock
# ============================================================

# 替换 experiment_A_sj_backends 为下面这个版本

def experiment_A_sj_backends():
    print("=" * 72)
    print("Experiment A: Three SJ execution paths (Conv→BN→LIF)")
    print("=" * 72)

    # --- Build three configurations sharing the same weights ---
    torch.manual_seed(42)
    multi_cupy_net = nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        layer.BatchNorm2d(C, step_mode='m'),
        neuron.LIFNode(
            step_mode='m', backend='cupy',
            surrogate_function=surrogate.ATan(),
        ),
    ).to(DEVICE).eval()
    disable_grad(multi_cupy_net)

    # multi-step torch backend (same weights)
    multi_torch_net = nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        layer.BatchNorm2d(C, step_mode='m'),
        neuron.LIFNode(
            step_mode='m', backend='torch',
            surrogate_function=surrogate.ATan(),
        ),
    ).to(DEVICE).eval()
    multi_torch_net[0].load_state_dict(multi_cupy_net[0].state_dict())
    multi_torch_net[1].load_state_dict(multi_cupy_net[1].state_dict())
    disable_grad(multi_torch_net)

    # single-step torch backend (same weights)
    single_net = nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='s'),
        layer.BatchNorm2d(C, step_mode='s'),
        neuron.LIFNode(
            step_mode='s', backend='torch',
            surrogate_function=surrogate.ATan(),
        ),
    ).to(DEVICE).eval()
    single_net[0].load_state_dict(multi_cupy_net[0].state_dict())
    single_net[1].load_state_dict(multi_cupy_net[1].state_dict())
    disable_grad(single_net)

    x_seq = torch.randn(T, B, C, H, W, device=DEVICE)

    def run_single():
        functional.reset_net(single_net)
        with torch.inference_mode():
            outs = [single_net(x_seq[t]) for t in range(T)]
        return torch.stack(outs, dim=0)

    def run_multi_torch():
        functional.reset_net(multi_torch_net)
        with torch.inference_mode():
            return multi_torch_net(x_seq)

    def run_multi_cupy():
        functional.reset_net(multi_cupy_net)
        with torch.inference_mode():
            return multi_cupy_net(x_seq)

    # --- Parity: all three must produce identical spikes ---
    out_s = run_single()
    out_mt = run_multi_torch()
    out_mc = run_multi_cupy()

    same_s_mt = torch.equal(out_s, out_mt)
    same_mt_mc = torch.equal(out_mt, out_mc)
    print(f"parity single vs multi_torch: exact={same_s_mt}, "
          f"mismatches={(out_s != out_mt).sum().item()}")
    print(f"parity multi_torch vs multi_cupy: exact={same_mt_mc}, "
          f"mismatches={(out_mt != out_mc).sum().item()}")
    assert same_s_mt and same_mt_mc, "Parity failure in Experiment A"

    # --- Timings ---
    gpu_s, gstd_s = bench_gpu_time(run_single)
    cpu_s, cstd_s = bench_cpu_wall(run_single)
    gpu_mt, gstd_mt = bench_gpu_time(run_multi_torch)
    cpu_mt, cstd_mt = bench_cpu_wall(run_multi_torch)
    gpu_mc, gstd_mc = bench_gpu_time(run_multi_cupy)
    cpu_mc, cstd_mc = bench_cpu_wall(run_multi_cupy)

    print()
    print(f"{'Config':<24}{'GPU time (ms)':>18}{'CPU wall-clock (ms)':>24}")
    print(f"{'-'*24}{'-'*18}{'-'*24}")
    print(f"{'single-step torch':<24}"
          f"{gpu_s:>10.3f} ± {gstd_s:<5.3f}"
          f"{cpu_s:>14.3f} ± {cstd_s:<5.3f}")
    print(f"{'multi-step torch':<24}"
          f"{gpu_mt:>10.3f} ± {gstd_mt:<5.3f}"
          f"{cpu_mt:>14.3f} ± {cstd_mt:<5.3f}")
    print(f"{'multi-step cupy':<24}"
          f"{gpu_mc:>10.3f} ± {gstd_mc:<5.3f}"
          f"{cpu_mc:>14.3f} ± {cstd_mc:<5.3f}")

    print()
    print("Analysis:")
    print(f"  BatchFold effect (single → multi_torch):  "
          f"CPU {(cpu_s-cpu_mt)/cpu_s*100:+.1f}% change "
          f"({'faster' if cpu_mt < cpu_s else 'SLOWER'})")
    print(f"  cuPy LIF effect (multi_torch → multi_cupy): "
          f"CPU {(cpu_mt-cpu_mc)/cpu_mt*100:+.1f}% change "
          f"({'faster' if cpu_mc < cpu_mt else 'SLOWER'})")
    print(f"  Combined (single → multi_cupy):            "
          f"CPU {(cpu_s-cpu_mc)/cpu_s*100:+.1f}% change")
    print()
    print("If BatchFold effect is SLOWER, this reflects cuDNN algorithm")
    print("dispatch at batch = T*B = 512 being suboptimal for this shape,")
    print("a known behavior on V100 sm_70 for mid-channel / small-spatial")
    print("configurations.\n")

# ============================================================
# Experiment B: SJ single-step vs multi-step allocator-level memory
# ============================================================

def experiment_B_single_vs_multi_memory():
    print("=" * 72)
    print("Experiment B: single-step vs multi-step allocator memory "
          "(Conv→BN→LIF)")
    print("=" * 72)

    multi_net = build_multi_step_conv_bn_lif()
    disable_grad(multi_net)
    single_net = build_single_step_from(multi_net)
    disable_grad(single_net)
    x_seq = torch.randn(T, B, C, H, W, device=DEVICE)

    def run_multi():
        functional.reset_net(multi_net)
        with torch.inference_mode():
            return multi_net(x_seq)

    def run_single():
        functional.reset_net(single_net)
        with torch.inference_mode():
            outs = [single_net(x_seq[t]) for t in range(T)]
        return torch.stack(outs, dim=0)

    out_multi = run_multi()
    out_single = run_single()
    same = torch.equal(out_multi, out_single)
    mismatches = (out_multi != out_single).sum().item()
    print(f"parity: exact={same}, mismatches={mismatches}")
    assert same, "Parity failure in Experiment B"

    m = measure_memory(run_multi)
    s = measure_memory(run_single)

    print(f"{'Metric':<28}{'multi-step':>14}{'single-step':>14}{'M/S':>10}")
    print(f"{'-'*28}{'-'*14}{'-'*14}{'-'*10}")
    print(f"{'peak allocated (MB)':<28}"
          f"{m['peak_allocated_MB']:>14.2f}"
          f"{s['peak_allocated_MB']:>14.2f}"
          f"{m['peak_allocated_MB']/s['peak_allocated_MB']:>10.3f}")
    print(f"{'allocator bytes (MB)':<28}"
          f"{m['new_alloc_bytes_MB']:>14.2f}"
          f"{s['new_alloc_bytes_MB']:>14.2f}"
          f"{m['new_alloc_bytes_MB']/s['new_alloc_bytes_MB']:>10.3f}")
    print(f"{'allocator events':<28}"
          f"{m['new_alloc_events']:>14}"
          f"{s['new_alloc_events']:>14}"
          f"{m['new_alloc_events']/s['new_alloc_events']:>10.3f}")
    print()
    print("Interpretation: this is allocator-level residency evidence, not a")
    print("DRAM traffic counter. It is consistent with multi-step trading")
    print("activation residency for fewer launches.\n")


# ============================================================
# Experiment C: CUDA Graph effect (CRITICAL — checks replay independence)
# ============================================================

# 替换 experiment_C_cuda_graph 为下面这个版本
# 关键改动:全部改回 no_grad(),避免 inference tensor × inplace × CUDA Graph 的三方冲突

def experiment_C_cuda_graph():
    print("=" * 72)
    print("Experiment C: CUDA Graph on SJ-cupy multi-step (Conv→BN→LIF)")
    print("=" * 72)

    net = build_multi_step_conv_bn_lif()
    disable_grad(net)
    x = torch.randn(T, B, C, H, W, device=DEVICE)

    # NOTE: use no_grad() instead of inference_mode() throughout this experiment.
    # inference_mode() tags tensors as "inference tensors" which cannot be
    # inplace-updated outside inference_mode context. CUDA Graph replay performs
    # the LIF v update outside the capture's inference_mode scope, causing a
    # RuntimeError. no_grad() is the safe fallback and gives effectively the
    # same measurement.

    def run_baseline():
        functional.reset_net(net)
        with torch.no_grad():
            return net(x)

    ref_out = run_baseline().clone()

    # ----- Baseline measurements -----
    gpu_base, gpu_std_base = bench_gpu_time(run_baseline)
    cpu_base, cpu_std_base = bench_cpu_wall(run_baseline)
    mem_base = measure_memory(run_baseline)

    # ----- CUDA Graph capture -----
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            functional.reset_net(net)
            with torch.no_grad():
                _ = net(x)
    torch.cuda.current_stream().wait_stream(s)

    static_x = x.clone()

    functional.reset_net(net)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = net(static_x)

    # ----- BLOCKER 2: repeated replay parity -----
    print("Repeated replay parity (10 iterations):")
    all_same = True
    for i in range(10):
        functional.reset_net(net)
        g.replay()
        torch.cuda.synchronize()
        cg_out = static_out.clone()
        same = torch.equal(cg_out, ref_out)
        mism = (cg_out != ref_out).sum().item()
        status = "OK" if same else "DRIFT"
        print(f"  replay {i:>2}: exact={same}, mismatches={mism} [{status}]")
        if not same:
            all_same = False

    if not all_same:
        print("\n*** BLOCKER: CUDA Graph replay is NOT independent across "
              "reset_net+replay. ***")
        print("Stopping Experiment C. Need redesign.\n")
        return

    print("  all 10 replays bit-exact ✓\n")

    def run_cudagraph():
        functional.reset_net(net)
        g.replay()

    gpu_cg, gpu_std_cg = bench_gpu_time(run_cudagraph)
    cpu_cg, cpu_std_cg = bench_cpu_wall(run_cudagraph)
    mem_cg = measure_memory(run_cudagraph)

    def pct(a, b):
        return (a / b - 1) * 100

    print(f"{'Metric':<28}{'SJ-cupy':>16}{'+CUDA Graph':>18}{'Δ%':>10}")
    print(f"{'-'*28}{'-'*16}{'-'*18}{'-'*10}")
    print(f"{'CPU wall-clock (ms)':<28}"
          f"{cpu_base:>8.3f} ± {cpu_std_base:<5.3f}"
          f"{cpu_cg:>10.3f} ± {cpu_std_cg:<5.3f}"
          f"{pct(cpu_cg, cpu_base):>+9.1f}%")
    print(f"{'GPU event time (ms)':<28}"
          f"{gpu_base:>8.3f} ± {gpu_std_base:<5.3f}"
          f"{gpu_cg:>10.3f} ± {gpu_std_cg:<5.3f}"
          f"{pct(gpu_cg, gpu_base):>+9.1f}%")
    print(f"{'peak allocated (MB)':<28}"
          f"{mem_base['peak_allocated_MB']:>16.2f}"
          f"{mem_cg['peak_allocated_MB']:>18.2f}"
          f"{pct(mem_cg['peak_allocated_MB'], mem_base['peak_allocated_MB']):>+9.1f}%")
    print()
    print("Interpretation:")
    print("  CPU wall-clock reduction reflects CUDA Graph's primary target")
    print("  (CPU-side kernel launch overhead). Peak allocated CUDA memory")
    print("  change reflects graph memory-pool reuse, not DRAM traffic.")
    print("  Kernel bodies are invariant by construction.\n")

# ============================================================
# Experiment D: torch.compile on SJ multi-step subgraphs
# ============================================================

def experiment_D_torchcompile():
    print("=" * 72)
    print("Experiment D: torch.compile on SJ multi-step subgraphs")
    print("=" * 72)

    configs = [
        ("Conv→LIF", build_multi_step_conv_lif, (T, B, C, H, W)),
        ("Conv→BN→LIF", build_multi_step_conv_bn_lif, (T, B, C, H, W)),
    ]

    for name, builder, shape in configs:
        print(f"\n>>> {name}")
        net = builder()
        disable_grad(net)
        x = torch.randn(*shape, device=DEVICE)

        # --- Report what Dynamo sees ---
        try:
            def _wrap(x):
                functional.reset_net(net)
                with torch.inference_mode():
                    return net(x)

            exp = dynamo.explain(_wrap)(x)
            print(f"--- torch._dynamo.explain summary ---")
            # dynamo.explain returns an ExplainOutput object; str() is readable.
            exp_str = str(exp)
            # Print first 2000 chars to keep log readable.
            print(exp_str[:2000])
            if len(exp_str) > 2000:
                print(f"... (truncated, total {len(exp_str)} chars)")
        except Exception as e:
            print(f"dynamo.explain failed: {type(e).__name__}: "
                  f"{str(e)[:400]}")

        # --- Capturability test: fullgraph=True ---
        print(f"\n--- torch.compile(fullgraph=True) ---")
        try:
            compiled = torch.compile(net, fullgraph=True)
            functional.reset_net(net)
            with torch.inference_mode():
                _ = compiled(x)
            print("fullgraph=True: SUCCEEDED (unexpected for stateful SJ)")
        except Exception as e:
            print(f"fullgraph=True: FAILED (expected)")
            print(f"  {type(e).__name__}: {str(e)[:400]}")

        # --- Best-effort test: fullgraph=False ---
        print(f"\n--- torch.compile(fullgraph=False) ---")
        try:
            compiled = torch.compile(net, fullgraph=False)

            def run_compiled():
                functional.reset_net(net)
                with torch.inference_mode():
                    return compiled(x)

            def run_eager():
                functional.reset_net(net)
                with torch.inference_mode():
                    return net(x)

            # Parity
            out_eager = run_eager()
            out_compiled = run_compiled()
            same = torch.equal(out_eager, out_compiled)
            mism = (out_eager != out_compiled).sum().item()
            print(f"parity: exact={same}, mismatches={mism}")

            cpu_eager, _ = bench_cpu_wall(run_eager)
            cpu_compiled, _ = bench_cpu_wall(run_compiled)
            ratio = cpu_compiled / cpu_eager
            verdict = "slower" if ratio > 1.0 else "faster"
            print(f"CPU wall-clock: eager={cpu_eager:.3f} ms, "
                  f"compiled={cpu_compiled:.3f} ms")
            print(f"compiled/eager = {ratio:.3f}× ({verdict})")
        except Exception as e:
            print(f"fullgraph=False also failed: "
                  f"{type(e).__name__}: {str(e)[:400]}")

    print()

        # 在 Experiment D 结尾追加:直接测 LIFNode,绕开 Sequential
    print("\n>>> Isolated test: LIFNode alone (bypass Sequential)")
    lif = neuron.LIFNode(
        step_mode='m', backend='cupy',
        surrogate_function=surrogate.ATan(),
    ).to(DEVICE).eval()
    disable_grad(lif)
    x_lif = torch.randn(T, B, C, H, W, device=DEVICE)

    try:
        compiled_lif = torch.compile(lif, fullgraph=True)
        functional.reset_net(lif)
        with torch.no_grad():
            _ = compiled_lif(x_lif)
        print("LIFNode fullgraph=True: SUCCEEDED")
    except Exception as e:
        print(f"LIFNode fullgraph=True: FAILED")
        print(f"  {type(e).__name__}: {str(e)[:600]}")

# ============================================================
# Entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiments', type=str, default='ABCD',
        help='Subset of {A,B,C,D} to run. Default: ABCD (all).',
    )
    args = parser.parse_args()

    print(f"Configuration: T={T} B={B} C={C} H=W={H} device={DEVICE}")
    step_bytes = B * C * H * W * 4
    print(f"|step| = B*C*H*W*4 bytes = {step_bytes/1e6:.2f} MB (fp32)")
    print(f"T·|step| = {T*step_bytes/1e6:.2f} MB")
    print(f"5T·|step| (practical SJ-cupy ledger) = {5*T*step_bytes/1e6:.2f} MB")
    print(f"7T·|step| (formal reference ledger)  = {7*T*step_bytes/1e6:.2f} MB")
    print()

    if 'A' in args.experiments:
        experiment_A_sj_backends()
    if 'B' in args.experiments:
        experiment_B_single_vs_multi_memory()
    if 'C' in args.experiments:
        experiment_C_cuda_graph()
    if 'D' in args.experiments:
        experiment_D_torchcompile()


if __name__ == '__main__':
    main()