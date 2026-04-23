"""
Verify CUDA Graph is a launch-dimension optimization:
- Q1: CUDA Graph + SJ-cupy reduces wall-clock (launch savings)
- Q2: CUDA Graph + SJ-cupy does NOT reduce HBM (peak/alloc unchanged)
"""
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, layer, surrogate, functional

# === 配置 (与 motivation 引用的数据一致) ===
T, B, C, H, W = 16, 32, 128, 16, 16
device = 'cuda'
torch.manual_seed(42)

def build_net():
    return nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        layer.BatchNorm2d(C, step_mode='m'),
        neuron.LIFNode(step_mode='m', backend='cupy', surrogate_function=surrogate.ATan()),
    ).to(device).eval()

net = build_net()
x = torch.randn(T, B, C, H, W, device=device)

# === 通用工具:trimmed mean N=10 of 12 ===
def bench(fn, n_warmup=20, n_measure=12, n_trim=1):
    """Returns (trimmed_mean_ms, std_ms)."""
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
    return float(np.mean(times)), float(np.std(times))

def measure_memory(fn, n_warmup=3):
    """Returns (peak_MB, alloc_bytes_MB, alloc_events)."""
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
    peak = after['allocated_bytes.all.peak'] / 1e6
    delta_bytes = (after['allocated_bytes.all.allocated'] - b_bytes) / 1e6
    delta_events = after['allocation.all.allocated'] - b_events
    return peak, delta_bytes, delta_events

# ============================================================
# 基线 1: 裸 SJ-cuPy multi-step (无 CUDA Graph)
# ============================================================
print("=" * 70)
print("Baseline 1: SJ-cuPy multi-step (no CUDA Graph)")
print("=" * 70)

def run_baseline():
    functional.reset_net(net)
    with torch.no_grad():
        return net(x)

# Parity reference
torch.cuda.synchronize()
ref_out = run_baseline().clone()

# Wall-clock
wc_base, std_base = bench(run_baseline)
print(f"wall-clock (trimmed mean): {wc_base:.3f} ± {std_base:.3f} ms")

# Memory
peak_base, bytes_base, events_base = measure_memory(run_baseline)
print(f"peak:          {peak_base:.2f} MB")
print(f"alloc bytes:   {bytes_base:.2f} MB")
print(f"alloc events:  {events_base}")

# ============================================================
# 基线 2: SJ-cuPy multi-step + CUDA Graph
# ============================================================
print()
print("=" * 70)
print("Baseline 2: SJ-cuPy multi-step + CUDA Graph (replay)")
print("=" * 70)

# Capture graph
# CUDA Graph 要求使用专用 stream,并且第一次"warmup" 在该 stream 上预跑
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    for _ in range(3):
        functional.reset_net(net)
        with torch.no_grad():
            _ = net(x)
torch.cuda.current_stream().wait_stream(s)

# 创建静态输入 buffer (CUDA Graph 要求输入 tensor 地址固定)
static_x = x.clone()
# CUDA Graph 不能捕获 functional.reset_net 里的 Python 控制流,
# 所以我们必须在 capture 前先 reset,capture 后保持状态前向滚动
functional.reset_net(net)

g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    static_out = net(static_x)

def run_cudagraph():
    # CUDA Graph replay: 输入已在 static_x,输出落在 static_out
    # 注意:LIF 的 v 状态会在多次 replay 间累积,这和"正常执行"语义不一致
    # 所以每次测量后需要在 graph 外 reset
    g.replay()
    return static_out

# Parity check (一次 replay 后 reset,对比 baseline)
functional.reset_net(net)
# 因为 reset 清了 net 里的 v,但 graph 已经捕获了第一次的 v 状态
# 所以 replay 语义上相当于"从 v=0 开始 forward 一次"——这与 baseline 一致
g.replay()
torch.cuda.synchronize()
cg_out = static_out.clone()

spike_match = (cg_out == ref_out).float().mean().item() * 100
max_diff = (cg_out - ref_out).abs().max().item()
print(f"parity vs baseline: spike_match={spike_match:.4f}%, max_diff={max_diff:.6e}")

# Wall-clock (每次 replay 前 reset net 以模拟独立 forward)
def run_cudagraph_with_reset():
    functional.reset_net(net)
    g.replay()

wc_cg, std_cg = bench(run_cudagraph_with_reset)
print(f"wall-clock (trimmed mean): {wc_cg:.3f} ± {std_cg:.3f} ms")

# Memory
peak_cg, bytes_cg, events_cg = measure_memory(run_cudagraph_with_reset)
print(f"peak:          {peak_cg:.2f} MB")
print(f"alloc bytes:   {bytes_cg:.2f} MB")
print(f"alloc events:  {events_cg}")

# ============================================================
# 对照与判读
# ============================================================
print()
print("=" * 70)
print("Comparison: CUDA Graph effect on launch vs HBM dimension")
print("=" * 70)

print(f"{'Metric':<24} {'SJ-cupy':>12} {'+ CUDA Graph':>14} {'Δ':>10} {'Δ%':>8}")
print(f"{'-'*24} {'-'*12} {'-'*14} {'-'*10} {'-'*8}")
print(f"{'wall-clock (ms)':<24} {wc_base:>12.3f} {wc_cg:>14.3f} {wc_cg-wc_base:>+10.3f} {(wc_cg/wc_base-1)*100:>+7.1f}%")
print(f"{'peak (MB)':<24} {peak_base:>12.2f} {peak_cg:>14.2f} {peak_cg-peak_base:>+10.2f} {(peak_cg/peak_base-1)*100:>+7.1f}%")
print(f"{'alloc bytes (MB)':<24} {bytes_base:>12.2f} {bytes_cg:>14.2f} {bytes_cg-bytes_base:>+10.2f} {(bytes_cg/bytes_base-1)*100:>+7.1f}%")
print(f"{'alloc events':<24} {events_base:>12} {events_cg:>14} {events_cg-events_base:>+10} {'--':>8}")

print()
print("Interpretation:")
print(f"  Q1 (launch dim): wall-clock reduction = {(1-wc_cg/wc_base)*100:.1f}%")
print(f"     Expected 5-15% if CUDA Graph works on launch dim")
print(f"  Q2 (HBM dim):    peak change = {(peak_cg/peak_base-1)*100:+.1f}%")
print(f"     Expected ≈ 0% if CUDA Graph does NOT touch HBM dim")