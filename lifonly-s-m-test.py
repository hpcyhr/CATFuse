import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional

# === 配置 (与之前 multi-step LIF-only 实验完全一致) ===
T, B, C, H, W = 16, 32, 128, 16, 16
device = 'cuda'
torch.manual_seed(42)

# === 构建两个 LIF: multi-step (cupy) 和 single-step (torch) ===
def build_lif_multi():
    return neuron.LIFNode(
        step_mode='m', backend='cupy',
        surrogate_function=surrogate.ATan()
    ).to(device).eval()

def build_lif_single():
    return neuron.LIFNode(
        step_mode='s', backend='torch',
        surrogate_function=surrogate.ATan()
    ).to(device).eval()

lif_m = build_lif_multi()
lif_s = build_lif_single()

# === 输入 (固定相同输入) ===
x_seq = torch.randn(T, B, C, H, W, device=device)

# ============================================================
# 测量 1: Multi-step LIF (一次调用处理 [T,B,C,H,W])
# ============================================================
print("=" * 60)
print("Multi-step LIF (cupy backend)")
print("=" * 60)

# Warmup
for _ in range(3):
    with torch.no_grad():
        functional.reset_net(lif_m)
        _ = lif_m(x_seq)
torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_stats()
before_alloc_bytes = before['allocated_bytes.all.allocated']
before_alloc_events = before['allocation.all.allocated']

with torch.no_grad():
    functional.reset_net(lif_m)
    out_m = lif_m(x_seq)
torch.cuda.synchronize()

after = torch.cuda.memory_stats()
multi_delta_bytes = (after['allocated_bytes.all.allocated'] - before_alloc_bytes) / 1e6
multi_delta_events = after['allocation.all.allocated'] - before_alloc_events
multi_peak = after['allocated_bytes.all.peak'] / 1e6

print(f"peak (since reset):           {multi_peak:.2f} MB")
print(f"allocated_bytes delta:        {multi_delta_bytes:.2f} MB")
print(f"allocation events delta:      {multi_delta_events} events")

# ============================================================
# 测量 2: Single-step LIF (Python loop T 步)
# ============================================================
print()
print("=" * 60)
print("Single-step LIF (torch backend, Python loop)")
print("=" * 60)

# Warmup
for _ in range(3):
    with torch.no_grad():
        functional.reset_net(lif_s)
        for t in range(T):
            _ = lif_s(x_seq[t])
torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_stats()
before_alloc_bytes = before['allocated_bytes.all.allocated']
before_alloc_events = before['allocation.all.allocated']

with torch.no_grad():
    functional.reset_net(lif_s)
    outs = []
    for t in range(T):
        outs.append(lif_s(x_seq[t]))
    out_s = torch.stack(outs, dim=0)
torch.cuda.synchronize()

after = torch.cuda.memory_stats()
single_delta_bytes = (after['allocated_bytes.all.allocated'] - before_alloc_bytes) / 1e6
single_delta_events = after['allocation.all.allocated'] - before_alloc_events
single_peak = after['allocated_bytes.all.peak'] / 1e6

print(f"peak (since reset):           {single_peak:.2f} MB")
print(f"allocated_bytes delta:        {single_delta_bytes:.2f} MB")
print(f"allocation events delta:      {single_delta_events} events")

# ============================================================
# Parity
# ============================================================
print()
print("=" * 60)
print("Parity (multi vs single)")
print("=" * 60)
spike_match = (out_m == out_s).float().mean().item() * 100
max_diff = (out_m - out_s).abs().max().item()
print(f"spike_match: {spike_match:.4f}%")
print(f"max_diff:    {max_diff:.6e}")

# ============================================================
# 对比与判读
# ============================================================
print()
print("=" * 60)
print("Comparison + interpretation")
print("=" * 60)
step_MB = B * C * H * W * 4 / 1e6
T_step_MB = T * step_MB

print(f"|step| = {step_MB:.2f} MB,  T·|step| = {T_step_MB:.2f} MB")
print()
print(f"{'Metric':<28} {'Multi':>10} {'Single':>10} {'M/S':>8} {'M-S':>10}")
print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
print(f"{'peak (MB)':<28} {multi_peak:>10.2f} {single_peak:>10.2f} {multi_peak/single_peak:>8.3f} {multi_peak-single_peak:>+10.2f}")
print(f"{'alloc bytes (MB)':<28} {multi_delta_bytes:>10.2f} {single_delta_bytes:>10.2f} {multi_delta_bytes/single_delta_bytes:>8.3f} {multi_delta_bytes-single_delta_bytes:>+10.2f}")
print(f"{'alloc events':<28} {multi_delta_events:>10} {single_delta_events:>10} {multi_delta_events/single_delta_events:>8.3f} {multi_delta_events-single_delta_events:>+10}")

print()
print("理论参考 (LIF-only ledger):")
print(f"  forward-only LIF per-step ledger = read z' (1) + read v (1) + write v (1) + write s (1) = 4 events × |step|")
print(f"  total per forward = 4 × T × |step| = {4 * T_step_MB:.2f} MB")
print(f"  额外: stack(outs) for single-step = +T·|step| = +{T_step_MB:.2f} MB")
print()

print("假设区分 (相对 multi/single 差额):")
print(f"  H_a: Multi 物化 v_seq (T·|step| = {T_step_MB:.2f} MB extra) → Multi 比 Single 多 ≈ {T_step_MB:.0f} MB")
print(f"  H_b: Multi 仅暴露 caching allocator 复用差异 → 差额 < {T_step_MB:.0f} MB,主要在 peak,不在累积 bytes")
print(f"  实测差额 (M - S):")
print(f"    peak     diff = {multi_peak - single_peak:+.2f} MB (T·|step| ≈ {T_step_MB:.0f})")
print(f"    bytes    diff = {multi_delta_bytes - single_delta_bytes:+.2f} MB")