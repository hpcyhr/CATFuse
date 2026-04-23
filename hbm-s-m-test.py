import torch
import torch.nn as nn
import copy
from spikingjelly.activation_based import neuron, layer, surrogate, functional

# === 配置 ===
T, B, C, H, W = 16, 32, 128, 16, 16
device = 'cuda'
torch.manual_seed(42)

# === 构建 multi-step 网络 ===
def build_multi_step():
    return nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
        layer.BatchNorm2d(C, step_mode='m'),
        neuron.LIFNode(step_mode='m', backend='cupy', surrogate_function=surrogate.ATan()),
    ).to(device).eval()

# === 构建 single-step 网络(权重必须与 multi-step 严格相同)===
def build_single_step(multi_net):
    single_net = nn.Sequential(
        layer.Conv2d(C, C, 3, padding=1, step_mode='s'),
        layer.BatchNorm2d(C, step_mode='s'),
        neuron.LIFNode(step_mode='s', backend='torch', surrogate_function=surrogate.ATan()),
    ).to(device).eval()
    # 权重对齐
    single_net[0].load_state_dict(multi_net[0].state_dict())
    single_net[1].load_state_dict(multi_net[1].state_dict())
    return single_net

multi_net = build_multi_step()
single_net = build_single_step(multi_net)

# === 输入 ===
x_seq = torch.randn(T, B, C, H, W, device=device)

# ============================================================
# 测量 1: multi-step 模式
# ============================================================
print("=" * 60)
print("Multi-step (step_mode='m', BatchFold inside SJ)")
print("=" * 60)

# Warmup
for _ in range(3):
    with torch.no_grad():
        functional.reset_net(multi_net)
        _ = multi_net(x_seq)
torch.cuda.synchronize()

# Reset stats AFTER warmup
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_stats()
before_alloc_bytes = before['allocated_bytes.all.allocated']
before_alloc_events = before['allocation.all.allocated']

# Single measurement
with torch.no_grad():
    functional.reset_net(multi_net)
    out_multi = multi_net(x_seq)
torch.cuda.synchronize()

after = torch.cuda.memory_stats()
multi_delta_bytes = (after['allocated_bytes.all.allocated'] - before_alloc_bytes) / 1e6
multi_delta_events = after['allocation.all.allocated'] - before_alloc_events
multi_peak = after['allocated_bytes.all.peak'] / 1e6

print(f"peak (since reset):           {multi_peak:.2f} MB")
print(f"allocated_bytes delta:        {multi_delta_bytes:.2f} MB")
print(f"allocation events delta:      {multi_delta_events} events")

# ============================================================
# 测量 2: single-step 模式 (Python loop T 步)
# ============================================================
print()
print("=" * 60)
print("Single-step (step_mode='s', Python loop T steps)")
print("=" * 60)

# Warmup
for _ in range(3):
    with torch.no_grad():
        functional.reset_net(single_net)
        for t in range(T):
            _ = single_net(x_seq[t])
torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_stats()
before_alloc_bytes = before['allocated_bytes.all.allocated']
before_alloc_events = before['allocation.all.allocated']

with torch.no_grad():
    functional.reset_net(single_net)
    outs = []
    for t in range(T):
        outs.append(single_net(x_seq[t]))
    out_single = torch.stack(outs, dim=0)
torch.cuda.synchronize()

after = torch.cuda.memory_stats()
single_delta_bytes = (after['allocated_bytes.all.allocated'] - before_alloc_bytes) / 1e6
single_delta_events = after['allocation.all.allocated'] - before_alloc_events
single_peak = after['allocated_bytes.all.peak'] / 1e6

print(f"peak (since reset):           {single_peak:.2f} MB")
print(f"allocated_bytes delta:        {single_delta_bytes:.2f} MB")
print(f"allocation events delta:      {single_delta_events} events")

# ============================================================
# Parity check
# ============================================================
print()
print("=" * 60)
print("Parity check (multi vs single output)")
print("=" * 60)
spike_match = (out_multi == out_single).float().mean().item() * 100
max_diff = (out_multi - out_single).abs().max().item()
print(f"spike_match: {spike_match:.4f}%")
print(f"max_diff:    {max_diff:.6e}")

# ============================================================
# 对比与判读
# ============================================================
print()
print("=" * 60)
print("Comparison")
print("=" * 60)
step_MB = B * C * H * W * 4 / 1e6

print(f"|step| = {step_MB:.2f} MB,  T·|step| = {T*step_MB:.2f} MB")
print()
print(f"{'Metric':<28} {'Multi-step':>12} {'Single-step':>14} {'Ratio M/S':>10}")
print(f"{'-'*28} {'-'*12} {'-'*14} {'-'*10}")
print(f"{'peak (MB)':<28} {multi_peak:>12.2f} {single_peak:>14.2f} {multi_peak/single_peak:>10.3f}")
print(f"{'alloc bytes (MB)':<28} {multi_delta_bytes:>12.2f} {single_delta_bytes:>14.2f} {multi_delta_bytes/single_delta_bytes:>10.3f}")
print(f"{'alloc events':<28} {multi_delta_events:>12} {single_delta_events:>14} {multi_delta_events/single_delta_events:>10.3f}")
print()

print("判读规则:")
print("  - 若 multi/single bytes ratio ≈ 1.0 → BatchFold 不节省 activation HBM (符合 ledger)")
print("  - 若 multi/single bytes ratio ≪ 1.0 → BatchFold 节省了 activation HBM (与 ledger 不符)")
print("  - alloc events 比例反映 launch 次数节省 (BatchFold 把 T 次 Conv launch 合成 1 次)")