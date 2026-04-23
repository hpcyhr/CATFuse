import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

# === 配置 ===
T, B, C, H, W = 16, 32, 128, 16, 16
device = 'cuda'

# === 构建一个 SJ multi-step Conv→BN→LIF (强制 cupy backend) ===
net = nn.Sequential(
    layer.Conv2d(C, C, 3, padding=1, step_mode='m'),
    layer.BatchNorm2d(C, step_mode='m'),
    neuron.LIFNode(step_mode='m', backend='cupy', surrogate_function=surrogate.ATan()),
).to(device).eval()

# === 输入 ===
x = torch.randn(T, B, C, H, W, device=device)

# === Warmup (排除 cuDNN algorithm selection 的首次 alloc 噪声) ===
for _ in range(3):
    with torch.no_grad():
        _ = net(x)
        for m in net.modules():
            if hasattr(m, 'reset'):
                m.reset()
torch.cuda.synchronize()

# === 开始记录显存事件 ===
torch.cuda.memory._record_memory_history(max_entries=100000)

with torch.no_grad():
    out = net(x)

torch.cuda.synchronize()

# === Dump 事件 ===
torch.cuda.memory._dump_snapshot("sj_cupy_lif_memory.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# === 同时打印峰值与累积分配统计,作为辅助证据 ===
stats = torch.cuda.memory_stats()
print("---- memory_stats summary ----")
print(f"allocated_bytes.peak:        {stats['allocated_bytes.all.peak'] / 1e6:.2f} MB")
print(f"allocated_bytes.allocated:   {stats['allocated_bytes.all.allocated'] / 1e6:.2f} MB")
print(f"allocation.allocated:        {stats['allocation.all.allocated']} events")

# === 计算理论值用于对照 ===
step_bytes = B * C * H * W * 4  # fp32
v_size = step_bytes  # v has same shape as one step's activation

print(f"\n---- 理论参考 ----")
print(f"|step| = {step_bytes / 1e6:.2f} MB")
print(f"T·|step| = {T * step_bytes / 1e6:.2f} MB")
print(f"")
print(f"H₀ 预测 (v in register, 5T ledger):")
print(f"  per-step v HBM流量 ≈ 2/T·|step| ≈ {2 / T * step_bytes / 1e6:.3f} MB")
print(f"  per-forward v 总 HBM 流量 ≈ 2·|step| = {2 * step_bytes / 1e6:.2f} MB")
print(f"  per-forward 总 ledger ≈ 5T·|step| = {5 * T * step_bytes / 1e6:.2f} MB")
print(f"")
print(f"H₁ 预测 (v 每步落 HBM, 7T ledger):")
print(f"  per-step v HBM流量 = 2·|step| = {2 * step_bytes / 1e6:.2f} MB")
print(f"  per-forward v 总 HBM 流量 = 2T·|step| = {2 * T * step_bytes / 1e6:.2f} MB")
print(f"  per-forward 总 ledger ≈ 7T·|step| = {7 * T * step_bytes / 1e6:.2f} MB")