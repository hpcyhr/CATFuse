import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

T, B, C, H, W = 16, 32, 128, 16, 16
device = 'cuda'

# LIF only, cupy backend, multi-step mode
lif = neuron.LIFNode(
    step_mode='m', backend='cupy',
    surrogate_function=surrogate.ATan()
).to(device).eval()

x = torch.randn(T, B, C, H, W, device=device)

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = lif(x)
        lif.reset()
torch.cuda.synchronize()

# Reset stats AFTER warmup to get clean per-forward delta
torch.cuda.reset_peak_memory_stats()
before_alloc_bytes = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
before_alloc_events = torch.cuda.memory_stats()['allocation.all.allocated']

# Single measurement forward
with torch.no_grad():
    out = lif(x)
    lif.reset()
torch.cuda.synchronize()

after = torch.cuda.memory_stats()
delta_bytes = (after['allocated_bytes.all.allocated'] - before_alloc_bytes) / 1e6
delta_events = after['allocation.all.allocated'] - before_alloc_events
peak = after['allocated_bytes.all.peak'] / 1e6

step_bytes = B * C * H * W * 4  # 4.19 MB
step_MB = step_bytes / 1e6

print(f"---- LIF-only per-forward delta ----")
print(f"peak (since reset):           {peak:.2f} MB")
print(f"allocated_bytes delta:        {delta_bytes:.2f} MB")
print(f"allocation events delta:      {delta_events} events")
print()
print(f"---- 理论参考 (LIF-only, no Conv/BN) ----")
print(f"|step| = {step_MB:.2f} MB,  T·|step| = {T*step_MB:.2f} MB")
print()
print(f"输入 x [T,B,C,H,W] 已在 HBM,不计 alloc")
print(f"输出 s [T,B,C,H,W] = {T*step_MB:.2f} MB,1 次 alloc")
print()
print(f"H₀ 预测 (v 在 multi-step kernel 内部 register 摊销):")
print(f"  per-forward LIF alloc ≈ {T*step_MB:.2f} MB (just s_seq) + 几 MB (v 的 init/final)")
print(f"  per-forward events ≈ 5-10")
print()
print(f"H₁ 预测 (v 每步落 HBM,Python loop):")
print(f"  per-forward LIF alloc ≈ {T*step_MB + 2*T*step_MB/T:.2f} MB (s_seq + v 每步覆盖)")
print(f"  per-forward events ≈ {3*T} (~50-60, ≈3 events × T)")