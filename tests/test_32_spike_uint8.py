"""
Test 32: Spike uint8 format — measure actual wall-clock savings.

Approach: wrap each fused module to output uint8 spike and accept uint8 input.
The HBM savings come from storing/transferring 1 byte instead of 4.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

# Simulate a pipeline of Conv+BN+LIF layers (like VGG/ResNet)
# Compare fp32 spike vs uint8 spike between layers

T, B = 4, 1
N_WARMUP = 100
N_ITER = 500

def bench(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e6

from catfuse.sparseflow.lif_seq_kernel import lif_sequential

# Test different shapes
configs = [
    ("Stem  C=64  H=112", 64,  112),
    ("L1    C=64  H=56",  64,   56),
    ("L2    C=128 H=28", 128,   28),
    ("L3    C=256 H=14", 256,   14),
    ("L4    C=512 H=7",  512,    7),
]

print("=" * 90)
print("Spike uint8 vs fp32: per-layer HBM transfer cost")
print(f"T={T}, B={B}")
print(f"{'Layer':<22s} {'spike_MB':>9s} {'fp32→fp32(μs)':>14s} {'u8→fp32(μs)':>13s} {'fp32→u8(μs)':>13s} {'savings':>8s}")
print("-" * 90)

for label, C, H in configs:
    # Spike tensor
    spike_fp32 = (torch.rand(T, B, C, H, H, device=device) > 0.5).float()
    spike_u8 = spike_fp32.to(torch.uint8)

    spike_mb = spike_fp32.numel() * 4 / 1024 / 1024

    # 1. fp32 → fp32: read fp32 spike, conv, write fp32 spike (current)
    # Simulate: read spike from HBM + simple elementwise (like conv reads it)
    w = torch.randn(C, C, 3, 3, device=device)
    inv_std = torch.ones(C, device=device)
    w_fused = w * inv_std.view(-1, 1, 1, 1)
    b_fused = torch.zeros(C, device=device)

    def fp32_pipeline():
        x_4d = spike_fp32.reshape(T*B, C, H, H)
        z = F.conv2d(x_4d, w_fused, b_fused, padding=1)
        v = torch.zeros(B, C, H, H, device=device)
        z5 = z.reshape(T, B, C, H, H)
        s, _ = lif_sequential(z5, v, tau=2.0, v_threshold=1.0, v_reset=0.0)
        return s  # fp32 spike output

    # 2. uint8 → fp32 → conv → fp32 → uint8
    def u8_pipeline():
        x_4d = spike_u8.reshape(T*B, C, H, H).float()  # uint8→fp32 in registers
        z = F.conv2d(x_4d, w_fused, b_fused, padding=1)
        v = torch.zeros(B, C, H, H, device=device)
        z5 = z.reshape(T, B, C, H, H)
        s, _ = lif_sequential(z5, v, tau=2.0, v_threshold=1.0, v_reset=0.0)
        return s.to(torch.uint8)  # fp32→uint8 at output

    t_fp32 = bench(fp32_pipeline)
    t_u8 = bench(u8_pipeline)

    # 3. Just the conversion costs
    t_to_fp32 = bench(lambda: spike_u8.float())
    t_to_u8 = bench(lambda: spike_fp32.to(torch.uint8))

    savings = (t_fp32 - t_u8) / t_fp32 * 100
    print(f"{label:<22s} {spike_mb:>8.2f}MB {t_fp32:>13.1f} {t_u8:>13.1f} {t_to_u8:>13.1f} {savings:>7.1f}%")

# Also test: what if we keep spike as uint8 in HBM but convert in Triton kernel?
print("\n--- Conversion overhead ---")
for label, C, H in configs:
    spike_fp32 = (torch.rand(T*B, C, H, H, device=device) > 0.5).float()
    spike_u8 = spike_fp32.to(torch.uint8)
    mb = spike_fp32.numel() * 4 / 1024 / 1024

    t_read_fp32 = bench(lambda: spike_fp32.sum())  # force full read
    t_read_u8 = bench(lambda: spike_u8.float().sum())  # read u8 + convert + sum
    t_write_u8 = bench(lambda: spike_fp32.to(torch.uint8))

    print(f"  {label:<22s} read_fp32={t_read_fp32:.1f}μs  read_u8+cast={t_read_u8:.1f}μs  write_u8={t_write_u8:.1f}μs  data={mb:.1f}MB")

print("=" * 90)
