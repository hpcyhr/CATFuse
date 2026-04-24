# conv_only_ablation.py
import torch, torch.nn as nn, numpy as np, time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
T, B, C, H, W = 16, 32, 128, 16, 16

conv_single = nn.Conv2d(C, C, 3, padding=1).cuda().eval()
x_single = torch.randn(B, C, H, W, device='cuda')

for _ in range(20):
    with torch.no_grad():
        _ = conv_single(x_single)
torch.cuda.synchronize()

def run():
    for _ in range(T):
        with torch.no_grad():
            _ = conv_single(x_single)

# Warmup
for _ in range(20): run()
torch.cuda.synchronize()

# Measure
times = []
for _ in range(12):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
times = sorted(times)[1:-1]
print(f"Conv-only T=16 iterations CPU wall-clock: {np.mean(times):.3f} ± {np.std(times, ddof=1):.3f} ms")