"""
Test 16: Prescan v1 vs v2 vs v3 — correctness + latency.
"""
import sys, time
sys.path.insert(0, '.')
import torch
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

from catfuse.sparseflow.sparse_conv2d_kernel import _select_tile_sizes, choose_group_size
from catfuse.sparseflow.fast_prescan import fast_spike_prescan_2d
from catfuse.sparseflow.fast_prescan_v2 import fast_spike_prescan_2d_v2, fast_spike_prescan_2d_v3

configs = [
    ("B=8  C=64  H=56  sp=95%",  8,  64,  56, 0.95),
    ("B=8  C=64  H=56  sp=99%",  8,  64,  56, 0.99),
    ("B=8  C=64  H=56  sp=50%",  8,  64,  56, 0.50),
    ("B=8  C=128 H=28  sp=95%",  8,  128, 28, 0.95),
    ("B=8  C=256 H=14  sp=95%",  8,  256, 14, 0.95),
    ("B=8  C=512 H=7   sp=95%",  8,  512, 7,  0.95),
    ("B=1  C=64  H=56  sp=95%",  1,  64,  56, 0.95),
]

N_WARMUP = 100
N_ITER = 1000

print("=" * 90)
print(f"{'Config':<28s} {'v1→v2':>7s} {'v1':>8s} {'v2':>8s} {'v3':>8s} {'v2 ok':>6s} {'v3 ok':>6s}")
print("-" * 90)

for label, B, C, H, sp in configs:
    W = H
    KS, S, P = 3, 1, 1
    H_OUT = (H + 2*P - KS)//S + 1
    W_OUT = (W + 2*P - KS)//S + 1
    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = B * GH * GW
    GSC = choose_group_size(C)

    x = (torch.rand(B, C, H, W, device=device) > sp).float()

    ag1 = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tc1 = torch.empty(N_TILES, dtype=torch.int32, device=device)
    ag2 = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tc2 = torch.empty(N_TILES, dtype=torch.int32, device=device)
    ag3 = torch.zeros(N_TILES, dtype=torch.int32, device=device)
    tc3 = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # Reference: v1
    fast_spike_prescan_2d(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                           ag_mask_out=ag1, tile_class_out=tc1)
    torch.cuda.synchronize()

    # v2
    fast_spike_prescan_2d_v2(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                              ag_mask_out=ag2, tile_class_out=tc2)
    torch.cuda.synchronize()
    v2_ok = (ag1[:N_TILES] == ag2[:N_TILES]).all().item()

    # v3
    fast_spike_prescan_2d_v3(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                              ag_mask_out=ag3, tile_class_out=tc3)
    torch.cuda.synchronize()
    v3_ok = (ag1[:N_TILES] == ag3[:N_TILES]).all().item()

    # Bench v1
    for _ in range(N_WARMUP):
        fast_spike_prescan_2d(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                               ag_mask_out=ag1, tile_class_out=tc1)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fast_spike_prescan_2d(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                               ag_mask_out=ag1, tile_class_out=tc1)
    torch.cuda.synchronize()
    v1_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # Bench v2
    for _ in range(N_WARMUP):
        fast_spike_prescan_2d_v2(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                                  ag_mask_out=ag2, tile_class_out=tc2)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        fast_spike_prescan_2d_v2(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                                  ag_mask_out=ag2, tile_class_out=tc2)
    torch.cuda.synchronize()
    v2_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # Bench v3
    for _ in range(N_WARMUP):
        ag3.zero_()
        fast_spike_prescan_2d_v3(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                                  ag_mask_out=ag3, tile_class_out=tc3)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.cuda.synchronize()
    for _ in range(N_ITER):
        ag3.zero_()
        fast_spike_prescan_2d_v3(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                                  ag_mask_out=ag3, tile_class_out=tc3)
    torch.cuda.synchronize()
    v3_ms = (time.perf_counter() - t0) / N_ITER * 1000

    speedup = v1_ms / v2_ms if v2_ms > 0 else 0
    print(f"{label:<28s} {speedup:>6.1f}x {v1_ms:>7.3f}ms {v2_ms:>7.3f}ms {v3_ms:>7.3f}ms "
          f"{'OK' if v2_ok else 'FAIL':>6s} {'OK' if v3_ok else 'FAIL':>6s}")

    if not v2_ok:
        diff = (ag1[:N_TILES] != ag2[:N_TILES]).nonzero(as_tuple=False).flatten()[:3]
        for i in diff:
            print(f"  v2 MISMATCH tile {i.item()}: ref=0x{ag1[i].item():08x} v2=0x{ag2[i].item():08x}")
    if not v3_ok:
        diff = (ag1[:N_TILES] != ag3[:N_TILES]).nonzero(as_tuple=False).flatten()[:3]
        for i in diff:
            print(f"  v3 MISMATCH tile {i.item()}: ref=0x{ag1[i].item():08x} v3=0x{ag3[i].item():08x}")

print("=" * 90)
