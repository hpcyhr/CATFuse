"""
Test 17: Lean path + prescan v2 — full SparseFlow optimized path.

Measures: fast_prescan_v2 + direct kernel launch vs cuDNN
across all ResNet18 layer shapes.
"""
import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import triton

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

from catfuse.sparseflow.sparse_conv2d_kernel import (
    sparse_conv3x3s1_nhwc_kernel_8x8,
    sparse_conv3x3s1_nhwc_kernel_8x16,
    _select_tile_sizes,
    choose_group_size,
)
from catfuse.sparseflow.fast_prescan_v2 import fast_spike_prescan_2d_v2

KS, S, P = 3, 1, 1
N_WARMUP = 100
N_ITER = 1000

configs = [
    # ResNet18 layers (stride=1 only, which SF handles)
    ("layer1: B=8 C=64  H=56", 8, 64,  64,  56),
    ("layer2: B=8 C=128 H=28", 8, 128, 128, 28),
    ("layer3: B=8 C=256 H=14", 8, 256, 256, 14),
    ("layer4: B=8 C=512 H=7",  8, 512, 512, 7),
    # B=1 latency
    ("layer1: B=1 C=64  H=56", 1, 64,  64,  56),
    ("layer2: B=1 C=128 H=28", 1, 128, 128, 28),
    ("layer3: B=1 C=256 H=14", 1, 256, 256, 14),
    ("layer4: B=1 C=512 H=7",  1, 512, 512, 7),
]

print("=" * 90)
print(f"{'Config':<28s} {'sp%':>4s}  {'cuDNN':>8s} {'lean+v2':>8s} {'ratio':>7s} {'kernel':>8s}")
print("-" * 90)

for label, B, C_IN, C_OUT, H in configs:
    W = H
    H_OUT = (H + 2*P - KS)//S + 1
    W_OUT = (W + 2*P - KS)//S + 1
    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = B * GH * GW
    GSC = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GSC)
    ALL_ONES = (1 << NUM_GROUPS) - 1
    DENSE_K = min(GSC * 2, 64)
    if DENSE_K < 16: DENSE_K = 16

    w = torch.randn(C_OUT, C_IN, KS, KS, device=device)
    b = torch.randn(C_OUT, device=device)
    w_cl = w.half().permute(0, 2, 3, 1).contiguous()
    b_f32 = b.float().contiguous()

    kernel = sparse_conv3x3s1_nhwc_kernel_8x16 if BW == 16 else sparse_conv3x3s1_nhwc_kernel_8x8

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

    ag_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    tc_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    y_buf = torch.empty(B, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)
    tile_ids = torch.arange(N_TILES, dtype=torch.int32, device=device)

    for sp in [95, 99]:
        x = (torch.rand(B, C_IN, H, W, device=device) > (sp/100.0)).float()
        x_f16 = x.half().contiguous()
        x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()

        # --- cuDNN ---
        for _ in range(N_WARMUP):
            _ = F.conv2d(x, w, b, stride=S, padding=P)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            _ = F.conv2d(x, w, b, stride=S, padding=P)
        torch.cuda.synchronize()
        cudnn_ms = (time.perf_counter() - t0) / N_ITER * 1000

        # --- lean + v2 prescan ---
        def run_lean():
            fast_spike_prescan_2d_v2(
                x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                ag_mask_out=ag_buf, tile_class_out=tc_buf)
            kernel[_grid](
                x_nhwc, w_cl, b_f32,
                ag_buf, tile_ids, y_buf, B,
                C_IN=C_IN, C_OUT=C_OUT,
                H_IN=H, W_IN=W,
                H_OUT=H_OUT, W_OUT=W_OUT,
                GH=GH, GW=GW,
                HAS_BIAS=True,
                GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
                ALL_ONES_MASK=ALL_ONES,
                DENSE_K=DENSE_K,
                USE_TILE_IDS=False)

        for _ in range(N_WARMUP):
            run_lean()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            run_lean()
        torch.cuda.synchronize()
        lean_ms = (time.perf_counter() - t0) / N_ITER * 1000

        # --- kernel only (no prescan) ---
        fast_spike_prescan_2d_v2(x, H_OUT, W_OUT, KS, S, P, BH, BW, GSC,
                                  ag_mask_out=ag_buf, tile_class_out=tc_buf)
        torch.cuda.synchronize()
        for _ in range(N_WARMUP):
            kernel[_grid](
                x_nhwc, w_cl, b_f32,
                ag_buf, tile_ids, y_buf, B,
                C_IN=C_IN, C_OUT=C_OUT,
                H_IN=H, W_IN=W,
                H_OUT=H_OUT, W_OUT=W_OUT,
                GH=GH, GW=GW,
                HAS_BIAS=True,
                GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
                ALL_ONES_MASK=ALL_ONES,
                DENSE_K=DENSE_K,
                USE_TILE_IDS=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            kernel[_grid](
                x_nhwc, w_cl, b_f32,
                ag_buf, tile_ids, y_buf, B,
                C_IN=C_IN, C_OUT=C_OUT,
                H_IN=H, W_IN=W,
                H_OUT=H_OUT, W_OUT=W_OUT,
                GH=GH, GW=GW,
                HAS_BIAS=True,
                GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
                ALL_ONES_MASK=ALL_ONES,
                DENSE_K=DENSE_K,
                USE_TILE_IDS=False)
        torch.cuda.synchronize()
        kern_ms = (time.perf_counter() - t0) / N_ITER * 1000

        ratio = cudnn_ms / lean_ms if lean_ms > 0 else 0
        win = "SF" if ratio > 1 else "cuDNN"
        print(f"{label:<28s} {sp:>3d}%  {cudnn_ms:>7.3f}ms {lean_ms:>7.3f}ms {ratio:>6.2f}x  {kern_ms:>7.3f}ms  {win}")

print("=" * 90)
