"""
Test 9: Run SparseFlow's ORIGINAL sparse_conv2d vs cuDNN
WITHOUT CATFuse wrapper, to verify SparseFlow's claimed speedups.

If standalone SparseFlow is fast but our integration is slow → integration bug.
If standalone SparseFlow is also slow → different benchmark conditions.
"""
import sys, os, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0'
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("SparseFlow standalone benchmark (no CATFuse wrapper)")
print("=" * 70)

# Import SparseFlow's original entry point
from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward

configs = [
    # (label, B, C_in, C_out, H, kernel_size, stride, padding)
    ("B=8  C=64  H=56 k=3", 8, 64, 64, 56, 3, 1, 1),
    ("B=8  C=128 H=28 k=3", 8, 128, 128, 28, 3, 1, 1),
    ("B=8  C=256 H=14 k=3", 8, 256, 256, 14, 3, 1, 1),
    ("B=8  C=512 H=7  k=3", 8, 512, 512, 7, 3, 1, 1),
    ("B=1  C=64  H=56 k=3", 1, 64, 64, 56, 3, 1, 1),
    ("B=1  C=128 H=28 k=3", 1, 128, 128, 28, 3, 1, 1),
    ("B=1  C=256 H=14 k=3", 1, 256, 256, 14, 3, 1, 1),
    ("B=1  C=512 H=7  k=3", 1, 512, 512, 7, 3, 1, 1),
]

print(f"\n  {'Config':<28s} {'Sparsity':>8s}  {'cuDNN':>8s}  {'SF-raw':>8s}  {'Ratio':>7s}")
print("  " + "-" * 70)

for label, B, cin, cout, H, ks, s, p in configs:
    conv = nn.Conv2d(cin, cout, ks, stride=s, padding=p, bias=True).to(device)
    w = conv.weight.detach()
    b = conv.bias.detach()
    
    for sp_pct in [95, 99]:
        thresh = sp_pct / 100.0
        x = (torch.rand(B, cin, H, H, device=device) > thresh).float()
        actual_sp = (1 - x.mean().item()) * 100
        
        # cuDNN
        torch.cuda.synchronize()
        # warmup
        for _ in range(50):
            with torch.no_grad():
                _ = F.conv2d(x, w, b, stride=s, padding=p)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            with torch.no_grad():
                _ = F.conv2d(x, w, b, stride=s, padding=p)
        torch.cuda.synchronize()
        cudnn_ms = (time.perf_counter() - t0) / 500 * 1000

        # SparseFlow raw kernel call
        x_f16 = x.half().contiguous()
        w_cl = w.half().permute(0, 2, 3, 1).contiguous()
        
        # warmup
        for _ in range(10):
            try:
                _ = sparse_conv2d_forward(
                    x_f16, w_cl, b, 
                    kernel_size=ks, stride=s, padding=p,
                    threshold=1e-6,
                )
            except Exception as e:
                print(f"  {label} sp={sp_pct}%: SF error: {e}")
                break
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                _ = sparse_conv2d_forward(
                    x_f16, w_cl, b,
                    kernel_size=ks, stride=s, padding=p,
                    threshold=1e-6,
                )
            torch.cuda.synchronize()
            sf_ms = (time.perf_counter() - t0) / 200 * 1000
            
            ratio = cudnn_ms / sf_ms if sf_ms > 0 else 0
            winner = "SF" if ratio > 1 else "cuDNN"
            print(f"  {label:<28s} {sp_pct:>6d}%  {cudnn_ms:>7.3f}ms {sf_ms:>7.3f}ms  {ratio:>6.2f}x  {winner}")
            continue
        # If we got here from the break, print error line
        print(f"  {label:<28s} {sp_pct:>6d}%  {cudnn_ms:>7.3f}ms     ERROR")

# Also test with SparseConv2d module wrapper (SparseFlow's own nn.Module)
print(f"\n--- SparseConv2d nn.Module wrapper ---")
try:
    from catfuse.sparseflow.ops.sparse_conv2d import SparseConv2d
    
    conv_dense = nn.Conv2d(64, 64, 3, padding=1, bias=True).to(device)
    sparse_conv = SparseConv2d.from_dense(conv_dense, block_size=16).to(device).eval()
    # Set inference mode to avoid calibration syncs
    sparse_conv.inference_mode = True
    
    for sp_pct in [95, 99]:
        x = (torch.rand(8, 64, 56, 56, device=device) > (sp_pct/100.0)).float()
        
        # warmup
        for _ in range(20):
            with torch.no_grad():
                _ = sparse_conv(x)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = sparse_conv(x)
        torch.cuda.synchronize()
        sc_ms = (time.perf_counter() - t0) / 200 * 1000
        
        # cuDNN baseline
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = conv_dense(x)
        torch.cuda.synchronize()
        cd_ms = (time.perf_counter() - t0) / 200 * 1000
        
        ratio = cd_ms / sc_ms if sc_ms > 0 else 0
        print(f"  B=8 C=64 H=56 sp={sp_pct}%: cuDNN={cd_ms:.3f}ms  SparseConv2d={sc_ms:.3f}ms  ratio={ratio:.2f}x")

except Exception as e:
    print(f"  SparseConv2d import/test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
