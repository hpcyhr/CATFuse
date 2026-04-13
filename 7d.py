"""
v7d: wall-clock baseline for v7c's Conv→LIF fused kernel
=========================================================

Goal: measure wall-clock for v7c (F1 fused Conv→LIF) on V100 across all K
values, compared against two baselines:
  - cudnn_naive: nn.Conv2d + per-step Python LIF (the "real" reference)
  - triton_naive: v7a conv + per-step Python LIF (shares gemm order with v7c)

This gives us the first "fused kernel speedup" data point for F1.
Does NOT attempt to extract DRAM bytes via wall-clock × BW inversion
(previous plan); we just report wall-clock and speedup directly. Cross-
hardware comparison against A100 will reuse this same script.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import statistics
import json

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Shape
B, C_in, C_out, H, W = 32, 128, 128, 16, 16
KS = 3
PAD = 1
STRIDE = 1
H_out = (H + 2 * PAD - KS) // STRIDE + 1
W_out = (W + 2 * PAD - KS) // STRIDE + 1

TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

T_TOTAL = 16
K_LIST = [1, 2, 4, 8, 16]

N_WARMUP = 20
N_ITER = 100
N_REPEAT = 11


# ============================================================
# v7a conv (used as gemm-correct reference)
# ============================================================

@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, z_ptr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_z_b, stride_z_c, stride_z_h, stride_z_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_total = C_in * KS * KS
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total
        c_in_idx = offs_k // (KS * KS)
        kh_kw = offs_k % (KS * KS)
        kh_idx = kh_kw // KS
        kw_idx = kh_kw % KS
        h_in = h_out_idx[:, None] * STRIDE + kh_idx[None, :] - PAD
        w_in = w_out_idx[:, None] * STRIDE + kw_idx[None, :] - PAD
        valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
        valid = valid & k_mask[None, :]
        valid = valid & (b_idx[:, None] < B)
        x_offs = (b_idx[:, None] * stride_x_b
                  + c_in_idx[None, :] * stride_x_c
                  + h_in * stride_x_h
                  + w_in * stride_x_w)
        x_tile = tl.load(x_ptr + x_offs, mask=valid, other=0.0)
        w_offs = (offs_n[:, None] * stride_w_o
                  + c_in_idx[None, :] * stride_w_c
                  + kh_idx[None, :] * stride_w_kh
                  + kw_idx[None, :] * stride_w_kw)
        w_mask = (offs_n[:, None] < C_out) & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, tl.trans(w_tile))
    z_offs = (b_idx[:, None] * stride_z_b
              + offs_n[None, :] * stride_z_c
              + h_out_idx[:, None] * stride_z_h
              + w_out_idx[:, None] * stride_z_w)
    z_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
    tl.store(z_ptr + z_offs, acc, mask=z_mask)


def triton_conv2d(x, w):
    B_, C_in_, H_, W_ = x.shape
    C_out_ = w.shape[0]
    H_out_ = (H_ + 2 * PAD - KS) // STRIDE + 1
    W_out_ = (W_ + 2 * PAD - KS) // STRIDE + 1
    z = torch.empty(B_, C_out_, H_out_, W_out_, device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B_ * H_out_ * W_out_, BLOCK_M), triton.cdiv(C_out_, BLOCK_N))
    conv2d_implicit_gemm_kernel[grid](
        x, w, z,
        B_, C_in_, H_, W_,
        C_out_, H_out_, W_out_,
        KS=KS, PAD=PAD, STRIDE=STRIDE,
        stride_x_b=x.stride(0), stride_x_c=x.stride(1),
        stride_x_h=x.stride(2), stride_x_w=x.stride(3),
        stride_w_o=w.stride(0), stride_w_c=w.stride(1),
        stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
        stride_z_b=z.stride(0), stride_z_c=z.stride(1),
        stride_z_h=z.stride(2), stride_z_w=z.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return z


# ============================================================
# v7c fused kernel (reused)
# ============================================================

@triton.jit
def conv_lif_block_kernel(
    x_ptr, w_ptr, s_ptr, v_carry_ptr,
    K: tl.constexpr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    KS: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_x_t, stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_o, stride_w_c, stride_w_kh, stride_w_kw,
    stride_s_t, stride_s_b, stride_s_c, stride_s_h, stride_s_w,
    stride_v_b, stride_v_c, stride_v_h, stride_v_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    hw_idx = offs_m % HW_out
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    v_offs = (b_idx[:, None] * stride_v_b
              + offs_n[None, :] * stride_v_c
              + h_out_idx[:, None] * stride_v_h
              + w_out_idx[:, None] * stride_v_w)
    v_mask = (offs_m[:, None] < B * HW_out) & (offs_n[None, :] < C_out)
    v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

    K_total = C_in * KS * KS
    for t in tl.static_range(K):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K_total, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_total
            c_in_idx = offs_k // (KS * KS)
            kh_kw = offs_k % (KS * KS)
            kh_idx = kh_kw // KS
            kw_idx = kh_kw % KS
            h_in = h_out_idx[:, None] * STRIDE + kh_idx[None, :] - PAD
            w_in = w_out_idx[:, None] * STRIDE + kw_idx[None, :] - PAD
            valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            valid = valid & k_mask[None, :]
            valid = valid & (b_idx[:, None] < B)
            x_offs = (t * stride_x_t
                      + b_idx[:, None] * stride_x_b
                      + c_in_idx[None, :] * stride_x_c
                      + h_in * stride_x_h
                      + w_in * stride_x_w)
            x_tile = tl.load(x_ptr + x_offs, mask=valid, other=0.0)
            w_offs = (offs_n[:, None] * stride_w_o
                      + c_in_idx[None, :] * stride_w_c
                      + kh_idx[None, :] * stride_w_kh
                      + kw_idx[None, :] * stride_w_kw)
            w_mask = (offs_n[:, None] < C_out) & k_mask[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
            acc += tl.dot(x_tile, tl.trans(w_tile))

        v = v + (acc - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        s_offs = (t * stride_s_t
                  + b_idx[:, None] * stride_s_b
                  + offs_n[None, :] * stride_s_c
                  + h_out_idx[:, None] * stride_s_h
                  + w_out_idx[:, None] * stride_s_w)
        tl.store(s_ptr + s_offs, spike, mask=v_mask)

    tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


def triton_conv_lif_blocked(x_seq, w, K):
    T_total = x_seq.shape[0]
    n_blocks = T_total // K
    s_seq = torch.empty(T_total, B, C_out, H_out, W_out,
                        device=device, dtype=torch.float32)
    v_carry = torch.zeros(B, C_out, H_out, W_out,
                          device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B * H_out * W_out, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    for block_idx in range(n_blocks):
        block_start = block_idx * K
        x_block = x_seq[block_start:block_start + K].contiguous()
        s_block = s_seq[block_start:block_start + K]

        conv_lif_block_kernel[grid](
            x_block, w, s_block, v_carry,
            K=K,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C_out, H_out=H_out, W_out=W_out,
            KS=KS, PAD=PAD, STRIDE=STRIDE,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_x_t=x_block.stride(0), stride_x_b=x_block.stride(1),
            stride_x_c=x_block.stride(2), stride_x_h=x_block.stride(3),
            stride_x_w=x_block.stride(4),
            stride_w_o=w.stride(0), stride_w_c=w.stride(1),
            stride_w_kh=w.stride(2), stride_w_kw=w.stride(3),
            stride_s_t=s_block.stride(0), stride_s_b=s_block.stride(1),
            stride_s_c=s_block.stride(2), stride_s_h=s_block.stride(3),
            stride_s_w=s_block.stride(4),
            stride_v_b=v_carry.stride(0), stride_v_c=v_carry.stride(1),
            stride_v_h=v_carry.stride(2), stride_v_w=v_carry.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    return s_seq


# ============================================================
# Baselines
# ============================================================

def cudnn_naive(x_seq, conv_module, T):
    """cudnn conv + py LIF, per step. The real naive reference."""
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        z_t = conv_module(x_seq[t])
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


def triton_naive(x_seq, w, T):
    """v7a conv + py LIF, per step. Same gemm order as v7c."""
    v = torch.zeros(B, C_out, H_out, W_out, device=device)
    spikes = []
    for t in range(T):
        z_t = triton_conv2d(x_seq[t], w)
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# Timing
# ============================================================

def cuda_time_one_shot(fn, n_iter):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def cuda_time_stats(fn, n_iter=N_ITER, n_repeat=N_REPEAT):
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'max': max(samples),
        'stdev': statistics.stdev(samples),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("v7d: wall-clock baseline for F1 (Conv→LIF fused kernel)")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C_in}, H=W={H}, k={KS}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    print()

    conv_module = nn.Conv2d(C_in, C_out, KS, padding=PAD, bias=False).to(device)
    with torch.no_grad():
        conv_module.weight.mul_(2.0)
    w = conv_module.weight.contiguous()
    x_seq = torch.randn(T_TOTAL, B, C_in, H, W, device=device)

    # Sanity check parity once
    print("Parity check (v7c vs triton_naive):")
    s_ref = triton_naive(x_seq, w, T_TOTAL)
    for K in K_LIST:
        s_tri = triton_conv_lif_blocked(x_seq.contiguous(), w, K)
        assert torch.equal(s_ref, s_tri), f"parity failed at K={K}"
    print("  all K bit-exact ✓")
    print()

    # Timing
    print("Wall-clock (median of 11 runs × 100 iters each):")
    print(f"{'config':<18} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs cudnn':<12} {'vs tri_naive':<14}")
    print('-' * 78)

    results = []

    # cudnn naive
    def run_cudnn():
        with torch.no_grad():
            _ = cudnn_naive(x_seq, conv_module, T_TOTAL)
    for _ in range(N_WARMUP):
        run_cudnn()
    torch.cuda.synchronize()
    t_cudnn = cuda_time_stats(run_cudnn)
    results.append(('cudnn_naive', t_cudnn))

    # triton naive
    def run_tri_naive():
        _ = triton_naive(x_seq, w, T_TOTAL)
    for _ in range(N_WARMUP):
        run_tri_naive()
    torch.cuda.synchronize()
    t_tri_naive = cuda_time_stats(run_tri_naive)
    results.append(('triton_naive', t_tri_naive))

    # v7c at each K
    for K in K_LIST:
        def run_ctf(K=K):
            _ = triton_conv_lif_blocked(x_seq.contiguous(), w, K)
        for _ in range(N_WARMUP):
            run_ctf()
        torch.cuda.synchronize()
        t_ctf = cuda_time_stats(run_ctf)
        results.append((f'v7c_K={K}', t_ctf))

    t_cudnn_med = t_cudnn['median']
    t_tri_naive_med = t_tri_naive['median']
    for name, stats in results:
        med = stats['median']
        std = stats['stdev']
        vs_cudnn = t_cudnn_med / med
        vs_tri = t_tri_naive_med / med
        print(f"{name:<18} {med:<14.3f} {std:<10.3f} "
              f"{vs_cudnn:<12.2f}x {vs_tri:<14.2f}x")
    print()

    # Save to json for future cross-hardware comparison
    out = {
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C_in': C_in, 'C_out': C_out,
                  'H': H, 'W': W, 'KS': KS},
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v7d_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v7d_v100_baseline.json")
    print()
    print("Interpretation:")
    print("  - 'vs cudnn' > 1: fused kernel beats real naive baseline")
    print("  - 'vs cudnn' < 1: Triton conv is losing to cudnn; speedup is")
    print("    bottlenecked by Triton's V100 conv performance, not by CTF.")
    print("    In that case 'vs tri_naive' is the fair comparison —")
    print("    it shows what CTF saves over the same-gemm non-fused baseline.")


if __name__ == '__main__':
    main()