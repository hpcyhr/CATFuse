"""
v14: F2 — Linear → LIF fused kernel (fair baseline: + sj_cupy)
===============================================================

v8 rerun with a SpikingJelly cupy baseline added, so F2's speedup is
measured against the SNN community's SOTA fused LIF kernel rather than
against a py-loop LIF.

Kernel code is identical to v8's. Only the baselines change:
  - sj_torch: SeqToANNContainer(Linear) → LIFNode(torch)
  - sj_cupy:  SeqToANNContainer(Linear) → LIFNode(cupy)  ← SOTA
  - v14 = v8: Triton fused Linear+LIF with TimeBlock(K) + StateCarry
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import statistics
import json

from spikingjelly.activation_based import neuron, functional as sj_func, surrogate, layer

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Shape (same as v8)
B = 32
C_in = 256
C_out = 512

# LIF
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

T_TOTAL = 16
K_LIST = [1, 2, 4, 8, 16]

N_WARMUP = 20
N_ITER = 100
N_REPEAT = 11


# ============================================================
# v8's fused kernel (copied verbatim)
# ============================================================

@triton.jit
def linear_lif_block_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    v_carry_ptr,
    K: tl.constexpr,
    B, C_in, C_out,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_x_t, stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,
    stride_s_t, stride_s_m, stride_s_n,
    stride_v_m, stride_v_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    v_offs = offs_m[:, None] * stride_v_m + offs_n[None, :] * stride_v_n
    v_mask = (offs_m[:, None] < B) & (offs_n[None, :] < C_out)
    v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

    for t in tl.static_range(K):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, C_in, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < C_in

            x_offs = (t * stride_x_t
                      + offs_m[:, None] * stride_x_m
                      + offs_k[None, :] * stride_x_k)
            x_mask = (offs_m[:, None] < B) & k_mask[None, :]
            x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

            w_offs = offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
            w_mask_inner = (offs_n[:, None] < C_out) & k_mask[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask_inner, other=0.0)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        v = v + (acc - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        s_offs = (t * stride_s_t
                  + offs_m[:, None] * stride_s_m
                  + offs_n[None, :] * stride_s_n)
        tl.store(s_ptr + s_offs, spike, mask=v_mask)

    tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


def triton_linear_lif_blocked(x_seq, w, K):
    T_total = x_seq.shape[0]
    assert T_total % K == 0
    n_blocks = T_total // K

    s_seq = torch.empty(T_total, B, C_out, device=device, dtype=torch.float32)
    v_carry = torch.zeros(B, C_out, device=device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    for block_idx in range(n_blocks):
        block_start = block_idx * K
        x_block = x_seq[block_start:block_start + K].contiguous()
        s_block = s_seq[block_start:block_start + K]

        linear_lif_block_kernel[grid](
            x_block, w, s_block, v_carry,
            K=K,
            B=B, C_in=C_in, C_out=C_out,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_x_t=x_block.stride(0), stride_x_m=x_block.stride(1),
            stride_x_k=x_block.stride(2),
            stride_w_n=w.stride(0), stride_w_k=w.stride(1),
            stride_s_t=s_block.stride(0), stride_s_m=s_block.stride(1),
            stride_s_n=s_block.stride(2),
            stride_v_m=v_carry.stride(0), stride_v_n=v_carry.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    return s_seq


# ============================================================
# Reference (for parity): triton_linear + py LIF
# ============================================================

@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, z_ptr,
    M, N, K_dim,
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,
    stride_z_m, stride_z_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K_dim, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_dim
        x_offs = offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
        w_offs = offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, tl.trans(w_tile))
    z_offs = offs_m[:, None] * stride_z_m + offs_n[None, :] * stride_z_n
    z_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(z_ptr + z_offs, acc, mask=z_mask)


def triton_linear(x, w):
    B_, C_in_ = x.shape
    C_out_ = w.shape[0]
    z = torch.empty(B_, C_out_, device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
    grid = (triton.cdiv(B_, BLOCK_M), triton.cdiv(C_out_, BLOCK_N))
    matmul_kernel[grid](
        x, w, z,
        B_, C_out_, C_in_,
        stride_x_m=x.stride(0), stride_x_k=x.stride(1),
        stride_w_n=w.stride(0), stride_w_k=w.stride(1),
        stride_z_m=z.stride(0), stride_z_n=z.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return z


def reference_triton_linear_lif(x_seq, w, T):
    v = torch.zeros(B, C_out, device=device)
    spikes = []
    for t in range(T):
        z_t = triton_linear(x_seq[t], w)
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines (the new part — Linear wrapped in SeqToANN)
# ============================================================

def make_sj_pipeline(linear_module, backend):
    linear_sj = nn.Linear(C_in, C_out, bias=False).to(device)
    with torch.no_grad():
        linear_sj.weight.copy_(linear_module.weight)
    linear_seq = layer.SeqToANNContainer(linear_sj)
    lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m', backend=backend,
    ).to(device)
    return linear_seq, lif


def sj_baseline_forward(x_seq, linear_seq, lif):
    sj_func.reset_net(lif)
    y_seq = linear_seq(x_seq)
    return lif(y_seq)


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
    print("=" * 80)
    print("v14: F2 Linear→LIF — fair baseline (+ sj_cupy)")
    print(f"Shape: T={T_TOTAL}, B={B}, C_in={C_in}, C_out={C_out}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print()

    linear = nn.Linear(C_in, C_out, bias=False).to(device)
    with torch.no_grad():
        linear.weight.mul_(2.0)
    w = linear.weight.contiguous()
    x_seq = torch.randn(T_TOTAL, B, C_in, device=device)

    # Parity vs triton_linear + py LIF
    print("Step 1: Parity — v14 fused vs triton_linear+py-LIF reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<14} {'spike_rate':<12}")
    print('-' * 60)
    s_ref = reference_triton_linear_lif(x_seq, w, T_TOTAL)
    all_pass = True
    for K in K_LIST:
        s_tri = triton_linear_lif_blocked(x_seq.contiguous(), w, K)
        exact = torch.equal(s_ref, s_tri)
        max_d = (s_ref - s_tri).abs().max().item()
        n_diff = (s_ref != s_tri).sum().item()
        rate = s_tri.mean().item()
        print(f"{K:<4} {str(exact):<8} {max_d:<12.2e} "
              f"{n_diff:<14,} {rate:<12.4f}")
        if not exact:
            all_pass = False
    print()
    if all_pass:
        print("PASS: all K bit-exact ✓")
    else:
        print("⚠ parity broken")
        return
    print()

    # v14 vs sj
    print("Step 2: v14 vs SpikingJelly (numerical equivalence)")
    linear_seq_t, lif_t = make_sj_pipeline(linear, 'torch')
    with torch.no_grad():
        s_sj_t = sj_baseline_forward(x_seq, linear_seq_t, lif_t)
    s_v14 = triton_linear_lif_blocked(x_seq.contiguous(), w, K=16)
    max_d = (s_sj_t - s_v14).abs().max().item()
    n_diff = (s_sj_t != s_v14).sum().item()
    flip_pct = n_diff / s_v14.numel() * 100
    print(f"  v14 vs SJ(torch) max_diff: {max_d:.2e}")
    print(f"  diff_spikes: {n_diff:,}/{s_v14.numel():,} ({flip_pct:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_t.mean().item():.4f}")
    print(f"  v14       spike rate: {s_v14.mean().item():.4f}")
    print()

    # Wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<22} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 78)
    results = []

    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline_forward(x_seq, linear_seq_t, lif_t)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_t = cuda_time_stats(run_sj_torch)
    results.append(('sj_torch', t_sj_t))

    cupy_available = False
    try:
        linear_seq_c, lif_c = make_sj_pipeline(linear, 'cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline_forward(x_seq, linear_seq_c, lif_c)
        for _ in range(N_WARMUP):
            run_sj_cupy()
        torch.cuda.synchronize()
        t_sj_c = cuda_time_stats(run_sj_cupy)
        results.append(('sj_cupy', t_sj_c))
        cupy_available = True
    except Exception as e:
        print(f"  (cupy backend unavailable: {e})")

    for K in K_LIST:
        def run_v14(K=K):
            _ = triton_linear_lif_blocked(x_seq.contiguous(), w, K)
        for _ in range(N_WARMUP):
            run_v14()
        torch.cuda.synchronize()
        t_v14 = cuda_time_stats(run_v14)
        results.append((f'v14_K={K}', t_v14))

    t_torch_med = t_sj_t['median']
    t_cupy_med = t_sj_c['median'] if cupy_available else float('nan')
    for name, stats in results:
        med = stats['median']
        std = stats['stdev']
        vs_torch = t_torch_med / med
        vs_cupy = t_cupy_med / med if cupy_available else float('nan')
        cupy_str = f"{vs_cupy:<12.2f}x" if cupy_available else "N/A"
        print(f"{name:<22} {med:<14.3f} {std:<10.3f} "
              f"{vs_torch:<12.2f}x {cupy_str}")
    print()

    out = {
        'kernel': 'F2_linear_lif_fair',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C_in': C_in, 'C_out': C_out},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v14_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v14_v100_baseline.json")


if __name__ == '__main__':
    main()