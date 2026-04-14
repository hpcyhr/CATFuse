"""
AvgPool(2,2) → LIF: fusion vs non-fusion, K-sweep.

Structural counterpart of add_lif_k_sweep.py, but with a spatial-reducing
TSI operator: average pooling with kernel=2, stride=2.

Key difference from Add→LIF:
  - The TSI output z has spatial shape (H/2, W/2), 1/4 the input size.
  - Therefore the analytic ratio is different: the baseline's z traffic
    is relatively smaller (output-sized instead of input-sized), so
    fusion's relative savings shrinks from 38.75% to ~27.7%.

Analytic formula (derived in chat):
    |step_in|  = B·C·H·W·dtype_bytes
    |step_out| = B·C·(H/2)·(W/2)·dtype_bytes = |step_in| / 4

    non-fusion:
        pool: read x (T·|step_in|) + write z (T·|step_out|)
        LIF : read z (T·|step_out|) + write spike (T·|step_out|)
        total = T·|step_in| + 3·T·|step_out| = (7T/4)·|step_in|

    fusion(K):
        read x             : T·|step_in|
        write spike        : T·|step_out|      (=T·|step_in|/4)
        v_carry traffic    : (2·T/K − 1)·|step_out|
                                (chunk 0 skips v-read; every chunk writes v,
                                 last write is wasted; same structure as
                                 Add→LIF's K kernel)
        total = T·|step_in| + T·|step_in|/4 + (2·T/K − 1)·|step_in|/4
              = [5T + 2·T/K − 1] / 4 · |step_in|

    ratio(T, K) = (5T + 2·T/K − 1) / (7T)

At T=16:
    K=1 : 111/112 = 0.9911  (0.89% savings)
    K=2 : 95/112  = 0.8482  (15.18% savings)
    K=4 : 87/112  = 0.7768  (22.32% savings)
    K=8 : 83/112  = 0.7411  (25.89% savings)
    K=16: 81/112  = 0.7232  (27.68% savings)

v_carry buffer size: B·C·(H/2)·(W/2)·4 bytes (output-spatial, one slab).

Output: single JSON file, one entry per K, plus the fixed non-fusion
baseline.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


# ============================================================
# Fusion kernel with TimeBlock(K), AvgPool(2,2) epilogue-absorbed
# ============================================================

@triton.jit
def _avgpool_lif_fused_k_kernel(
    x_ptr, spike_ptr, v_carry_ptr,
    T, K,
    B, C, H_in, W_in,           # input spatial
    H_out, W_out,               # output spatial (H_in/2, W_in/2)
    tau, v_th, v_reset,
    stride_t_in, stride_t_out,  # bytes-per-timestep for input and output
    BLOCK: tl.constexpr,
):
    """
    Each program handles BLOCK output-spatial elements (over the B·C·H_out·W_out
    flat index space). For each output element, it loads the 2×2 input patch,
    computes the average in registers, feeds it to LIF, stores the spike.

    Membrane potential v is indexed by the output-spatial position and
    carried across TimeBlock chunks via v_carry_ptr.

    HBM reads per program per time step:
      - 4 input loads (the 2×2 patch), gathered via index arithmetic
    HBM writes per program per time step:
      - 1 spike store
    HBM touches per chunk boundary:
      - 1 v_carry read at chunk start (skipped on chunk 0)
      - 1 v_carry write at chunk end (last chunk's write is wasted)
    """
    pid = tl.program_id(0)
    offs_out = pid * BLOCK + tl.arange(0, BLOCK)
    N_out_per_t = B * C * H_out * W_out
    mask = offs_out < N_out_per_t

    # Decompose flat output index into (b, c, h_out, w_out)
    # Layout: contiguous in w_out, then h_out, then c, then b
    w_out = offs_out % W_out
    tmp = offs_out // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    b = tmp // C

    # Corresponding input top-left pixel of the 2×2 patch
    h_in = h_out * 2
    w_in = w_out * 2

    # Flat input offsets for the 4 patch pixels (within one time step)
    # Input layout: (b, c, h_in, w_in), stride ordering W_in fastest
    base_in = ((b * C + c) * H_in + h_in) * W_in + w_in
    off_tl = base_in                     # top-left
    off_tr = base_in + 1                 # top-right
    off_bl = base_in + W_in              # bottom-left
    off_br = base_in + W_in + 1          # bottom-right

    inv_tau = 1.0 / tau
    n_chunks = T // K

    for chunk in range(n_chunks):
        if chunk == 0:
            v = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            v = tl.load(v_carry_ptr + offs_out, mask=mask, other=0.0)

        for k in range(K):
            t = chunk * K + k
            # Input reads for time step t
            tl_in = t * stride_t_in
            p_tl = tl.load(x_ptr + tl_in + off_tl, mask=mask, other=0.0)
            p_tr = tl.load(x_ptr + tl_in + off_tr, mask=mask, other=0.0)
            p_bl = tl.load(x_ptr + tl_in + off_bl, mask=mask, other=0.0)
            p_br = tl.load(x_ptr + tl_in + off_br, mask=mask, other=0.0)

            z = (p_tl + p_tr + p_bl + p_br) * 0.25          # register-resident

            v = v + (z - v) * inv_tau
            s = (v >= v_th).to(tl.float32)
            v = v * (1.0 - s) + v_reset * s

            spike_offset = t * stride_t_out + offs_out
            tl.store(spike_ptr + spike_offset, s, mask=mask)

        tl.store(v_carry_ptr + offs_out, v, mask=mask)


# ============================================================
# Non-fusion kernels
# ============================================================

@triton.jit
def _avgpool_only_kernel(
    x_ptr, z_ptr,
    T,
    B, C, H_in, W_in,
    H_out, W_out,
    stride_t_in, stride_t_out,
    BLOCK: tl.constexpr,
):
    """
    Program over flat (t, b, c, h_out, w_out) space. Each thread reads 2×2
    input patch, writes 1 output pixel to z.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    N_total_out = T * B * C * H_out * W_out
    mask = offs < N_total_out

    # Decompose flat offset into (t, b, c, h_out, w_out)
    w_out = offs % W_out
    tmp = offs // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    tmp = tmp // C
    b = tmp % B
    t = tmp // B

    h_in = h_out * 2
    w_in = w_out * 2
    base_in = t * stride_t_in + ((b * C + c) * H_in + h_in) * W_in + w_in

    p_tl = tl.load(x_ptr + base_in, mask=mask, other=0.0)
    p_tr = tl.load(x_ptr + base_in + 1, mask=mask, other=0.0)
    p_bl = tl.load(x_ptr + base_in + W_in, mask=mask, other=0.0)
    p_br = tl.load(x_ptr + base_in + W_in + 1, mask=mask, other=0.0)
    z = (p_tl + p_tr + p_bl + p_br) * 0.25

    tl.store(z_ptr + offs, z, mask=mask)


@triton.jit
def _lif_only_kernel(
    z_ptr, spike_ptr,
    T, N_spatial_out,
    tau, v_th, v_reset,
    stride_t_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial_out
    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau
    for t in range(T):
        base = t * stride_t_out + offs
        z = tl.load(z_ptr + base, mask=mask, other=0.0)
        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s
        tl.store(spike_ptr + base, s, mask=mask)


BLOCK_DEFAULT = 1024


# ============================================================
# Python wrappers
# ============================================================

def run_fusion_k(x, K, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    assert x.is_contiguous()
    T, B, C, H_in, W_in = x.shape
    assert H_in % 2 == 0 and W_in % 2 == 0, "H, W must be even for AvgPool(2,2)"
    assert T % K == 0
    H_out, W_out = H_in // 2, W_in // 2
    N_out_per_t = B * C * H_out * W_out
    stride_t_in = B * C * H_in * W_in
    stride_t_out = N_out_per_t

    spike = torch.empty(T, B, C, H_out, W_out, device=x.device, dtype=x.dtype)
    v_carry = torch.empty(N_out_per_t, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N_out_per_t, BLOCK),)
    _avgpool_lif_fused_k_kernel[grid](
        x, spike, v_carry,
        T, K,
        B, C, H_in, W_in,
        H_out, W_out,
        tau, v_th, v_reset,
        stride_t_in, stride_t_out,
        BLOCK=BLOCK,
    )
    return spike


def run_no_fusion(x, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    assert x.is_contiguous()
    T, B, C, H_in, W_in = x.shape
    H_out, W_out = H_in // 2, W_in // 2
    N_total_out = T * B * C * H_out * W_out
    N_spatial_out = B * C * H_out * W_out
    stride_t_in = B * C * H_in * W_in
    stride_t_out = N_spatial_out

    z = torch.empty(T, B, C, H_out, W_out, device=x.device, dtype=x.dtype)
    spike = torch.empty_like(z)

    # Pass 1: pool
    grid_pool = (triton.cdiv(N_total_out, BLOCK),)
    _avgpool_only_kernel[grid_pool](
        x, z,
        T, B, C, H_in, W_in, H_out, W_out,
        stride_t_in, stride_t_out,
        BLOCK=BLOCK,
    )
    # Pass 2: LIF
    grid_lif = (triton.cdiv(N_spatial_out, BLOCK),)
    _lif_only_kernel[grid_lif](
        z, spike,
        T, N_spatial_out,
        tau, v_th, v_reset,
        stride_t_out,
        BLOCK=BLOCK,
    )
    return spike


def run_reference(x, tau=2.0, v_th=1.0, v_reset=0.0):
    """PyTorch reference."""
    T, B, C, H_in, W_in = x.shape
    # AvgPool2d(kernel=2, stride=2) via torch functional
    z = torch.nn.functional.avg_pool2d(
        x.reshape(T * B, C, H_in, W_in), kernel_size=2, stride=2
    ).reshape(T, B, C, H_in // 2, W_in // 2)
    v = torch.zeros_like(z[0])
    spikes = torch.empty_like(z)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z[t] - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


# ============================================================
# Analytic formula
# ============================================================

def analytic_hbm_bytes(T, K, B, C, H_in, W_in, dtype_bytes=4):
    step_in = B * C * H_in * W_in * dtype_bytes
    step_out = step_in // 4
    n_chunks = T // K
    v_traffic_units = 2 * n_chunks - 1

    # non-fusion: T·step_in + 3·T·step_out = T·step_in·(1 + 3/4) = 7T·step_in/4
    no_fusion = T * step_in + 3 * T * step_out

    # fusion: T·step_in + T·step_out + v_traffic_units·step_out
    fusion = T * step_in + T * step_out + v_traffic_units * step_out

    return {
        'T': T, 'K': K, 'n_chunks': n_chunks,
        'step_in_bytes': step_in,
        'step_out_bytes': step_out,
        'v_traffic_units': v_traffic_units,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


# ============================================================
# Helpers
# ============================================================

def parity_check_k(x, K, tau, v_th, v_reset):
    out_ref = run_reference(x, tau, v_th, v_reset)
    out_fus = run_fusion_k(x, K, tau, v_th, v_reset)
    out_nof = run_no_fusion(x, tau, v_th, v_reset)

    def diff(a, b):
        d = (a - b).abs()
        return {
            'max_diff': d.max().item(),
            'mean_abs_diff': d.mean().item(),
            'n_diff': (a != b).sum().item(),
            'n_total': a.numel(),
            'bit_exact': bool((a == b).all().item()),
        }

    return {
        'spike_rate_reference': out_ref.mean().item(),
        'fusion_vs_reference': diff(out_fus, out_ref),
        'no_fusion_vs_reference': diff(out_nof, out_ref),
        'fusion_vs_no_fusion': diff(out_fus, out_nof),
    }


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


def cuda_time_stats(fn, n_iter=100, n_repeat=11, n_warmup=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'mean_ms': statistics.mean(samples),
        'stdev_ms': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'n_iter': n_iter,
        'n_repeat': n_repeat,
    }


def peak_memory_one_run(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return {
        'peak_allocated_bytes': peak,
        'peak_allocated_MB': peak / 1024 / 1024,
    }


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--C', type=int, default=128)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--W', type=int, default=16)
    p.add_argument('--K_list', type=str, default='1,2,4,8,16')
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--block', type=int, default=BLOCK_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=11)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/avgpool_lif_k_sweep.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C, H, W = args.T, args.B, args.C, args.H, args.W
    assert H % 2 == 0 and W % 2 == 0, "H, W must be even for AvgPool(2,2)"
    K_list = [int(k) for k in args.K_list.split(',')]
    for K in K_list:
        assert T % K == 0, f"T={T} must be divisible by K={K}"

    print(f"{'='*90}")
    print(f"AvgPool(2,2) → LIF: K-sweep verifying §3.9 structure (spatial-reducing TSI)")
    print(f"{'='*90}")
    print(f"shape:  T={T} B={B} C={C} H={H} W={W}")
    print(f"        input spatial: {H}x{W}, output spatial: {H//2}x{W//2}")
    print(f"K list: {K_list}")
    print(f"GPU:    {torch.cuda.get_device_name(device)}")
    print()

    # --- input ---
    x = (torch.randn(T, B, C, H, W, device=device,
                     dtype=torch.float32) * 0.5).contiguous()

    # --- baseline ---
    print("[baseline] non-fusion (AvgPool kernel + LIF kernel)")

    def bench_nof():
        _ = run_no_fusion(x, args.tau, args.v_th, args.v_reset, BLOCK=args.block)

    wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)
    mem_nof = peak_memory_one_run(bench_nof)
    nof_analytic = 7 * T * B * C * H * W * 4 // 4  # (7T/4)·|step_in|
    print(f"  analytic HBM   : {nof_analytic / 1024 / 1024:.2f} MB")
    print(f"  wall-clock     : {wall_nof['median_ms']:.4f} ms "
          f"± {wall_nof['stdev_ms']:.4f}")
    print(f"  peak memory    : {mem_nof['peak_allocated_MB']:.2f} MB")
    print()

    # --- K sweep ---
    k_results = []
    print(f"{'K':>4}  {'analytic':>9}  {'fusion_MB':>10}  {'nof_MB':>8}  "
          f"{'wall_ms':>9}  {'speedup':>9}  {'peak_MB':>9}  {'parity':>10}")
    print('-' * 92)

    for K in K_list:
        hbm = analytic_hbm_bytes(T, K, B, C, H, W, dtype_bytes=4)
        parity = parity_check_k(x, K, args.tau, args.v_th, args.v_reset)

        def bench_fus(K=K):
            _ = run_fusion_k(x, K, args.tau, args.v_th, args.v_reset,
                             BLOCK=args.block)

        wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
        mem_fus = peak_memory_one_run(bench_fus)
        speedup = wall_nof['median_ms'] / wall_fus['median_ms']

        parity_tag = (
            'BIT-EXACT' if parity['fusion_vs_reference']['bit_exact']
            else f"max={parity['fusion_vs_reference']['max_diff']:.1e}"
        )

        print(f"{K:>4}  {hbm['ratio_fusion_over_no_fusion']:>9.4f}  "
              f"{hbm['fusion_hbm_MB']:>10.2f}  {hbm['no_fusion_hbm_MB']:>8.2f}  "
              f"{wall_fus['median_ms']:>9.4f}  {speedup:>9.4f}  "
              f"{mem_fus['peak_allocated_MB']:>9.2f}  {parity_tag:>10}")

        k_results.append({
            'K': K,
            'analytic_hbm': hbm,
            'parity': parity,
            'wall_clock_fusion': wall_fus,
            'peak_memory_fusion': mem_fus,
            'speedup_no_fusion_over_fusion': speedup,
        })

    print()

    # --- Save ---
    result = {
        'experiment': 'avgpool_lif_k_sweep',
        'section_ref': '§3.9 structural verification on AvgPool(2,2)→LIF',
        'config': {
            'shape': {'T': T, 'B': B, 'C': C, 'H': H, 'W': W},
            'avgpool': {'kernel': 2, 'stride': 2},
            'K_list': K_list,
            'lif': {'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset},
            'triton_block': args.block,
            'seed': args.seed,
            'dtype': 'float32',
            'device': torch.cuda.get_device_name(device),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'triton_version': triton.__version__,
        },
        'no_fusion_baseline': {
            'analytic_hbm_bytes': nof_analytic,
            'analytic_hbm_MB': nof_analytic / 1024 / 1024,
            'wall_clock': wall_nof,
            'peak_memory': mem_nof,
        },
        'k_sweep': k_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == '__main__':
    main()