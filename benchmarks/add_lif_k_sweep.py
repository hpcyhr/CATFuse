"""
Add → LIF: T-sweep at fixed K=4, verifying T-scaling of §3.9-style formula.

Builds on add_lif_k_sweep.py. Same kernels, same shapes except T varies.
K is fixed at 4 (4 chunks for T=16, 16 chunks for T=64, 1 chunk for T=4).

Analytic formula for Add→LIF with fixed K:
    ratio(T, K) = (3·T + 2·T/K - 1) / (5·T)
                = 0.60 + 2/(5·K) - 1/(5·T)

At K=4:
    T=4  : 0.60 + 0.10 - 0.05    = 0.65   (35.0% savings)
    T=8  : 0.60 + 0.10 - 0.025   = 0.675  (32.5% savings)
    T=16 : 0.60 + 0.10 - 0.0125  = 0.6875 (31.25% savings, matches yesterday)
    T=32 : 0.60 + 0.10 - 0.00625 = 0.69375 (30.625% savings)
    T=64 : 0.60 + 0.10 - 0.003   = 0.6969  (30.3% savings)

Converges to 0.70 as T→∞ at K=4. Curve is expected to be very flat; the
interesting question is whether wall-clock speedup stays stable (cache and
launch overhead behavior) or shows T-dependent drift.

Output: single JSON file, one entry per T, plus matched non-fusion baseline
(baseline is T-dependent, must be re-measured for each T).
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
# Kernels (identical to add_lif_k_sweep.py)
# ============================================================

@triton.jit
def _add_lif_fused_k_kernel(
    x1_ptr, x2_ptr, spike_ptr, v_carry_ptr,
    T, K, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial

    inv_tau = 1.0 / tau
    n_chunks = T // K

    for c in range(n_chunks):
        if c == 0:
            v = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            v = tl.load(v_carry_ptr + offs, mask=mask, other=0.0)

        for k in range(K):
            t = c * K + k
            base = t * stride_t + offs
            x1 = tl.load(x1_ptr + base, mask=mask, other=0.0)
            x2 = tl.load(x2_ptr + base, mask=mask, other=0.0)
            z = x1 + x2
            v = v + (z - v) * inv_tau
            s = (v >= v_th).to(tl.float32)
            v = v * (1.0 - s) + v_reset * s
            tl.store(spike_ptr + base, s, mask=mask)

        tl.store(v_carry_ptr + offs, v, mask=mask)


@triton.jit
def _add_only_kernel(
    x1_ptr, x2_ptr, z_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0)
    tl.store(z_ptr + offs, x1 + x2, mask=mask)


@triton.jit
def _lif_only_kernel(
    z_ptr, spike_ptr,
    T, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial
    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau
    for t in range(T):
        base = t * stride_t + offs
        z = tl.load(z_ptr + base, mask=mask, other=0.0)
        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s
        tl.store(spike_ptr + base, s, mask=mask)


BLOCK_DEFAULT = 1024


def run_fusion_k(x1, x2, K, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    T = x1.shape[0]
    assert T % K == 0, f"T={T} must be divisible by K={K}"
    N_total = x1.numel()
    N_spatial = N_total // T
    stride_t = N_spatial
    spike = torch.empty_like(x1)
    v_carry = torch.empty(N_spatial, device=x1.device, dtype=x1.dtype)
    grid = (triton.cdiv(N_spatial, BLOCK),)
    _add_lif_fused_k_kernel[grid](
        x1, x2, spike, v_carry,
        T, K, N_spatial,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


def run_no_fusion(x1, x2, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    T = x1.shape[0]
    N_total = x1.numel()
    N_spatial = N_total // T
    stride_t = N_spatial
    z = torch.empty_like(x1)
    spike = torch.empty_like(x1)
    grid_add = (triton.cdiv(N_total, BLOCK),)
    _add_only_kernel[grid_add](x1, x2, z, N_total, BLOCK=BLOCK)
    grid_lif = (triton.cdiv(N_spatial, BLOCK),)
    _lif_only_kernel[grid_lif](
        z, spike, T, N_spatial, tau, v_th, v_reset, stride_t, BLOCK=BLOCK,
    )
    return spike


def run_reference(x1, x2, tau=2.0, v_th=1.0, v_reset=0.0):
    T = x1.shape[0]
    z = x1 + x2
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
# Analytic formula at fixed K
# ============================================================

def analytic_hbm_bytes(T, K, B, C, H, W, dtype_bytes=4):
    step = B * C * H * W * dtype_bytes
    n_chunks = T // K
    v_traffic_units = 2 * n_chunks - 1
    fusion = (3 * T + v_traffic_units) * step
    no_fusion = 5 * T * step
    return {
        'T': T, 'K': K, 'n_chunks': n_chunks,
        'step_bytes': step,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


# ============================================================
# Parity / timing / memory helpers
# ============================================================

def parity_check(x1, x2, K, tau, v_th, v_reset):
    out_ref = run_reference(x1, x2, tau, v_th, v_reset)
    out_fus = run_fusion_k(x1, x2, K, tau, v_th, v_reset)
    out_nof = run_no_fusion(x1, x2, tau, v_th, v_reset)

    def diff(a, b):
        d = (a - b).abs()
        return {
            'max_diff': d.max().item(),
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
    p.add_argument('--T_list', type=str, default='4,8,16,32,64',
                   help='comma-separated T values to sweep')
    p.add_argument('--K', type=int, default=4,
                   help='fixed K (TimeBlock size) for the sweep')
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--C', type=int, default=128)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--W', type=int, default=16)
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
                   default='./results/add_lif_t_sweep.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    K = args.K
    T_list = [int(t) for t in args.T_list.split(',')]
    for T in T_list:
        assert T % K == 0, f"T={T} must be divisible by K={K}"

    B, C, H, W = args.B, args.C, args.H, args.W

    print(f"{'='*86}")
    print(f"Add → LIF: T-sweep at fixed K={K}")
    print(f"{'='*86}")
    print(f"T list: {T_list}")
    print(f"shape:  B={B} C={C} H={H} W={W}")
    print(f"K:      {K}  (fixed)")
    print(f"GPU:    {torch.cuda.get_device_name(device)}")
    print()

    results = []
    print(f"{'T':>4}  {'n_chunks':>8}  {'analytic':>9}  {'fusion_MB':>10}  "
          f"{'nof_MB':>8}  {'nof_ms':>8}  {'fus_ms':>8}  {'speedup':>9}  "
          f"{'peak_MB':>9}  {'parity':>10}")
    print('-' * 102)

    for T in T_list:
        torch.manual_seed(args.seed)
        x1 = (torch.randn(T, B, C, H, W, device=device,
                          dtype=torch.float32) * 0.5).contiguous()
        x2 = (torch.randn(T, B, C, H, W, device=device,
                          dtype=torch.float32) * 0.5).contiguous()

        hbm = analytic_hbm_bytes(T, K, B, C, H, W, dtype_bytes=4)
        parity = parity_check(x1, x2, K, args.tau, args.v_th, args.v_reset)

        def bench_fus(T=T, K=K):
            _ = run_fusion_k(x1, x2, K, args.tau, args.v_th, args.v_reset,
                             BLOCK=args.block)

        def bench_nof(T=T):
            _ = run_no_fusion(x1, x2, args.tau, args.v_th, args.v_reset,
                              BLOCK=args.block)

        wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
        wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)
        mem_fus = peak_memory_one_run(bench_fus)
        mem_nof = peak_memory_one_run(bench_nof)

        speedup = wall_nof['median_ms'] / wall_fus['median_ms']

        parity_tag = (
            'BIT-EXACT' if parity['fusion_vs_reference']['bit_exact']
            else f"max={parity['fusion_vs_reference']['max_diff']:.1e}"
        )

        print(f"{T:>4}  {hbm['n_chunks']:>8}  {hbm['ratio_fusion_over_no_fusion']:>9.4f}  "
              f"{hbm['fusion_hbm_MB']:>10.2f}  {hbm['no_fusion_hbm_MB']:>8.2f}  "
              f"{wall_nof['median_ms']:>8.4f}  {wall_fus['median_ms']:>8.4f}  "
              f"{speedup:>9.4f}  {mem_fus['peak_allocated_MB']:>9.2f}  "
              f"{parity_tag:>10}")

        results.append({
            'T': T,
            'analytic_hbm': hbm,
            'parity': parity,
            'wall_clock_fusion': wall_fus,
            'wall_clock_no_fusion': wall_nof,
            'peak_memory_fusion': mem_fus,
            'peak_memory_no_fusion': mem_nof,
            'speedup_no_fusion_over_fusion': speedup,
        })

    print()

    # --- Save ---
    result = {
        'experiment': 'add_lif_t_sweep',
        'section_ref': '§3.9 T-scaling verification at fixed K, Add→LIF',
        'config': {
            'T_list': T_list,
            'K_fixed': K,
            'shape_BCHW': {'B': B, 'C': C, 'H': H, 'W': W},
            'lif': {'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset},
            'triton_block': args.block,
            'seed': args.seed,
            'dtype': 'float32',
            'device': torch.cuda.get_device_name(device),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'triton_version': triton.__version__,
        },
        't_sweep': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == '__main__':
    main()