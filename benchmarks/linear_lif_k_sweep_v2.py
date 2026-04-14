"""
Linear → LIF: K-sweep, v3.

Change from v2:
  - T is runtime (not constexpr), so Triton does NOT unroll the T loop.
    This prevents the 16-way explosion of the inner matmul code and
    should drastically reduce register pressure.
  - K is still constexpr, so t%K checks collapse at compile time.
  - The v_carry read branch (t > 0 and t % K == 0) is rewritten as a
    single condition to help the compiler.

Rationale: v2 had T=16 fully unrolled, causing each fused kernel to contain
16 copies of a full (I/BLOCK_I)-step matmul. On V100 sm_70 this exceeds
register file capacity and spills to local memory, making fusion slower
than non-fusion by 5x. v3 keeps T as a runtime loop, so the inner matmul
is emitted once in PTX and executed T times.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_lif_fused_kernel(
    x_ptr, w_ptr, spike_ptr, v_carry_ptr,
    T, B, I_dim, O_dim,                  # T is runtime now
    tau, v_th, v_reset,
    stride_t_x, stride_t_s,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Single T-step loop (runtime, not unrolled). K is compile-time so the
    t % K branches collapse at compile time for constant K values.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_b = offs_b < B
    mask_o = offs_o < O_dim

    v = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
    inv_tau = 1.0 / tau

    v_offs = offs_b[:, None] * O_dim + offs_o[None, :]
    v_mask = mask_b[:, None] & mask_o[None, :]

    for t in range(T):
        # Chunk-boundary v_carry read: t > 0 and t % K == 0
        # Note: t=0 is the first step, no read needed.
        load_v = (t > 0) & (t % K == 0)
        if load_v:
            v = tl.load(v_carry_ptr + v_offs, mask=v_mask, other=0.0)

        # Matmul: z[b, o] = sum_i x[t, b, i] * w[i, o]
        acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
        for i_start in range(0, I_dim, BLOCK_I):
            offs_i = i_start + tl.arange(0, BLOCK_I)
            mask_i = offs_i < I_dim

            x_offs = (t * stride_t_x
                      + offs_b[:, None] * I_dim
                      + offs_i[None, :])
            x_mask_2d = mask_b[:, None] & mask_i[None, :]
            x_tile = tl.load(x_ptr + x_offs, mask=x_mask_2d, other=0.0)

            w_offs = offs_i[:, None] * O_dim + offs_o[None, :]
            w_mask_2d = mask_i[:, None] & mask_o[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask_2d, other=0.0)

            acc += tl.dot(x_tile, w_tile)

        z = acc

        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s

        spike_offs = (t * stride_t_s
                      + offs_b[:, None] * O_dim
                      + offs_o[None, :])
        tl.store(spike_ptr + spike_offs, s, mask=v_mask)

        # Chunk-boundary v_carry write
        if t % K == K - 1:
            tl.store(v_carry_ptr + v_offs, v, mask=v_mask)


# ============================================================
# Non-fusion kernels (unchanged, known good)
# ============================================================

@triton.jit
def _matmul_only_kernel(
    x_ptr, w_ptr, z_ptr,
    T, B, I_dim, O_dim,
    stride_t_x, stride_t_z,
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_o = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_b = offs_b < B
    mask_o = offs_o < O_dim

    acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
    for i_start in range(0, I_dim, BLOCK_I):
        offs_i = i_start + tl.arange(0, BLOCK_I)
        mask_i = offs_i < I_dim

        x_offs = (pid_t * stride_t_x
                  + offs_b[:, None] * I_dim
                  + offs_i[None, :])
        x_mask = mask_b[:, None] & mask_i[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        w_offs = offs_i[:, None] * O_dim + offs_o[None, :]
        w_mask = mask_i[:, None] & mask_o[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    z_offs = (pid_t * stride_t_z
              + offs_b[:, None] * O_dim
              + offs_o[None, :])
    z_mask = mask_b[:, None] & mask_o[None, :]
    tl.store(z_ptr + z_offs, acc, mask=z_mask)


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


BLOCK_B_DEFAULT = 32
BLOCK_O_DEFAULT = 64
BLOCK_I_DEFAULT = 64
BLOCK_LIF_DEFAULT = 1024


def run_fusion_k(x, w, K, tau=2.0, v_th=1.0, v_reset=0.0,
                 BLOCK_B=BLOCK_B_DEFAULT, BLOCK_O=BLOCK_O_DEFAULT,
                 BLOCK_I=BLOCK_I_DEFAULT):
    T, B, I_dim = x.shape
    I2, O_dim = w.shape
    assert I_dim == I2
    assert T % K == 0
    stride_t_x = B * I_dim
    stride_t_s = B * O_dim

    spike = torch.empty(T, B, O_dim, device=x.device, dtype=x.dtype)
    v_carry = torch.empty(B * O_dim, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(O_dim, BLOCK_O))
    _linear_lif_fused_kernel[grid](
        x, w, spike, v_carry,
        T, B, I_dim, O_dim,
        tau, v_th, v_reset,
        stride_t_x, stride_t_s,
        K=K,
        BLOCK_B=BLOCK_B, BLOCK_O=BLOCK_O, BLOCK_I=BLOCK_I,
    )
    return spike


def run_no_fusion(x, w, tau=2.0, v_th=1.0, v_reset=0.0,
                  BLOCK_B=BLOCK_B_DEFAULT, BLOCK_O=BLOCK_O_DEFAULT,
                  BLOCK_I=BLOCK_I_DEFAULT, BLOCK_LIF=BLOCK_LIF_DEFAULT):
    T, B, I_dim = x.shape
    I2, O_dim = w.shape
    assert I_dim == I2
    stride_t_x = B * I_dim
    stride_t_z = B * O_dim

    z = torch.empty(T, B, O_dim, device=x.device, dtype=x.dtype)
    spike = torch.empty_like(z)

    grid_mm = (T, triton.cdiv(B, BLOCK_B), triton.cdiv(O_dim, BLOCK_O))
    _matmul_only_kernel[grid_mm](
        x, w, z,
        T, B, I_dim, O_dim,
        stride_t_x, stride_t_z,
        BLOCK_B=BLOCK_B, BLOCK_O=BLOCK_O, BLOCK_I=BLOCK_I,
    )

    N_spatial = B * O_dim
    grid_lif = (triton.cdiv(N_spatial, BLOCK_LIF),)
    _lif_only_kernel[grid_lif](
        z, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        N_spatial,
        BLOCK=BLOCK_LIF,
    )
    return spike


def run_reference(x, w, tau=2.0, v_th=1.0, v_reset=0.0):
    T, B, I_dim = x.shape
    O_dim = w.shape[1]
    z = torch.matmul(x, w)
    v = torch.zeros(B, O_dim, device=x.device, dtype=x.dtype)
    spikes = torch.empty_like(z)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z[t] - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


def analytic_hbm_bytes(T, K, B, I_dim, O_dim, dtype_bytes=4):
    step_x = B * I_dim * dtype_bytes
    step_z = B * O_dim * dtype_bytes
    w_bytes = I_dim * O_dim * dtype_bytes
    n_chunks = T // K
    v_traffic_units = 2 * n_chunks - 1

    no_fusion = T * step_x + w_bytes + 3 * T * step_z
    fusion = T * step_x + w_bytes + T * step_z + v_traffic_units * step_z

    return {
        'T': T, 'K': K, 'n_chunks': n_chunks,
        'step_x_bytes': step_x,
        'step_z_bytes': step_z,
        'w_bytes': w_bytes,
        'v_traffic_units': v_traffic_units,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


def parity_check(x, w, K, tau, v_th, v_reset):
    out_ref = run_reference(x, w, tau, v_th, v_reset)
    out_fus = run_fusion_k(x, w, K, tau, v_th, v_reset)
    out_nof = run_no_fusion(x, w, tau, v_th, v_reset)

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
        'spike_rate_fusion': out_fus.mean().item(),
        'spike_rate_no_fusion': out_nof.mean().item(),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--I', type=int, default=512)
    p.add_argument('--O', type=int, default=512)
    p.add_argument('--K_list', type=str, default='1,2,4,8,16')
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--block_b', type=int, default=BLOCK_B_DEFAULT)
    p.add_argument('--block_o', type=int, default=BLOCK_O_DEFAULT)
    p.add_argument('--block_i', type=int, default=BLOCK_I_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=11)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/linear_lif_k_sweep_v3.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, I_dim, O_dim = args.T, args.B, args.I, args.O
    K_list = [int(k) for k in args.K_list.split(',')]
    for K in K_list:
        assert T % K == 0

    print(f"{'='*90}")
    print(f"Linear → LIF: K-sweep (v3: T runtime, K constexpr)")
    print(f"{'='*90}")
    print(f"shape:   T={T} B={B} I={I_dim} O={O_dim}")
    print(f"K list:  {K_list}")
    print(f"tiles:   BLOCK_B={args.block_b} BLOCK_O={args.block_o} "
          f"BLOCK_I={args.block_i}")
    print(f"scale:   {args.input_scale}")
    print(f"GPU:     {torch.cuda.get_device_name(device)}")
    print()

    x = (torch.randn(T, B, I_dim, device=device, dtype=torch.float32)
         * args.input_scale).contiguous()
    w = (torch.randn(I_dim, O_dim, device=device, dtype=torch.float32)
         * args.input_scale).contiguous()

    print("[baseline] non-fusion (matmul kernel + LIF kernel)")

    def bench_nof():
        _ = run_no_fusion(x, w, args.tau, args.v_th, args.v_reset,
                          BLOCK_B=args.block_b, BLOCK_O=args.block_o,
                          BLOCK_I=args.block_i)

    print("  smoke test...")
    _ = run_no_fusion(x, w, args.tau, args.v_th, args.v_reset,
                      BLOCK_B=args.block_b, BLOCK_O=args.block_o,
                      BLOCK_I=args.block_i)
    for K in K_list:
        _ = run_fusion_k(x, w, K, args.tau, args.v_th, args.v_reset,
                         BLOCK_B=args.block_b, BLOCK_O=args.block_o,
                         BLOCK_I=args.block_i)
        torch.cuda.synchronize()
    print("  smoke test passed")

    wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)
    mem_nof = peak_memory_one_run(bench_nof)

    nof_analytic = (T * B * I_dim + I_dim * O_dim + 3 * T * B * O_dim) * 4
    print(f"  analytic HBM : {nof_analytic / 1024 / 1024:.3f} MB")
    print(f"  wall-clock   : {wall_nof['median_ms']:.4f} ms "
          f"± {wall_nof['stdev_ms']:.4f}")
    print(f"  peak memory  : {mem_nof['peak_allocated_MB']:.3f} MB")
    print()

    k_results = []
    print(f"{'K':>4}  {'analytic':>9}  {'fusion_MB':>10}  {'nof_MB':>8}  "
          f"{'wall_ms':>9}  {'speedup':>9}  {'peak_MB':>9}  {'spike':>8}  "
          f"{'max_diff':>10}")
    print('-' * 98)

    for K in K_list:
        hbm = analytic_hbm_bytes(T, K, B, I_dim, O_dim, dtype_bytes=4)
        parity = parity_check(x, w, K, args.tau, args.v_th, args.v_reset)

        def bench_fus(K=K):
            _ = run_fusion_k(x, w, K, args.tau, args.v_th, args.v_reset,
                             BLOCK_B=args.block_b, BLOCK_O=args.block_o,
                             BLOCK_I=args.block_i)

        wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
        mem_fus = peak_memory_one_run(bench_fus)

        speedup = wall_nof['median_ms'] / wall_fus['median_ms']
        max_diff = parity['fusion_vs_reference']['max_diff']
        spike_rate = parity['spike_rate_reference']

        print(f"{K:>4}  {hbm['ratio_fusion_over_no_fusion']:>9.4f}  "
              f"{hbm['fusion_hbm_MB']:>10.3f}  {hbm['no_fusion_hbm_MB']:>8.3f}  "
              f"{wall_fus['median_ms']:>9.4f}  {speedup:>9.4f}  "
              f"{mem_fus['peak_allocated_MB']:>9.3f}  {spike_rate:>8.4f}  "
              f"{max_diff:>10.2e}")

        k_results.append({
            'K': K,
            'analytic_hbm': hbm,
            'parity': parity,
            'wall_clock_fusion': wall_fus,
            'peak_memory_fusion': mem_fus,
            'speedup_no_fusion_over_fusion': speedup,
        })

    print()

    result = {
        'experiment': 'linear_lif_k_sweep_v3',
        'section_ref': '§3.9 structural verification on Linear(matmul)→LIF, v3',
        'change_from_v2': 'T made runtime (not constexpr) to prevent T-loop unroll',
        'config': {
            'shape': {'T': T, 'B': B, 'I': I_dim, 'O': O_dim},
            'K_list': K_list,
            'lif': {'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset},
            'triton_tiles': {
                'BLOCK_B': args.block_b,
                'BLOCK_O': args.block_o,
                'BLOCK_I': args.block_i,
            },
            'input_scale': args.input_scale,
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