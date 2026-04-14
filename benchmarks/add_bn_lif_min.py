"""
Add → BN → LIF: fusion vs non-fusion, K=T only.

Memory-bound chained-TSI counterpart to conv_bn_lif_min.py. Both 'Add'
operands are independent input tensors; BN is inference-mode per-channel
affine; LIF is the same leaky integrator used in earlier experiments.

Pipeline:
    z[t,b,c,h,w]  = a[t,b,c,h,w] + x[t,b,c,h,w]      # Add (TSI, elementwise)
    z'[t,b,c,h,w] = z * scale[c] + bias[c]            # BN  (TSI, per-ch affine)
    spike, v      = LIF(z')                            # CSR

Variable being tested:
  do z and z' get materialized to HBM, or fused in registers?

  - fusion:     a + x → z in registers, z*scale+bias → z' in registers,
                LIF on z' in registers, only spike (and one v_carry write) hit HBM
  - non-fusion: 3 separate kernels (add, BN, LIF), z and z' both hit HBM

Analytic HBM bytes at K=T:
    |step|     = B·C·H·W·4
    bn_params  = 2·C·4

    non-fusion = 2·T·|step|              (read a, read x)
               + T·|step|                (write z)
               + T·|step|                (read z by BN)
               + T·|step|                (write z' by BN)
               + T·|step|                (read z' by LIF)
               + T·|step|                (write spike by LIF)
               + bn_params
             = 7·T·|step| + bn_params

    fusion    = 2·T·|step|               (read a, read x)
               + T·|step|                (write spike)
               + |step|                  (v_carry wasted write)
               + bn_params
             = 3·T·|step| + |step| + bn_params

For default shape (T=16, B=32, C=128, H=W=16):
    |step|     = 32·128·16·16·4 = 4 MB
    bn_params  = 1024 bytes (negligible)
    no_fusion  = 7·16·4 = 448 MB
    fusion     = 3·16·4 + 4 = 196 MB
    ratio      = 196/448 = 0.4375
    savings    = 56.25%

Compare with Conv→BN→LIF: same chained structure, but here the TSI is
elementwise (memory-bound). Wall-clock should reflect this — fusion should
realize the full HBM savings as a wall-clock speedup, unlike Conv→BN→LIF
where Triton's tl.dot codegen overhead masked the savings.

Parity policy:
  - fusion vs no_fusion : require bit-exact (no tl.dot, no FP order issues)
  - fusion vs reference : also expected bit-exact for the same reason
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
# Fusion kernel: Add + BN + LIF, all in registers
# ============================================================

@triton.jit
def _add_bn_lif_fused_kernel(
    a_ptr, x_ptr, scale_ptr, bias_ptr, spike_ptr, v_carry_ptr,
    T,
    N_total,                # B*C*H*W (per-step element count)
    HW,                     # H*W (for channel index decode)
    C,                      # C (for channel index decode)
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    """
    Grid: (cdiv(N_total, BLOCK),)
    Each program owns BLOCK contiguous spatial elements within one time step,
    and loops over t internally to keep v in registers.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total

    # Channel index for each element: c = (offs // HW) % C
    # offs is a flat (b, c, h, w) index within one time step
    c_idx = (offs // HW) % C

    # Load BN params for this block of elements (per-element gather is fine
    # because C is small relative to N_total and L1 cache absorbs reuse)
    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau

    for t in range(T):
        base = t * stride_t + offs

        a = tl.load(a_ptr + base, mask=mask, other=0.0)
        x = tl.load(x_ptr + base, mask=mask, other=0.0)

        # Add
        z = a + x

        # BN
        z_bn = z * scale + bias

        # LIF
        v = v + (z_bn - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s

        tl.store(spike_ptr + base, s, mask=mask)

    # v_carry wasted write (K=T case, single chunk boundary at end)
    tl.store(v_carry_ptr + offs, v, mask=mask)


# ============================================================
# Non-fusion kernels: Add, BN, LIF as 3 separate kernels
# ============================================================

@triton.jit
def _add_only_kernel(
    a_ptr, x_ptr, z_ptr,
    N_total_full,             # T*B*C*H*W
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total_full
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    z = a + x
    tl.store(z_ptr + offs, z, mask=mask)


@triton.jit
def _bn_only_kernel(
    z_ptr, scale_ptr, bias_ptr, z_out_ptr,
    N_total_full,             # T*B*C*H*W
    HW, C,
    BLOCK: tl.constexpr,
):
    """
    z layout: [T, B, C, H, W] flat. Channel index = (offs // HW) % C.
    Note: this works because (offs % (C*HW)) // HW = c_idx within each (t,b),
    and modding by C gives the same result.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total_full

    c_idx = (offs // HW) % C
    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    z = tl.load(z_ptr + offs, mask=mask, other=0.0)
    z_out = z * scale + bias
    tl.store(z_out_ptr + offs, z_out, mask=mask)


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


# ============================================================
# Wrappers
# ============================================================

def run_fusion(a, x, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0,
               BLOCK=BLOCK_DEFAULT):
    assert a.is_contiguous() and x.is_contiguous()
    assert scale.is_contiguous() and bias.is_contiguous()
    T, B, C, H, W = a.shape
    assert x.shape == a.shape

    spike = torch.empty_like(a)
    v_carry = torch.empty(B * C * H * W, device=a.device, dtype=a.dtype)

    N_total = B * C * H * W   # per-step element count
    HW = H * W
    stride_t = N_total

    grid = (triton.cdiv(N_total, BLOCK),)
    _add_bn_lif_fused_kernel[grid](
        a, x, scale, bias, spike, v_carry,
        T,
        N_total,
        HW,
        C,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


def run_no_fusion(a, x, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0,
                  BLOCK=BLOCK_DEFAULT):
    assert a.is_contiguous() and x.is_contiguous()
    T, B, C, H, W = a.shape

    z = torch.empty_like(a)
    z_bn = torch.empty_like(a)
    spike = torch.empty_like(a)

    N_total_full = T * B * C * H * W
    HW = H * W
    stride_t = B * C * H * W

    # Pass 1: Add
    grid_add = (triton.cdiv(N_total_full, BLOCK),)
    _add_only_kernel[grid_add](
        a, x, z,
        N_total_full,
        BLOCK=BLOCK,
    )

    # Pass 2: BN
    grid_bn = (triton.cdiv(N_total_full, BLOCK),)
    _bn_only_kernel[grid_bn](
        z, scale, bias, z_bn,
        N_total_full,
        HW, C,
        BLOCK=BLOCK,
    )

    # Pass 3: LIF
    N_spatial = B * C * H * W
    grid_lif = (triton.cdiv(N_spatial, BLOCK),)
    _lif_only_kernel[grid_lif](
        z_bn, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


def run_reference(a, x, scale, bias, tau=2.0, v_th=1.0, v_reset=0.0):
    T, B, C, H, W = a.shape
    z = a + x
    z_bn = z * scale.view(1, 1, C, 1, 1) + bias.view(1, 1, C, 1, 1)
    v = torch.zeros(B, C, H, W, device=a.device, dtype=a.dtype)
    spikes = torch.empty_like(z_bn)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z_bn[t] - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


def analytic_hbm_bytes(T, B, C, H, W, dtype_bytes=4):
    step = B * C * H * W * dtype_bytes
    bn_params = 2 * C * dtype_bytes
    no_fusion = 7 * T * step + bn_params
    fusion = 3 * T * step + step + bn_params
    return {
        'step_MB': step / 1024 / 1024,
        'bn_params_bytes': bn_params,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'ratio_fusion_over_no_fusion': fusion / no_fusion,
        'savings_pct': (1 - fusion / no_fusion) * 100,
    }


def parity_check(a, x, scale, bias, tau, v_th, v_reset):
    out_ref = run_reference(a, x, scale, bias, tau, v_th, v_reset)
    out_fus = run_fusion(a, x, scale, bias, tau, v_th, v_reset)
    out_nof = run_no_fusion(a, x, scale, bias, tau, v_th, v_reset)

    def diff(p, q):
        d = (p - q).abs()
        return {
            'max_diff': d.max().item(),
            'mean_abs_diff': d.mean().item(),
            'n_diff': (p != q).sum().item(),
            'n_total': p.numel(),
            'bit_exact': bool((p == q).all().item()),
        }

    def spike_match_pct(p, q):
        return ((p == q).float().mean().item()) * 100

    return {
        'spike_rate_reference': out_ref.mean().item(),
        'spike_rate_fusion': out_fus.mean().item(),
        'spike_rate_no_fusion': out_nof.mean().item(),
        'fusion_vs_reference': {
            **diff(out_fus, out_ref),
            'spike_match_pct': spike_match_pct(out_fus, out_ref),
        },
        'no_fusion_vs_reference': {
            **diff(out_nof, out_ref),
            'spike_match_pct': spike_match_pct(out_nof, out_ref),
        },
        'fusion_vs_no_fusion': {
            **diff(out_fus, out_nof),
            'spike_match_pct': spike_match_pct(out_fus, out_nof),
        },
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
    p.add_argument('--C', type=int, default=128)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--W', type=int, default=16)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--block', type=int, default=BLOCK_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=11)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/add_bn_lif_min.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C, H, W = args.T, args.B, args.C, args.H, args.W

    print(f"{'='*86}")
    print(f"Add → BN → LIF: fusion vs non-fusion")
    print(f"{'='*86}")
    print(f"shape  : T={T} B={B} C={C} H={H} W={W}")
    print(f"BN     : per-channel affine (scale, bias)")
    print(f"LIF    : tau={args.tau} v_th={args.v_th} v_reset={args.v_reset}")
    print(f"BLOCK  : {args.block}")
    print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print()

    a = (torch.randn(T, B, C, H, W, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    x = (torch.randn(T, B, C, H, W, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    scale = (1.0 + torch.randn(C, device=device, dtype=torch.float32) * 0.1).contiguous()
    bias = (torch.randn(C, device=device, dtype=torch.float32) * 0.1).contiguous()

    print("[1/4] Analytic HBM bytes")
    hbm = analytic_hbm_bytes(T, B, C, H, W, dtype_bytes=4)
    print(f"  step         : {hbm['step_MB']:.2f} MB")
    print(f"  bn_params    : {hbm['bn_params_bytes']} bytes")
    print(f"  no-fusion    : {hbm['no_fusion_hbm_MB']:.2f} MB")
    print(f"  fusion (K=T) : {hbm['fusion_hbm_MB']:.2f} MB")
    print(f"  ratio        : {hbm['ratio_fusion_over_no_fusion']:.4f}")
    print(f"  savings      : {hbm['savings_pct']:.2f}%")
    print()

    print("[2/4] Smoke test")
    try:
        out_nof = run_no_fusion(a, x, scale, bias,
                                args.tau, args.v_th, args.v_reset,
                                BLOCK=args.block)
        torch.cuda.synchronize()
        print(f"  non-fusion OK, output shape: {list(out_nof.shape)}")
    except Exception as e:
        print(f"  non-fusion FAILED: {type(e).__name__}: {e}")
        raise

    try:
        out_fus = run_fusion(a, x, scale, bias,
                             args.tau, args.v_th, args.v_reset,
                             BLOCK=args.block)
        torch.cuda.synchronize()
        print(f"  fusion     OK, output shape: {list(out_fus.shape)}")
    except Exception as e:
        print(f"  fusion FAILED: {type(e).__name__}: {e}")
        raise
    print()

    print("[3/4] Parity check")
    parity = parity_check(a, x, scale, bias,
                          args.tau, args.v_th, args.v_reset)
    print(f"  spike_rate (reference) : {parity['spike_rate_reference']:.4f}")
    print(f"  spike_rate (fusion)    : {parity['spike_rate_fusion']:.4f}")
    print(f"  spike_rate (no_fusion) : {parity['spike_rate_no_fusion']:.4f}")
    for name, key in [
        ('fusion    vs reference ', 'fusion_vs_reference'),
        ('no_fusion vs reference ', 'no_fusion_vs_reference'),
        ('fusion    vs no_fusion ', 'fusion_vs_no_fusion'),
    ]:
        d = parity[key]
        tag = ' [BIT-EXACT]' if d['bit_exact'] else ''
        print(f"  {name}: max_diff={d['max_diff']:.2e}, "
              f"spike_match={d['spike_match_pct']:.4f}%{tag}")
    print()

    print("[4/4] Wall-clock")

    def bench_fus():
        _ = run_fusion(a, x, scale, bias,
                       args.tau, args.v_th, args.v_reset,
                       BLOCK=args.block)

    def bench_nof():
        _ = run_no_fusion(a, x, scale, bias,
                          args.tau, args.v_th, args.v_reset,
                          BLOCK=args.block)

    wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
    wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)
    mem_fus = peak_memory_one_run(bench_fus)
    mem_nof = peak_memory_one_run(bench_nof)

    speedup = wall_nof['median_ms'] / wall_fus['median_ms']

    print(f"  fusion    : {wall_fus['median_ms']:.4f} ms "
          f"± {wall_fus['stdev_ms']:.4f}  peak {mem_fus['peak_allocated_MB']:.1f} MB")
    print(f"  no-fusion : {wall_nof['median_ms']:.4f} ms "
          f"± {wall_nof['stdev_ms']:.4f}  peak {mem_nof['peak_allocated_MB']:.1f} MB")
    print(f"  speedup   : {speedup:.4f}×  "
          f"({(speedup-1)*100:+.2f}% {'faster' if speedup>1 else 'slower'})")
    print()

    result = {
        'experiment': 'add_bn_lif_min',
        'section_ref': '§3.9 chained TSI memory-bound: Add → BN → LIF, K=T',
        'config': {
            'shape': {'T': T, 'B': B, 'C': C, 'H': H, 'W': W},
            'bn': {'mode': 'inference', 'param_init_scale_std': 0.1,
                   'param_init_bias_std': 0.1},
            'lif': {'tau': args.tau, 'v_th': args.v_th, 'v_reset': args.v_reset},
            'block': args.block,
            'input_scale': args.input_scale,
            'seed': args.seed,
            'dtype': 'float32',
            'device': torch.cuda.get_device_name(device),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'triton_version': triton.__version__,
        },
        'analytic_hbm': hbm,
        'parity': parity,
        'wall_clock': {
            'fusion': wall_fus,
            'no_fusion': wall_nof,
            'speedup_no_fusion_over_fusion': speedup,
        },
        'peak_memory': {
            'fusion': mem_fus,
            'no_fusion': mem_nof,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == '__main__':
    main()