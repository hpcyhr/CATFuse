"""
Add → LIF fusion vs non-fusion controlled comparison.

Target section: §3.8 Form 2 (StreamFuse for a simple TSI→CSR chain).

Variable being tested:
    does the intermediate tensor z = x1 + x2 go through HBM?
  - fusion version:     z lives in registers, never touches HBM
                         (single Triton kernel, TimeBlock ∘ StreamFuse(Add, LIF)
                          ∘ StateCarry(LIF) with K=T, one block covering all steps)
  - non-fusion version: z is materialized to HBM between two kernels
                         (kernel 1: add, writes z to HBM;
                          kernel 2: LIF, reads z from HBM)

Everything else is held fixed across the two versions:
  - same Triton runtime
  - same LIF hyperparameters (tau, v_th, v_reset)
  - same input data (seed=0)
  - same shape (T, B, C, H, W)
  - same BLOCK size for LIF pass
  - same dtype (fp32)

The only difference is whether z passes through HBM. The wall-clock /
peak-memory / analytic HBM bytes delta between the two is therefore
attributable to fusion alone (modulo small noise from kernel launch overhead
and cache state).

Output: a single JSON file with analytic formulas, parity check, wall-clock
stats, and peak memory stats for both versions.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


# ============================================================
# Fusion kernel: single pass, z stays in registers
# ============================================================

@triton.jit
def _add_lif_fused_kernel(
    x1_ptr, x2_ptr, spike_ptr,
    T, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    """
    One program handles a contiguous slab of BLOCK spatial elements,
    streaming T time steps of Add → LIF in a register-resident loop.
    z = x1[t] + x2[t] is computed and consumed within registers.
    Only x1, x2 (reads) and spike (writes) touch HBM per step.
    Membrane potential v lives in registers across the T loop.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial

    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau

    for t in range(T):
        base = t * stride_t + offs
        x1 = tl.load(x1_ptr + base, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr + base, mask=mask, other=0.0)
        z = x1 + x2                              # register-resident
        v = v + (z - v) * inv_tau                # LIF integration
        s = (v >= v_th).to(tl.float32)           # spike
        v = v * (1.0 - s) + v_reset * s          # hard reset
        tl.store(spike_ptr + base, s, mask=mask)


# ============================================================
# Non-fusion kernels: two passes, z materialized to HBM
# ============================================================

@triton.jit
def _add_only_kernel(
    x1_ptr, x2_ptr, z_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    """Elementwise add, flat layout over the whole [T,B,C,H,W] tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0)
    tl.store(z_ptr + offs, x1 + x2, mask=mask)   # z ← HBM


@triton.jit
def _lif_only_kernel(
    z_ptr, spike_ptr,
    T, N_spatial,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    """LIF over T steps, reading z from HBM each step."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial

    v = tl.zeros((BLOCK,), dtype=tl.float32)
    inv_tau = 1.0 / tau

    for t in range(T):
        base = t * stride_t + offs
        z = tl.load(z_ptr + base, mask=mask, other=0.0)   # z ← HBM
        v = v + (z - v) * inv_tau
        s = (v >= v_th).to(tl.float32)
        v = v * (1.0 - s) + v_reset * s
        tl.store(spike_ptr + base, s, mask=mask)


# ============================================================
# Python wrappers
# ============================================================

BLOCK_DEFAULT = 1024


def run_fusion(x1, x2, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    assert x1.is_contiguous() and x2.is_contiguous()
    assert x1.shape == x2.shape
    T = x1.shape[0]
    N_total = x1.numel()
    N_spatial = N_total // T
    stride_t = N_spatial
    spike = torch.empty_like(x1)
    grid = (triton.cdiv(N_spatial, BLOCK),)
    _add_lif_fused_kernel[grid](
        x1, x2, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


def run_no_fusion(x1, x2, tau=2.0, v_th=1.0, v_reset=0.0, BLOCK=BLOCK_DEFAULT):
    assert x1.is_contiguous() and x2.is_contiguous()
    assert x1.shape == x2.shape
    T = x1.shape[0]
    N_total = x1.numel()
    N_spatial = N_total // T
    stride_t = N_spatial
    z = torch.empty_like(x1)
    spike = torch.empty_like(x1)
    # pass 1: add
    grid_add = (triton.cdiv(N_total, BLOCK),)
    _add_only_kernel[grid_add](
        x1, x2, z,
        N_total,
        BLOCK=BLOCK,
    )
    # pass 2: LIF
    grid_lif = (triton.cdiv(N_spatial, BLOCK),)
    _lif_only_kernel[grid_lif](
        z, spike,
        T, N_spatial,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


def run_reference(x1, x2, tau=2.0, v_th=1.0, v_reset=0.0):
    """PyTorch reference, ground truth for parity."""
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
# Analytic HBM bytes formula
# ============================================================

def analytic_hbm_bytes(T, B, C, H, W, dtype_bytes=4):
    """
    Analytic HBM traffic for one forward pass, in bytes.

    fusion (single kernel, z in registers):
        - read  x1[T,B,C,H,W] : T·step
        - read  x2[T,B,C,H,W] : T·step
        - write spike[T,B,C,H,W]: T·step
        total = 3·T·step

    non-fusion (two kernels, z materialized):
        kernel 1 (add):
            - read  x1 : T·step
            - read  x2 : T·step
            - write z  : T·step
        kernel 2 (LIF):
            - read  z     : T·step
            - write spike : T·step
        total = 5·T·step

    Note: v stays in registers in both versions (K=T, no TimeBlock split),
    so it does not contribute to HBM traffic.
    """
    step = B * C * H * W * dtype_bytes
    fusion = 3 * T * step
    no_fusion = 5 * T * step
    ratio = fusion / no_fusion
    savings_abs = no_fusion - fusion        # bytes saved
    savings_pct = (1 - ratio) * 100         # % saved
    return {
        'T': T, 'B': B, 'C': C, 'H': H, 'W': W,
        'dtype_bytes': dtype_bytes,
        'step_bytes': step,
        'fusion_hbm_bytes': fusion,
        'no_fusion_hbm_bytes': no_fusion,
        'fusion_hbm_MB': fusion / 1024 / 1024,
        'no_fusion_hbm_MB': no_fusion / 1024 / 1024,
        'ratio_fusion_over_no_fusion': ratio,
        'savings_bytes': savings_abs,
        'savings_MB': savings_abs / 1024 / 1024,
        'savings_pct': savings_pct,
    }


# ============================================================
# Parity check
# ============================================================

def parity_check(x1, x2, tau, v_th, v_reset):
    """
    Three-way parity:
      - fusion    vs PyTorch reference
      - no_fusion vs PyTorch reference
      - fusion    vs no_fusion (should be bit-exact, same math order)
    """
    out_ref = run_reference(x1, x2, tau, v_th, v_reset)
    out_fus = run_fusion(x1, x2, tau, v_th, v_reset)
    out_nof = run_no_fusion(x1, x2, tau, v_th, v_reset)

    def diff(a, b):
        d = (a - b).abs()
        return {
            'max_diff': d.max().item(),
            'mean_abs_diff': d.mean().item(),
            'n_diff': (a != b).sum().item(),
            'n_total': a.numel(),
            'bit_exact': bool((a == b).all().item()),
        }

    spike_rate = out_ref.mean().item()
    return {
        'spike_rate_reference': spike_rate,
        'fusion_vs_reference': diff(out_fus, out_ref),
        'no_fusion_vs_reference': diff(out_nof, out_ref),
        'fusion_vs_no_fusion': diff(out_fus, out_nof),
    }


# ============================================================
# Wall-clock benchmark
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
        'n_warmup': n_warmup,
        'all_samples_ms': samples,
    }


# ============================================================
# Peak memory measurement
# ============================================================

def peak_memory_one_run(fn):
    """Measure peak GPU memory allocated during a single forward call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserv = torch.cuda.max_memory_reserved()
    del out
    return {
        'peak_allocated_bytes': peak_alloc,
        'peak_allocated_MB': peak_alloc / 1024 / 1024,
        'peak_reserved_bytes': peak_reserv,
        'peak_reserved_MB': peak_reserv / 1024 / 1024,
    }


# ============================================================
# Main: run everything, dump JSON
# ============================================================

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
    p.add_argument('--block', type=int, default=BLOCK_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=11)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str,
                   default='./results/add_lif_compare.json')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C, H, W = args.T, args.B, args.C, args.H, args.W
    tau, v_th, v_reset = args.tau, args.v_th, args.v_reset

    print(f"{'='*70}")
    print(f"Add → LIF: fusion vs non-fusion")
    print(f"{'='*70}")
    print(f"shape: T={T} B={B} C={C} H={H} W={W}")
    print(f"LIF:   tau={tau} v_th={v_th} v_reset={v_reset}")
    print(f"GPU:   {torch.cuda.get_device_name(device)}")
    print()

    # --- inputs ---
    x1 = (torch.randn(T, B, C, H, W, device=device,
                      dtype=torch.float32) * 0.5).contiguous()
    x2 = (torch.randn(T, B, C, H, W, device=device,
                      dtype=torch.float32) * 0.5).contiguous()

    # --- analytic HBM bytes ---
    print("[1/4] Analytic HBM bytes")
    hbm = analytic_hbm_bytes(T, B, C, H, W, dtype_bytes=4)
    print(f"  fusion    : {hbm['fusion_hbm_MB']:.2f} MB")
    print(f"  no-fusion : {hbm['no_fusion_hbm_MB']:.2f} MB")
    print(f"  ratio     : {hbm['ratio_fusion_over_no_fusion']:.4f}")
    print(f"  savings   : {hbm['savings_MB']:.2f} MB ({hbm['savings_pct']:.2f}%)")
    print()

    # --- parity ---
    print("[2/4] Parity check")
    parity = parity_check(x1, x2, tau, v_th, v_reset)
    print(f"  spike rate (reference): {parity['spike_rate_reference']:.4f}")
    for name, key in [
        ('fusion    vs reference', 'fusion_vs_reference'),
        ('no-fusion vs reference', 'no_fusion_vs_reference'),
        ('fusion    vs no-fusion', 'fusion_vs_no_fusion'),
    ]:
        d = parity[key]
        tag = ' [BIT-EXACT]' if d['bit_exact'] else ''
        print(f"  {name}: max_diff={d['max_diff']:.2e}, "
              f"n_diff={d['n_diff']}/{d['n_total']}{tag}")
    print()

    # --- wall-clock ---
    print("[3/4] Wall-clock")

    def bench_fus():
        _ = run_fusion(x1, x2, tau, v_th, v_reset, BLOCK=args.block)

    def bench_nof():
        _ = run_no_fusion(x1, x2, tau, v_th, v_reset, BLOCK=args.block)

    wall_fus = cuda_time_stats(bench_fus, args.n_iter, args.n_repeat, args.n_warmup)
    wall_nof = cuda_time_stats(bench_nof, args.n_iter, args.n_repeat, args.n_warmup)

    print(f"  fusion    : {wall_fus['median_ms']:.4f} ms "
          f"± {wall_fus['stdev_ms']:.4f}")
    print(f"  no-fusion : {wall_nof['median_ms']:.4f} ms "
          f"± {wall_nof['stdev_ms']:.4f}")
    speedup_wall = wall_nof['median_ms'] / wall_fus['median_ms']
    print(f"  speedup   : {speedup_wall:.4f}x  "
          f"(fusion is {(speedup_wall-1)*100:.2f}% faster)")
    print()

    # --- peak memory ---
    print("[4/4] Peak memory (single forward)")
    mem_fus = peak_memory_one_run(bench_fus)
    mem_nof = peak_memory_one_run(bench_nof)
    print(f"  fusion    : peak_allocated = {mem_fus['peak_allocated_MB']:.2f} MB")
    print(f"  no-fusion : peak_allocated = {mem_nof['peak_allocated_MB']:.2f} MB")
    mem_ratio = (mem_fus['peak_allocated_bytes']
                 / mem_nof['peak_allocated_bytes']) if mem_nof['peak_allocated_bytes'] > 0 else float('nan')
    print(f"  ratio     : {mem_ratio:.4f}")
    print(f"  savings   : {(mem_nof['peak_allocated_MB']-mem_fus['peak_allocated_MB']):.2f} MB "
          f"({(1-mem_ratio)*100:.2f}%)")
    print()

    # --- assemble result ---
    result = {
        'experiment': 'add_lif_fusion_vs_no_fusion',
        'section_ref': '§3.8 Form 2 (StreamFuse for Add→LIF)',
        'config': {
            'shape': {'T': T, 'B': B, 'C': C, 'H': H, 'W': W},
            'lif': {'tau': tau, 'v_th': v_th, 'v_reset': v_reset},
            'triton_block': args.block,
            'seed': args.seed,
            'dtype': 'float32',
            'device': torch.cuda.get_device_name(device),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'triton_version': triton.__version__,
        },
        'analytic_hbm_bytes': hbm,
        'parity': parity,
        'wall_clock': {
            'fusion': wall_fus,
            'no_fusion': wall_nof,
            'speedup_no_fusion_over_fusion': speedup_wall,
            'fusion_faster_pct': (speedup_wall - 1) * 100,
        },
        'peak_memory': {
            'fusion': mem_fus,
            'no_fusion': mem_nof,
            'ratio_fusion_over_no_fusion': mem_ratio,
            'savings_MB': mem_nof['peak_allocated_MB'] - mem_fus['peak_allocated_MB'],
            'savings_pct': (1 - mem_ratio) * 100,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # --- write JSON ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")
    print()

    # --- quick summary ---
    print(f"{'='*70}")
    print(f"Summary (shape T={T} B={B} C={C} H={H} W={W}):")
    print(f"{'='*70}")
    print(f"  analytic HBM savings:  {hbm['savings_pct']:.2f}%  "
          f"(predict: fusion is 40% smaller, ratio 0.60)")
    print(f"  wall-clock speedup:    {speedup_wall:.4f}x  "
          f"({(speedup_wall-1)*100:.2f}% faster)")
    print(f"  peak memory savings:   {(1-mem_ratio)*100:.2f}%")
    print(f"  parity (fus vs ref):   "
          f"{'BIT-EXACT' if parity['fusion_vs_reference']['bit_exact'] else 'NOT bit-exact'}")
    print(f"  parity (nof vs ref):   "
          f"{'BIT-EXACT' if parity['no_fusion_vs_reference']['bit_exact'] else 'NOT bit-exact'}")


if __name__ == '__main__':
    main()