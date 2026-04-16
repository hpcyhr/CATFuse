"""
Phase 2 Task 2.2b — Add→BN→LIF ablation, 5 configs, K-sweep.

Configs:
  A  : per-step torch add + per-step torch BN-affine + per-step torch LIF
  B  : BatchFold(Add) + BatchFold(BN) per block, then per-step torch LIF
       (K ∈ {1, 2, 4, 8, 16})
  B' : Triton add kernel + Triton BN kernel + Triton LIF kernel
       (all T steps batched in each kernel; z and z_bn through HBM;
        v in register across T within LIF kernel)
       This is Phase 0 `add_bn_lif_min.py::run_no_fusion`.
  C  : K=T single block, fused Triton kernel (add + BN + LIF + state carry
       all in one kernel, z/z_bn/v all in register).
       This is Phase 0 `add_bn_lif_min.py::run_fusion`.
  D  : K ∈ {2, 4, 8} multi-block fused Triton kernel
       (same as C but v carried through HBM at block boundaries)
       NEW KERNEL here — Phase 0 doesn't have a K-parameterized Add+BN+LIF.

Parity reference: run_reference (per-step torch Add + BN + LIF)

Data points (per subgraph):
  A  (K=T canonical)          : 1
  B  (K=1, 2, 4, 8, 16)       : 5
  B' (K=T canonical)          : 1
  C  (K=T)                    : 1
  D  (K=2, 4, 8)              : 3
  Total                       : 11

Shape: T=16, B=32, C=128, H=W=16 (matches Phase 0 add_bn_lif_min.py default)
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

# Phase 0 Add→BN→LIF kernels (K=T only)
from add_bn_lif_min import (
    run_fusion as phase0_abl_fusion_KT,
    run_no_fusion as phase0_abl_no_fusion,
)


# ============================================================
# NEW kernel: K-parameterized Add + BN + LIF + StateCarry
# ============================================================
# Based on add_lif_k_sweep.py's _add_lif_fused_k_kernel structure, but with
# BN affine fused in (scale + bias per channel via c_idx decode).
# ============================================================

@triton.jit
def _add_bn_lif_fused_k_kernel(
    a_ptr, x_ptr, scale_ptr, bias_ptr, spike_ptr, v_carry_ptr,
    T, K, N_spatial, HW, C,
    tau, v_th, v_reset,
    stride_t,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_spatial

    # Channel index for this block of elements
    c_idx = (offs // HW) % C
    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

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

            a = tl.load(a_ptr + base, mask=mask, other=0.0)
            x = tl.load(x_ptr + base, mask=mask, other=0.0)

            # Add
            z = a + x
            # BN affine
            z_bn = z * scale + bias
            # LIF
            v = v + (z_bn - v) * inv_tau
            s = (v >= v_th).to(tl.float32)
            v = v * (1.0 - s) + v_reset * s
            tl.store(spike_ptr + base, s, mask=mask)

        tl.store(v_carry_ptr + offs, v, mask=mask)


def _run_add_bn_lif_fused_k(a, x, scale, bias, K,
                             tau=2.0, v_th=1.0, v_reset=0.0,
                             BLOCK=1024):
    assert a.is_contiguous() and x.is_contiguous()
    T, B, C, H, W = a.shape
    assert T % K == 0
    N_spatial = B * C * H * W
    stride_t = N_spatial
    HW = H * W

    spike = torch.empty_like(a)
    v_carry = torch.empty(N_spatial, device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(N_spatial, BLOCK),)
    _add_bn_lif_fused_k_kernel[grid](
        a, x, scale, bias, spike, v_carry,
        T, K, N_spatial, HW, C,
        tau, v_th, v_reset,
        stride_t,
        BLOCK=BLOCK,
    )
    return spike


# ============================================================
# Config A: per-step torch ops (no fusion)
# ============================================================

def run_config_A(a, x, scale, bias, tau, v_th, v_reset):
    T, B, C, H, W = a.shape
    scale_r = scale.view(1, C, 1, 1)
    bias_r = bias.view(1, C, 1, 1)
    v = torch.zeros(B, C, H, W, device=a.device, dtype=a.dtype)
    spikes = torch.empty_like(a)
    inv_tau = 1.0 / tau
    for t in range(T):
        z_t = a[t] + x[t]
        z_bn_t = z_t * scale_r + bias_r
        v = v + (z_bn_t - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s
    return spikes


# ============================================================
# Config B: BatchFold(Add, BN) per block + per-step torch LIF
# ============================================================

def run_config_B(a, x, scale, bias, K, tau, v_th, v_reset):
    T, B, C, H, W = a.shape
    scale_r = scale.view(1, 1, C, 1, 1)
    bias_r = bias.view(1, 1, C, 1, 1)
    v = torch.zeros(B, C, H, W, device=a.device, dtype=a.dtype)
    spikes = torch.empty_like(a)
    inv_tau = 1.0 / tau
    for block_start in range(0, T, K):
        block_end = block_start + K
        # BatchFold: one Add and one BN for all K time steps in block
        z_block = a[block_start:block_end] + x[block_start:block_end]
        z_bn_block = z_block * scale_r + bias_r
        # Per-step LIF within block
        for k in range(K):
            v = v + (z_bn_block[k] - v) * inv_tau
            s = (v >= v_th).float()
            v = v * (1.0 - s) + v_reset * s
            spikes[block_start + k] = s
    return spikes


# ============================================================
# Config B': Phase 0 run_no_fusion (3 separate Triton kernels)
# ============================================================

def run_config_B_prime(a, x, scale, bias, tau, v_th, v_reset):
    return phase0_abl_no_fusion(a, x, scale, bias, tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Config C: Phase 0 run_fusion (K=T single fused kernel)
# ============================================================

def run_config_C(a, x, scale, bias, tau, v_th, v_reset):
    return phase0_abl_fusion_KT(a, x, scale, bias, tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Config D: new K-parameterized fused kernel (K < T)
# ============================================================

def run_config_D(a, x, scale, bias, K, tau, v_th, v_reset):
    return _run_add_bn_lif_fused_k(a, x, scale, bias, K,
                                    tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Reference
# ============================================================

def run_reference(a, x, scale, bias, tau, v_th, v_reset):
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


def cuda_time_trimmed(fn, n_iter=100, n_repeat=12, n_warmup=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    sorted_s = sorted(samples)
    trimmed = sorted_s[1:-1]
    return {
        'trimmed_mean_ms': statistics.mean(trimmed),
        'trimmed_stdev_ms': statistics.stdev(trimmed),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'all_samples_ms': samples,
    }


# ============================================================
# Analytic HBM model for Add→BN→LIF
# ============================================================
# Reference (per step): 2 reads (a, x) + 1 write z + 1 read z + 1 write z_bn
#                       + 1 read z_bn + 2 LIF (v r/w) + 1 write s
#                     = 9 |step| per step? Too many. Let me recount carefully.
#
# Actually the Phase 0 analytic for Add→BN→LIF reference uses 7*T*step:
#   - 2 reads (a + x) = 2
#   - 1 write z = 1
#   - 1 read z (BN input) = 1
#   - 1 write z_bn = 1
#   - 1 read z_bn (LIF input) = 1
#   - 1 write spike = 1
#   Total: 7 |step| per step, ignoring v HBM in the ref (ref has v in memory too,
#   but Phase 0 analytic ignores it for simplicity)
#
# This counts 7 units. For our 5-config ablation we extend this:
#   - A : 7 T step (same as ref)
#   - B : 7 T step (same, BatchFold doesn't reduce HBM)
#   - B': 5 T step (LIF v in register across T, eliminates 2 v r/w per step)
#   - C : T step + |step| for s + bn_params ≈ 3T step + 1 + bn (fusion path of Phase 0)
#        Phase 0 says 3T + 1 for fusion at K=T, so savings = 4/7 = 57.1%
#   - D : 3T step + (2*(T/K)-1)*step for v_carry boundaries

def analytic_hbm_bytes(config: str, K: int, T: int, B: int, C: int, H: int, W: int,
                        dtype_bytes: int = 4) -> dict:
    step = B * C * H * W * dtype_bytes
    bn_params = 2 * C * dtype_bytes

    ref = 7 * T * step + bn_params

    if config == 'A':
        bytes_ = ref
    elif config == 'B':
        bytes_ = ref  # BatchFold doesn't change HBM
    elif config == "B'":
        # LIF kernel keeps v in register across T
        # Savings vs ref: 2*T*step (v r/w eliminated)
        bytes_ = 5 * T * step + bn_params
    elif config == 'C':
        # K=T single fused kernel: all of z, z_bn, v in register
        # Remaining HBM: a read + x read + s write + 1 v_carry final write
        bytes_ = 3 * T * step + step + bn_params
    elif config == 'D':
        # C + (2*(T/K)-1)*step for v_carry at block boundaries
        n_blocks = T // K
        v_traffic = (2 * n_blocks - 1) * step
        bytes_ = 3 * T * step + v_traffic + bn_params
    else:
        raise ValueError(f"Unknown config: {config}")

    return {
        'bytes': bytes_,
        'MB': bytes_ / (1024 * 1024),
        'ratio_vs_ref': bytes_ / ref,
        'savings_pct': (1 - bytes_ / ref) * 100,
    }


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--B', type=int, default=32)
    p.add_argument('--C', type=int, default=128)
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--W', type=int, default=16)
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--v_th', type=float, default=1.0)
    p.add_argument('--input_scale', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_iter', type=int, default=100)
    p.add_argument('--n_repeat', type=int, default=12)
    p.add_argument('--n_warmup', type=int, default=20)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f'./results/phase2/add_bn_lif_ablation_v100_seed{args.seed}.json'

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, C, H, W = args.T, args.B, args.C, args.H, args.W

    print(f"Phase 2 task 2.2b — Add->BN->LIF 5-config ablation, K-sweep")
    print(f"  GPU   : {torch.cuda.get_device_name(device)}")
    print(f"  shape : T={T} B={B} C={C} H=W={H}")
    print(f"  LIF   : tau={args.tau} v_th={args.v_th}")
    print(f"  seed  : {args.seed}")

    a = (torch.randn(T, B, C, H, W, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    x = (torch.randn(T, B, C, H, W, device=device,
                     dtype=torch.float32) * args.input_scale).contiguous()
    scale = (1.0 + torch.randn(C, device=device,
                                dtype=torch.float32) * 0.1).contiguous()
    bias = (torch.randn(C, device=device,
                         dtype=torch.float32) * 0.1).contiguous()

    # Reference
    ref_spikes = run_reference(a, x, scale, bias, args.tau, args.v_th, 0.0)
    ref_rate = ref_spikes.mean().item()
    print(f"  reference spike rate: {ref_rate:.6f}")

    # ----- Parity -----
    print(f"\n{'='*92}")
    print(f"Parity check (all configs vs reference)")
    print(f"{'='*92}")

    def check_parity(tag, cand):
        flips = (cand != ref_spikes).sum().item()
        total = ref_spikes.numel()
        match = (1.0 - flips / total) * 100
        status = 'PASS' if match > 99.9 else 'FAIL'
        print(f"  [{tag:<20}] match={match:.6f}%  flips={flips}/{total}  {status}")
        return {'spike_match_pct': match, 'spike_flips': flips}

    parity_results = {}

    with torch.no_grad():
        s_A = run_config_A(a, x, scale, bias, args.tau, args.v_th, 0.0)
    parity_results['A'] = check_parity('A', s_A)

    K_list = [1, 2, 4, 8, 16]
    for K in K_list:
        with torch.no_grad():
            s_B_K = run_config_B(a, x, scale, bias, K, args.tau, args.v_th, 0.0)
        parity_results[f'B_K{K}'] = check_parity(f"B (K={K})", s_B_K)

    with torch.no_grad():
        s_Bp = run_config_B_prime(a, x, scale, bias, args.tau, args.v_th, 0.0)
    parity_results["B'"] = check_parity("B'", s_Bp)

    with torch.no_grad():
        s_C = run_config_C(a, x, scale, bias, args.tau, args.v_th, 0.0)
    parity_results['C'] = check_parity('C (K=T)', s_C)

    for K in [2, 4, 8]:
        with torch.no_grad():
            s_D_K = run_config_D(a, x, scale, bias, K, args.tau, args.v_th, 0.0)
        parity_results[f'D_K{K}'] = check_parity(f"D (K={K})", s_D_K)

    # ----- Wall-clock -----
    print(f"\n{'='*92}")
    print(f"Wall-clock (trimmed mean N={args.n_repeat-2} of {args.n_repeat}, "
          f"{args.n_iter} inner iters, {args.n_warmup} warmup)")
    print(f"{'='*92}")

    wall_results = {}
    hbm_results = {}

    def time_and_report(tag, fn, config_name, K):
        stats = cuda_time_trimmed(fn, n_iter=args.n_iter,
                                   n_repeat=args.n_repeat, n_warmup=args.n_warmup)
        wall_results[tag] = stats
        hbm = analytic_hbm_bytes(config_name, K, T, B, C, H, W)
        hbm_results[tag] = hbm
        print(f"  {tag:<18} wall={stats['trimmed_mean_ms']:>8.4f} ms "
              f"(stdev {stats['trimmed_stdev_ms']:.4f})  "
              f"hbm_ratio={hbm['ratio_vs_ref']:.4f}  save={hbm['savings_pct']:.1f}%")

    print(f"  {'tag':<18} {'wall':>16}  {'hbm_ratio':>10}  save")
    print(f"  {'-'*82}")

    bench_A = lambda: run_config_A(a, x, scale, bias, args.tau, args.v_th, 0.0)
    time_and_report('A_KT', bench_A, 'A', T)

    for K in K_list:
        bench_B_K = lambda K=K: run_config_B(a, x, scale, bias, K, args.tau, args.v_th, 0.0)
        time_and_report(f'B_K{K}', bench_B_K, 'B', K)

    bench_Bp = lambda: run_config_B_prime(a, x, scale, bias, args.tau, args.v_th, 0.0)
    time_and_report('Bp_KT', bench_Bp, "B'", T)

    bench_C = lambda: run_config_C(a, x, scale, bias, args.tau, args.v_th, 0.0)
    time_and_report('C_KT', bench_C, 'C', T)

    for K in [2, 4, 8]:
        bench_D_K = lambda K=K: run_config_D(a, x, scale, bias, K, args.tau, args.v_th, 0.0)
        time_and_report(f'D_K{K}', bench_D_K, 'D', K)

    # ----- Speedup table -----
    print(f"\n{'='*92}")
    print(f"Speedup table (Config A as baseline)")
    print(f"{'='*92}")
    base_ms = wall_results['A_KT']['trimmed_mean_ms']
    print(f"  {'tag':<18} {'wall_ms':>10}  {'speedup_vs_A':>12}  {'hbm_ratio':>10}")
    print(f"  {'-'*62}")
    for tag in wall_results:
        ms = wall_results[tag]['trimmed_mean_ms']
        speedup = base_ms / ms
        hbm_ratio = hbm_results[tag]['ratio_vs_ref']
        print(f"  {tag:<18} {ms:>10.4f}  {speedup:>11.4f}x  {hbm_ratio:>10.4f}")

    # ----- Save JSON -----
    output = {
        'experiment': 'phase2_task2.2b_add_bn_lif_ablation',
        'device': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'reference_spike_rate': ref_rate,
        'parity': parity_results,
        'wall_clock': wall_results,
        'hbm_analytic': hbm_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()