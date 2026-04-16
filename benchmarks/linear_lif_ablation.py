"""
Phase 2 Task 2.2a — Linear->LIF ablation, 5 configs, K-sweep.

Configs:
  A  : Per-step F.linear + per-step torch-op LIF (no BatchFold, no StreamFuse,
       no StateCarry).  K not meaningful for A (K=T canonical only).
  B  : BatchFold(Linear) within block K + per-step torch-op LIF.
       Linear has T/K launches, LIF has T launches. z materialized in HBM.
  B' : BatchFold(Linear, all T) one Triton matmul kernel + one Triton LIF
       kernel (LIF steps collapsed into a single kernel, v in register
       across all T within the single LIF kernel).  z through HBM.
       Phase 0 `run_no_fusion` is this.
  C  : BatchFold + StreamFuse + StateCarry-implicit.  One fused Triton kernel
       for all T, z in register, v in register.  K=T only.
       Phase 0 `run_fusion_k(K=T)` is this.
  D  : BatchFold + StreamFuse + StateCarry-explicit.  Multi-block version,
       v carried through HBM at block boundaries.  K < T.
       Phase 0 `run_fusion_k(K)` is this for K in {2, 4, 8}.

Parity reference: run_reference (torch.matmul + python for-loop LIF)

Data points (per subgraph):
  Config A (K=T only)         : 1
  Config B (K=1, 2, 4, 8, 16) : 5
  Config B' (K=T only)        : 1
  Config C (K=T only)         : 1
  Config D (K=2, 4, 8)        : 3
  Total                       : 11

Total with 3 seeds: 33 runs per subgraph.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

# Import Phase 0 Linear->LIF kernels
from linear_lif_k_sweep_v3 import (
    run_fusion_k as phase0_run_fusion_k,
    run_no_fusion as phase0_run_no_fusion,
)


# ============================================================
# Config A: per-step F.linear + per-step torch-op LIF
# ============================================================
# This is the "nothing is fused" baseline. T launches of F.linear + T
# launches of torch LIF ops (add/mul/ge/etc).
# ============================================================

def run_config_A(x: torch.Tensor,
                 w: torch.Tensor,
                 tau: float, v_th: float, v_reset: float) -> torch.Tensor:
    """
    Args:
      x: [T, B, I]
      w: [I, O]  (Phase 0 convention, not PyTorch [O, I] convention)
    """
    T, B, I_dim = x.shape
    O_dim = w.shape[1]
    w_pt = w.T.contiguous()  # F.linear expects [O, I]

    v = torch.zeros(B, O_dim, device=x.device, dtype=x.dtype)
    spikes = torch.empty(T, B, O_dim, device=x.device, dtype=x.dtype)
    inv_tau = 1.0 / tau

    for t in range(T):
        z_t = F.linear(x[t], w_pt)  # [B, O]
        v = v + (z_t - v) * inv_tau
        s = (v >= v_th).float()
        v = v * (1.0 - s) + v_reset * s
        spikes[t] = s

    return spikes


# ============================================================
# Config B: BatchFold(Linear, K) + per-step torch-op LIF
# ============================================================
# For each block of K consecutive time steps, merge Linear into one batched
# call; then do LIF step-by-step within the block.  T/K Linear launches, T
# LIF launches.  z materialized in HBM.
# ============================================================

def run_config_B(x: torch.Tensor,
                 w: torch.Tensor,
                 K: int,
                 tau: float, v_th: float, v_reset: float) -> torch.Tensor:
    T, B, I_dim = x.shape
    O_dim = w.shape[1]
    assert T % K == 0
    w_pt = w.T.contiguous()

    v = torch.zeros(B, O_dim, device=x.device, dtype=x.dtype)
    spikes = torch.empty(T, B, O_dim, device=x.device, dtype=x.dtype)
    inv_tau = 1.0 / tau

    for block_start in range(0, T, K):
        # BatchFold: one F.linear call for K time steps
        x_block = x[block_start:block_start + K].reshape(K * B, I_dim)
        z_block = F.linear(x_block, w_pt).reshape(K, B, O_dim)

        # Per-step LIF within the block
        for k in range(K):
            z_k = z_block[k]
            v = v + (z_k - v) * inv_tau
            s = (v >= v_th).float()
            v = v * (1.0 - s) + v_reset * s
            spikes[block_start + k] = s

    return spikes


# ============================================================
# Config B': Phase 0 run_no_fusion (Triton matmul + Triton LIF)
# ============================================================
# Wrapping for naming clarity in the ablation.

def run_config_B_prime(x, w, tau, v_th, v_reset):
    return phase0_run_no_fusion(x, w, tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Config C: Phase 0 run_fusion_k at K=T
# ============================================================

def run_config_C(x, w, T, tau, v_th, v_reset):
    return phase0_run_fusion_k(x, w, K=T, tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Config D: Phase 0 run_fusion_k at K<T
# ============================================================

def run_config_D(x, w, K, tau, v_th, v_reset):
    return phase0_run_fusion_k(x, w, K=K, tau=tau, v_th=v_th, v_reset=v_reset)


# ============================================================
# Reference
# ============================================================

def run_reference(x, w, tau, v_th, v_reset):
    """torch.matmul + python for-loop LIF."""
    T, B, I_dim = x.shape
    O_dim = w.shape[1]
    z = torch.matmul(x, w)  # [T, B, O]
    v = torch.zeros(B, O_dim, device=x.device, dtype=x.dtype)
    spikes = torch.empty_like(z)
    inv_tau = 1.0 / tau
    for t in range(T):
        v = v + (z[t] - v) * inv_tau
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
# Analytic HBM model for Linear->LIF
# ============================================================
# Reference HBM (per 4-byte element):
#   x read : T * B * I
#   w read : I * O  (amortized once, assume L2 cold in the analytic model)
#   z write+read : 2 * T * B * O    (between Linear and LIF)
#   v read+write : 2 * T * B * O    (LIF recurrence per step)
#   s write : T * B * O
# Total ref ~ T*B*I + I*O + 5*T*B*O
#
# For each config, report what's eliminated:
#   A : same as ref
#   B : z still in HBM (noop for HBM, only Linear launches change)
#   B': z still in HBM (Triton LIF kernel has v in register across T, so
#       v HBM roundtrip eliminated: T*B*I + I*O + 3*T*B*O + T*B*O (s)
#       = T*B*I + I*O + 4*T*B*O  (saves 1 step vs ref? let's keep simple)
#       Actually since Phase 0 run_no_fusion has one LIF kernel that keeps
#       v in register, the 2*T*B*O v roundtrip becomes 0. Savings: 2 T B O.
#   C : z in register too (streamfuse). Savings vs B': 2*T*B*O (z roundtrip)
#       Total HBM: T*B*I + I*O + T*B*O (s write only) + 0 (z, v in register)
#   D : C + block-boundary v roundtrip. HBM: same as C + (2*(T/K) - 1)*B*O
#       for the v_carry roundtrip at each block boundary.

def analytic_hbm_bytes(config: str, K: int, T: int, B: int, I_dim: int, O_dim: int,
                        dtype_bytes: int = 4) -> dict:
    step_x = B * I_dim * dtype_bytes
    step_z = B * O_dim * dtype_bytes
    w_bytes = I_dim * O_dim * dtype_bytes

    ref = T * step_x + w_bytes + 5 * T * step_z
    # components:
    # - x read (T * step_x)
    # - w read (w_bytes, amortized)
    # - z write (T * step_z)
    # - z read (T * step_z, LIF reads z)
    # - v read (T * step_z) — LIF recurrence (step 1 reads v_prev)
    # - v write (T * step_z) — LIF recurrence (write v_new)
    # - s write (T * step_z)
    # Total: T*x + w + 5*T*z   (combining z_write+z_read+v_read+v_write+s_write)

    if config == 'A':
        # same as ref
        bytes_ = ref
    elif config == 'B':
        # BatchFold only: Linear launches reduce, HBM identical to A (z/v/s still thru HBM)
        bytes_ = ref
    elif config == "B'":
        # LIF kernel collapses v into register across T. z still through HBM.
        # Savings: 2 * T * step_z  (v read + v write eliminated)
        bytes_ = T * step_x + w_bytes + 3 * T * step_z
        # components: x + w + z_write + z_read + s_write
    elif config == 'C':
        # StreamFuse: z also in register. z write+read eliminated.
        # Savings vs B': 2 * T * step_z (z write + z read)
        bytes_ = T * step_x + w_bytes + T * step_z
        # components: x + w + s_write only
    elif config == 'D':
        # C + block-boundary v_carry: (2 * (T/K) - 1) * step_z for v_carry
        # (writes at end of each block except last + reads at start of each block except first)
        n_blocks = T // K
        v_traffic = (2 * n_blocks - 1) * step_z if n_blocks > 1 else 0
        bytes_ = T * step_x + w_bytes + T * step_z + v_traffic
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
    p.add_argument('--I', type=int, default=512)
    p.add_argument('--O', type=int, default=512)
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
        args.output = f'./results/phase2/linear_lif_ablation_v100_seed{args.seed}.json'

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    T, B, I_dim, O_dim = args.T, args.B, args.I, args.O

    print(f"Phase 2 task 2.2a — Linear->LIF 5-config ablation, K-sweep")
    print(f"  GPU   : {torch.cuda.get_device_name(device)}")
    print(f"  shape : T={T} B={B} I={I_dim} O={O_dim}")
    print(f"  LIF   : tau={args.tau} v_th={args.v_th}")
    print(f"  seed  : {args.seed}")

    x = (torch.randn(T, B, I_dim, device=device, dtype=torch.float32)
         * args.input_scale).contiguous()
    w = (torch.randn(I_dim, O_dim, device=device, dtype=torch.float32)
         * args.input_scale).contiguous()

    # Reference for parity
    ref_spikes = run_reference(x, w, args.tau, args.v_th, 0.0)
    ref_spike_rate = ref_spikes.mean().item()
    print(f"  reference spike rate: {ref_spike_rate:.6f}")

    # ----- Parity check -----
    print(f"\n{'='*92}")
    print(f"Parity check (all configs vs reference)")
    print(f"{'='*92}")

    def check_parity(tag, candidate):
        flips = (candidate != ref_spikes).sum().item()
        total = ref_spikes.numel()
        match = (1.0 - flips / total) * 100
        status = 'PASS' if match > 99.9 else 'FAIL'
        print(f"  [{tag:<20}] match={match:.6f}%  flips={flips}/{total}  {status}")
        return {'spike_match_pct': match, 'spike_flips': flips}

    parity_results = {}

    # Config A
    with torch.no_grad():
        s_A = run_config_A(x, w, args.tau, args.v_th, 0.0)
    parity_results['A'] = check_parity('A', s_A)

    # Config B at each K
    K_list = [1, 2, 4, 8, 16]
    for K in K_list:
        with torch.no_grad():
            s_B_K = run_config_B(x, w, K, args.tau, args.v_th, 0.0)
        parity_results[f'B_K{K}'] = check_parity(f"B (K={K})", s_B_K)

    # Config B'
    with torch.no_grad():
        s_Bp = run_config_B_prime(x, w, args.tau, args.v_th, 0.0)
    parity_results["B'"] = check_parity("B'", s_Bp)

    # Config C at K=T
    with torch.no_grad():
        s_C = run_config_C(x, w, T, args.tau, args.v_th, 0.0)
    parity_results['C'] = check_parity('C (K=T)', s_C)

    # Config D at K in {2, 4, 8}
    for K in [2, 4, 8]:
        with torch.no_grad():
            s_D_K = run_config_D(x, w, K, args.tau, args.v_th, 0.0)
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
        hbm = analytic_hbm_bytes(config_name, K, T, B, I_dim, O_dim)
        hbm_results[tag] = hbm
        print(f"  {tag:<18} wall={stats['trimmed_mean_ms']:>8.4f} ms "
              f"(stdev {stats['trimmed_stdev_ms']:.4f})  "
              f"hbm_ratio={hbm['ratio_vs_ref']:.4f}  save={hbm['savings_pct']:.1f}%")

    print(f"  {'tag':<18} {'wall':>16}  {'hbm_ratio':>10}  save")
    print(f"  {'-'*82}")

    # A
    bench_A = lambda: run_config_A(x, w, args.tau, args.v_th, 0.0)
    time_and_report('A_KT', bench_A, 'A', T)

    # B at each K
    for K in K_list:
        bench_B_K = lambda K=K: run_config_B(x, w, K, args.tau, args.v_th, 0.0)
        time_and_report(f'B_K{K}', bench_B_K, 'B', K)

    # B'
    bench_Bp = lambda: run_config_B_prime(x, w, args.tau, args.v_th, 0.0)
    time_and_report("Bp_KT", bench_Bp, "B'", T)

    # C
    bench_C = lambda: run_config_C(x, w, T, args.tau, args.v_th, 0.0)
    time_and_report('C_KT', bench_C, 'C', T)

    # D at K in {2, 4, 8}
    for K in [2, 4, 8]:
        bench_D_K = lambda K=K: run_config_D(x, w, K, args.tau, args.v_th, 0.0)
        time_and_report(f'D_K{K}', bench_D_K, 'D', K)

    # ----- Summary table: speedup vs Config A -----
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
        'experiment': 'phase2_task2.2a_linear_lif_ablation',
        'device': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'reference_spike_rate': ref_spike_rate,
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