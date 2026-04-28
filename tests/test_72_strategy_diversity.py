"""
Test 72: Phase 0 Strategy Diversity Heatmap
============================================

Purpose
-------
This is the GO/NO-GO experiment for the new v2 thesis:
  "Per-layer fusion strategy choice is the primary optimization axis;
   no single fusion strategy dominates all layers."

If a single strategy (e.g., DK) dominates all layers in our test grid,
the thesis dies. If different layers prefer different strategies, the
thesis has empirical support and routing has measurable value.

This is the most critical experiment — must run BEFORE writing any
new fusion kernel (Hybrid PostFuse, Cross-site Fused) or routing
framework. If RED, those work items don't make sense.

Setup
-----
- Real SEW-RN18 layer shapes (~16 unique layers)
- Realistic SNN sparsity (Bernoulli input, sparsity 0.85-0.95)
- T ∈ {4, 8, 16} — important because state carry benefit scales with T
- B ∈ {1, 2, 4}
- Two implementations available now:
    * DK: cuDNN + Triton LIF sequential
    * FullFuse-Triton: existing fused_conv_bn_lif kernel
      (NOT relying on sparsity skip — just fusion benefit)

Decision rule (Go/No-Go from GPT review):
  GREEN: ≥30% layers prefer non-majority impl AND oracle/best-fixed > 1.05×
  YELLOW: 10-30% layers prefer non-majority
  RED: <10% non-majority OR oracle/best-fixed < 1.05×

Outputs
-------
  artifacts/phase0_v2/diversity_data.npz
  artifacts/phase0_v2/diversity_decision.md  ← key file
  artifacts/phase0_v2/figures/
    heatmap_t4.pdf — layer × implementation latency, T=4
    heatmap_t8.pdf — same for T=8
    heatmap_t16.pdf
    regret_summary.pdf — fixed-vs-oracle gap distribution
"""

import os, sys, time, statistics, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 42

N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3


# ============================================================
# SEW-RN18 real layer shapes
# ============================================================
# (name, C_in, C_out, H, W, kernel_size, stride)
# Covers all unique LIF site shapes in SEW-RN18

SEW_RN18_LAYERS = [
    # Stem
    ('stem',        3,    64, 112, 112, 7, 2),  # large input, but stride=2 (skip if cudnn 7x7 path is unstable)
    
    # Layer 1: 64 → 64, 56×56
    ('l1.0.conv1',  64,   64, 56, 56, 3, 1),
    ('l1.0.conv2',  64,   64, 56, 56, 3, 1),
    
    # Layer 2: 64 → 128, downsample
    ('l2.0.conv1',  64,  128, 56, 56, 3, 2),  # stride=2 downsample
    ('l2.0.conv2',  128, 128, 28, 28, 3, 1),
    ('l2.0.down',   64,  128, 56, 56, 1, 2),  # 1x1 projection
    ('l2.1.conv1',  128, 128, 28, 28, 3, 1),
    
    # Layer 3
    ('l3.0.conv1',  128, 256, 28, 28, 3, 2),
    ('l3.0.conv2',  256, 256, 14, 14, 3, 1),
    ('l3.0.down',   128, 256, 28, 28, 1, 2),
    ('l3.1.conv1',  256, 256, 14, 14, 3, 1),
    
    # Layer 4
    ('l4.0.conv1',  256, 512, 14, 14, 3, 2),
    ('l4.0.conv2',  512, 512, 7,  7,  3, 1),
    ('l4.0.down',   256, 512, 14, 14, 1, 2),
    ('l4.1.conv1',  512, 512, 7,  7,  3, 1),
]

# But for now, only test 3x3 stride=1 shapes (FullFuse kernel constraint)
# 3x3 stride=2 falls back to dense in current SF implementation
LAYERS_3X3_S1 = [l for l in SEW_RN18_LAYERS if l[5] == 3 and l[6] == 1]


# ============================================================
# Test grid
# ============================================================

T_VALUES = [4, 8, 16]
B_VALUES = [1, 2]

# Realistic SNN sparsity — based on trained SNN literature
# Don't sweep, just use representative values
SPARSITY = 0.90


# ============================================================
# Implementations
# ============================================================

def make_input(C, H, W, T, B, sparsity, generator):
    """Real binary input simulating spike with given sparsity."""
    keep = 1.0 - sparsity
    rand = torch.rand(T, B, C, H, W, device=DEVICE, generator=generator)
    return (rand < keep).float()


def bench_dk(x, weight, bn_scale, bn_bias):
    """DenseKeep: cuDNN conv + Triton LIF sequential.
    
    z is materialized in HBM. LIF runs over T steps with state carry in registers.
    """
    T, B_size, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH = weight.shape[2]
    pad = kH // 2
    
    # BN folding into conv weight
    w_fused = weight * bn_scale.view(-1, 1, 1, 1)
    b_fused = bn_bias
    
    x_4d = x.reshape(T * B_size, C_in, H, W)
    
    try:
        from catfuse.sparseflow.lif_seq_kernel import lif_sequential
        use_triton_lif = True
    except ImportError:
        use_triton_lif = False
    
    def _fwd():
        z_4d = F.conv2d(x_4d, w_fused, bias=b_fused, stride=1, padding=pad)
        z = z_4d.reshape(T, B_size, C_out, H, W)
        v_in = torch.zeros(B_size, C_out, H, W, device=DEVICE, dtype=torch.float32)
        if use_triton_lif:
            spikes, v_out = lif_sequential(
                z, v_in, tau=2.0, v_threshold=1.0, v_reset=0.0,
            )
        else:
            v = v_in
            spikes = []
            for t in range(T):
                v = v + (z[t] - v) / 2.0
                spike = (v >= 1.0).float()
                v = v * (1.0 - spike)
                spikes.append(spike)
            spikes = torch.stack(spikes)
            v_out = v
        return spikes, v_out
    
    return _measure(_fwd)


def bench_fullfuse(x, weight, bn_scale, bn_bias):
    """FullFuse-Triton: existing Triton fused conv-bn-lif kernel.
    
    Conv + BN + LIF all in one Triton kernel. z does NOT materialize to HBM.
    Note: This implementation has a sparsity skip path internally, but we
    are NOT testing whether sparsity skip works — we're testing fusion benefit
    vs DK across different (shape, T, B).
    """
    try:
        from catfuse.sparseflow.fused_conv_bn_lif_kernel import sparse_fused_conv_bn_lif_forward
    except ImportError as e:
        return None, f"FullFuse import: {e}"
    
    T, B_size, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH = weight.shape[2]
    H_out, W_out = H, W
    
    def _fwd():
        v = torch.zeros(B_size, C_out, H_out, W_out, device=DEVICE, dtype=torch.float32)
        spikes = []
        for t in range(T):
            x_t = x[t]
            try:
                result = sparse_fused_conv_bn_lif_forward(
                    x_t, v, weight,
                    bias=None,
                    bn_scale=bn_scale,
                    bn_bias=bn_bias,
                    tau=2.0,
                    v_threshold=1.0,
                    v_reset=0.0,
                    kernel_size=kH,
                )
                if isinstance(result, tuple):
                    spike = result[0]
                    v = result[1]
                else:
                    spike = result
            except Exception as e:
                raise RuntimeError(f"FullFuse: {e}")
            spikes.append(spike)
        return torch.stack(spikes), v
    
    # Verify it runs
    try:
        with torch.no_grad():
            _fwd()
    except Exception as e:
        return None, f"FullFuse fwd: {e}"
    
    return _measure(_fwd), None


def _measure(forward_fn):
    for _ in range(N_WARMUP):
        with torch.no_grad():
            forward_fn()
    torch.cuda.synchronize()
    
    medians = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            with torch.no_grad():
                forward_fn()
        torch.cuda.synchronize()
        medians.append((time.perf_counter() - t0) / N_ITER * 1e6)
    
    return statistics.median(medians)


# ============================================================
# Sweep
# ============================================================

def run_sweep():
    weight_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    input_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    
    print(f"{'='*120}")
    header = f"{'Layer':<14}{'C_in':<5}{'C_out':<6}{'H':<4}{'W':<4}{'T':<3}{'B':<3}"
    header += f"{'T_DK_us':<10}{'T_FF_us':<10}{'oracle':<9}{'DK_reg':<8}{'FF_reg':<8}{'winner':<8}"
    print(header)
    print('-' * 120)
    
    results = []
    
    for layer_name, C_in, C_out, H, W, kH, stride in LAYERS_3X3_S1:
        weight = torch.randn(C_out, C_in, kH, kH, device=DEVICE, generator=weight_gen) * 0.1
        bn_scale = torch.ones(C_out, device=DEVICE)
        bn_bias = torch.zeros(C_out, device=DEVICE)
        
        for T in T_VALUES:
            for B in B_VALUES:
                # Per-config input
                local_gen = torch.Generator(device=DEVICE).manual_seed(
                    SEED + hash((layer_name, T, B)) % 100000
                )
                x = make_input(C_in, H, W, T, B, SPARSITY, local_gen)
                
                # Bench DK
                try:
                    t_dk = bench_dk(x, weight, bn_scale, bn_bias)
                except Exception as e:
                    print(f"  DK fail at {layer_name} T={T} B={B}: {e}")
                    t_dk = float('nan')
                
                # Bench FullFuse
                try:
                    res = bench_fullfuse(x, weight, bn_scale, bn_bias)
                    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], (int, float, type(None))):
                        # Error case
                        t_ff, err = res
                        if t_ff is None:
                            print(f"  FF fail at {layer_name} T={T} B={B}: {err}")
                            t_ff = float('nan')
                    else:
                        # res is just a number (the median)
                        t_ff = res if isinstance(res, (int, float)) else res[0] if res[0] is not None else float('nan')
                except Exception as e:
                    print(f"  FF fail at {layer_name} T={T} B={B}: {e}")
                    t_ff = float('nan')
                
                # Compute oracle / regret
                if not np.isnan(t_dk) and not np.isnan(t_ff):
                    oracle = min(t_dk, t_ff)
                    dk_regret = t_dk / oracle
                    ff_regret = t_ff / oracle
                    winner = 'DK' if t_dk < t_ff else 'FF'
                else:
                    oracle = t_dk if not np.isnan(t_dk) else t_ff
                    dk_regret = float('nan') if np.isnan(t_dk) else 1.0 if np.isnan(t_ff) else float('nan')
                    ff_regret = float('nan')
                    winner = '?'
                
                row = f"{layer_name:<14}{C_in:<5}{C_out:<6}{H:<4}{W:<4}{T:<3}{B:<3}"
                row += f"{t_dk:<10.1f}{t_ff:<10.1f}{oracle:<9.1f}"
                row += f"{dk_regret:<8.3f}{ff_regret:<8.3f}{winner:<8}"
                print(row)
                
                results.append({
                    'layer': layer_name,
                    'C_in': C_in, 'C_out': C_out, 'H': H, 'W': W,
                    'T': T, 'B': B,
                    't_dk_us': t_dk,
                    't_ff_us': t_ff,
                    'oracle_us': oracle,
                    'dk_regret': dk_regret,
                    'ff_regret': ff_regret,
                    'winner': winner,
                })
    
    return results


# ============================================================
# Analysis & Decision
# ============================================================

def analyze(results, out_dir):
    valid = [r for r in results if not np.isnan(r['dk_regret']) and not np.isnan(r['ff_regret'])]
    
    if not valid:
        print("\n[ERROR] No valid measurements")
        return
    
    n = len(valid)
    n_dk_wins = sum(1 for r in valid if r['winner'] == 'DK')
    n_ff_wins = sum(1 for r in valid if r['winner'] == 'FF')
    
    # majority impl is whichever wins more
    if n_dk_wins >= n_ff_wins:
        majority = 'DK'
        non_majority_count = n_ff_wins
    else:
        majority = 'FF'
        non_majority_count = n_dk_wins
    
    non_majority_frac = non_majority_count / n if n > 0 else 0
    
    # oracle vs best-fixed speedup
    # best-fixed = whichever single impl gives best total time
    total_dk = sum(r['t_dk_us'] for r in valid)
    total_ff = sum(r['t_ff_us'] for r in valid)
    total_oracle = sum(r['oracle_us'] for r in valid)
    
    best_fixed_total = min(total_dk, total_ff)
    oracle_speedup_over_fixed = best_fixed_total / total_oracle
    
    # Per-T breakdown — does T matter?
    by_T = {}
    for r in valid:
        by_T.setdefault(r['T'], []).append(r)
    
    # Build report
    lines = []
    lines.append("# Phase 0 Strategy Diversity — Decision Report\n\n")
    lines.append("## Setup\n")
    lines.append(f"- SEW-RN18 3x3 stride=1 layers tested: {len(LAYERS_3X3_S1)}\n")
    lines.append(f"- T values: {T_VALUES}\n")
    lines.append(f"- B values: {B_VALUES}\n")
    lines.append(f"- Total configs: {len(results)}\n")
    lines.append(f"- Valid measurements: {n}\n")
    lines.append(f"- Sparsity: {SPARSITY} (Bernoulli, simulating realistic SNN)\n\n")
    
    lines.append("## Strategy diversity\n\n")
    lines.append(f"| Implementation | Wins | Win% |\n")
    lines.append(f"|---|---|---|\n")
    lines.append(f"| DK (cuDNN + Triton LIF) | {n_dk_wins} | {100*n_dk_wins/n:.1f}% |\n")
    lines.append(f"| FullFuse-Triton | {n_ff_wins} | {100*n_ff_wins/n:.1f}% |\n\n")
    
    lines.append(f"- Majority impl: **{majority}** ({max(n_dk_wins, n_ff_wins)}/{n})\n")
    lines.append(f"- Non-majority fraction: **{non_majority_frac:.1%}**\n")
    lines.append(f"- Oracle vs best-fixed speedup: **{oracle_speedup_over_fixed:.3f}×**\n\n")
    
    # T sensitivity
    lines.append(f"## T sensitivity (does state carry matter more at larger T?)\n\n")
    lines.append(f"| T | DK wins | FF wins | total | FF win% |\n")
    lines.append(f"|---|---|---|---|---|\n")
    for T_val, items in sorted(by_T.items()):
        dk_w = sum(1 for r in items if r['winner'] == 'DK')
        ff_w = sum(1 for r in items if r['winner'] == 'FF')
        total = len(items)
        lines.append(f"| {T_val} | {dk_w} | {ff_w} | {total} | {100*ff_w/total:.1f}% |\n")
    lines.append("\n")
    
    # Per-layer winners
    lines.append(f"## Per-layer winners\n\n")
    by_layer = {}
    for r in valid:
        by_layer.setdefault(r['layer'], []).append(r)
    lines.append(f"| Layer | Shape | DK win count | FF win count |\n")
    lines.append(f"|---|---|---|---|\n")
    for layer_name, items in by_layer.items():
        c_in = items[0]['C_in']; c_out = items[0]['C_out']
        h = items[0]['H']; w = items[0]['W']
        dk_w = sum(1 for r in items if r['winner'] == 'DK')
        ff_w = sum(1 for r in items if r['winner'] == 'FF')
        lines.append(f"| {layer_name} | ({c_in}x{c_out}, {h}x{w}) | {dk_w} | {ff_w} |\n")
    lines.append("\n")
    
    # Decision (GPT's Go/No-Go thresholds)
    lines.append(f"## Decision\n\n")
    
    if non_majority_frac >= 0.30 and oracle_speedup_over_fixed >= 1.05:
        verdict = "GREEN"
        reason = f"Non-majority impl wins {non_majority_frac:.0%} of cases; routing gives {(oracle_speedup_over_fixed-1)*100:.1f}% speedup over best-fixed."
        action = "Continue v2 plan: write Hybrid PostFuse, Cross-site Fused, build routing framework."
    elif non_majority_frac >= 0.10:
        verdict = "YELLOW"
        reason = f"Non-majority impl wins only {non_majority_frac:.0%} of cases; oracle speedup {(oracle_speedup_over_fixed-1)*100:.1f}%."
        action = "Marginal evidence for routing. Consider narrowing thesis to specific regimes (e.g., specific T or shape ranges)."
    else:
        verdict = "RED"
        reason = f"Single impl ({majority}) dominates {1-non_majority_frac:.0%} of cases; oracle speedup only {(oracle_speedup_over_fixed-1)*100:.1f}%."
        action = "Routing thesis lacks empirical support. DO NOT proceed with new kernel work or routing framework. Reassess."
    
    lines.append(f"### Verdict: **{verdict}**\n\n")
    lines.append(f"{reason}\n\n")
    lines.append(f"**Recommendation:** {action}\n\n")
    
    lines.append(f"## Notes\n\n")
    lines.append(f"- This experiment uses ONLY 2 implementations (DK, FullFuse-Triton)\n")
    lines.append(f"- Adding more strategies (Hybrid PostFuse, Cross-site) can only INCREASE diversity\n")
    lines.append(f"- If GREEN here: thesis safely holds when adding more strategies later\n")
    lines.append(f"- If RED here with 2 strategies: more strategies unlikely to save it,\n")
    lines.append(f"  because if FullFuse already loses to DK on every layer, adding Hybrid\n")
    lines.append(f"  PostFuse (which has even less fusion than FullFuse) won't help\n")
    lines.append(f"- Tested 3x3 stride=1 layers only; 1x1 and stride=2 may show different patterns\n")
    
    out_path = os.path.join(out_dir, 'diversity_decision.md')
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"\n[saved] {out_path}")
    
    print(f"\n{'='*70}")
    print(f"PHASE 0 STRATEGY DIVERSITY VERDICT: {verdict}")
    print(f"{'='*70}")
    print(f"Non-majority impl wins: {non_majority_frac:.1%} of {n} configs")
    print(f"Oracle vs best-fixed speedup: {oracle_speedup_over_fixed:.3f}×")
    print(f"Recommendation: {action}")


def save_results(results, path):
    keys = list(results[0].keys())
    arr_dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        if k in ('layer', 'winner'):
            arr_dict[k] = np.array(vals, dtype=object)
        else:
            arr_dict[k] = np.array(vals, dtype=np.float64)
    np.savez(path, **arr_dict)
    print(f"[saved] {path}")


def plot_heatmap(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    valid = [r for r in results 
             if not np.isnan(r['t_dk_us']) and not np.isnan(r['t_ff_us'])]
    if not valid:
        return
    
    # one plot per T
    for T_val in T_VALUES:
        items = [r for r in valid if r['T'] == T_val and r['B'] == B_VALUES[0]]
        if not items:
            continue
        
        layer_names = [r['layer'] for r in items]
        dk_times = [r['t_dk_us'] for r in items]
        ff_times = [r['t_ff_us'] for r in items]
        
        fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.8), 5))
        x = np.arange(len(items))
        w = 0.4
        ax.bar(x - w/2, dk_times, w, label='DK (cuDNN+Triton LIF)', color='steelblue')
        ax.bar(x + w/2, ff_times, w, label='FullFuse-Triton', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_ylabel('Latency (μs)')
        ax.set_title(f'Strategy Diversity Heatmap (T={T_val}, B={B_VALUES[0]}, sparsity={SPARSITY})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'figures', f'heatmap_t{T_val}.pdf'))
        fig.savefig(os.path.join(out_dir, 'figures', f'heatmap_t{T_val}.png'), dpi=150)
        plt.close(fig)
    
    # regret summary
    fig, ax = plt.subplots(figsize=(8, 6))
    dk_regrets = [r['dk_regret'] for r in valid]
    ff_regrets = [r['ff_regret'] for r in valid]
    ax.scatter(dk_regrets, ff_regrets, alpha=0.6, s=50)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('DK regret')
    ax.set_ylabel('FullFuse regret')
    ax.set_title('Regret distribution: each point is one (layer, T, B) config')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_summary.pdf'))
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_summary.png'), dpi=150)
    plt.close(fig)
    
    print(f"[saved] {out_dir}/figures/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='artifacts/phase0_v2')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    
    print("=" * 80)
    print("Test 72 — Phase 0 Strategy Diversity Heatmap")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Layers: {len(LAYERS_3X3_S1)} (3x3 stride=1)")
    print(f"T values: {T_VALUES}")
    print(f"B values: {B_VALUES}")
    print(f"Sparsity: {SPARSITY}")
    print(f"Total configs: {len(LAYERS_3X3_S1) * len(T_VALUES) * len(B_VALUES)}")
    print()
    
    torch.backends.cudnn.benchmark = True
    
    results = run_sweep()
    save_results(results, os.path.join(args.out_dir, 'diversity_data.npz'))
    plot_heatmap(results, args.out_dir)
    analyze(results, args.out_dir)


if __name__ == '__main__':
    main()