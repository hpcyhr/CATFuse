"""
Test 70: Phase 0 Oracle Gap Microbenchmark
==========================================

Purpose
-------
Validate the core thesis of CATFuse v2 routing:
  "No single implementation dominates all (layer, sparsity, pattern) combinations,
   so per-layer routing has measurable value."

This experiment must run BEFORE writing any new kernel or routing code.
If fixed-DK or fixed-SF already dominates ≥95% of cases, the thesis is dead
and we need to redesign.

Design
------
For a grid of (shape, sparsity, pattern):
  - DK (cuDNN + Triton LIF) latency
  - SF v1 latency
  - oracle = min(DK, SF)
  - DK regret = DK / oracle
  - SF regret = SF / oracle

Sparsity patterns tested:
  - bernoulli: each position independently 0/1 with prob (1-sparsity)
  - blocky:    entire 8x8 tiles are all-0 or all-1 (SF prescan friendly)
  - channel:   entire channel is all-0 or all-1 (group-level skip friendly)
  - temporal:  some time steps all-0, others dense (SNN-realistic)

Critical decision rule (after experiment):
  - if fixed-SF regret p50 < 1.05 across MOST cases:
      SF alone is fine, routing has limited value → thesis at risk
  - if fixed-DK regret p50 < 1.05 across MOST cases:
      DK alone is fine, routing has limited value → thesis at risk
  - if BOTH have high regret in different regimes:
      routing has value → thesis holds → continue v2 plan

Outputs
-------
  artifacts/phase0/oracle_gap_data.npz       — raw measurements
  artifacts/phase0/regret_summary.md         — human-readable summary
  artifacts/phase0/figures/                  — plots
  artifacts/phase0/decision.md               — GO / NO-GO judgment

Usage
-----
    python tests/test_70_oracle_gap.py
    
    # quick mode
    python tests/test_70_oracle_gap.py --quick
"""

import os, sys, time, statistics, argparse, json
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 42

T_TIME = 4
B = 1

N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3   # how many times we re-measure to get distribution


# ============================================================
# Test grid
# ============================================================

# 4 representative SEW-RN18 layer shapes, span the actual layer space
SHAPES = [
    # (name, C_in, C_out, H, W)
    ('layer1.0', 64,  64, 56, 56),
    ('layer2.0', 64,  128, 28, 28),
    ('layer3.0', 128, 256, 14, 14),
    ('layer4.0', 256, 512, 7,  7),
]

# Focus on realistic SNN sparsity range (>0.7)
# trained SNN typically has 85-95%; outside this range less interesting
SPARSITY_LEVELS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

# Sparsity patterns
PATTERNS = ['bernoulli', 'blocky', 'channel', 'temporal']


# ============================================================
# Sparsity injection — different patterns
# ============================================================

def make_input(shape, sparsity, pattern, generator):
    """Generate a binary tensor with target sparsity and given spatial pattern.
    
    shape: (T, B, C, H, W)
    sparsity: 0..1, fraction of zeros
    pattern: 'bernoulli', 'blocky', 'channel', 'temporal'
    """
    T, B, C, H, W = shape
    keep = 1.0 - sparsity
    
    if pattern == 'bernoulli':
        # Each position independent
        rand = torch.rand(shape, device=DEVICE, generator=generator)
        x = (rand < keep).float()
    
    elif pattern == 'blocky':
        # 8x8 spatial blocks are all-0 or all-1
        # Simulates SF prescan-friendly case (whole tile zero)
        TILE = 8
        Hb = (H + TILE - 1) // TILE
        Wb = (W + TILE - 1) // TILE
        # Generate random per-(t, b, c, hb, wb) decisions
        block_rand = torch.rand(T, B, C, Hb, Wb, device=DEVICE, generator=generator)
        block_active = (block_rand < keep).float()  # [T, B, C, Hb, Wb]
        # Upsample to spatial size
        block_active = block_active.repeat_interleave(TILE, dim=3).repeat_interleave(TILE, dim=4)
        # Crop to target H, W (in case TILE doesn't divide)
        x = block_active[:, :, :, :H, :W].contiguous()
    
    elif pattern == 'channel':
        # Some channels are all-0 across spatial dims
        # Simulates group-level-skip friendly case
        chan_rand = torch.rand(T, B, C, device=DEVICE, generator=generator)
        chan_active = (chan_rand < keep).float()  # [T, B, C]
        # Each active channel: density set to 1 (full active)
        # Each inactive channel: density 0
        x = chan_active.unsqueeze(-1).unsqueeze(-1).expand(T, B, C, H, W).contiguous()
    
    elif pattern == 'temporal':
        # Some time steps all-0, others dense
        # Simulates SNN where spike rate varies across time
        # Half time steps active (with double density in active steps)
        t_rand = torch.rand(T, device=DEVICE, generator=generator)
        n_active_t = max(1, int(round(T * keep * 2)))  # roughly half active
        active_t = t_rand.argsort()[:n_active_t]
        x = torch.zeros(T, B, C, H, W, device=DEVICE)
        # In active time steps: dense (sparsity = 0)
        for t in active_t:
            x[t] = 1.0  # all-active in active steps
        # Note: actual sparsity ≈ 1 - n_active_t/T (close to target)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return x


# ============================================================
# Implementations
# ============================================================

def bench_densekeep(x, weight, bn_scale, bn_bias):
    """DK: cuDNN conv + Triton lif_sequential."""
    T, B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH = weight.shape[2]
    pad = kH // 2
    
    # BN folding into conv weight
    w_fused = weight * bn_scale.view(-1, 1, 1, 1)
    b_fused = bn_bias
    
    x_4d = x.reshape(T * B, C_in, H, W)
    
    try:
        from catfuse.sparseflow.lif_seq_kernel import lif_sequential
        use_triton_lif = True
    except ImportError:
        use_triton_lif = False
    
    def _forward():
        z_4d = F.conv2d(x_4d, w_fused, bias=b_fused, stride=1, padding=pad)
        z = z_4d.reshape(T, B, C_out, H, W)
        v_in = torch.zeros(B, C_out, H, W, device=DEVICE, dtype=torch.float32)
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
    
    return _measure(_forward)


def bench_sparseflow(x, weight, bn_scale, bn_bias):
    """SF v1: existing Triton fused kernel."""
    try:
        from catfuse.sparseflow.fused_conv_bn_lif_kernel import sparse_fused_conv_bn_lif_forward
    except ImportError as e:
        return None, f"SF kernel import failed: {e}"
    
    T, B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH = weight.shape[2]
    H_out, W_out = H, W
    
    def _forward():
        v = torch.zeros(B, C_out, H_out, W_out, device=DEVICE, dtype=torch.float32)
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
                raise RuntimeError(f"SF forward error: {e}")
            spikes.append(spike)
        return torch.stack(spikes), v
    
    # Test if it runs at all
    try:
        with torch.no_grad():
            _forward()
    except Exception as e:
        return None, f"SF call failed: {e}"
    
    return _measure(_forward), None


def _measure(forward_fn):
    """Repeat measurement N_REPEAT times. Return list of medians."""
    # Warmup
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
        medians.append((time.perf_counter() - t0) / N_ITER * 1e6)  # us
    
    # Return median across repeats + p10/p90
    return {
        'median': statistics.median(medians),
        'min': min(medians),
        'max': max(medians),
        'all': medians,
    }


# ============================================================
# Sweep
# ============================================================

def run_sweep(shapes, sparsities, patterns):
    """Run all (shape, sparsity, pattern) combinations."""
    
    weight_gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    input_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    
    print(f"{'='*120}")
    print(f"{'Shape':<14}{'Sp':<6}{'Pattern':<11}{'T_DK':<10}{'T_SF':<10}{'Oracle':<10}{'DK_reg':<10}{'SF_reg':<10}{'Winner':<8}")
    print('-' * 120)
    
    results = []
    
    for shape_name, C_in, C_out, H, W in shapes:
        # Fix weights for this shape
        weight = torch.randn(C_out, C_in, 3, 3, device=DEVICE, generator=weight_gen) * 0.1
        bn_scale = torch.ones(C_out, device=DEVICE)
        bn_bias = torch.zeros(C_out, device=DEVICE)
        
        for sp in sparsities:
            for pattern in patterns:
                # Per-config input seed
                local_gen = torch.Generator(device=DEVICE).manual_seed(
                    SEED + hash((shape_name, sp, pattern)) % 100000
                )
                
                shape_5d = (T_TIME, B, C_in, H, W)
                x = make_input(shape_5d, sp, pattern, local_gen)
                actual_sp = 1.0 - x.float().mean().item()
                
                # Bench DK
                try:
                    dk_meas = bench_densekeep(x, weight, bn_scale, bn_bias)
                    t_dk = dk_meas['median']
                except Exception as e:
                    print(f"  DK fail: {e}")
                    t_dk = float('nan')
                    dk_meas = None
                
                # Bench SF
                try:
                    sf_meas, sf_err = bench_sparseflow(x, weight, bn_scale, bn_bias)
                    if sf_meas is None:
                        t_sf = float('nan')
                    else:
                        t_sf = sf_meas['median']
                except Exception as e:
                    print(f"  SF fail: {e}")
                    t_sf = float('nan')
                    sf_meas = None
                
                # Oracle, regret
                if not np.isnan(t_dk) and not np.isnan(t_sf):
                    oracle = min(t_dk, t_sf)
                    dk_regret = t_dk / oracle
                    sf_regret = t_sf / oracle
                    winner = 'DK' if t_dk < t_sf else 'SF'
                elif not np.isnan(t_dk):
                    oracle = t_dk
                    dk_regret = 1.0
                    sf_regret = float('nan')
                    winner = 'DK only'
                elif not np.isnan(t_sf):
                    oracle = t_sf
                    dk_regret = float('nan')
                    sf_regret = 1.0
                    winner = 'SF only'
                else:
                    oracle = float('nan')
                    dk_regret = float('nan')
                    sf_regret = float('nan')
                    winner = '?'
                
                print(f"{shape_name:<14}{actual_sp:<6.3f}{pattern:<11}"
                      f"{t_dk:<10.1f}{t_sf:<10.1f}{oracle:<10.1f}"
                      f"{dk_regret:<10.3f}{sf_regret:<10.3f}{winner:<8}")
                
                results.append({
                    'shape': shape_name,
                    'C_in': C_in, 'C_out': C_out, 'H': H, 'W': W,
                    'target_sparsity': sp,
                    'actual_sparsity': actual_sp,
                    'pattern': pattern,
                    't_dk_us': t_dk,
                    't_sf_us': t_sf,
                    'oracle_us': oracle,
                    'dk_regret': dk_regret,
                    'sf_regret': sf_regret,
                    'winner': winner,
                })
    
    return results


# ============================================================
# Analysis & Decision
# ============================================================

def analyze(results, out_dir):
    """Compute regret distribution and make GO/NO-GO decision."""
    
    valid = [r for r in results 
             if not np.isnan(r['dk_regret']) and not np.isnan(r['sf_regret'])]
    
    if not valid:
        print("\n[ERROR] No valid measurements")
        return
    
    dk_regrets = [r['dk_regret'] for r in valid]
    sf_regrets = [r['sf_regret'] for r in valid]
    
    # Per-pattern breakdown
    by_pattern = {}
    for r in valid:
        by_pattern.setdefault(r['pattern'], []).append(r)
    
    # Build report
    lines = []
    lines.append("# Phase 0 Oracle Gap — Decision Report\n\n")
    lines.append(f"## Setup\n")
    lines.append(f"- Networks: SEW-RN18 representative layers\n")
    lines.append(f"- Shapes tested: {len(SHAPES)}\n")
    lines.append(f"- Sparsity levels: {SPARSITY_LEVELS}\n")
    lines.append(f"- Patterns: {PATTERNS}\n")
    lines.append(f"- Total configs: {len(results)}\n")
    lines.append(f"- Valid measurements: {len(valid)}\n\n")
    
    # Overall regret distribution
    lines.append(f"## Overall regret distribution\n\n")
    lines.append(f"| Metric | DK regret | SF regret |\n")
    lines.append(f"|---|---|---|\n")
    lines.append(f"| p50 (median) | {np.median(dk_regrets):.3f} | {np.median(sf_regrets):.3f} |\n")
    lines.append(f"| p75          | {np.percentile(dk_regrets, 75):.3f} | {np.percentile(sf_regrets, 75):.3f} |\n")
    lines.append(f"| p90          | {np.percentile(dk_regrets, 90):.3f} | {np.percentile(sf_regrets, 90):.3f} |\n")
    lines.append(f"| max          | {max(dk_regrets):.3f} | {max(sf_regrets):.3f} |\n\n")
    
    # Cases where each is bad
    dk_bad_5 = sum(1 for r in dk_regrets if r > 1.05) / len(dk_regrets)
    dk_bad_10 = sum(1 for r in dk_regrets if r > 1.10) / len(dk_regrets)
    sf_bad_5 = sum(1 for r in sf_regrets if r > 1.05) / len(sf_regrets)
    sf_bad_10 = sum(1 for r in sf_regrets if r > 1.10) / len(sf_regrets)
    
    lines.append(f"## Sub-optimality fraction\n\n")
    lines.append(f"| | DK | SF |\n")
    lines.append(f"|---|---|---|\n")
    lines.append(f"| Cases where regret > 1.05 (5%+ slowdown) | {dk_bad_5:.1%} | {sf_bad_5:.1%} |\n")
    lines.append(f"| Cases where regret > 1.10 (10%+ slowdown) | {dk_bad_10:.1%} | {sf_bad_10:.1%} |\n\n")
    
    # Pattern breakdown
    lines.append(f"## Per-pattern breakdown\n\n")
    lines.append(f"| Pattern | n | DK regret p50 | SF regret p50 | DK wins | SF wins |\n")
    lines.append(f"|---|---|---|---|---|---|\n")
    for pattern, items in by_pattern.items():
        dk_r = [r['dk_regret'] for r in items]
        sf_r = [r['sf_regret'] for r in items]
        dk_wins = sum(1 for r in items if r['t_dk_us'] < r['t_sf_us'])
        sf_wins = len(items) - dk_wins
        lines.append(f"| {pattern} | {len(items)} | {np.median(dk_r):.3f} | {np.median(sf_r):.3f} | "
                     f"{dk_wins} | {sf_wins} |\n")
    lines.append("\n")
    
    # Decision
    lines.append(f"## Decision\n\n")
    
    # Conditions for thesis to hold:
    # - DK is suboptimal in non-trivial fraction (≥10%) OR
    # - SF is suboptimal in non-trivial fraction (≥10%)
    # AND
    # - at least one regime where DK loses to SF, and one where SF loses to DK
    
    dk_dominates = (sf_bad_5 >= 0.5)  # SF is bad in majority of cases → DK alone is fine
    sf_dominates = (dk_bad_5 >= 0.5)  # DK is bad in majority → SF alone is fine
    routing_valuable = (dk_bad_5 >= 0.10) and (sf_bad_5 >= 0.10)
    
    has_dk_wins = sum(1 for r in valid if r['t_dk_us'] < r['t_sf_us']) > 0
    has_sf_wins = sum(1 for r in valid if r['t_sf_us'] < r['t_dk_us']) > 0
    has_split_regime = has_dk_wins and has_sf_wins
    
    lines.append(f"- DK regret > 5% in {dk_bad_5:.1%} of cases\n")
    lines.append(f"- SF regret > 5% in {sf_bad_5:.1%} of cases\n")
    lines.append(f"- DK wins in some regime: {has_dk_wins}\n")
    lines.append(f"- SF wins in some regime: {has_sf_wins}\n\n")
    
    if dk_dominates and not has_sf_wins:
        verdict = "RED — fixed-DK dominates; SF v1 has no profitable regime; routing thesis dead"
    elif sf_dominates and not has_dk_wins:
        verdict = "RED — fixed-SF dominates; DK has no profitable regime; routing thesis dead"
    elif not has_split_regime:
        verdict = "YELLOW — only one implementation ever wins; routing has limited value"
    elif routing_valuable:
        verdict = "GREEN — both have profitable regimes, routing has measurable value, thesis holds"
    else:
        verdict = "YELLOW — partial value; one impl close to optimal in most cases but routing helps in tail"
    
    lines.append(f"### Verdict: **{verdict}**\n\n")
    
    if "GREEN" in verdict:
        lines.append("→ Continue v2 plan: build routing framework, expand implementation set\n")
    elif "YELLOW" in verdict:
        lines.append("→ v2 plan needs scope adjustment:\n")
        lines.append("  - Routing thesis still has value but needs to be narrowed\n")
        lines.append("  - Consider what regime CATFuse routing is specifically valuable in\n")
        lines.append("  - May need to add cross-site / Helios-style impls before routing has paper value\n")
    else:
        lines.append("→ v2 thesis at risk:\n")
        lines.append("  - Need to redesign SF or add fundamentally different implementations\n")
        lines.append("  - Possibly pivot from 'sparsity-aware routing' to other niche\n")
        lines.append("  - DO NOT continue routing implementation work until thesis revalidated\n")
    
    lines.append("\n## Notes\n\n")
    lines.append("- Current SF v1 has known issues (group-level skip not effective)\n")
    lines.append("- This experiment uses SF v1 as-is; if thesis fails here it may still hold\n")
    lines.append("  with improved SF (v2) or different impl set\n")
    lines.append("- Trained checkpoint sparsity may differ from synthetic patterns tested here;\n")
    lines.append("  verify on real model after Phase 0 if GREEN\n")
    
    out_path = os.path.join(out_dir, 'decision.md')
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"\n[saved] {out_path}")
    
    # Console verdict
    print(f"\n{'='*70}")
    print(f"PHASE 0 VERDICT")
    print(f"{'='*70}")
    print(verdict)
    print(f"DK suboptimal: {dk_bad_5:.1%} of cases (>5% regret)")
    print(f"SF suboptimal: {sf_bad_5:.1%} of cases (>5% regret)")
    print(f"DK wins in some regime: {has_dk_wins}")
    print(f"SF wins in some regime: {has_sf_wins}")


def save_results(results, path):
    keys = list(results[0].keys())
    arr_dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        if k in ('shape', 'pattern', 'winner'):
            arr_dict[k] = np.array(vals, dtype=object)
        else:
            arr_dict[k] = np.array(vals, dtype=np.float64)
    np.savez(path, **arr_dict)
    print(f"[saved] {path}")


def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    valid = [r for r in results 
             if not np.isnan(r['dk_regret']) and not np.isnan(r['sf_regret'])]
    if not valid:
        return
    
    # Plot 1: DK vs SF regret scatter, colored by pattern
    fig, ax = plt.subplots(figsize=(8, 8))
    pattern_colors = {'bernoulli': 'blue', 'blocky': 'red', 'channel': 'green', 'temporal': 'orange'}
    for pattern, color in pattern_colors.items():
        items = [r for r in valid if r['pattern'] == pattern]
        if not items:
            continue
        dk_r = [r['dk_regret'] for r in items]
        sf_r = [r['sf_regret'] for r in items]
        ax.scatter(dk_r, sf_r, c=color, alpha=0.6, label=pattern, s=50)
    ax.axhline(1.05, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
    ax.axvline(1.05, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('DK regret (DK_time / oracle)')
    ax.set_ylabel('SF regret (SF_time / oracle)')
    ax.set_title('Per-config regret: each point is one (shape, sparsity, pattern)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_scatter.pdf'))
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_scatter.png'), dpi=150)
    plt.close(fig)
    
    # Plot 2: per-shape, per-pattern, regret vs sparsity
    fig, axes = plt.subplots(len(SHAPES), len(PATTERNS), figsize=(16, 4 * len(SHAPES)),
                              sharey=True)
    for i, (shape_name, *_) in enumerate(SHAPES):
        for j, pattern in enumerate(PATTERNS):
            ax = axes[i, j] if len(SHAPES) > 1 else axes[j]
            items = sorted(
                [r for r in valid if r['shape'] == shape_name and r['pattern'] == pattern],
                key=lambda x: x['actual_sparsity'],
            )
            if not items:
                continue
            sps = [r['actual_sparsity'] for r in items]
            dk_rs = [r['dk_regret'] for r in items]
            sf_rs = [r['sf_regret'] for r in items]
            ax.plot(sps, dk_rs, '-o', label='DK', color='blue')
            ax.plot(sps, sf_rs, '-s', label='SF', color='red')
            ax.axhline(1.05, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'{shape_name} / {pattern}')
            ax.set_xlabel('Sparsity')
            if j == 0:
                ax.set_ylabel('Regret (impl_time / oracle)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_curves.pdf'))
    fig.savefig(os.path.join(out_dir, 'figures', 'regret_curves.png'), dpi=150)
    plt.close(fig)
    
    print(f"[saved] {out_dir}/figures/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Test only 1 shape')
    parser.add_argument('--out_dir', default='artifacts/phase0')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    
    if args.quick:
        shapes = SHAPES[:1]
        sparsities = SPARSITY_LEVELS[::2]
        patterns = PATTERNS
    else:
        shapes = SHAPES
        sparsities = SPARSITY_LEVELS
        patterns = PATTERNS
    
    print("=" * 80)
    print("CATFuse v2 — Phase 0 Oracle Gap Microbenchmark")
    print("=" * 80)
    print(f"Device:        {DEVICE}")
    print(f"T x B:         {T_TIME} x {B}")
    print(f"Shapes:        {len(shapes)}")
    print(f"Sparsities:    {len(sparsities)}")
    print(f"Patterns:      {patterns}")
    print(f"Total configs: {len(shapes) * len(sparsities) * len(patterns)}")
    print()
    
    torch.backends.cudnn.benchmark = True
    
    results = run_sweep(shapes, sparsities, patterns)
    
    save_results(results, os.path.join(args.out_dir, 'oracle_gap_data.npz'))
    plot_results(results, args.out_dir)
    analyze(results, args.out_dir)


if __name__ == '__main__':
    main()