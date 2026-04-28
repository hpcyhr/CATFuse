"""
Test 71: SF Path B Diagnosis
============================

Purpose
-------
We discovered that v1 SF has TWO independent kernel paths:

  Path A: catfuse.sparseflow.fused_conv_bn_lif_kernel
          sparse_fused_conv_bn_lif_forward
          ├─ NCHW layout
          ├─ Fused Conv+BN+LIF in one kernel
          └─ for g in range(NUM_GROUPS): tl.dot(masked, masked)
             (FAKE skip — tl.dot always executes)

  Path B: catfuse.sparseflow.sparse_conv2d_kernel  
          sparse_conv2d_forward
          ├─ NHWC layout
          ├─ Conv only (BN/LIF separately)
          └─ for g: if g_active != 0: tl.dot(...)
             (REAL skip — tl.dot conditionally executed)

Test 70 only measured Path A and got RED. This test measures Path B,
which has the real skip mechanism, to see if it changes with sparsity.

If Path B latency varies meaningfully with sparsity:
  → Real skip mechanism works on V100 + Triton
  → SF v2 = port Path B's mechanism into Path A's fused kernel
If Path B latency does NOT vary:
  → Real skip is also broken in this environment
  → SF v2 needs a fundamentally different approach

Usage
-----
    python tests/test_71_sf_path_b.py
"""

import os, sys, time, statistics, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 42

# ============================================================
# Setup — same shape grid as test_70 for comparable numbers
# ============================================================

SHAPES = [
    ('layer1.0', 64,  64, 56, 56),
    ('layer2.0', 64,  128, 28, 28),
    ('layer3.0', 128, 256, 14, 14),
    ('layer4.0', 256, 512, 7,  7),
]

# Realistic SNN sparsity range, with extreme cases
SPARSITY_LEVELS = [0.0, 0.50, 0.85, 0.90, 0.95, 0.98, 0.99]

# Test both kernel sizes (Path B has separate 1x1 and 3x3 kernels)
KERNELS = [1, 3]

# Patterns
PATTERNS = ['bernoulli', 'blocky', 'channel']

T_TIME = 4
B = 1

N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3


# ============================================================
# Sparsity injection (same as test_70)
# ============================================================

def make_input(shape, sparsity, pattern, generator):
    T, B, C, H, W = shape
    keep = 1.0 - sparsity
    
    if pattern == 'bernoulli':
        rand = torch.rand(shape, device=DEVICE, generator=generator)
        x = (rand < keep).float()
    elif pattern == 'blocky':
        TILE = 8
        Hb = (H + TILE - 1) // TILE
        Wb = (W + TILE - 1) // TILE
        block_rand = torch.rand(T, B, C, Hb, Wb, device=DEVICE, generator=generator)
        block_active = (block_rand < keep).float()
        block_active = block_active.repeat_interleave(TILE, dim=3).repeat_interleave(TILE, dim=4)
        x = block_active[:, :, :, :H, :W].contiguous()
    elif pattern == 'channel':
        chan_rand = torch.rand(T, B, C, device=DEVICE, generator=generator)
        chan_active = (chan_rand < keep).float()
        x = chan_active.unsqueeze(-1).unsqueeze(-1).expand(T, B, C, H, W).contiguous()
    return x


# ============================================================
# Bench DK (cuDNN)
# ============================================================

def bench_dk(x_5d, weight, kernel_size):
    T, B_size, C_in, H, W = x_5d.shape
    pad = kernel_size // 2
    x_4d = x_5d.reshape(T * B_size, C_in, H, W)
    
    def _fwd():
        return F.conv2d(x_4d, weight, stride=1, padding=pad)
    
    return _measure(_fwd)


# ============================================================
# Bench SF Path B (sparse_conv2d_forward)
# ============================================================

def bench_sf_path_b(x_5d, weight, kernel_size):
    """Test Path B: sparse_conv2d_forward (NHWC, real conditional skip)."""
    try:
        from catfuse.sparseflow.sparse_conv2d_kernel import sparse_conv2d_forward
    except ImportError as e:
        return None, f"Path B import fail: {e}"
    
    T, B_size, C_in, H, W = x_5d.shape
    pad = kernel_size // 2
    x_4d = x_5d.reshape(T * B_size, C_in, H, W)
    
    # Path B accepts standard NCHW input internally converts to NHWC
    def _fwd():
        try:
            result = sparse_conv2d_forward(
                x_4d, weight, bias=None,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            )
            if isinstance(result, tuple):
                return result[0]
            return result
        except Exception as e:
            raise RuntimeError(f"Path B call: {e}")
    
    # Verify it runs
    try:
        with torch.no_grad():
            _fwd()
    except Exception as e:
        return None, f"Path B fwd failed: {e}"
    
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
    
    results = []
    
    print(f"{'='*120}")
    print(f"{'Shape':<14}{'K':<3}{'Pattern':<11}{'Sp':<6}{'T_DK_us':<10}{'T_SF_B_us':<12}{'SF_B/DK':<10}{'oracle':<10}")
    print('-' * 120)
    
    for shape_name, C_in, C_out, H, W in SHAPES:
        for kernel_size in KERNELS:
            # Fix weights for this (shape, kernel) combo
            weight = torch.randn(C_out, C_in, kernel_size, kernel_size,
                                  device=DEVICE, generator=weight_gen) * 0.1
            
            for pattern in PATTERNS:
                for sp in SPARSITY_LEVELS:
                    local_gen = torch.Generator(device=DEVICE).manual_seed(
                        SEED + hash((shape_name, kernel_size, pattern, sp)) % 100000
                    )
                    
                    shape_5d = (T_TIME, B, C_in, H, W)
                    x = make_input(shape_5d, sp, pattern, local_gen)
                    actual_sp = 1.0 - x.float().mean().item()
                    
                    # Bench DK
                    try:
                        t_dk = bench_dk(x, weight, kernel_size)
                    except Exception as e:
                        t_dk = float('nan')
                    
                    # Bench SF Path B
                    try:
                        t_sf_b, err = bench_sf_path_b(x, weight, kernel_size)
                        if t_sf_b is None:
                            t_sf_b = float('nan')
                    except Exception as e:
                        t_sf_b = float('nan')
                    
                    if not np.isnan(t_dk) and not np.isnan(t_sf_b):
                        ratio = t_sf_b / t_dk
                        oracle = min(t_dk, t_sf_b)
                    else:
                        ratio = float('nan')
                        oracle = t_dk if not np.isnan(t_dk) else t_sf_b
                    
                    print(f"{shape_name:<14}{kernel_size:<3}{pattern:<11}{actual_sp:<6.3f}"
                          f"{t_dk:<10.1f}{t_sf_b:<12.1f}{ratio:<10.3f}{oracle:<10.1f}")
                    
                    results.append({
                        'shape': shape_name,
                        'kernel_size': kernel_size,
                        'pattern': pattern,
                        'target_sp': sp,
                        'actual_sp': actual_sp,
                        't_dk_us': t_dk,
                        't_sf_b_us': t_sf_b,
                        'ratio': ratio,
                    })
    
    return results


def analyze(results, out_dir):
    """Analyze whether Path B latency varies with sparsity."""
    
    valid = [r for r in results if not np.isnan(r['t_sf_b_us']) and not np.isnan(r['t_dk_us'])]
    
    if not valid:
        print("\n[ERROR] No valid measurements.")
        return
    
    # Group by (shape, kernel_size, pattern)
    groups = {}
    for r in valid:
        key = (r['shape'], r['kernel_size'], r['pattern'])
        groups.setdefault(key, []).append(r)
    
    lines = []
    lines.append("# Test 71 — SF Path B vs Sparsity Decision\n\n")
    lines.append("## Question\n")
    lines.append("Does Path B (sparse_conv2d_forward, with real conditional skip) latency vary with sparsity?\n\n")
    lines.append("If YES → SF v2 design = port Path B's `if g_active != 0:` mechanism into Path A's fused kernel.\n")
    lines.append("If NO → Path B's real skip is also ineffective on V100+Triton, need fundamentally different approach.\n\n")
    
    lines.append("## Per-(shape, kernel, pattern) Path B latency variation\n\n")
    lines.append("| Shape | K | Pattern | T_min | T_max | max/min ratio | Conclusion |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    
    variations = []
    sf_wins_dk = []  # cases where SF Path B wins DK
    
    for key, items in groups.items():
        shape, k, pattern = key
        ts = [r['t_sf_b_us'] for r in items]
        t_min, t_max = min(ts), max(ts)
        var = t_max / t_min
        variations.append((key, var))
        
        if var > 1.5:
            conclusion = "✓ varies"
        elif var < 1.2:
            conclusion = "✗ flat"
        else:
            conclusion = "△ slight"
        
        lines.append(f"| {shape} | {k} | {pattern} | {t_min:.1f} | {t_max:.1f} | {var:.2f}× | {conclusion} |\n")
        
        for r in items:
            if r['t_sf_b_us'] < r['t_dk_us']:
                sf_wins_dk.append((shape, k, pattern, r['actual_sp'], r['t_sf_b_us'], r['t_dk_us']))
    
    lines.append("\n")
    
    # SF wins DK cases
    lines.append(f"## SF Path B winning over DK\n\n")
    lines.append(f"Out of {len(valid)} configs, SF Path B beats DK in **{len(sf_wins_dk)}**.\n\n")
    if sf_wins_dk:
        lines.append("### Specific winning configs:\n\n")
        lines.append("| Shape | K | Pattern | Sparsity | T_SF_B | T_DK |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for shape, k, pattern, sp, tsf, tdk in sf_wins_dk:
            lines.append(f"| {shape} | {k} | {pattern} | {sp:.3f} | {tsf:.1f} | {tdk:.1f} |\n")
    
    # Decision
    n_strong = sum(1 for _, v in variations if v > 1.5)
    n_total = len(variations)
    median_var = np.median([v for _, v in variations])
    n_sf_wins = len(sf_wins_dk)
    
    lines.append(f"\n## Decision\n\n")
    lines.append(f"- Configs with >1.5x latency variation across sparsity: {n_strong}/{n_total}\n")
    lines.append(f"- Median variation: {median_var:.2f}×\n")
    lines.append(f"- SF Path B wins DK count: {n_sf_wins}/{len(valid)}\n\n")
    
    if n_sf_wins == 0 and median_var < 1.2:
        verdict = "RED — Path B's real skip is also ineffective. Even with `if g_active != 0:`, SF doesn't beat DK and doesn't vary with sparsity. Triton/V100 cannot effectively exploit fine-grained group skip."
        recommendation = """
**SF v2 needs fundamentally different approach.** Possible directions:
- Hybrid SF: keep cuDNN conv, only fuse post-op (BN+Add+LIF). Sacrifice sparsity utilization for fusion savings.
- Tile-level branching: dispatch entire CTAs to different sub-kernels based on prescan
- Bit-packed input: change data representation (Helios-style)
"""
    elif n_sf_wins > 0 and median_var >= 1.2:
        verdict = "GREEN — Path B's real skip mechanism IS effective. SF v2 = port this into fused kernel."
        recommendation = """
**SF v2 design plan:**
1. Take fused_conv_bn_lif_kernel.py (Path A)
2. Replace `for g in range(NUM_GROUPS): tl.dot(masked,masked)` with Path B's `for g: if g_active != 0: tl.dot(...)`
3. Add prescan granularity tuning (smaller tiles for higher zero-tile probability)
4. Merge with existing BN+LIF fusion logic
"""
    else:
        verdict = "YELLOW — Path B has some sparsity-related variation but doesn't consistently beat DK."
        recommendation = """
**SF v2 needs combined approach:**
- Port Path B's real skip (helps in some regimes)
- Add tile-level dispatch (stronger skip at high sparsity)
- Accept that DK will dominate medium-sparsity regimes
- Focus on extreme high-sparsity regimes (>0.95) where real skip pays off
"""
    
    lines.append(f"### Verdict: **{verdict}**\n")
    lines.append(recommendation)
    
    out_path = os.path.join(out_dir, 'path_b_decision.md')
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"\n[saved] {out_path}")
    
    # Console summary
    print(f"\n{'='*70}")
    print(f"PATH B VERDICT")
    print(f"{'='*70}")
    print(verdict)
    print(f"Configs with >1.5x sparsity variation: {n_strong}/{n_total}")
    print(f"Median variation: {median_var:.2f}x")
    print(f"SF Path B wins DK: {n_sf_wins}/{len(valid)}")


def save_results(results, path):
    keys = list(results[0].keys())
    arr_dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        if k in ('shape', 'pattern'):
            arr_dict[k] = np.array(vals, dtype=object)
        else:
            arr_dict[k] = np.array(vals, dtype=np.float64)
    np.savez(path, **arr_dict)
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='artifacts/phase0')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Test 71 — SF Path B Diagnosis")
    print("=" * 80)
    print(f"Device:      {DEVICE}")
    print(f"Shapes:      {len(SHAPES)}")
    print(f"Kernels:     {KERNELS}")
    print(f"Patterns:    {PATTERNS}")
    print(f"Sparsities:  {len(SPARSITY_LEVELS)}")
    print(f"Total:       {len(SHAPES) * len(KERNELS) * len(PATTERNS) * len(SPARSITY_LEVELS)}")
    print()
    
    torch.backends.cudnn.benchmark = True
    
    results = run_sweep()
    save_results(results, os.path.join(args.out_dir, 'path_b_data.npz'))
    analyze(results, args.out_dir)


if __name__ == '__main__':
    main()