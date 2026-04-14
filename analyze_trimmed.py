import json, glob, statistics, sys
from pathlib import Path

def trimmed_stats(values):
    """12 numbers in, drop max & min, return (mean, std, n)."""
    sv = sorted(values)
    trimmed = sv[1:-1]
    return statistics.mean(trimmed), statistics.stdev(trimmed), len(trimmed)

def load_speedup(path):
    """Try several JSON shapes used by our scripts."""
    d = json.load(open(path))
    # conv_lif_min_v2, conv_bn_lif_min, add_bn_lif_min: top-level wall_clock
    if 'wall_clock' in d:
        return (
            d['wall_clock']['fusion']['median_ms'],
            d['wall_clock']['no_fusion']['median_ms'],
            d['wall_clock']['speedup_no_fusion_over_fusion'],
        )
    # linear_lif_k_sweep_v3: pick K=16 entry from k_sweep list
    if 'k_sweep' in d:
        for entry in d['k_sweep']:
            if entry['K'] == 16:
                return (
                    entry['wall_clock_fusion']['median_ms'],
                    d['no_fusion_baseline']['wall_clock']['median_ms'],
                    entry['speedup_no_fusion_over_fusion'],
                )
    # add_lif_k_sweep: t_sweep entries, pick T=16
    if 't_sweep' in d:
        for entry in d['t_sweep']:
            if entry['T'] == 16:
                return (
                    entry['wall_clock_fusion']['median_ms'],
                    entry['wall_clock_no_fusion']['median_ms'],
                    entry['speedup_no_fusion_over_fusion'],
                )
    raise ValueError(f"Unknown JSON shape in {path}")

experiments = {
    'conv_lif':       'results/a100_repeat/conv_lif_a100_run*.json',
    'conv_bn_lif':    'results/a100_repeat/conv_bn_lif_a100_run*.json',
    'linear_lif':     'results/a100_repeat/linear_lif_a100_B256_run*.json',
    'add_lif':        'results/a100_repeat/add_lif_a100_run*.json',
    'add_bn_lif':     'results/a100_repeat/add_bn_lif_a100_run*.json',
}

print(f"{'experiment':<20} {'n':>3} {'fus_mean':>10} {'fus_std':>9} "
      f"{'nof_mean':>10} {'nof_std':>9} {'speedup_mean':>14} {'speedup_std':>13}")
print('-' * 100)

for name, pattern in experiments.items():
    files = sorted(glob.glob(pattern))
    if len(files) < 3:
        print(f"{name:<20} ONLY {len(files)} files found, skipping")
        continue
    fus_list, nof_list, sp_list = [], [], []
    for f in files:
        try:
            fus, nof, sp = load_speedup(f)
            fus_list.append(fus)
            nof_list.append(nof)
            sp_list.append(sp)
        except Exception as e:
            print(f"  ! {f}: {e}")
    if len(fus_list) < 3:
        continue
    fus_m, fus_s, n = trimmed_stats(fus_list)
    nof_m, nof_s, _ = trimmed_stats(nof_list)
    sp_m, sp_s, _ = trimmed_stats(sp_list)
    print(f"{name:<20} {n:>3} {fus_m:>10.4f} {fus_s:>9.4f} "
          f"{nof_m:>10.4f} {nof_s:>9.4f} {sp_m:>14.4f} {sp_s:>13.4f}")

print()
print("Note: trimmed mean drops 1 max + 1 min from N runs, then averages the rest.")
