"""
Test 63: Sparsity Sensitivity Microbenchmark
============================================

Purpose
-------
GPT review 指出: 我们必须实测 T_SF / T_DK 是否真的随 input sparsity 变化.
如果不变化, 整个 "sparsity-driven dispatch" thesis 就站不住, sFR 的 y 轴应该改
成 shape × operator × hardware. 这个实验直接决定 v2 的 thesis.

Setup
-----
- 固定 operator shape (Conv-BN-LIF, 几个代表性 shape)
- 固定 conv weights (随机初始化, 同一 seed)
- 固定 BN params (identity: scale=1, bias=0, 避免 BN 引入额外噪声)
- 人工注入可控 sparsity: ρ ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}
  注: ρ 是 sparsity, 即 1 - spike_rate. ρ=0.99 意味着 99% 位置是 0.
- 测两条路径的 latency:
    DenseKeep: F.conv2d (cuDNN) + Triton lif_sequential
    SparseFlow: sparse_fused_conv_bn_lif_forward (Triton 全融合)
- 输出 T_SF / T_DK 比率 vs sparsity 曲线

Decision Rule
-------------
  如果 T_SF/T_DK 随 sparsity 显著变化 (跨 sparsity 比率变化 > 1.5x):
      → thesis 站住, sFR y 轴是 sparsity, 继续 v2 计划
  如果 T_SF/T_DK 几乎不变 (< 1.2x 变化):
      → thesis 死, sFR y 轴改成 shape × operator
  介于之间:
      → thesis 部分成立, 需要更精细分析

Notes
-----
- 关键: 输入 tensor 必须是真 binary (0/1), 不是连续的 0.5 之类
  因为 SparseFlow 的 prescan 是基于 tile 内 nnz 决策的
- 用足够多的 warmup + iter 抑制 jitter
- 所有 shape 测多次取 median

Usage
-----
    python tests/test_63_sparsity_sensitivity.py
    
    # 只测一个 shape 快速验证
    python tests/test_63_sparsity_sensitivity.py --quick

Outputs
-------
    artifacts/sparsity_v2/sensitivity_data.npz
    artifacts/sparsity_v2/figures/fig_sensitivity.pdf
    artifacts/sparsity_v2/figures/fig_sensitivity.png
    artifacts/sparsity_v2/sensitivity_decision.md  ← 关键决策报告
"""

import os, sys, time, statistics, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 42

# ============================================================
# 配置
# ============================================================

# T 步, B batch (CATFuse 典型)
T_TIME = 4
B = 1

# 测的 shape: SEW-RN18 真实 layer 中代表性几个
# (C_in, C_out, H, W)
SHAPES = [
    # name, C_in, C_out, H, W
    ('layer1.0.conv1',  64,  64, 56, 56),  # block 1, mid problem size
    ('layer2.0.conv1',  64, 128, 28, 28),  # block 2 entry, downsample-like
    ('layer3.0.conv1', 128, 256, 14, 14),  # block 3, getting smaller
    ('layer4.0.conv1', 256, 512,  7,  7),  # block 4, smallest
]

# Sparsity sweep
SPARSITY_SCAN = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.99]

# Bench params
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3

# LIF params
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0


# ============================================================
# Sparsity injection (binary)
# ============================================================

def make_sparse_binary_input(shape, sparsity, generator):
    """生成 binary tensor, 满足指定 sparsity.
    
    sparsity ∈ [0, 1] 是 0 的比例; 1 - sparsity 是 1 的比例.
    保证是真 binary (元素只有 0 或 1), 模拟 spike 输入.
    """
    keep_prob = 1.0 - sparsity
    rand = torch.rand(shape, device=DEVICE, generator=generator)
    binary = (rand < keep_prob).float()
    return binary


# ============================================================
# DenseKeep path
# ============================================================

def bench_densekeep(x, weight, bn_scale, bn_bias, conv_bias):
    """DenseKeep: F.conv2d (cuDNN) + Triton lif_sequential.
    
    x:           [T, B, C_in, H, W]
    weight:      [C_out, C_in, kH, kW]
    bn_scale:    [C_out]
    bn_bias:     [C_out]
    conv_bias:   [C_out] or None
    
    返回 median latency in microseconds.
    """
    # Pre-fold BN into conv weight + bias (跟 v1 DenseKeep 一致)
    # w_fused = weight * bn_scale.view(C_out, 1, 1, 1)
    # b_fused = (conv_bias - 0) * bn_scale + bn_bias  (BN running_mean=0 假设)
    w_fused = weight * bn_scale.view(-1, 1, 1, 1)
    if conv_bias is not None:
        b_fused = conv_bias * bn_scale + bn_bias
    else:
        b_fused = bn_bias.clone()
    
    T, B_size = x.shape[0], x.shape[1]
    C_in = x.shape[2]
    H, W = x.shape[3], x.shape[4]
    C_out = weight.shape[0]
    kH = weight.shape[2]
    pad = kH // 2
    H_out = (H + 2 * pad - kH) + 1
    W_out = H_out
    
    x_4d = x.reshape(T * B_size, C_in, H, W)
    
    # Try import lif_sequential, fallback to pure pytorch
    try:
        from catfuse.sparseflow.lif_seq_kernel import lif_sequential
        use_triton_lif = True
    except ImportError:
        use_triton_lif = False
    
    def _forward():
        z_4d = F.conv2d(x_4d, w_fused, bias=b_fused, stride=1, padding=pad)
        z = z_4d.reshape(T, B_size, C_out, H_out, W_out)
        v_in = torch.zeros(B_size, C_out, H_out, W_out, device=DEVICE, dtype=torch.float32)
        if use_triton_lif:
            spikes, v_out = lif_sequential(
                z, v_in, tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
            )
        else:
            v = v_in
            spikes_list = []
            for t in range(T):
                v = v + (z[t] - (v - V_RESET)) / TAU
                spike = (v >= V_TH).to(z.dtype)
                v = v * (1.0 - spike) + V_RESET * spike
                spikes_list.append(spike)
            spikes = torch.stack(spikes_list, dim=0)
            v_out = v
        return spikes, v_out
    
    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _forward()
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            with torch.no_grad():
                _forward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1e6)  # us
    
    return statistics.median(times)


# ============================================================
# SparseFlow path
# ============================================================

def bench_sparseflow(x, weight, bn_scale, bn_bias, conv_bias):
    """SparseFlow: sparse_fused_conv_bn_lif_forward (Triton 全融合).
    
    注意接口: SF kernel 一次处理 [N, C_in, H, W] (N = T*B), 内部循环不展开 T.
    所以我们要在外层循环 T 步, 每步 chain v_prev -> v_next.
    """
    try:
        from catfuse.sparseflow.fused_conv_bn_lif_kernel import sparse_fused_conv_bn_lif_forward
    except ImportError as e:
        print(f"  [SF kernel import failed]: {e}")
        return None
    
    T, B_size = x.shape[0], x.shape[1]
    C_in = x.shape[2]
    H, W = x.shape[3], x.shape[4]
    C_out = weight.shape[0]
    kH = weight.shape[2]
    
    H_out = H  # padding=1 for k=3, stride=1
    W_out = W
    
    def _forward():
        v = torch.zeros(B_size, C_out, H_out, W_out, device=DEVICE, dtype=torch.float32)
        spikes_per_step = []
        for t in range(T):
            x_t = x[t]  # [B, C_in, H, W]
            try:
                result = sparse_fused_conv_bn_lif_forward(
                    x_t, v, weight,
                    bias=conv_bias,
                    bn_scale=bn_scale,
                    bn_bias=bn_bias,
                    tau=TAU,
                    v_threshold=V_TH,
                    v_reset=V_RESET,
                    kernel_size=kH,
                )
                # result is (spike, v_next, sparse_ms) typically
                if isinstance(result, tuple):
                    spike = result[0]
                    v = result[1]
                else:
                    spike = result
            except Exception as e:
                # SF kernel failed — return None to signal can't bench this config
                raise
            spikes_per_step.append(spike)
        return torch.stack(spikes_per_step, dim=0), v
    
    # Warmup (also catches errors)
    try:
        for _ in range(min(N_WARMUP, 5)):
            with torch.no_grad():
                _forward()
    except Exception as e:
        print(f"  [SF kernel call failed in warmup]: {type(e).__name__}: {e}")
        return None
    
    # Full warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _forward()
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            with torch.no_grad():
                _forward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1e6)
    
    return statistics.median(times)


# ============================================================
# Sweep
# ============================================================

def run_sweep(shapes, sparsities, seed=SEED):
    """对每个 (shape, sparsity) 测 DK 和 SF 延迟"""
    
    results = []
    
    # 固定 weights generator (跨 shape 不同, 同一 shape 跨 sparsity 相同)
    weight_generator = torch.Generator(device=DEVICE).manual_seed(seed)
    input_generator = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    
    print(f"{'='*100}")
    print(f"{'Shape':<25} {'Sp':>6} {'T_DK(us)':>10} {'T_SF(us)':>10} {'SF/DK':>8} {'DK/SF':>8} {'winner':>8}")
    print('-' * 100)
    
    for shape_name, C_in, C_out, H, W in shapes:
        kH = 3  # 默认 3x3
        
        # 固定 weights for this shape
        weight = torch.randn(C_out, C_in, kH, kH, device=DEVICE, generator=weight_generator) * 0.1
        # BN params: identity (scale=1, bias=0) 避免引入噪声
        bn_scale = torch.ones(C_out, device=DEVICE)
        bn_bias = torch.zeros(C_out, device=DEVICE)
        conv_bias = None  # 简化, 无 conv bias
        
        for sp in sparsities:
            # 固定 input randomness for this (shape, sparsity)
            input_generator_local = torch.Generator(device=DEVICE).manual_seed(seed + hash((shape_name, sp)) % 1000)
            x_4d = make_sparse_binary_input(
                (T_TIME, B, C_in, H, W), sp,
                input_generator_local,
            )
            
            # Sanity check on actual sparsity
            actual_sp = 1.0 - x_4d.float().mean().item()
            
            # Bench DK
            try:
                t_dk = bench_densekeep(x_4d, weight, bn_scale, bn_bias, conv_bias)
            except Exception as e:
                print(f"{shape_name:<25} {sp:>6.2f}  DK FAILED: {e}")
                t_dk = float('nan')
            
            # Bench SF
            try:
                t_sf = bench_sparseflow(x_4d, weight, bn_scale, bn_bias, conv_bias)
                if t_sf is None:
                    t_sf = float('nan')
            except Exception as e:
                print(f"{shape_name:<25} {sp:>6.2f}  SF FAILED: {e}")
                t_sf = float('nan')
            
            # Ratios
            sf_over_dk = (t_sf / t_dk) if (not np.isnan(t_sf) and not np.isnan(t_dk) and t_dk > 0) else float('nan')
            dk_over_sf = (t_dk / t_sf) if (not np.isnan(t_sf) and not np.isnan(t_dk) and t_sf > 0) else float('nan')
            
            # Winner
            if not np.isnan(sf_over_dk):
                winner = 'SF' if sf_over_dk < 1.0 else 'DK'
            else:
                winner = '?'
            
            print(f"{shape_name:<25} {actual_sp:>6.3f} {t_dk:>10.1f} {t_sf:>10.1f} {sf_over_dk:>8.3f} {dk_over_sf:>8.3f} {winner:>8}")
            
            results.append({
                'shape': shape_name,
                'C_in': C_in,
                'C_out': C_out,
                'H': H,
                'W': W,
                'target_sparsity': sp,
                'actual_sparsity': actual_sp,
                't_dk_us': t_dk,
                't_sf_us': t_sf,
                'sf_over_dk': sf_over_dk,
                'dk_over_sf': dk_over_sf,
            })
    
    return results


# ============================================================
# Save & Plot
# ============================================================

def save_results(results, out_path):
    keys = list(results[0].keys())
    arr_dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        if k == 'shape':
            arr_dict[k] = np.array(vals, dtype=object)
        else:
            arr_dict[k] = np.array(vals, dtype=np.float64 if k != 'shape' else None)
    np.savez(out_path, **arr_dict)
    print(f"\n[saved] {out_path}")


def plot_sensitivity(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Group by shape
    shape_set = []
    for r in results:
        if r['shape'] not in shape_set:
            shape_set.append(r['shape'])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    cmap = plt.cm.tab10
    
    # Left: T_SF / T_DK vs sparsity
    ax = axes[0]
    for i, shape_name in enumerate(shape_set):
        sps, ratios = [], []
        for r in results:
            if r['shape'] == shape_name and not np.isnan(r['sf_over_dk']):
                sps.append(r['actual_sparsity'])
                ratios.append(r['sf_over_dk'])
        ax.plot(sps, ratios, '-o', label=shape_name, color=cmap(i), linewidth=2, markersize=6)
    
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='SF/DK = 1 (tie)')
    ax.set_xlabel('Input sparsity')
    ax.set_ylabel('T_SF / T_DK (lower = SF wins)')
    ax.set_title('SparseFlow vs DenseKeep latency ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # log scale to see big variation
    
    # Right: absolute T_SF and T_DK separately
    ax = axes[1]
    for i, shape_name in enumerate(shape_set):
        sps, t_dks, t_sfs = [], [], []
        for r in results:
            if r['shape'] == shape_name:
                if not np.isnan(r['t_dk_us']):
                    sps.append(r['actual_sparsity'])
                    t_dks.append(r['t_dk_us'])
        # plot DK (solid) and SF (dashed)
        sps_dk = [r['actual_sparsity'] for r in results if r['shape'] == shape_name and not np.isnan(r['t_dk_us'])]
        t_dk_l = [r['t_dk_us'] for r in results if r['shape'] == shape_name and not np.isnan(r['t_dk_us'])]
        sps_sf = [r['actual_sparsity'] for r in results if r['shape'] == shape_name and not np.isnan(r['t_sf_us'])]
        t_sf_l = [r['t_sf_us'] for r in results if r['shape'] == shape_name and not np.isnan(r['t_sf_us'])]
        ax.plot(sps_dk, t_dk_l, '-', color=cmap(i), label=f'{shape_name} DK', linewidth=2)
        ax.plot(sps_sf, t_sf_l, '--', color=cmap(i), label=f'{shape_name} SF', linewidth=2)
    
    ax.set_xlabel('Input sparsity')
    ax.set_ylabel('Latency (us)')
    ax.set_title('Absolute latency: DK (solid) vs SF (dashed)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Decision
# ============================================================

def make_decision(results, out_path):
    """关键判断: thesis 是否成立"""
    
    lines = []
    lines.append("# CATFuse v2 - Sparsity Sensitivity Decision Report\n\n")
    lines.append("## Question\n")
    lines.append("Does T_SF / T_DK ratio vary with input sparsity?\n\n")
    lines.append("If YES (>1.5x variation): sFR y-axis is sparsity → v2 thesis holds.\n")
    lines.append("If NO (<1.2x variation): sFR y-axis is shape × operator → thesis revision needed.\n\n")
    
    lines.append("## Per-shape variation analysis\n\n")
    lines.append("| Shape | min(T_SF/T_DK) | max(T_SF/T_DK) | max/min ratio | conclusion |\n")
    lines.append("|---|---|---|---|---|\n")
    
    shape_set = []
    for r in results:
        if r['shape'] not in shape_set:
            shape_set.append(r['shape'])
    
    overall_variations = []
    for shape_name in shape_set:
        ratios = [r['sf_over_dk'] for r in results 
                  if r['shape'] == shape_name and not np.isnan(r['sf_over_dk'])]
        if not ratios:
            continue
        r_min = min(ratios)
        r_max = max(ratios)
        variation = r_max / r_min if r_min > 0 else float('inf')
        overall_variations.append(variation)
        
        if variation > 1.5:
            conclusion = "✓ sparsity matters"
        elif variation < 1.2:
            conclusion = "✗ shape-dominated"
        else:
            conclusion = "△ partial"
        
        lines.append(f"| {shape_name} | {r_min:.3f} | {r_max:.3f} | {variation:.2f}× | {conclusion} |\n")
    
    lines.append("\n## Overall decision\n\n")
    
    if not overall_variations:
        lines.append("**ERROR**: No valid measurements. SF kernel may have failed to run.\n")
    else:
        median_var = np.median(overall_variations)
        max_var = max(overall_variations)
        n_strong = sum(1 for v in overall_variations if v > 1.5)
        n_total = len(overall_variations)
        
        lines.append(f"- Number of shapes tested: {n_total}\n")
        lines.append(f"- Shapes with >1.5x variation: {n_strong}/{n_total}\n")
        lines.append(f"- Median variation: {median_var:.2f}×\n")
        lines.append(f"- Max variation: {max_var:.2f}×\n\n")
        
        if n_strong >= n_total * 0.75:
            lines.append("### **GREEN**: thesis 站住\n\n")
            lines.append("T_SF/T_DK 在大部分 shape 上随 sparsity 显著变化 (>1.5x).\n")
            lines.append("**v2 plan 可以按 sparsity-driven dispatch 继续推进**.\n")
            lines.append("sFR 的 y 轴是 sparsity, x 轴是 problem size.\n")
        elif n_strong >= n_total * 0.4:
            lines.append("### **YELLOW**: thesis 部分成立\n\n")
            lines.append(f"只有 {n_strong}/{n_total} 的 shape 显示 sparsity 强相关.\n")
            lines.append("v2 sFR 抽象需要扩展 — sparsity 是其中一个维度, shape × operator 也是.\n")
            lines.append("**建议**: paper framing 改成 'sparsity AND shape-aware dispatch'.\n")
        else:
            lines.append("### **RED**: thesis 死了\n\n")
            lines.append(f"只有 {n_strong}/{n_total} 的 shape 显示 sparsity 影响.\n")
            lines.append("**T_SF/T_DK 主要由 shape 决定, sparsity 影响很小**.\n")
            lines.append("v2 plan 需要重写:\n")
            lines.append("- sFR 的 y 轴改成 'memory materialization ratio' 或 'operator group'\n")
            lines.append("- paper framing 从 'sparsity-driven' 改成 'shape-aware fusion dispatch'\n")
            lines.append("- 不能再 claim 'spike sparsity is first-class signal'\n")
            lines.append("- **强烈建议跟老师讨论方向调整, 不要继续在 sparsity-driven 路线投入**\n")
    
    lines.append("\n---\n\n## Notes\n\n")
    lines.append("- 这个实验用 untrained random weights + 人工注入 sparsity, 测试纯 GPU kernel 行为\n")
    lines.append("- 结果反映 SF kernel 实现在不同 sparsity 下的性能特性\n")
    lines.append("- 真实 SNN 的 sparsity 影响可能不止于此 (还有跟 weights 的耦合等), 但本实验给出 lower bound\n")
    lines.append("- 如果 GREEN, 后续仍需用 trained checkpoint 验证 (Layer 1)\n")
    
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"[saved] {out_path}")
    
    # Print to console
    print("\n" + "=" * 70)
    print("SPARSITY SENSITIVITY DECISION")
    print("=" * 70)
    if not overall_variations:
        print("ERROR: No valid measurements")
    else:
        median_var = np.median(overall_variations)
        n_strong = sum(1 for v in overall_variations if v > 1.5)
        n_total = len(overall_variations)
        print(f"Shapes with >1.5x variation: {n_strong}/{n_total}")
        print(f"Median variation: {median_var:.2f}x")
        if n_strong >= n_total * 0.75:
            print("Verdict: GREEN — thesis holds, continue v2 plan")
        elif n_strong >= n_total * 0.4:
            print("Verdict: YELLOW — thesis partial, framing needs broadening")
        else:
            print("Verdict: RED — thesis dead, v2 needs major rework")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='只测一个 shape')
    parser.add_argument('--out_dir', default='artifacts/sparsity_v2')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    
    if args.quick:
        shapes = SHAPES[:1]
        sparsities = SPARSITY_SCAN[::2]  # 只测 6 个 sparsity
    else:
        shapes = SHAPES
        sparsities = SPARSITY_SCAN
    
    print("=" * 100)
    print("CATFuse v2 — T_SF / T_DK vs Sparsity Sensitivity Microbenchmark")
    print("=" * 100)
    print(f"Device:        {DEVICE}")
    print(f"T x B:         {T_TIME} x {B}")
    print(f"Shapes:        {len(shapes)}")
    print(f"Sparsity pts:  {len(sparsities)}")
    print(f"Warmup/Iter:   {N_WARMUP} / {N_ITER} (x{N_REPEAT} repeats)")
    print()
    
    # 全局 cuDNN 配置
    torch.backends.cudnn.benchmark = True  # 让 cuDNN 选最优 algo (DenseKeep 最佳条件)
    torch.backends.cudnn.deterministic = False
    
    # Sweep
    results = run_sweep(shapes, sparsities)
    
    # Save raw
    save_results(results, os.path.join(args.out_dir, 'sensitivity_data.npz'))
    
    # Plot
    plot_sensitivity(results, os.path.join(args.out_dir, 'figures', 'fig_sensitivity.pdf'))
    
    # Decision
    make_decision(results, os.path.join(args.out_dir, 'sensitivity_decision.md'))
    
    print("\nDone.")


if __name__ == '__main__':
    main()