import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
import statistics
import json

device = 'cuda'

# 关键:关掉 cudnn.benchmark,用确定性 algorithm,换取稳定性
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# V100 HBM2 理论带宽 (GB/s)
HBM_BW = 900.0

# 主战场:ResNet18 中层代表性 shape
B, C_in, C_out, H, W = 32, 128, 128, 16, 16
T_list = [4, 8, 16, 32]

# 统计参数
N_WARMUP = 50
N_ITER = 100
N_REPEAT = 11  # 奇数,便于取 median


def cuda_time_one_shot(fn, n_iter):
    """单次测量:连续 n_iter 次,返回每次迭代的平均时间 (ms)。"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def cuda_time_stats(fn, n_iter=N_ITER, n_repeat=N_REPEAT):
    """多次测量,返回 (median, min, max, stdev)。"""
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'max': max(samples),
        'stdev': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'samples': samples,
    }


def make_conv():
    return nn.Conv2d(C_in, C_out, 3, padding=1, bias=False).to(device)


def make_lif(backend):
    return neuron.LIFNode(
        tau=2.0, surrogate_function=surrogate.ATan(),
        step_mode='m', backend=backend,
    ).to(device)


def benchmark_config(T, backend):
    """对单个 (T, backend) 配置做完整测量,返回结果字典。"""
    # 一次性创建所有对象,全程复用
    conv = make_conv()
    lif = make_lif(backend)
    x = torch.randn(T, B, C_in, H, W, device=device)
    with torch.no_grad():
        z_precomputed = conv(x.flatten(0, 1)).reshape(T, B, C_out, H, W).contiguous()

    # 定义三段测量的 step 函数
    def step_total():
        functional.reset_net(lif)
        x_flat = x.flatten(0, 1)
        z_flat = conv(x_flat)
        z = z_flat.reshape(T, B, C_out, H, W)
        lif(z)

    def step_conv():
        x_flat = x.flatten(0, 1)
        z_flat = conv(x_flat)
        z_flat.reshape(T, B, C_out, H, W)

    def step_lif():
        functional.reset_net(lif)
        lif(z_precomputed)

    # 统一大 warmup:按 total 路径跑,让所有状态稳定
    for _ in range(N_WARMUP):
        step_total()
    torch.cuda.synchronize()

    # 三段测量
    total_stats = cuda_time_stats(step_total)
    conv_stats = cuda_time_stats(step_conv)
    lif_stats = cuda_time_stats(step_lif)

    total_ms = total_stats['median']
    conv_ms = conv_stats['median']
    lif_ms = lif_stats['median']

    # 派生指标
    inter_GB = T * B * C_out * H * W * 4 * 2 / 1e9
    lif_traffic_GB = T * B * C_out * H * W * 4 * 4 / 1e9
    lif_BW = lif_traffic_GB / (lif_ms / 1e3)

    # Fuse ceiling:两种模型
    fuse_lb_bw = total_ms - inter_GB / HBM_BW * 1000  # 保守:只省 HBM 往返
    fuse_lb_max = max(conv_ms, lif_ms)                 # 激进:lif 被吸收
    fuse_pct_bw = (1 - fuse_lb_bw / total_ms) * 100
    fuse_pct_max = max(0, (1 - fuse_lb_max / total_ms) * 100)

    sum_ms = conv_ms + lif_ms
    overshoot = (sum_ms - total_ms) / total_ms * 100
    consistent = abs(overshoot) < 10  # sum 与 total 差 <10% 认为自洽

    return {
        'T': T, 'backend': backend,
        'total_ms': total_ms, 'total_stdev': total_stats['stdev'],
        'conv_ms': conv_ms, 'conv_stdev': conv_stats['stdev'],
        'lif_ms': lif_ms, 'lif_stdev': lif_stats['stdev'],
        'sum_ms': sum_ms,
        'overshoot_pct': overshoot,
        'consistent': consistent,
        'conv_frac': conv_ms / sum_ms * 100,
        'lif_per_T': lif_ms / T,
        'lif_BW_GBps': lif_BW,
        'inter_GB': inter_GB,
        'fuse_lb_bw': fuse_lb_bw,
        'fuse_lb_max': fuse_lb_max,
        'fuse_pct_bw': fuse_pct_bw,
        'fuse_pct_max': fuse_pct_max,
        'raw_samples': {
            'total': total_stats['samples'],
            'conv': conv_stats['samples'],
            'lif': lif_stats['samples'],
        },
    }


def print_header():
    print(f"\nShape: B={B}, C={C_in}, H=W={H}  (ResNet18 middle-layer representative)")
    print(f"Setup: warmup={N_WARMUP}, iter/meas={N_ITER}, repeat={N_REPEAT}, cudnn.benchmark=OFF")
    print(f"V100 HBM peak: {HBM_BW} GB/s\n")

    hdr = (f"{'T':<4} {'bk':<6} {'total':<14} {'conv':<14} {'lif':<14} "
           f"{'ok':<3} {'conv%':<6} {'lif/T':<8} {'lif_BW':<9} "
           f"{'fuse%_bw':<10} {'fuse%_max':<10}")
    print(hdr)
    print('-' * len(hdr))


def print_row(r):
    total_str = f"{r['total_ms']:.3f}±{r['total_stdev']:.2f}"
    conv_str = f"{r['conv_ms']:.3f}±{r['conv_stdev']:.2f}"
    lif_str = f"{r['lif_ms']:.3f}±{r['lif_stdev']:.2f}"
    ok = '✓' if r['consistent'] else '✗'
    print(f"{r['T']:<4} {r['backend']:<6} {total_str:<14} {conv_str:<14} {lif_str:<14} "
          f"{ok:<3} {r['conv_frac']:<6.1f} {r['lif_per_T']:<8.3f} "
          f"{r['lif_BW_GBps']:<9.1f} {r['fuse_pct_bw']:<10.1f} {r['fuse_pct_max']:<10.1f}")


def main():
    print_header()
    all_results = []
    for backend in ['torch', 'cupy']:
        for T in T_list:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            r = benchmark_config(T, backend)
            all_results.append(r)
            print_row(r)
        print()

    # 额外:生成一张 torch→cupy 加速比和 fuse ceiling 的紧凑总结
    print("\n=== Summary: speedup & fuse ceiling ===")
    print(f"{'T':<4} {'torch_total':<12} {'cupy_total':<12} {'cupy_speedup':<14} "
          f"{'cupy_fuse_lb':<14} {'max_theoretical_speedup_vs_torch':<32}")
    torch_results = {r['T']: r for r in all_results if r['backend'] == 'torch'}
    cupy_results = {r['T']: r for r in all_results if r['backend'] == 'cupy'}
    for T in T_list:
        tr = torch_results[T]['total_ms']
        cr = cupy_results[T]['total_ms']
        cr_lb = cupy_results[T]['fuse_lb_max']
        print(f"{T:<4} {tr:<12.3f} {cr:<12.3f} {tr/cr:<14.2f}x "
              f"{cr_lb:<14.3f} {tr/cr_lb:<32.2f}x")

    # 保存原始数据到 json,方便后面画图
    with open('benchmark_middle.json', 'w') as f:
        # 去掉 samples 让文件小一点,只保留统计量
        clean = [{k: v for k, v in r.items() if k != 'raw_samples'} for r in all_results]
        json.dump(clean, f, indent=2)
    print("\nSaved to benchmark_middle.json")


if __name__ == '__main__':
    main()