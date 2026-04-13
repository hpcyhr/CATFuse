"""
v2: forward + backward benchmark for Conv -> LIF
Same shape/stability settings as v1, adds:
  - forward-only timing (same as v1, for regression check)
  - forward+backward timing
  - backward-only (derived: fwdbwd - fwd)
  - fuse ceiling analysis for both fwd and fwdbwd
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
import statistics
import json

device = 'cuda:0'  # 新机器上只有一张卡,用 cuda:0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

HBM_BW = 900.0  # V100 HBM2 GB/s

B, C_in, C_out, H, W = 32, 128, 128, 16, 16
T_list = [4, 8, 16, 32]

N_WARMUP = 50
N_ITER = 100
N_REPEAT = 11


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


def cuda_time_stats(fn, n_iter=N_ITER, n_repeat=N_REPEAT):
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
    conv = make_conv()
    lif = make_lif(backend)

    # forward-only: 不需要 requires_grad
    x_fwd = torch.randn(T, B, C_in, H, W, device=device)

    # forward+backward: 需要一个 leaf tensor 且 requires_grad=True
    # 这里让 x 作为 leaf,conv 权重本身也是 leaf
    x_fwdbwd = torch.randn(T, B, C_in, H, W, device=device, requires_grad=True)

    # 预计算一个 z 给 lif-only 测量(仅 fwd 用)
    with torch.no_grad():
        z_precomputed = conv(x_fwd.flatten(0, 1)).reshape(T, B, C_out, H, W).contiguous()

    # ---------- forward-only steps ----------
    def step_total_fwd():
        functional.reset_net(lif)
        with torch.no_grad():
            x_flat = x_fwd.flatten(0, 1)
            z_flat = conv(x_flat)
            z = z_flat.reshape(T, B, C_out, H, W)
            lif(z)

    def step_conv_fwd():
        with torch.no_grad():
            x_flat = x_fwd.flatten(0, 1)
            z_flat = conv(x_flat)
            z_flat.reshape(T, B, C_out, H, W)

    def step_lif_fwd():
        functional.reset_net(lif)
        with torch.no_grad():
            lif(z_precomputed)

    # ---------- forward+backward steps ----------
    # 注意:每次迭代都要重新前向,因为 backward 会消耗 graph
    def step_total_fwdbwd():
        functional.reset_net(lif)
        conv.zero_grad(set_to_none=True)
        if x_fwdbwd.grad is not None:
            x_fwdbwd.grad = None
        x_flat = x_fwdbwd.flatten(0, 1)
        z_flat = conv(x_flat)
        z = z_flat.reshape(T, B, C_out, H, W)
        s = lif(z)
        # 用 sum 作为 loss,避免额外的 reduction kernel 开销影响测量
        loss = s.sum()
        loss.backward()

    def step_conv_fwdbwd():
        conv.zero_grad(set_to_none=True)
        if x_fwdbwd.grad is not None:
            x_fwdbwd.grad = None
        x_flat = x_fwdbwd.flatten(0, 1)
        z_flat = conv(x_flat)
        z = z_flat.reshape(T, B, C_out, H, W)
        loss = z.sum()
        loss.backward()

    def step_lif_fwdbwd():
        functional.reset_net(lif)
        # lif 单独测 backward 需要一个 requires_grad 的输入
        # 这里用 z_precomputed 的 detach + requires_grad 版本
        z_in = z_precomputed.detach().clone().requires_grad_(True)
        s = lif(z_in)
        loss = s.sum()
        loss.backward()

    # ---------- warmup(按最重的路径) ----------
    for _ in range(N_WARMUP):
        step_total_fwdbwd()
    torch.cuda.synchronize()

    # ---------- 测量 ----------
    # forward-only
    total_fwd = cuda_time_stats(step_total_fwd)
    conv_fwd = cuda_time_stats(step_conv_fwd)
    lif_fwd = cuda_time_stats(step_lif_fwd)

    # forward+backward
    total_fb = cuda_time_stats(step_total_fwdbwd)
    conv_fb = cuda_time_stats(step_conv_fwdbwd)
    lif_fb = cuda_time_stats(step_lif_fwdbwd)

    # 派生
    def pack(fwd_stats, fb_stats, conv_stats_fwd, lif_stats_fwd,
             conv_stats_fb, lif_stats_fb):
        total_ms_fwd = fwd_stats['median']
        total_ms_fb = fb_stats['median']
        bwd_only_ms = total_ms_fb - total_ms_fwd  # 推断出来的 backward 时间

        conv_fwd_ms = conv_stats_fwd['median']
        lif_fwd_ms = lif_stats_fwd['median']
        conv_fb_ms = conv_stats_fb['median']
        lif_fb_ms = lif_stats_fb['median']

        # Fuse ceiling (aggressive: lif absorbed)
        fuse_lb_fwd = max(conv_fwd_ms, lif_fwd_ms)
        fuse_lb_fb = max(conv_fb_ms, lif_fb_ms)
        fuse_pct_fwd = max(0, (1 - fuse_lb_fwd / total_ms_fwd) * 100)
        fuse_pct_fb = max(0, (1 - fuse_lb_fb / total_ms_fb) * 100)

        return {
            'total_fwd': total_ms_fwd, 'total_fwd_std': fwd_stats['stdev'],
            'total_fb': total_ms_fb, 'total_fb_std': fb_stats['stdev'],
            'bwd_only': bwd_only_ms,
            'bwd_ratio': bwd_only_ms / total_ms_fwd if total_ms_fwd > 0 else 0,
            'conv_fwd': conv_fwd_ms, 'lif_fwd': lif_fwd_ms,
            'conv_fb': conv_fb_ms, 'lif_fb': lif_fb_ms,
            'fuse_lb_fwd': fuse_lb_fwd, 'fuse_lb_fb': fuse_lb_fb,
            'fuse_pct_fwd': fuse_pct_fwd, 'fuse_pct_fb': fuse_pct_fb,
        }

    derived = pack(total_fwd, total_fb, conv_fwd, lif_fwd, conv_fb, lif_fb)
    return {'T': T, 'backend': backend, **derived}


def print_row(r):
    print(f"{r['T']:<4} {r['backend']:<6} "
          f"{r['total_fwd']:<8.3f} {r['total_fb']:<8.3f} {r['bwd_only']:<8.3f} "
          f"{r['bwd_ratio']:<7.2f} "
          f"{r['conv_fwd']:<8.3f} {r['lif_fwd']:<8.3f} "
          f"{r['conv_fb']:<8.3f} {r['lif_fb']:<8.3f} "
          f"{r['fuse_pct_fwd']:<9.1f} {r['fuse_pct_fb']:<9.1f}")


def main():
    print(f"\nShape: B={B}, C={C_in}, H=W={H}")
    print(f"Setup: warmup={N_WARMUP}, iter/meas={N_ITER}, repeat={N_REPEAT}, "
          f"cudnn.benchmark=OFF")
    print(f"V100 HBM peak: {HBM_BW} GB/s\n")

    hdr = (f"{'T':<4} {'bk':<6} "
           f"{'tot_fwd':<8} {'tot_fb':<8} {'bwd':<8} {'bwd/fwd':<7} "
           f"{'conv_f':<8} {'lif_f':<8} {'conv_fb':<8} {'lif_fb':<8} "
           f"{'fuse%_f':<9} {'fuse%_fb':<9}")
    print(hdr)
    print('-' * len(hdr))

    all_results = []
    for backend in ['torch', 'cupy']:
        for T in T_list:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            r = benchmark_config(T, backend)
            all_results.append(r)
            print_row(r)
        print()

    # Summary: speedup & fuse ceiling, fwd vs fwdbwd
    print("\n=== Summary: fwd vs fwd+bwd, speedup & fuse ceiling ===")
    print(f"{'T':<4} {'mode':<8} {'torch':<10} {'cupy':<10} {'cupy_spd':<10} "
          f"{'cupy_fuseLB':<13} {'spd_vs_torch':<14}")
    torch_res = {r['T']: r for r in all_results if r['backend'] == 'torch'}
    cupy_res = {r['T']: r for r in all_results if r['backend'] == 'cupy'}
    for T in T_list:
        for mode, tkey, lbkey in [('fwd', 'total_fwd', 'fuse_lb_fwd'),
                                    ('fwd+bwd', 'total_fb', 'fuse_lb_fb')]:
            tr = torch_res[T][tkey]
            cr = cupy_res[T][tkey]
            cr_lb = cupy_res[T][lbkey]
            cupy_spd = tr / cr if cr > 0 else 0
            vs_torch = tr / cr_lb if cr_lb > 0 else 0
            print(f"{T:<4} {mode:<8} {tr:<10.3f} {cr:<10.3f} {cupy_spd:<10.2f}x "
                  f"{cr_lb:<13.3f} {vs_torch:<14.2f}x")
        print()

    with open('benchmark_v2_fwdbwd.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to benchmark_v2_fwdbwd.json")


if __name__ == '__main__':
    main()