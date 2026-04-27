"""test_58_table15_remeasure.py — 重测 Table 12 + Table 15

目的:
  Table 15 报告的 L_multi / L_消除物化后 数字,跟 Table 1 的 SJ-multi / CATFuse 数字
  系统性偏高 ~22% (例如 VGG11-BN: Table 15 L_multi=7.68 vs Table 1 SJ=6.31)。
  这是跨表口径不一致问题,reviewer 会怀疑 cherry-pick。

  本脚本用跟 Table 1 完全一致的协议重测 4 个网络:
  - cudnn.benchmark = False
  - cudnn.deterministic = True
  - 20 warmup + 50 iter × 3 repeats
  - 取 median
  - 网络配置跟 Table 1 完全一致

  产出:
  - 新的 Table 12 (HBM 实测节省) 和 Table 15 (物化开销 φ) 数字
  - 跟 Table 1 数字交叉验证 (L_multi 必须等于 Table 1 SJ-multi)

注意:
  Table 15 的 "L_multi" 在 §3.9 定义里就是 SJ multi-step 延迟,
  Table 15 的 "L_消除物化后" 在 §3.9 定义里就是 CATFuse 延迟。
  所以本质上 L_multi == Table 1 SJ, L_after == Table 1 CF, 节省 = SJ - CF。
  Table 12 的"实测节省"列也是 SJ - CF。

  正确的 reproduction 应该是: 直接用 Table 1 的 SJ - CF 作为节省值,
  而不是另起一个 baseline。本脚本验证这一点。
"""
import sys, time, statistics, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_resnet, spiking_vgg
from catfuse.substitute import substitute_sf as substitute
from models.spiking_alexnet import SpikingAlexNet

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 跟 Table 1 完全一致的协议
T = 4
B = 2
N_W, N_I, N_R = 20, 50, 3   # V100 上 N_R=3 (跟 Table 1 完全一致)


def bench(model, x):
    """返回 median latency (ms) over N_R repeats. 跟 Table 1 完全一致。"""
    model.eval()
    times = []
    for _ in range(N_R):
        # warmup
        for _ in range(N_W):
            functional.reset_net(model)
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        # timed
        t0 = time.perf_counter()
        for _ in range(N_I):
            functional.reset_net(model)
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_I * 1000)
    return statistics.median(times)


# === Table 1 的 4 个网络 ===
configs = [
    ('VGG11-BN', lambda: spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('SEW-RN18', lambda: sew_resnet.sew_resnet18(
        pretrained=False, num_classes=10, cnf='ADD',
        spiking_neuron=neuron.LIFNode, tau=2.0)),
    ('SpikRN18', lambda: spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0, num_classes=10)),
    ('AlexNet', lambda: SpikingAlexNet(
        num_classes=10, spiking_neuron=neuron.LIFNode, tau=2.0, input_size=32)),
]

# === Table 1 V100 已发表数字 (cross-check 用) ===
table1_v100 = {
    'VGG11-BN': (6.31, 3.30),
    'SEW-RN18': (12.32, 5.93),
    'SpikRN18': (11.19, 5.88),
    'AlexNet':  (4.48, 2.74),
}

# === Table 15 之前发布的数字 (cross-check 用) ===
table15_old = {
    # (L_multi, L_after, phi)
    'VGG11-BN': (7.68, 3.34, 0.565),
    'SEW-RN18': (14.91, 5.66, 0.620),
    'SpikRN18': (13.38, 5.67, 0.576),
    'AlexNet':  (5.30, 2.36, 0.555),
}

gpu = torch.cuda.get_device_name(0)
print('=' * 100)
print(f"Table 15 重测 — 物化开销 φ (口径对齐 Table 1)")
print(f"GPU: {gpu}")
print(f"协议: T={T}, B={B}, cudnn.benchmark=False, {N_W} warmup + {N_I} iter × {N_R} repeats, median")
print('=' * 100)
print(f"{'Network':<12} {'SJ_new':>9} {'CF_new':>9} {'节省_new':>9} {'φ_new':>8}  "
      f"{'T1_SJ':>8} {'T1_CF':>8}  {'差异_SJ':>8} {'差异_CF':>8}")
print('-' * 100)

results = {}

for name, build_fn in configs:
    x = torch.rand(T, B, 3, 32, 32, device=device)

    # SJ multi-step (= Table 15 的 L_multi, = Table 1 的 SJ)
    m_sj = build_fn().to(device).eval()
    functional.set_step_mode(m_sj, 'm')
    L_multi_new = bench(m_sj, x)

    # CATFuse (= Table 15 的 L_消除物化后, = Table 1 的 CF)
    m_cf = build_fn().to(device).eval()
    functional.set_step_mode(m_cf, 'm')
    m_cf, _ = substitute(m_cf, T=T)
    m_cf = m_cf.to(device).eval()
    functional.set_step_mode(m_cf, 'm')
    L_after_new = bench(m_cf, x)

    save_new = L_multi_new - L_after_new
    phi_new = save_new / L_multi_new

    t1_sj, t1_cf = table1_v100[name]
    diff_sj = (L_multi_new / t1_sj - 1) * 100  # 百分比偏差
    diff_cf = (L_after_new / t1_cf - 1) * 100

    results[name] = {
        'L_multi_new': L_multi_new,
        'L_after_new': L_after_new,
        'save_new': save_new,
        'phi_new': phi_new,
        'diff_sj_pct': diff_sj,
        'diff_cf_pct': diff_cf,
    }

    print(f"{name:<12} {L_multi_new:>9.2f} {L_after_new:>9.2f} {save_new:>9.2f} {phi_new:>7.1%}  "
          f"{t1_sj:>8.2f} {t1_cf:>8.2f}  {diff_sj:>+7.1f}% {diff_cf:>+7.1f}%")

    del m_sj, m_cf, x
    torch.cuda.empty_cache()

print('-' * 100)
mean_phi = statistics.mean(r['phi_new'] for r in results.values())
print(f"{'Mean':<12} {'':>9} {'':>9} {'':>9} {mean_phi:>7.1%}")
print('=' * 100)

# === Cross-check: Table 15 vs Table 1 一致性 ===
print()
print("=" * 100)
print("Cross-check: 新测 baseline 跟 Table 1 是否一致 (差异应该 < 5%)")
print("=" * 100)
print(f"{'Network':<12} {'差异_SJ':>10} {'差异_CF':>10} {'verdict':<30}")
print('-' * 100)
all_consistent = True
for name in results:
    r = results[name]
    ok_sj = abs(r['diff_sj_pct']) < 5
    ok_cf = abs(r['diff_cf_pct']) < 5
    verdict = '✓ consistent' if (ok_sj and ok_cf) else '✗ inconsistent (调查!)'
    if not (ok_sj and ok_cf):
        all_consistent = False
    print(f"{name:<12} {r['diff_sj_pct']:>+9.1f}% {r['diff_cf_pct']:>+9.1f}% {verdict:<30}")
print('-' * 100)

if all_consistent:
    print("\n✓ 全部 4 个网络 baseline 跟 Table 1 一致 (差异 < 5%)")
    print("→ Table 15 应该用 (Table 1 SJ - Table 1 CF) / Table 1 SJ 重新计算")
    print("→ 论文修改: 删掉 Table 15 独立 baseline,改用 Table 1 派生公式")
else:
    print("\n✗ 部分网络 baseline 跟 Table 1 不一致")
    print("→ 调查原因: 是 substitute() 的 deepcopy 影响,还是 cudnn dispatch 不同?")
    print("→ 必须解决后才能投 ASPLOS/MLSys")

print()
print("=" * 100)
print("Table 15 重测后的新数字 (paper 直接用):")
print("=" * 100)
print(f"{'Network':<12} {'L_multi':>10} {'L_after':>10} {'save':>10} {'phi':>10}")
print('-' * 60)
for name in results:
    r = results[name]
    print(f"{name:<12} {r['L_multi_new']:>10.2f} {r['L_after_new']:>10.2f} "
          f"{r['save_new']:>10.2f} {r['phi_new']:>9.1%}")
print('-' * 60)
print(f"{'Mean':<12} {'':>10} {'':>10} {'':>10} {mean_phi:>9.1%}")

# === CSV 输出方便复制到论文 ===
print()
print("# CSV: name,L_multi_ms,L_after_ms,save_ms,phi")
for name in results:
    r = results[name]
    print(f"{name},{r['L_multi_new']:.2f},{r['L_after_new']:.2f},"
          f"{r['save_new']:.2f},{r['phi_new']:.4f}")