"""
CTF §3.8 Form 1 — save% vs weight/activation ratio across shapes.

Previous round's §3.8 revision added an "applicability conditions" section
claiming "BatchFold benefit grows when weights are large relative to
activations". That was an unsupported inference. This script scans 4 shapes
spanning the typical range of SNN layers and tabulates:

  weight_frac = |θ| / (|θ| + |activation_per_step|)
  save_pct    = analytic I/O savings / I/O_ref

If save_pct monotonically increases with weight_frac, the claim holds.
If not, the §3.8 revision needs another pass.
"""
import torch
import torch.nn as nn
import gc

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

T = 16
ELEM = 4
K_SIZE = 3

# (name, B, C_in, C_out, H, W, k)
# Chosen to span low->high weight/activation ratio
SHAPES = [
    ('shallow 3x3',    32,  64,  64, 32, 32, 3),   # large HW, medium C, low weight frac
    ('middle 3x3',     32, 128, 128, 16, 16, 3),   # our main testbed
    ('deep 3x3',       32, 512, 512,  4,  4, 3),   # small HW, large C, high weight frac
    ('middle 1x1',     32, 128, 128, 16, 16, 1),   # 1x1 conv — minimal weight
    ('bottleneck 1x1', 32, 256, 256,  8,  8, 1),
    ('expansion 1x1',  32, 128, 512, 16, 16, 1),
]


def io_ref(T, B, C_in, C_out, H, W, k):
    x = B * C_in * H * W * ELEM
    w = C_out * C_in * k * k * ELEM
    z = B * C_out * H * W * ELEM
    return T * (x + w + z)


def io_fold(T, B, C_in, C_out, H, W, k):
    x = T * B * C_in * H * W * ELEM
    w = C_out * C_in * k * k * ELEM
    z = T * B * C_out * H * W * ELEM
    return x + w + z


def main():
    print("=" * 95)
    print("§3.8 Form 1 — BatchFold savings vs weight/activation ratio")
    print(f"T={T}, all shapes use fp32")
    print("=" * 95)
    print()

    print(f"{'shape':<16} {'B':<4} {'Cin':<5} {'Cout':<5} {'H':<4} {'k':<3} "
          f"{'|θ| (KB)':<10} {'|x_t|+|z_t| (KB)':<18} {'w_frac':<8} "
          f"{'save%':<8}")
    print('-' * 95)

    rows = []
    for name, B, C_in, C_out, H, W, k in SHAPES:
        w_bytes = C_out * C_in * k * k * ELEM
        act_bytes = (B * C_in * H * W + B * C_out * H * W) * ELEM
        w_frac = w_bytes / (w_bytes + act_bytes)

        ref = io_ref(T, B, C_in, C_out, H, W, k)
        fold = io_fold(T, B, C_in, C_out, H, W, k)
        save_pct = (ref - fold) / ref * 100

        print(f"{name:<16} {B:<4} {C_in:<5} {C_out:<5} {H:<4} {k:<3} "
              f"{w_bytes/1024:<10.1f} {act_bytes/1024:<18.1f} "
              f"{w_frac:<8.3f} {save_pct:<8.2f}")
        rows.append((name, w_frac, save_pct))

    print()
    print("Sorted by weight_frac (ascending):")
    print('-' * 60)
    rows.sort(key=lambda r: r[1])
    for name, wf, sp in rows:
        bar = '█' * int(sp)
        print(f"  {name:<16} w_frac={wf:<7.3f} save%={sp:<6.2f} {bar}")

    print()
    print("Claim: save% should monotonically increase with weight_frac.")
    is_monotonic = all(rows[i][2] <= rows[i+1][2] for i in range(len(rows)-1))
    print(f"Monotonic: {is_monotonic}")

    if not is_monotonic:
        print()
        print("NOTE: non-monotonic means the §3.8 revised 'applicability condition'")
        print("      is too simplified. Save% depends on the full ratio structure,")
        print("      not just weight_frac. Needs another revision.")


if __name__ == '__main__':
    main()