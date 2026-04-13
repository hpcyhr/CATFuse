"""
CTF §3.8 Form-1: TimeBlock(K) ∘ BatchFold(Conv)

Goal: derive and verify the I/O formula for BatchFold on a single Conv layer.
This is the simplest CTF transform, establishing the measurement methodology
before moving to §3.9 StreamFuse.

No kernels are written. Two schedules are built from stock PyTorch ops:
  σ_ref : per-step Python loop over T, one Conv call per step
  σ_fold: reshape T·B into batch dim, one Conv call total
"""
import torch
import torch.nn as nn
import statistics

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ResNet18 middle-layer shape
B, C_in, C_out, H, W = 32, 128, 128, 16, 16
K_SIZE = 3  # conv kernel
T_list = [4, 8, 16, 32]
ELEM = 4  # float32 bytes


# ============================================================
# Analytical I/O formulas
# ============================================================

def io_ref(T, B, C_in, C_out, H, W, k=K_SIZE):
    """σ_ref: per-step execution. W is (re)loaded T times."""
    per_step_x = B * C_in * H * W * ELEM
    per_step_w = C_out * C_in * k * k * ELEM
    per_step_z = B * C_out * H * W * ELEM
    return T * (per_step_x + per_step_w + per_step_z)


def io_fold(T, B, C_in, C_out, H, W, k=K_SIZE):
    """σ_fold: BatchFold over T. W loaded once."""
    x_total = T * B * C_in * H * W * ELEM
    w_total = C_out * C_in * k * k * ELEM
    z_total = T * B * C_out * H * W * ELEM
    return x_total + w_total + z_total


def io_savings(T, B, C_in, C_out, H, W, k=K_SIZE):
    """Analytic savings: (T-1) · weight bytes."""
    return (T - 1) * C_out * C_in * k * k * ELEM


# ============================================================
# Two schedules built from stock PyTorch
# ============================================================

def run_ref(conv, x):
    """σ_ref: Python for-loop over T. One Conv call per step."""
    T = x.shape[0]
    outs = []
    for t in range(T):
        outs.append(conv(x[t]))
    return torch.stack(outs, dim=0)


def run_fold(conv, x):
    """σ_fold: reshape [T,B,...] to [T·B,...], single Conv call."""
    T, B = x.shape[0], x.shape[1]
    x_flat = x.reshape(T * B, *x.shape[2:])
    z_flat = conv(x_flat)
    return z_flat.reshape(T, B, *z_flat.shape[1:])


# ============================================================
# Parity check: schedules must be bit-exact per Theorem 3.9
# ============================================================

def check_parity():
    torch.manual_seed(0)
    conv = nn.Conv2d(C_in, C_out, K_SIZE, padding=1, bias=False).to(device)
    x = torch.randn(8, B, C_in, H, W, device=device)

    with torch.no_grad():
        z_ref = run_ref(conv, x)
        z_fold = run_fold(conv, x)

    max_diff = (z_ref - z_fold).abs().max().item()
    exact = torch.equal(z_ref, z_fold)
    print(f"Parity check (§3.8 Form-1):")
    print(f"  σ_ref vs σ_fold  exact={exact}, max_diff={max_diff}")
    print(f"  Theorem 3.9 requires exact equality.")
    print()
    return exact


# ============================================================
# Wall-clock measurement (secondary, cross-checks formula direction)
# ============================================================

def cuda_time(fn, n_warmup=20, n_iter=50, n_repeat=7):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / n_iter)
    return statistics.median(times)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("CTF §3.8 Form-1: TimeBlock(K) ∘ BatchFold(Conv)")
    print(f"Shape: B={B}, C_in={C_in}, C_out={C_out}, H=W={H}, k={K_SIZE}")
    print("=" * 78)
    print()

    check_parity()

    # Analytical table
    print("Analytical I/O (bytes), from formulas:")
    print(f"{'T':<4} {'I/O_ref':<14} {'I/O_fold':<14} {'savings':<12} {'save %':<8} {'save / weight_size':<20}")
    print('-' * 78)
    weight_size = C_out * C_in * K_SIZE * K_SIZE * ELEM
    for T in T_list:
        ref = io_ref(T, B, C_in, C_out, H, W)
        fold = io_fold(T, B, C_in, C_out, H, W)
        save = io_savings(T, B, C_in, C_out, H, W)
        pct = save / ref * 100
        # sanity: savings should be (T-1) * weight_size
        assert save == (T - 1) * weight_size
        print(f"{T:<4} {ref:<14,} {fold:<14,} {save:<12,} {pct:<8.2f} "
              f"{save/weight_size:<20.1f} × W")
    print()

    # Wall-clock cross-check
    print("Wall-clock (ms), cross-checking that σ_fold ≤ σ_ref:")
    print(f"{'T':<4} {'σ_ref':<12} {'σ_fold':<12} {'fold/ref':<10}")
    print('-' * 78)
    for T in T_list:
        torch.cuda.empty_cache()
        conv = nn.Conv2d(C_in, C_out, K_SIZE, padding=1, bias=False).to(device)
        x = torch.randn(T, B, C_in, H, W, device=device)

        t_ref = cuda_time(lambda: run_ref(conv, x))
        t_fold = cuda_time(lambda: run_fold(conv, x))
        print(f"{T:<4} {t_ref:<12.3f} {t_fold:<12.3f} {t_fold/t_ref:<10.3f}")

    print()
    print("Interpretation:")
    print("  The analytical savings ((T-1)·|W|) is the upper bound assuming")
    print("  σ_ref actually re-reads W every step. In practice cudnn's L2")
    print("  caching and workspace reuse mean the real savings is smaller.")
    print("  Wall-clock comparison is only a direction check: σ_fold should")
    print("  not be slower than σ_ref, otherwise our formula is wrong about")
    print("  which direction the savings go.")


if __name__ == '__main__':
    main()