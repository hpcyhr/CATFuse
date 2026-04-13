"""
CTF §3.9 ratio formula — scan T.

Last round verified ratio = (1 + 2/K)/5 at T=16. This script scans T to
confirm the formula is T-independent (it should be, because the derivation
never uses T as a variable — T cancels out between I/O_naive and I/O_ctf).

If the empirical peak ratio shifts with T, it means either:
  (a) there's a T-dependent term in the derivation we missed, or
  (b) constant per-step overhead (~9 step for CTF) dilutes differently at
      different T — this would be a measurement artifact, not a theory issue
"""
import torch
import torch.nn as nn
import gc

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

B, C_in, C, H, W = 32, 128, 128, 16, 16
T_list = [4, 8, 16, 32]
K_list = [1, 2, 4, 8]  # K values that divide all T
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0
ELEM = 4


def io_naive(T, B, C, H, W):
    return 5 * T * B * C * H * W * ELEM


def io_ctf(T, K, B, C, H, W):
    step = B * C * H * W * ELEM
    n_blocks = T // K
    return n_blocks * K * step + n_blocks * 2 * step


def schedule_naive(conv, x_seq, T):
    z_seq = torch.empty(T, B, C, H, W, device=device)
    v_seq = torch.empty(T, B, C, H, W, device=device)
    s_seq = torch.empty(T, B, C, H, W, device=device)
    v_prev = torch.zeros(B, C, H, W, device=device)
    for t in range(T):
        z_seq[t] = conv(x_seq[t])
        z_t = z_seq[t]
        v_t = v_prev + (z_t - (v_prev - V_RESET)) / TAU
        s_t = (v_t >= V_TH).float()
        v_t = v_t * (1 - s_t) + V_RESET * s_t
        v_seq[t] = v_t
        s_seq[t] = s_t
        v_prev = v_t
    return s_seq


def schedule_ctf(conv, x_seq, T, K):
    assert T % K == 0
    spikes = torch.empty(T, B, C, H, W, device=device)
    v = torch.zeros(B, C, H, W, device=device)
    for block_start in range(0, T, K):
        for t in range(block_start, block_start + K):
            z_t = conv(x_seq[t])
            v = v + (z_t - (v - V_RESET)) / TAU
            s = (v >= V_TH).float()
            v = v * (1 - s) + V_RESET * s
            spikes[t] = s
        v = v.clone()
    return spikes


def measure_peak(fn):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def main():
    print("=" * 82)
    print("§3.9 ratio formula — scan T")
    print(f"Shape: B={B}, C={C}, H=W={H}")
    print("=" * 82)
    print()

    torch.manual_seed(0)
    conv = nn.Conv2d(C_in, C, 3, padding=1, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(2.0)
    step_size = B * C * H * W * ELEM

    print(f"{'T':<4} {'K':<4} {'analytic':<12} {'peak':<12} "
          f"{'analytic/5T':<14} {'peak/5T':<12}")
    print('-' * 82)

    for T in T_list:
        x = torch.randn(T, B, C_in, H, W, device=device)

        # Parity check first
        with torch.no_grad():
            s_ref = schedule_naive(conv, x, T)
        for K in K_list:
            if T % K != 0:
                continue
            with torch.no_grad():
                s_ctf = schedule_ctf(conv, x, T, K)
            assert torch.equal(s_ref, s_ctf), f"parity fail T={T} K={K}"

        # Naive peak
        def run_naive():
            with torch.no_grad():
                _ = schedule_naive(conv, x, T)
        for _ in range(3):
            run_naive()
        peak_naive = measure_peak(run_naive)
        naive_bytes = io_naive(T, B, C, H, W)
        print(f"{T:<4} {'naive':<4} {naive_bytes:<12,} {peak_naive:<12,} "
              f"{naive_bytes/(5*T*step_size):<14.3f} "
              f"{peak_naive/(5*T*step_size):<12.3f}")

        for K in K_list:
            if T % K != 0:
                continue
            def run_ctf(K=K, T=T):
                with torch.no_grad():
                    _ = schedule_ctf(conv, x, T, K)
            for _ in range(3):
                run_ctf()
            peak_ctf = measure_peak(run_ctf)
            ctf_bytes = io_ctf(T, K, B, C, H, W)

            analytic_ratio = ctf_bytes / naive_bytes
            empirical_ratio = peak_ctf / peak_naive
            predicted_ratio = (1 + 2/K) / 5

            flag = "✓" if abs(analytic_ratio - predicted_ratio) < 1e-9 else "✗"
            print(f"{T:<4} K={K:<2} {ctf_bytes:<12,} {peak_ctf:<12,} "
                  f"{analytic_ratio:<14.3f} {empirical_ratio:<12.3f} {flag}")
        print()

    print("What to check:")
    print("  - 'analytic' column matches predicted (1+2/K)/5 for all T")
    print("  - 'peak' column's ratio is T-dependent due to per-step constant")
    print("    overhead: CTF has ~9-step constant, so at small T it shows")
    print("    more overhead relative to T*step")


if __name__ == '__main__':
    main()