"""
CTF §3.9 — cumulative allocation at block boundaries.

P1 revealed that peak memory is K-invariant for CTF because v.clone() is
immediately freed. But cumulative allocated bytes DO see every alloc/free
event, so they can witness the block-boundary v carry cost that peak misses.

Specifically, torch.cuda.memory_stats() exposes:
  - allocated_bytes.all.peak: what P1 already measured
  - allocated_bytes.all.allocated: cumulative bytes EVER allocated
  - allocation.all.allocated: cumulative COUNT of allocation events

The cumulative 'allocated' is what captures the clone() overhead: each
clone allocates a fresh buffer, even if the old one is freed moments later.
"""
import torch
import torch.nn as nn
import gc

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

B, C_in, C, H, W = 32, 128, 128, 16, 16
T = 16
K_list = [1, 2, 4, 8, 16]
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0


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
        v = v.clone()  # block boundary: explicit state carry
    return spikes


def measure_cumulative(fn, n_iter=20):
    """
    Run fn() n_iter times and measure the cumulative bytes allocated during
    all iterations. Using many iterations amplifies the per-block clone
    count so it becomes distinguishable from per-iteration noise.
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Snapshot stats before
    stats_before = torch.cuda.memory_stats()
    alloc_bytes_before = stats_before['allocated_bytes.all.allocated']
    alloc_count_before = stats_before['allocation.all.allocated']

    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()

    stats_after = torch.cuda.memory_stats()
    alloc_bytes = stats_after['allocated_bytes.all.allocated'] - alloc_bytes_before
    alloc_count = stats_after['allocation.all.allocated'] - alloc_count_before

    return alloc_bytes / n_iter, alloc_count / n_iter


def main():
    print("=" * 82)
    print("§3.9 — cumulative allocation vs K")
    print(f"Shape: T={T}, B={B}, C={C}, H=W={H}")
    print("=" * 82)
    print()

    torch.manual_seed(0)
    conv = nn.Conv2d(C_in, C, 3, padding=1, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(2.0)
    x = torch.randn(T, B, C_in, H, W, device=device)
    step_size = B * C * H * W * 4

    print(f"{'K':<4} {'n_blocks':<10} {'alloc_count/iter':<18} "
          f"{'alloc_bytes/iter':<18} {'alloc_bytes/step':<18}")
    print('-' * 82)

    for K in K_list:
        n_blocks = T // K

        def run_ctf(K=K):
            with torch.no_grad():
                _ = schedule_ctf(conv, x, T, K)

        for _ in range(5):
            run_ctf()

        alloc_bytes, alloc_count = measure_cumulative(run_ctf, n_iter=20)

        print(f"{K:<4} {n_blocks:<10} {alloc_count:<18.1f} "
              f"{alloc_bytes:<18,.0f} {alloc_bytes/step_size:<18.2f}")

    print()
    print("Expected pattern:")
    print("  - alloc_count monotonically DECREASES as K increases")
    print("    (fewer blocks → fewer clone events → fewer allocations)")
    print("  - alloc_bytes/step should also show a K-dependent component:")
    print("    each clone contributes 1 step-worth of allocation")
    print("  - If alloc_bytes is flat across K, the clone() isn't actually")
    print("    being captured by the cumulative counter — in which case")
    print("    we need a different instrumentation")


if __name__ == '__main__':
    main()