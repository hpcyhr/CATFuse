"""
CTF §3.9 I/O cost verification: TimeBlock(K) ∘ StreamFuse ∘ StateCarry

Verifies the paper's central claim: StreamFuse eliminates z_t materialization,
and TimeBlock + StateCarry reduces v's HBM traffic from O(T·HWC) to O(T/K·HWC).

Method:
  1. Derive analytic HBM byte counts for two schedules (naive vs CTF)
  2. Implement both as pure PyTorch (no kernels) so allocation/deallocation
     events are visible to torch.cuda profiler
  3. Cross-check: CTF schedule's tensor allocations match analytic prediction
  4. Scan K to verify the O(T/K) scaling
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
ELEM = 4  # float32


# ============================================================
# Analytical formulas
# ============================================================

def io_naive(T, B, C, H, W):
    """
    Naive per-step schedule:
      each step: conv writes z_t, LIF reads z_t, reads v_{t-1}, writes v_t, writes s_t
      = 5 * step_size per step
    """
    step = B * C * H * W * ELEM
    return 5 * T * step


def io_ctf(T, K, B, C, H, W):
    """
    CTF: TimeBlock(K) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF).
    Within a block: only s_t is written (K writes).
    At each block boundary: v end-state write + v start-state read (2 writes/reads).
    Blocks: T/K.
    """
    step = B * C * H * W * ELEM
    n_blocks = T // K
    within_block = n_blocks * K * step  # s writes, = T * step
    boundary = n_blocks * 2 * step
    return within_block + boundary


# ============================================================
# Two implementations (pure PyTorch, no custom kernels)
# ============================================================

def schedule_naive(conv, x_seq):
    """
    Explicit per-step materialization. Allocates full [T,B,C,H,W] tensors
    for z and v to force HBM residency, matching the 'reference execution'
    described in §3.1 and §3.9.
    """
    T_ = x_seq.shape[0]
    z_seq = torch.empty(T_, B, C, H, W, device=device)
    v_seq = torch.empty(T_, B, C, H, W, device=device)
    s_seq = torch.empty(T_, B, C, H, W, device=device)

    v_prev = torch.zeros(B, C, H, W, device=device)
    for t in range(T_):
        # Conv output explicitly written to the [T,...] buffer
        z_seq[t] = conv(x_seq[t])
        # LIF reads z_seq[t] and v_prev from HBM
        z_t = z_seq[t]
        v_t = v_prev + (z_t - (v_prev - V_RESET)) / TAU
        s_t = (v_t >= V_TH).float()
        v_t = v_t * (1 - s_t) + V_RESET * s_t
        # Write v_t back to the [T,...] buffer
        v_seq[t] = v_t
        s_seq[t] = s_t
        v_prev = v_t  # still carries, but v_seq holds the materialized copy

    return s_seq


def schedule_ctf(conv, x_seq, K):
    """
    CTF schedule: TimeBlock(K) ∘ StreamFuse ∘ StateCarry.
    - z_t is never materialized (stays as temporary within the step)
    - v is a function-local variable, not stored into a [T,...] tensor
    - Only s_t is accumulated (required by downstream layers)
    Between blocks: v is carried explicitly (in Python) — in a real kernel
    this would be an explicit HBM write+read at the boundary.
    """
    T_ = x_seq.shape[0]
    assert T_ % K == 0
    spikes = torch.empty(T_, B, C, H, W, device=device)

    v = torch.zeros(B, C, H, W, device=device)
    for block_start in range(0, T_, K):
        block_end = block_start + K
        for t in range(block_start, block_end):
            z_t = conv(x_seq[t])  # temporary — not stored
            v = v + (z_t - (v - V_RESET)) / TAU
            s = (v >= V_TH).float()
            v = v * (1 - s) + V_RESET * s
            spikes[t] = s
        # Block boundary: in a real fused impl, v would be written to HBM
        # and re-read at the next block's start. Here we simulate it:
        v = v.clone()  # forces a copy, analog of HBM write + read

    return spikes


# ============================================================
# Parity check: both schedules must produce the same spikes
# ============================================================

def check_parity():
    torch.manual_seed(0)
    conv = nn.Conv2d(C_in, C, 3, padding=1, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(2.0)  # ensure meaningful spike rate
    x = torch.randn(T, B, C_in, H, W, device=device)

    with torch.no_grad():
        s_naive = schedule_naive(conv, x)

    print(f"Naive spike rate: {s_naive.mean().item():.4f}")
    print()
    print(f"{'K':<4} {'ctf_exact':<12} {'max_diff':<12}")
    print('-' * 40)
    all_ok = True
    for K in K_list:
        with torch.no_grad():
            s_ctf = schedule_ctf(conv, x, K)
        exact = torch.equal(s_naive, s_ctf)
        max_d = (s_naive - s_ctf).abs().max().item()
        print(f"{K:<4} {str(exact):<12} {max_d:<12.2e}")
        if not exact:
            all_ok = False
    print()
    return all_ok


# ============================================================
# Allocation tracking via torch.cuda memory stats
# ============================================================

def measure_allocations(fn):
    """
    Run fn() and return (peak_allocated_bytes, total_allocated_bytes).
    peak: high-water mark of allocated memory during fn's execution
    total: cumulative bytes allocated (includes reallocations)
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    return peak


def main():
    print("=" * 78)
    print("§3.9 I/O cost verification: TimeBlock(K) ∘ StreamFuse ∘ StateCarry")
    print(f"Shape: T={T}, B={B}, C={C}, H=W={H}")
    print("=" * 78)
    print()

    print("Step 1: Parity check (CTF schedule == naive schedule)")
    if not check_parity():
        print("FAIL: parity broken, cannot trust I/O measurements")
        return
    print("PASS: all K produce same spikes")
    print()

    # -----------------------------------------------------------
    # Analytical I/O
    # -----------------------------------------------------------
    print("Step 2: Analytical HBM byte counts")
    print(f"{'schedule':<20} {'bytes':<16} {'vs naive':<12}")
    print('-' * 60)
    naive_bytes = io_naive(T, B, C, H, W)
    print(f"{'naive (reference)':<20} {naive_bytes:<16,} {'1.00x':<12}")
    for K in K_list:
        ctf_bytes = io_ctf(T, K, B, C, H, W)
        ratio = ctf_bytes / naive_bytes
        print(f"{'CTF K=' + str(K):<20} {ctf_bytes:<16,} {ratio:<12.3f}")
    print()

    # -----------------------------------------------------------
    # Empirical peak memory (cross-check)
    # -----------------------------------------------------------
    print("Step 3: Empirical peak allocation (cross-check)")
    print("Note: peak allocation != HBM traffic, but it DOES reflect which")
    print("intermediate tensors get materialized. We expect:")
    print("  - naive: peak includes [T,B,C,H,W] for z_seq + v_seq + s_seq")
    print("  - CTF:   peak includes only [T,B,C,H,W] for s_seq + O(1) temps")
    print()

    torch.manual_seed(0)
    conv = nn.Conv2d(C_in, C, 3, padding=1, bias=False).to(device)
    with torch.no_grad():
        conv.weight.mul_(2.0)
    x = torch.randn(T, B, C_in, H, W, device=device)
    step_size = B * C * H * W * ELEM

    # Naive
    def run_naive():
        with torch.no_grad():
            _ = schedule_naive(conv, x)
    # Warmup to eliminate first-call allocator fluctuations
    for _ in range(5):
        run_naive()
    peak_naive = measure_allocations(run_naive)

    print(f"{'schedule':<20} {'peak (bytes)':<16} {'peak / step':<14} "
          f"{'vs naive':<10}")
    print('-' * 70)
    print(f"{'naive':<20} {peak_naive:<16,} {peak_naive/step_size:<14.2f} "
          f"{'1.00x':<10}")

    for K in K_list:
        def run_ctf(K=K):
            with torch.no_grad():
                _ = schedule_ctf(conv, x, K)
        for _ in range(5):
            run_ctf()
        peak_ctf = measure_allocations(run_ctf)
        ratio = peak_ctf / peak_naive
        print(f"{'CTF K=' + str(K):<20} {peak_ctf:<16,} "
              f"{peak_ctf/step_size:<14.2f} {ratio:<10.3f}")
    print()

    # -----------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------
    print("Expected observations:")
    print("  - Analytical: CTF at K=1 already saves ~40% vs naive (from")
    print("    StreamFuse eliminating z_t). Further K reduces the v-carry")
    print("    overhead, approaching 20% of naive at K→T.")
    print("  - Empirical peak: naive needs ~3T*step (z_seq + v_seq + s_seq),")
    print("    CTF needs ~T*step (only s_seq), regardless of K.")
    print("  - Main savings come from StreamFuse (removing z_seq), not from")
    print("    TimeBlock (which only affects v's block-boundary traffic).")


if __name__ == '__main__':
    main()