"""
v9: F6 — Add → LIF fused kernel
================================

CTF transform: TimeBlock(K) ∘ StreamFuse(Add, LIF) ∘ StateCarry(LIF)

Add is a TSI operator (elementwise, no state, no time dependence), LIF is
a CSR operator. The fused kernel reads x1[t] and x2[t] from HBM, adds them
in registers, feeds the sum directly into the LIF update, and writes only
the spike back to HBM. z_t (the sum) never materializes in HBM.

This corresponds to ResNet-style residual connections:
    y = LIF(Conv(x)_main + x_identity)
where after the conv path produces main activations and the shortcut passes
identity, both are fed into an elementwise sum, then the LIF neuron.

Baselines (SNN community reference):
  - torch_naive: torch add + spikingjelly.LIFNode(backend='torch')
  - cupy_naive:  torch add + spikingjelly.LIFNode(backend='cupy')
    ← the real SOTA for SNN: a fused multi-step LIF kernel in CuPy

Shape: we use a mid-sized residual feature map — B=32, C=128, H=W=16,
matching the middle ResNet-18 layer the earlier v7c used.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import statistics
import json

from spikingjelly.activation_based import neuron, functional, surrogate

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# Shape: mid-size residual path
B = 32
C = 128
H = 16
W = 16

# LIF parameters (SpikingJelly uses tau differently; see setup below)
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

T_TOTAL = 16
K_LIST = [1, 2, 4, 8, 16]

N_WARMUP = 20
N_ITER = 100
N_REPEAT = 11


# ============================================================
# v9 fused Add+LIF kernel
# ============================================================

@triton.jit
def add_lif_block_kernel(
    x1_ptr,         # [K, numel] — flattened view of [K, B, C, H, W]
    x2_ptr,         # [K, numel]
    s_ptr,          # [K, numel]
    v_carry_ptr,    # [numel]
    K: tl.constexpr,
    numel,
    tau: tl.constexpr, v_th: tl.constexpr, v_reset: tl.constexpr,
    stride_t,
    BLOCK: tl.constexpr,
):
    """
    Each program handles a contiguous BLOCK of neurons along the flattened
    (B, C, H, W) dimension. v lives in registers across K time steps.

    CTF mapping:
      - StreamFuse(Add, LIF): z_t = x1[t] + x2[t] happens in registers,
        never written to HBM
      - StateCarry(LIF): v is loaded from v_carry_ptr at block start
        and written back at block end
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    # Load v from v_carry at block start
    v = tl.load(v_carry_ptr + offs, mask=mask, other=0.0)

    for t in tl.static_range(K):
        # Load x1[t], x2[t] from HBM
        x1_t = tl.load(x1_ptr + t * stride_t + offs, mask=mask, other=0.0)
        x2_t = tl.load(x2_ptr + t * stride_t + offs, mask=mask, other=0.0)

        # StreamFuse: z_t is a register-only intermediate
        z_t = x1_t + x2_t

        # LIF update
        v = v + (z_t - (v - v_reset)) / tau
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike) + v_reset * spike

        # Store spike
        tl.store(s_ptr + t * stride_t + offs, spike, mask=mask)

    # StateCarry: v travels through HBM to the next block
    tl.store(v_carry_ptr + offs, v, mask=mask)


def triton_add_lif_blocked(x1_seq, x2_seq, K):
    """
    x1_seq, x2_seq: [T_total, B, C, H, W]
    K:              block size
    returns s_seq:  [T_total, B, C, H, W]
    """
    T_total = x1_seq.shape[0]
    assert T_total % K == 0
    n_blocks = T_total // K

    s_seq = torch.empty_like(x1_seq)
    v_carry = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)

    numel_per_step = B * C * H * W
    BLOCK = 1024
    grid = (triton.cdiv(numel_per_step, BLOCK),)

    # Flatten views for the kernel (rely on contiguity)
    for block_idx in range(n_blocks):
        bs = block_idx * K
        x1_block = x1_seq[bs:bs + K].contiguous().view(K, -1)
        x2_block = x2_seq[bs:bs + K].contiguous().view(K, -1)
        s_block = s_seq[bs:bs + K].view(K, -1)
        v_flat = v_carry.view(-1)

        add_lif_block_kernel[grid](
            x1_block, x2_block, s_block, v_flat,
            K=K,
            numel=numel_per_step,
            tau=TAU, v_th=V_TH, v_reset=V_RESET,
            stride_t=numel_per_step,
            BLOCK=BLOCK,
        )
    return s_seq


# ============================================================
# Reference for bit-exact parity: pure torch add + py LIF
# (This is NOT one of the SpikingJelly baselines — it's a
#  numerical reference that shares floating-point ops with v9.)
# ============================================================

def reference_add_lif(x1_seq, x2_seq, T):
    v = torch.zeros(B, C, H, W, device=device)
    spikes = []
    for t in range(T):
        z_t = x1_seq[t] + x2_seq[t]
        v = v + (z_t - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


# ============================================================
# SpikingJelly baselines
# ============================================================

def make_sj_lif(backend):
    """
    Build a SpikingJelly LIFNode with v_th=1, v_reset=0, tau=2.
    SpikingJelly's LIFNode uses:
        v_{t} = v_{t-1} + (x_t - (v_{t-1} - v_reset)) / tau
    which matches our reference LIF exactly.
    """
    lif = neuron.LIFNode(
        tau=TAU,
        v_threshold=V_TH,
        v_reset=V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode='m',     # multi-step mode
        backend=backend,
    ).to(device)
    return lif


def sj_baseline(x1_seq, x2_seq, lif_node):
    """
    Standard SpikingJelly residual pattern:
        y_seq = x1_seq + x2_seq      # torch add (NOT fused)
        s_seq = lif_node(y_seq)      # multi-step LIF (fused across T
                                     # by the chosen backend)
    """
    functional.reset_net(lif_node)
    y_seq = x1_seq + x2_seq
    s_seq = lif_node(y_seq)
    return s_seq


# ============================================================
# Timing
# ============================================================

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
        'stdev': statistics.stdev(samples),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("v9: F6 Add→LIF fused kernel")
    print(f"Shape: T={T_TOTAL}, B={B}, C={C}, H=W={H}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    print()

    x1_seq = torch.randn(T_TOTAL, B, C, H, W, device=device)
    x2_seq = torch.randn(T_TOTAL, B, C, H, W, device=device)

    # Step 1: parity vs local reference (bit-exact expected)
    print("Step 1: Parity — v9 fused vs local torch reference")
    print(f"{'K':<4} {'exact':<8} {'max_diff':<12} {'diff_spikes':<14} {'spike_rate':<12}")
    print('-' * 60)
    with torch.no_grad():
        s_ref = reference_add_lif(x1_seq, x2_seq, T_TOTAL)

    all_pass = True
    for K in K_LIST:
        s_tri = triton_add_lif_blocked(x1_seq, x2_seq, K)
        exact = torch.equal(s_ref, s_tri)
        max_d = (s_ref - s_tri).abs().max().item()
        n_diff = (s_ref != s_tri).sum().item()
        rate = s_tri.mean().item()
        print(f"{K:<4} {str(exact):<8} {max_d:<12.2e} "
              f"{n_diff:<14,} {rate:<12.4f}")
        if not exact:
            all_pass = False
    print()
    if not all_pass:
        print("FAIL: parity broken")
        return
    print("PASS: all K bit-exact vs local reference ✓")
    print()

    # Step 2: sanity-check v9 vs SpikingJelly
    # SpikingJelly uses its own internal ops so we don't expect bit-exact
    # match, just numerical equivalence
    print("Step 2: v9 vs SpikingJelly (numerical equivalence)")
    lif_torch = make_sj_lif('torch')
    with torch.no_grad():
        s_sj_torch = sj_baseline(x1_seq, x2_seq, lif_torch)
    s_v9 = triton_add_lif_blocked(x1_seq, x2_seq, K=16)
    max_d = (s_sj_torch - s_v9).abs().max().item()
    n_diff_sj = (s_sj_torch != s_v9).sum().item()
    print(f"  v9 vs SJ(torch) max_diff: {max_d:.2e}, "
          f"diff_spikes: {n_diff_sj:,}/{s_v9.numel():,} "
          f"({n_diff_sj/s_v9.numel()*100:.4f}%)")
    print(f"  SJ(torch) spike rate: {s_sj_torch.mean().item():.4f}")
    print(f"  v9       spike rate: {s_v9.mean().item():.4f}")
    if max_d < 1e-4:
        print("  numerical equivalence OK")
    else:
        print("  ⚠ larger than expected — check SJ's LIF formula")
    print()

    # Step 3: wall-clock
    print("Step 3: Wall-clock vs SpikingJelly baselines")
    print(f"{'config':<22} {'wall (ms)':<14} {'stdev':<10} "
          f"{'vs torch':<12} {'vs cupy':<12}")
    print('-' * 78)
    results = []

    # Baseline 1: SJ LIF torch backend
    def run_sj_torch():
        with torch.no_grad():
            _ = sj_baseline(x1_seq, x2_seq, lif_torch)
    for _ in range(N_WARMUP):
        run_sj_torch()
    torch.cuda.synchronize()
    t_sj_torch = cuda_time_stats(run_sj_torch)
    results.append(('sj_naive_torch', t_sj_torch))

    # Baseline 2: SJ LIF cupy backend (real SOTA)
    try:
        lif_cupy = make_sj_lif('cupy')
        def run_sj_cupy():
            with torch.no_grad():
                _ = sj_baseline(x1_seq, x2_seq, lif_cupy)
        for _ in range(N_WARMUP):
            run_sj_cupy()
        torch.cuda.synchronize()
        t_sj_cupy = cuda_time_stats(run_sj_cupy)
        results.append(('sj_naive_cupy', t_sj_cupy))
        cupy_available = True
    except Exception as e:
        print(f"  (cupy backend unavailable: {e})")
        t_sj_cupy = None
        cupy_available = False

    # v9 at each K
    for K in K_LIST:
        def run_v9(K=K):
            _ = triton_add_lif_blocked(x1_seq, x2_seq, K)
        for _ in range(N_WARMUP):
            run_v9()
        torch.cuda.synchronize()
        t_v9 = cuda_time_stats(run_v9)
        results.append((f'v9_K={K}', t_v9))

    t_torch_med = t_sj_torch['median']
    t_cupy_med = t_sj_cupy['median'] if cupy_available else float('nan')
    for name, stats in results:
        med = stats['median']
        std = stats['stdev']
        vs_torch = t_torch_med / med
        vs_cupy = t_cupy_med / med if cupy_available else float('nan')
        cupy_str = f"{vs_cupy:<12.2f}x" if cupy_available else "N/A"
        print(f"{name:<22} {med:<14.3f} {std:<10.3f} "
              f"{vs_torch:<12.2f}x {cupy_str}")
    print()

    # Save
    out = {
        'kernel': 'F6_add_lif',
        'device': torch.cuda.get_device_name(0),
        'shape': {'T': T_TOTAL, 'B': B, 'C': C, 'H': H, 'W': W},
        'cupy_available': cupy_available,
        'results': [{'name': n, **s} for n, s in results],
    }
    with open('v9_v100_baseline.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved to v9_v100_baseline.json")


if __name__ == '__main__':
    main()