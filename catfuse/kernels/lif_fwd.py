"""
v6a: Triton multi-step LIF forward kernel

Implements a stateless LIF forward kernel in Triton and benchmarks it against:
  - SpikingJelly torch backend
  - SpikingJelly cupy backend
  - Our pure PyTorch stateless LIF (from v5)

Target: match or beat sj_cupy on forward-only, T=4..32.
No backward yet (that's v6b).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
from spikingjelly.activation_based import neuron, functional, surrogate
import statistics
import json

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

B, C_in, C_out, H, W = 32, 128, 128, 16, 16
T_list = [4, 8, 16, 32]
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0

N_WARMUP = 50
N_ITER = 100
N_REPEAT = 11


# ============================================================
# Triton LIF forward kernel
# ============================================================

@triton.jit
def lif_fwd_kernel(
    x_ptr,       # [T, N] input current
    s_ptr,       # [T, N] output spikes
    T: tl.constexpr,
    N,           # total neurons = B*C*H*W
    tau: tl.constexpr,
    v_th: tl.constexpr,
    v_reset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Persistent membrane potential in registers
    v = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over T steps; T is constexpr so Triton will unroll
    for t in tl.static_range(T):
        # Load input current for this time step
        x = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0)

        # Charge: v += (x - (v - v_reset)) / tau
        v = v + (x - (v - v_reset)) / tau

        # Fire
        spike = (v >= v_th).to(tl.float32)

        # Hard reset: v = v_reset where spike, else v unchanged
        v = v * (1.0 - spike) + v_reset * spike

        # Store spike
        tl.store(s_ptr + t * N + offs, spike, mask=mask)


def triton_lif_forward(x_seq, tau=TAU, v_th=V_TH, v_reset=V_RESET):
    """
    x_seq: [T, B, C, H, W] float32, CUDA
    Returns: spike_seq [T, B, C, H, W]
    """
    assert x_seq.is_cuda and x_seq.dtype == torch.float32
    assert x_seq.is_contiguous()

    T = x_seq.shape[0]
    N = x_seq.numel() // T
    s_seq = torch.empty_like(x_seq)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    lif_fwd_kernel[grid](
        x_seq, s_seq,
        T=T, N=N,
        tau=tau, v_th=v_th, v_reset=v_reset,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return s_seq


# ============================================================
# Parity check
# ============================================================

def check_parity(T=8):
    torch.manual_seed(0)

    sj_lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(alpha=2.0),
        step_mode='m', backend='torch',
    ).to(device)

    x = torch.randn(T, B, C_out, H, W, device=device) * 2.0  # scaled to get good spike rate

    # SJ reference
    functional.reset_net(sj_lif)
    with torch.no_grad():
        s_sj = sj_lif(x.clone())

    # Triton
    s_triton = triton_lif_forward(x.contiguous())

    max_diff = (s_sj - s_triton).abs().max().item()
    match = torch.equal(s_sj, s_triton)
    print(f"Parity check (T={T}): max_diff={max_diff}, exact_match={match}")
    print(f"  SJ spike rate:     {s_sj.mean().item():.4f}")
    print(f"  Triton spike rate: {s_triton.mean().item():.4f}")
    return match


# ============================================================
# Full Conv -> LIF pipelines for benchmarking
# ============================================================

class SJModel(nn.Module):
    def __init__(self, backend, T):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
            surrogate_function=surrogate.ATan(alpha=2.0),
            step_mode='m', backend=backend,
        )
        self.T = T

    def forward(self, x):
        functional.reset_net(self.lif)
        z = self.conv(x.flatten(0, 1)).reshape(self.T, B, C_out, H, W)
        return self.lif(z)


class TritonModel(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=False)
        self.T = T

    def forward(self, x):
        z = self.conv(x.flatten(0, 1)).reshape(self.T, B, C_out, H, W).contiguous()
        return triton_lif_forward(z)


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
        'stdev': statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_model(model_fn, T):
    """model_fn: () -> nn.Module (fresh). Returns fwd-only timing."""
    model = model_fn().to(device)
    x = torch.randn(T, B, C_in, H, W, device=device)

    def step():
        with torch.no_grad():
            model(x)

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


def benchmark_lif_only(lif_fn, T):
    """lif_fn: (z) -> spikes. Benchmarks just the LIF part on precomputed z."""
    z = torch.randn(T, B, C_out, H, W, device=device).contiguous()

    def step():
        with torch.no_grad():
            lif_fn(z)

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


def fmt(v, std):
    if v != v:
        return 'FAIL'
    return f"{v:.3f}±{std:.2f}"


# ============================================================
# Main
# ============================================================

def main():
    print(f"\nShape: B={B}, C={C_in}, H=W={H}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Setup: warmup={N_WARMUP}, iter={N_ITER}, repeat={N_REPEAT}\n")

    print("Step 1: Parity check")
    check_parity(T=4)
    check_parity(T=16)
    print()

    print("Step 2: LIF-only benchmark (precomputed z, isolates LIF cost)")
    hdr = f"{'T':<4} {'sj_torch':<12} {'sj_cupy':<12} {'triton':<12} {'tri/cupy':<10} {'tri/torch':<10}"
    print(hdr)
    print('-' * len(hdr))

    # Build SJ lif modules once for reuse in lif-only timing
    def make_sj_lif_fn(backend):
        lif = neuron.LIFNode(
            tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
            surrogate_function=surrogate.ATan(alpha=2.0),
            step_mode='m', backend=backend,
        ).to(device)
        def fn(z):
            functional.reset_net(lif)
            return lif(z)
        return fn

    lif_only_results = []
    for T in T_list:
        torch.cuda.empty_cache()
        sjt = benchmark_lif_only(make_sj_lif_fn('torch'), T)
        sjc = benchmark_lif_only(make_sj_lif_fn('cupy'), T)
        tri = benchmark_lif_only(triton_lif_forward, T)
        tri_vs_cupy = sjc['median'] / tri['median']
        tri_vs_torch = sjt['median'] / tri['median']
        lif_only_results.append({
            'T': T,
            'sj_torch_lif': sjt['median'], 'sj_torch_lif_std': sjt['stdev'],
            'sj_cupy_lif': sjc['median'], 'sj_cupy_lif_std': sjc['stdev'],
            'triton_lif': tri['median'], 'triton_lif_std': tri['stdev'],
            'tri_vs_cupy': tri_vs_cupy, 'tri_vs_torch': tri_vs_torch,
        })
        print(f"{T:<4} {fmt(sjt['median'], sjt['stdev']):<12} "
              f"{fmt(sjc['median'], sjc['stdev']):<12} "
              f"{fmt(tri['median'], tri['stdev']):<12} "
              f"{tri_vs_cupy:<10.2f} {tri_vs_torch:<10.2f}")
    print()

    print("Step 3: Full Conv -> LIF forward (end-to-end)")
    hdr = f"{'T':<4} {'sj_torch':<12} {'sj_cupy':<12} {'triton':<12} {'tri/cupy':<10} {'tri/torch':<10}"
    print(hdr)
    print('-' * len(hdr))

    e2e_results = []
    for T in T_list:
        torch.cuda.empty_cache()
        sjt = benchmark_model(lambda: SJModel('torch', T), T)
        sjc = benchmark_model(lambda: SJModel('cupy', T), T)
        tri = benchmark_model(lambda: TritonModel(T), T)
        tri_vs_cupy = sjc['median'] / tri['median']
        tri_vs_torch = sjt['median'] / tri['median']
        e2e_results.append({
            'T': T,
            'sj_torch_e2e': sjt['median'], 'sj_torch_e2e_std': sjt['stdev'],
            'sj_cupy_e2e': sjc['median'], 'sj_cupy_e2e_std': sjc['stdev'],
            'triton_e2e': tri['median'], 'triton_e2e_std': tri['stdev'],
            'tri_vs_cupy': tri_vs_cupy, 'tri_vs_torch': tri_vs_torch,
        })
        print(f"{T:<4} {fmt(sjt['median'], sjt['stdev']):<12} "
              f"{fmt(sjc['median'], sjc['stdev']):<12} "
              f"{fmt(tri['median'], tri['stdev']):<12} "
              f"{tri_vs_cupy:<10.2f} {tri_vs_torch:<10.2f}")
    print()

    with open('benchmark_v6a_triton_fwd.json', 'w') as f:
        json.dump({'lif_only': lif_only_results, 'e2e': e2e_results}, f, indent=2)
    print("Saved to benchmark_v6a_triton_fwd.json")


if __name__ == '__main__':
    main()