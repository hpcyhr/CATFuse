"""
v6b: Triton LIF forward + backward kernel with autograd integration

- fwd kernel: stores v_raw_seq for backward
- bwd kernel: reverse-time BPTT, computes grad_x
- torch.autograd.Function wraps both
- parity check: forward and backward vs SJ torch backend
- benchmark: fwd and fwd+bwd, triton vs sj_torch vs sj_cupy
"""
import math
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
ALPHA = 2.0  # ATan surrogate alpha

N_WARMUP = 50
N_ITER = 100
N_REPEAT = 11


# ============================================================
# Triton kernels
# ============================================================

@triton.jit
def lif_fwd_kernel(
    x_ptr,         # [T, N] input current
    s_ptr,         # [T, N] output spikes
    v_raw_ptr,     # [T, N] saved pre-reset membrane (for backward)
    T: tl.constexpr,
    N,
    tau: tl.constexpr,
    v_th: tl.constexpr,
    v_reset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    v = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in tl.static_range(T):
        x = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0)
        # Charge
        v_raw = v + (x - (v - v_reset)) / tau
        # Fire
        spike = (v_raw >= v_th).to(tl.float32)
        # Save pre-reset v (needed by backward)
        tl.store(v_raw_ptr + t * N + offs, v_raw, mask=mask)
        tl.store(s_ptr + t * N + offs, spike, mask=mask)
        # Hard reset
        v = v_raw * (1.0 - spike) + v_reset * spike


@triton.jit
def lif_bwd_kernel(
    grad_s_ptr,    # [T, N] incoming grad w.r.t. spikes
    v_raw_ptr,     # [T, N] saved from forward
    grad_x_ptr,    # [T, N] output grad w.r.t. x
    T: tl.constexpr,
    N,
    tau: tl.constexpr,
    v_th: tl.constexpr,
    v_reset: tl.constexpr,
    half_alpha: tl.constexpr,      # alpha / 2
    pi_half_alpha: tl.constexpr,   # pi / 2 * alpha
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    grad_v_prev = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t_rev in tl.static_range(T):
        t = T - 1 - t_rev

        v_raw = tl.load(v_raw_ptr + t * N + offs, mask=mask, other=0.0)
        grad_s = tl.load(grad_s_ptr + t * N + offs, mask=mask, other=0.0)

        # Recompute spike from v_raw
        spike = (v_raw >= v_th).to(tl.float32)

        # Grad flowing back from the reset gate at step t
        grad_v_next = grad_v_prev
        grad_v_raw_from_reset = grad_v_next * (1.0 - spike)
        grad_s_from_reset = grad_v_next * (v_reset - v_raw)

        # Total grad on spike at step t
        grad_s_total = grad_s + grad_s_from_reset

        # ATan surrogate: (alpha/2) / (1 + (pi/2 * alpha * (v_raw - v_th))^2)
        diff = pi_half_alpha * (v_raw - v_th)
        sg_denom = 1.0 + diff * diff
        surrogate_grad = half_alpha / sg_denom
        grad_v_raw_from_s = grad_s_total * surrogate_grad

        # Total grad on v_raw at step t
        grad_v_raw = grad_v_raw_from_reset + grad_v_raw_from_s

        # Charge: v_raw = v_{t-1} + (x - (v_{t-1} - v_reset)) / tau
        # d(v_raw)/d(x) = 1/tau
        # d(v_raw)/d(v_{t-1}) = 1 - 1/tau
        grad_x = grad_v_raw / tau
        grad_v_prev = grad_v_raw * (1.0 - 1.0 / tau)

        tl.store(grad_x_ptr + t * N + offs, grad_x, mask=mask)


# ============================================================
# Autograd function
# ============================================================

class TritonLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, tau, v_th, v_reset, alpha):
        assert x_seq.is_cuda and x_seq.dtype == torch.float32 and x_seq.is_contiguous()
        T = x_seq.shape[0]
        N = x_seq.numel() // T

        s_seq = torch.empty_like(x_seq)
        v_raw_seq = torch.empty_like(x_seq)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        lif_fwd_kernel[grid](
            x_seq, s_seq, v_raw_seq,
            T=T, N=N, tau=tau, v_th=v_th, v_reset=v_reset,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(v_raw_seq)
        ctx.tau = tau
        ctx.v_th = v_th
        ctx.v_reset = v_reset
        ctx.alpha = alpha
        ctx.T = T
        ctx.N = N
        return s_seq

    @staticmethod
    def backward(ctx, grad_s_seq):
        (v_raw_seq,) = ctx.saved_tensors
        T = ctx.T
        N = ctx.N

        grad_s_seq = grad_s_seq.contiguous()
        grad_x_seq = torch.empty_like(grad_s_seq)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        lif_bwd_kernel[grid](
            grad_s_seq, v_raw_seq, grad_x_seq,
            T=T, N=N,
            tau=ctx.tau, v_th=ctx.v_th, v_reset=ctx.v_reset,
            half_alpha=ctx.alpha / 2.0,
            pi_half_alpha=(math.pi / 2.0) * ctx.alpha,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_x_seq, None, None, None, None


def triton_lif(x_seq, tau=TAU, v_th=V_TH, v_reset=V_RESET, alpha=ALPHA):
    return TritonLIF.apply(x_seq, tau, v_th, v_reset, alpha)


# ============================================================
# Parity check (forward + backward)
# ============================================================

def check_parity(T=8):
    torch.manual_seed(0)
    x_init = torch.randn(T, B, C_out, H, W, device=device) * 2.0

    # SJ torch reference
    x_sj = x_init.detach().clone().requires_grad_(True)
    lif_sj = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(alpha=ALPHA),
        step_mode='m', backend='torch',
    ).to(device)
    functional.reset_net(lif_sj)
    s_sj = lif_sj(x_sj)
    grad_out = torch.ones_like(s_sj)
    s_sj.backward(grad_out)
    gx_sj = x_sj.grad.clone()

    # Triton
    x_tri = x_init.detach().clone().requires_grad_(True)
    s_tri = triton_lif(x_tri.contiguous())
    s_tri.backward(torch.ones_like(s_tri))
    gx_tri = x_tri.grad.clone()

    fwd_exact = torch.equal(s_sj, s_tri)
    fwd_max_diff = (s_sj - s_tri).abs().max().item()
    bwd_max_abs = (gx_sj - gx_tri).abs().max().item()
    bwd_max_rel = ((gx_sj - gx_tri).abs() / (gx_sj.abs() + 1e-8)).max().item()

    print(f"Parity T={T}:")
    print(f"  Forward exact: {fwd_exact}, max diff: {fwd_max_diff}")
    print(f"  Backward max abs diff: {bwd_max_abs:.2e}")
    print(f"  Backward max rel diff: {bwd_max_rel:.2e}")
    print(f"  grad_sj     stats: mean={gx_sj.mean().item():.6f}, std={gx_sj.std().item():.6f}")
    print(f"  grad_triton stats: mean={gx_tri.mean().item():.6f}, std={gx_tri.std().item():.6f}")

    passed = fwd_exact and bwd_max_abs < 1e-5
    print(f"  Verdict: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================
# Models
# ============================================================

class SJModel(nn.Module):
    def __init__(self, backend, T):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
            surrogate_function=surrogate.ATan(alpha=ALPHA),
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
        return triton_lif(z)


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


def fmt(v, std):
    if v != v:
        return 'FAIL'
    return f"{v:.3f}±{std:.2f}"


def bench_fwd(model_factory, T):
    model = model_factory().to(device)
    x = torch.randn(T, B, C_in, H, W, device=device)

    def step():
        with torch.no_grad():
            model(x)

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


def bench_fwdbwd(model_factory, T):
    model = model_factory().to(device)
    x = torch.randn(T, B, C_in, H, W, device=device, requires_grad=True)

    def step():
        model.conv.zero_grad(set_to_none=True)
        x.grad = None
        out = model(x)
        out.sum().backward()

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


# ============================================================
# Main
# ============================================================

def main():
    print(f"\nShape: B={B}, C={C_in}, H=W={H}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}\n")

    print("Step 1: Parity check (fwd + bwd)")
    all_pass = True
    for T in [4, 8, 16, 32]:
        if not check_parity(T=T):
            all_pass = False
    if not all_pass:
        print("WARNING: parity check failed for some T. Benchmarks below may be meaningless.")
        print()

    print("Step 2: End-to-end benchmark")
    hdr = (f"{'T':<4} {'mode':<7} "
           f"{'sj_torch':<12} {'sj_cupy':<12} {'triton':<12} "
           f"{'tri/cupy':<9} {'tri/torch':<10}")
    print(hdr)
    print('-' * len(hdr))

    results = []
    for T in T_list:
        for mode in ['fwd', 'fwdbwd']:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            bench_fn = bench_fwd if mode == 'fwd' else bench_fwdbwd
            sjt = bench_fn(lambda: SJModel('torch', T), T)
            sjc = bench_fn(lambda: SJModel('cupy', T), T)
            tri = bench_fn(lambda: TritonModel(T), T)

            tri_vs_cupy = sjc['median'] / tri['median']
            tri_vs_torch = sjt['median'] / tri['median']

            results.append({
                'T': T, 'mode': mode,
                'sj_torch': sjt['median'], 'sj_torch_std': sjt['stdev'],
                'sj_cupy': sjc['median'], 'sj_cupy_std': sjc['stdev'],
                'triton': tri['median'], 'triton_std': tri['stdev'],
                'tri_vs_cupy': tri_vs_cupy,
                'tri_vs_torch': tri_vs_torch,
            })

            print(f"{T:<4} {mode:<7} "
                  f"{fmt(sjt['median'], sjt['stdev']):<12} "
                  f"{fmt(sjc['median'], sjc['stdev']):<12} "
                  f"{fmt(tri['median'], tri['stdev']):<12} "
                  f"{tri_vs_cupy:<9.2f} {tri_vs_torch:<10.2f}")
        print()

    print("\n=== Headline numbers (fwd+bwd) ===")
    print(f"{'T':<4} {'sj_torch':<10} {'sj_cupy':<10} {'triton':<10} "
          f"{'vs torch':<10} {'vs cupy':<10}")
    for r in results:
        if r['mode'] == 'fwdbwd':
            print(f"{r['T']:<4} {r['sj_torch']:<10.3f} {r['sj_cupy']:<10.3f} "
                  f"{r['triton']:<10.3f} "
                  f"{r['tri_vs_torch']:<10.2f}x {r['tri_vs_cupy']:<10.2f}x")

    with open('benchmark_v6b_triton_fwdbwd.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_v6b_triton_fwdbwd.json")


if __name__ == '__main__':
    main()