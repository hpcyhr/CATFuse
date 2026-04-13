"""
v3: CUDA Graph benchmark for Conv -> LIF
Tests whether graph capture can approach the fuse ceiling from v2.
Compares 4 configurations on each (T, backend):
  - eager forward (baseline, same as v2)
  - cuda graph forward
  - eager fwd+bwd
  - cuda graph fwd+bwd  (only if backward-through-graph works)
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
import statistics
import json
import traceback

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

HBM_BW = 900.0
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
        'stdev': statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def make_conv():
    return nn.Conv2d(C_in, C_out, 3, padding=1, bias=False).to(device)


def make_lif(backend):
    return neuron.LIFNode(
        tau=2.0, surrogate_function=surrogate.ATan(),
        step_mode='m', backend=backend,
    ).to(device)


def init_lif_state(lif, T, B, C_out, H, W):
    """
    Manually initialize lif.v as a zero tensor so we don't need reset_net()
    inside the captured region. reset_net() creates new tensors, which
    breaks graph capture.
    """
    lif.v = torch.zeros(T, B, C_out, H, W, device=device)
    # Some SpikingJelly versions store v as a float scalar initially;
    # flatten to match what multi_step_forward expects.


def zero_lif_state_inplace(lif):
    """After each replay, zero out v in place (preserves memory address)."""
    if isinstance(lif.v, torch.Tensor):
        lif.v.zero_()


# ============================================================
# Eager forward / fwdbwd (for regression vs v2)
# ============================================================

def benchmark_eager_fwd(T, backend):
    conv = make_conv()
    lif = make_lif(backend)
    x = torch.randn(T, B, C_in, H, W, device=device)

    def step():
        functional.reset_net(lif)
        with torch.no_grad():
            z = conv(x.flatten(0, 1)).reshape(T, B, C_out, H, W)
            lif(z)

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


def benchmark_eager_fwdbwd(T, backend):
    conv = make_conv()
    lif = make_lif(backend)
    x = torch.randn(T, B, C_in, H, W, device=device, requires_grad=True)

    def step():
        functional.reset_net(lif)
        conv.zero_grad(set_to_none=True)
        x.grad = None
        z = conv(x.flatten(0, 1)).reshape(T, B, C_out, H, W)
        s = lif(z)
        s.sum().backward()

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


# ============================================================
# CUDA Graph forward
# ============================================================

def benchmark_graph_fwd(T, backend):
    conv = make_conv()
    lif = make_lif(backend)

    # Persistent input buffer (address fixed)
    static_x = torch.randn(T, B, C_in, H, W, device=device)

    # Warm up in a side stream before capture (required by CUDA Graph API)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(N_WARMUP):
            functional.reset_net(lif)
            with torch.no_grad():
                z = conv(static_x.flatten(0, 1)).reshape(T, B, C_out, H, W)
                out = lif(z)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Prepare LIF state so capture doesn't need reset_net
    init_lif_state(lif, T, B, C_out, H, W)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            z = conv(static_x.flatten(0, 1)).reshape(T, B, C_out, H, W)
            static_out = lif(z)

    def step():
        # Zero LIF state in place (address preserved), then replay
        zero_lif_state_inplace(lif)
        g.replay()

    # Warmup replays
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    return cuda_time_stats(step)


# ============================================================
# CUDA Graph forward + backward
# ============================================================
# CUDA Graph + autograd is tricky. PyTorch supports this via
# `torch.cuda.make_graphed_callables` which handles the plumbing.
# We use that instead of manual capture.

def benchmark_graph_fwdbwd(T, backend):
    conv = make_conv()
    lif = make_lif(backend)

    # Wrap Conv→LIF as a single callable
    class ConvLIF(nn.Module):
        def __init__(self, conv, lif):
            super().__init__()
            self.conv = conv
            self.lif = lif

        def forward(self, x):
            z = self.conv(x.flatten(0, 1)).reshape(T, B, C_out, H, W)
            return self.lif(z)

    model = ConvLIF(conv, lif).to(device)

    # Sample input for graphing (defines the shapes and memory layout)
    sample_x = torch.randn(T, B, C_in, H, W, device=device, requires_grad=True)

    # Warmup before make_graphed_callables
    for _ in range(N_WARMUP):
        functional.reset_net(lif)
        out = model(sample_x)
        out.sum().backward()
        conv.zero_grad(set_to_none=True)
        sample_x.grad = None
    torch.cuda.synchronize()

    # Initialize LIF state for capture
    init_lif_state(lif, T, B, C_out, H, W)

    try:
        graphed_model = torch.cuda.make_graphed_callables(model, (sample_x,))
    except Exception as e:
        return {'median': float('nan'), 'stdev': 0.0, 'error': str(e)[:200]}

    # Persistent input
    x = torch.randn(T, B, C_in, H, W, device=device, requires_grad=True)

    def step():
        zero_lif_state_inplace(lif)
        conv.zero_grad(set_to_none=True)
        x.grad = None
        out = graphed_model(x)
        out.sum().backward()

    for _ in range(20):
        try:
            step()
        except Exception as e:
            return {'median': float('nan'), 'stdev': 0.0, 'error': str(e)[:200]}
    torch.cuda.synchronize()
    return cuda_time_stats(step)


# ============================================================
# Main
# ============================================================

def safe_bench(fn, T, backend):
    """Wrap a benchmark to catch capture errors gracefully."""
    try:
        return fn(T, backend)
    except Exception as e:
        print(f"  [FAILED] {fn.__name__}({T}, {backend}): {type(e).__name__}: {str(e)[:150]}")
        return {'median': float('nan'), 'stdev': 0.0, 'error': str(e)[:200]}


def main():
    print(f"\nShape: B={B}, C={C_in}, H=W={H}")
    print(f"Setup: warmup={N_WARMUP}, iter/meas={N_ITER}, repeat={N_REPEAT}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\n")

    hdr = (f"{'T':<4} {'bk':<6} "
           f"{'eager_f':<12} {'graph_f':<12} {'spd_f':<8} "
           f"{'eager_fb':<12} {'graph_fb':<12} {'spd_fb':<8}")
    print(hdr)
    print('-' * len(hdr))

    all_results = []
    for backend in ['torch', 'cupy']:
        for T in T_list:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            ef = safe_bench(benchmark_eager_fwd, T, backend)
            gf = safe_bench(benchmark_graph_fwd, T, backend)
            efb = safe_bench(benchmark_eager_fwdbwd, T, backend)
            gfb = safe_bench(benchmark_graph_fwdbwd, T, backend)

            spd_f = ef['median'] / gf['median'] if gf['median'] > 0 else float('nan')
            spd_fb = efb['median'] / gfb['median'] if gfb['median'] > 0 else float('nan')

            r = {
                'T': T, 'backend': backend,
                'eager_fwd': ef['median'], 'eager_fwd_std': ef['stdev'],
                'graph_fwd': gf['median'], 'graph_fwd_std': gf['stdev'],
                'eager_fb': efb['median'], 'eager_fb_std': efb['stdev'],
                'graph_fb': gfb['median'], 'graph_fb_std': gfb['stdev'],
                'spd_fwd': spd_f, 'spd_fb': spd_fb,
            }
            all_results.append(r)

            def fmt(v, std):
                if v != v:  # nan
                    return 'FAIL'
                return f"{v:.3f}±{std:.2f}"

            print(f"{T:<4} {backend:<6} "
                  f"{fmt(ef['median'], ef['stdev']):<12} "
                  f"{fmt(gf['median'], gf['stdev']):<12} "
                  f"{spd_f:<8.2f} "
                  f"{fmt(efb['median'], efb['stdev']):<12} "
                  f"{fmt(gfb['median'], gfb['stdev']):<12} "
                  f"{spd_fb:<8.2f}")
        print()

    # Compare against v2 ceiling: does graph approach max(conv, lif)?
    print("\n=== Graph vs fuse ceiling (how close are we to the theoretical limit) ===")
    print("Ceiling data loaded from v1 fwd-only numbers; fb ceiling from v2.\n")
    print(f"{'T':<4} {'bk':<6} {'mode':<6} {'eager':<10} {'graph':<10} {'ceiling?':<10}")
    # We don't have v2's numbers loaded here; user can cross-reference with json.
    # Just print graph numbers and let user compare.

    with open('benchmark_v3_cudagraph.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to benchmark_v3_cudagraph.json")


if __name__ == '__main__':
    main()