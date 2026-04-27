"""Test 30b: §3.10 K-sweep using DEPLOYMENT path (Stage 3-6 APIs).

This is the corrected version of test_30_k_sweep_hbm.py per Stage 8.

The original test_30 used an INLINE Python loop for the LIF dynamics in
its `hybrid_fn` benchmark — `v = v * 0.5 + z_block[t] * 0.5` rather than
the actual lif_sequential Triton kernel that production patterns use.
This made test_30's wall-clock numbers unrepresentative of CATFuse's
actual deployment behavior.

This test_30b replaces that with the real Implementation APIs:

  - analytic I/O bytes:    impl.analytic_io_cost(spec, T, B, H, W, K)
                           (Stage 4 — exact §3.9 formula)
  - empirical K-sweep:     impl.forward_with_k(x, ..., K=K)
                           (Stage 6 — chunks T into ceil(T/K) blocks,
                            relies on StateBuffer for StateCarry(LIF))
  - parity within K:       compare against forward_with_k(K=T), expect
                           max_diff = 0 (Corollary 3.17 + §3.13 Lemma 3.14)

Output: per (config, T, K) row showing analytic intermediate_io,
empirical wall-clock, and bit-exact parity vs K=T.

Run:
    cd /path/to/CATFuse
    python tests/test_30b_k_sweep_real.py
"""
import os
import sys
import time
import statistics

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer as sj_layer, functional

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
from catfuse.implementations import SparseFlow, DenseKeep

DEVICE = "cuda:0"
torch.backends.cudnn.benchmark = False
N_WARMUP = 30
N_ITER = 100


def bench(fn, n_warmup=N_WARMUP, n_iter=N_ITER, n_repeat=3):
    """Median wall-clock per fn() call, in microseconds."""
    times = []
    for _ in range(n_repeat):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_iter * 1e6)
    return statistics.median(times)


def build_layer(B, Cin, Cout, H, T):
    """Build a fused STFusion layer + sparse input."""
    torch.manual_seed(42)
    conv = sj_layer.Conv2d(Cin, Cout, 3, padding=1, bias=False).to(DEVICE)
    bn = sj_layer.BatchNorm2d(Cout).to(DEVICE)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(DEVICE)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(DEVICE).eval()
    return fused


def main():
    # Configurations selected to overlap with SEW-RN18 / VGG11 layer shapes
    configs = [
        # (label, B, Cin, Cout, H, sparsity)
        ("layer3-shape  256x14",  2, 256, 256, 14, 0.85),
        ("layer4-shape  512x7",   2, 512, 512,  7, 0.85),
        ("layer3-shape (denser)", 2, 256, 256, 14, 0.50),
    ]
    T_values = [4, 8]

    print("=" * 110)
    print("Test 30b: §3.10 K-sweep using DEPLOYMENT path (Stage 3-6 APIs)")
    print("=" * 110)
    print()
    print("All numbers are per forward() call.")
    print("  analytic_io  = SparseFlow.analytic_io_cost(...).intermediate_io  (z_io + v_io)")
    print("  wall_us      = median of forward_with_k(K) over",
          f"{N_ITER} iterations × 3 repeats")
    print("  parity       = max |forward_with_k(K=K) - forward_with_k(K=T)|")
    print("                 (expected: 0.00e+00, §3.13 Lemma 3.14)")
    print()

    for label, B, Cin, Cout, H, sp in configs:
        print(f"\n{'─' * 110}")
        print(f"Config: {label}  (B={B}, Cin={Cin}, Cout={Cout}, H={H}, sparsity={sp:.0%})")
        print(f"{'─' * 110}")

        for T in T_values:
            fused = build_layer(B, Cin, Cout, H, T)
            # Sparse input
            torch.manual_seed(99)
            x = (torch.rand(T, B, Cin, H, H, device=DEVICE) > sp).float()
            spec = fused.spec
            params = fused._ensure_params()

            # Reference output: K=T
            functional.reset_net(fused)
            y_ref = fused._impl_sparse.forward_with_k(
                x, spec, params, fused.state, K=T).clone()

            print(f"\n  T={T}")
            print(f"  {'K':>3s} {'#blk':>4s} "
                  f"{'analytic_io_KB':>14s} {'analytic_total_KB':>17s} "
                  f"{'wall_us':>10s} {'wall_per_step_us':>17s} "
                  f"{'parity':>10s}")
            print(f"  {'─' * 88}")

            K_values = [K for K in [1, 2, 4, 8] if K <= T]
            for K in K_values:
                # Analytic via Stage 4 IOCost
                cost = fused._impl_sparse.analytic_io_cost(
                    spec, T=T, B=B, H_in=H, W_in=H, K=K)

                # Empirical wall-clock via Stage 6 forward_with_k
                def _fn():
                    functional.reset_net(fused)
                    return fused._impl_sparse.forward_with_k(
                        x, spec, params, fused.state, K=K)
                t_us = bench(_fn)

                # Parity vs K=T
                functional.reset_net(fused)
                y_K = fused._impl_sparse.forward_with_k(
                    x, spec, params, fused.state, K=K)
                max_diff = (y_K - y_ref).abs().max().item()

                print(f"  {K:>3d} {cost.num_blocks:>4d} "
                      f"{cost.intermediate_io/1024:>14.2f} "
                      f"{cost.total/1024:>17.2f} "
                      f"{t_us:>10.2f} "
                      f"{t_us/T:>17.2f} "
                      f"{max_diff:>10.2e}")

            # DenseKeep baseline for comparison (one row at K=T)
            dk_cost = fused._impl_dense.analytic_io_cost(
                spec, T=T, B=B, H_in=H, W_in=H)

            def _fn_dense():
                functional.reset_net(fused)
                return fused._impl_dense.forward(
                    x, spec, params, fused.state)
            t_dk = bench(_fn_dense)
            print(f"  {'DK':>3s} {'  -':>4s} "
                  f"{dk_cost.intermediate_io/1024:>14.2f} "
                  f"{dk_cost.total/1024:>17.2f} "
                  f"{t_dk:>10.2f} "
                  f"{t_dk/T:>17.2f} "
                  f"{'  baseline':>10s}")

            # §3.9 ratio at K=T
            sf_cost_KT = fused._impl_sparse.analytic_io_cost(
                spec, T=T, B=B, H_in=H, W_in=H, K=T)
            ratio = sf_cost_KT.intermediate_io / dk_cost.intermediate_io
            print(f"  §3.9 prediction at K=T={T}: SF intermediate_io / DK = "
                  f"{ratio:.4f} (= 1/(T+1) = {1.0/(T+1):.4f})")

    print()
    print("=" * 110)
    print("Notes:")
    print("  - parity column should be 0 for every K — empirical proof of")
    print("    §3.13 Lemma 3.14 on the live deployment kernel.")
    print("  - analytic_io is upper-bound (worst-case dense input);")
    print("    sparsity-aware kernel may move fewer bytes empirically.")
    print("  - wall_us mixes input loading, conv compute, BN, LIF, output")
    print("    writes — it's a holistic latency, not pure HBM-bound metric.")
    print("=" * 110)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Test 30b requires CUDA")
        sys.exit(0)
    main()
