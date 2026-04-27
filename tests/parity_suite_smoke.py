"""parity_suite_smoke: smoke-level bit-exact matrix for the §3.10 substrate.

Runs SEW-ResNet18 end-to-end, swaps each STFusion layer's K dynamically via
forward_with_k, and verifies that the network's final output is bit-exact
identical to the K=T reference for every K choice. This is a smoke-level
version of the full parity_suite (which would also iterate over more
networks, T values, batch sizes, and CTF backends).

Stage 8 deliverable. Future work: extend to the 19+17 network coverage
matrix (Stage 8 full version per the audit's problem 6) — only after the
§3.10 K-sweep experiment has produced primary data, since that's the
load-bearing experiment for §5.2.

Run:
    cd /path/to/CATFuse
    python tests/parity_suite_smoke.py
"""
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron, surrogate
        from spikingjelly.activation_based.model import sew_resnet
        from catfuse import optimize
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception as e:
        print(f"SKIP: imports failed: {e}")
        return 0

    if not torch.cuda.is_available() or SparseFlow is None:
        print("SKIP: needs CUDA + SparseFlow")
        return 0

    device = "cuda:0"

    # Build SEW-ResNet18 ADD
    net = sew_resnet.sew_resnet18(
        pretrained=False, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    functional.set_step_mode(net, "m")
    net = net.to(device).eval()

    print("=" * 70)
    print("parity_suite_smoke: SEW-RN18 full-network K-sweep bit-exact matrix")
    print("=" * 70)

    matrix_pass = True

    for T in [4, 8]:
        for B in [1, 2]:
            torch.manual_seed(0)
            x = torch.randn(T, B, 3, 32, 32, device=device)

            # Build a fresh fused network for this (T, B) combination
            fused, _ = optimize(net, T=T, use_sparseflow=True)
            fused = fused.to(device).eval()

            # Reference: default forward (every STFusion uses its single-block
            # forward, which == forward_with_k(K=T))
            with torch.no_grad():
                functional.reset_net(fused)
                y_ref = fused(x).clone()

            st_layers = [m for m in fused.modules()
                         if isinstance(m, STFusionConvBNLIF)]
            n_st = len(st_layers)

            print(f"\n  T={T}, B={B}, STFusion layers={n_st}")
            print(f"    {'K':>3s}  {'max_diff_vs_K=T':>20s}  status")

            K_values = [K for K in [1, 2, 4, 8] if K <= T]
            for K in K_values:
                # Monkey-patch every STFusion's _batchfold_forward to use K=K
                originals = []
                for layer in st_layers:
                    originals.append(layer._batchfold_forward)
                    def _make_K_forward(layer, K=K):
                        def _bf(x):
                            if layer._impl_sparse is not None:
                                return layer._impl_sparse.forward_with_k(
                                    x, layer.spec, layer._ensure_params(),
                                    layer.state, K=K)
                            return layer._impl_dense.forward(
                                x, layer.spec, layer._ensure_params(),
                                layer.state)
                        return _bf
                    layer._batchfold_forward = _make_K_forward(layer)

                with torch.no_grad():
                    functional.reset_net(fused)
                    y_K = fused(x)

                # Restore
                for layer, orig in zip(st_layers, originals):
                    layer._batchfold_forward = orig

                max_diff = (y_K - y_ref).abs().max().item()
                ok = max_diff == 0
                mark = "OK" if ok else "FAIL"
                print(f"    {K:>3d}  {max_diff:>20.2e}  {mark}")
                if not ok:
                    matrix_pass = False

    print()
    print("=" * 70)
    if matrix_pass:
        print("PASS: every (T, B, K) combination produces bit-exact output")
        print("=" * 70)
        return 0
    else:
        print("FAIL: some (T, B, K) combination diverged from K=T reference")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
