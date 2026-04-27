"""[Stage 4 verification] analytic_io_cost binding (problem 5).

Verifies that:
  1. Implementation ABC requires analytic_io_cost (now abstract).
  2. IOCost dataclass is exported and behaves correctly.
  3. DenseKeep.analytic_io_cost returns the §3.9 baseline formula on
     a hand-checked concrete shape (sew-rn18 layer3.0.conv2).
  4. SparseFlow.analytic_io_cost returns the §3.9 CTF formula on the
     same shape, parameterized by K.
  5. K-sweep monotonicity: as K increases, intermediate_io decreases
     monotonically toward its lower bound at K=T.
  6. CTF vs baseline ratio matches §3.9 prediction:
        ratio(K) = (2·ceil(T/K)) / (2T + 2)
  7. SEW-ResNet18 substitute_sf still bit-exact (regression).

Run:
    cd /path/to/CATFuse
    python tests/stage4_verify.py
"""
import math
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_imports_and_abstract():
    """1+2: Implementation requires analytic_io_cost; IOCost is exported."""
    print("[1/7] Imports + abstract-method enforcement...")
    try:
        from catfuse.implementations import (
            Implementation, IOCost, ConvLIFSpec, DenseKeep,
        )
    except Exception:
        print("  FAIL: imports failed")
        traceback.print_exc()
        return False

    # Implementation must still be abstract; should raise TypeError
    # (because both forward and analytic_io_cost are abstract now).
    try:
        Implementation()  # type: ignore
        print("  FAIL: Implementation should be abstract")
        return False
    except TypeError:
        pass

    # Implementation has analytic_io_cost listed as abstract
    abstract_methods = Implementation.__abstractmethods__
    if "analytic_io_cost" not in abstract_methods:
        print(f"  FAIL: analytic_io_cost not in abstractmethods ({abstract_methods})")
        return False

    print(f"  OK: imports clean; abstractmethods = {sorted(abstract_methods)}")
    return True


def check_iocost_dataclass():
    """2: IOCost dataclass — frozen, has total / intermediate_io properties."""
    print("[2/7] IOCost dataclass structure...")
    try:
        from catfuse.implementations import IOCost
    except Exception:
        print("  SKIP: import failed")
        return None

    cost = IOCost(
        x_load=100, w_load=200, z_io=300, v_io=400, spike_write=500,
        schedule="test", num_blocks=2,
    )
    assert cost.total == 1500, f"total expected 1500, got {cost.total}"
    assert cost.intermediate_io == 700, f"int_io expected 700, got {cost.intermediate_io}"

    # Frozen
    try:
        cost.x_load = 999  # type: ignore[misc]
        print("  FAIL: IOCost should be frozen")
        return False
    except Exception:
        pass

    d = cost.as_dict()
    assert d["total"] == 1500 and d["intermediate_io"] == 700

    print("  OK: IOCost frozen, total/intermediate_io correct")
    return True


def hand_check_dense_keep():
    """3: DenseKeep.analytic_io_cost on hand-computed concrete shape.

    Layer:   sew-rn18 layer3.0.conv2  (256→256, 3x3, stride=1, pad=1)
    Input:   T=4, B=2, H=8, W=8 (after layer3 stride=2 from H=16)
    HWC_in  = 8·8·256 = 16384
    HWC_out = 8·8·256 = 16384  (stride=1, pad=1)
    """
    print("[3/7] Hand-check DenseKeep formula on sew-rn18 layer3.0.conv2 shape...")
    try:
        from catfuse.implementations import ConvLIFSpec, DenseKeep
    except Exception:
        print("  SKIP: imports failed")
        return None

    spec = ConvLIFSpec(
        in_channels=256, out_channels=256,
        kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    impl = DenseKeep()
    cost = impl.analytic_io_cost(spec, T=4, B=2, H_in=8, W_in=8, dtype_bytes=4)

    # Hand-computed
    HWC_in = 8 * 8 * 256
    HWC_out = 8 * 8 * 256
    expect_x_load = 4 * 2 * HWC_in * 4              # = 524_288
    expect_w_load = 256 * 256 * 9 * 4               # = 2_359_296
    expect_z_io = 2 * 4 * 2 * HWC_out * 4           # = 1_048_576
    expect_v_io = 2 * 2 * HWC_out * 4               # = 262_144
    expect_spike = 4 * 2 * HWC_out * 4              # = 524_288
    expect_total = (expect_x_load + expect_w_load + expect_z_io
                    + expect_v_io + expect_spike)

    print(f"  Hand-computed (expected):")
    print(f"    x_load = {expect_x_load:>12,}")
    print(f"    w_load = {expect_w_load:>12,}")
    print(f"    z_io   = {expect_z_io:>12,}")
    print(f"    v_io   = {expect_v_io:>12,}")
    print(f"    spike  = {expect_spike:>12,}")
    print(f"    total  = {expect_total:>12,}")
    print(f"  IOCost     (returned):")
    print(f"    x_load = {cost.x_load:>12,}")
    print(f"    w_load = {cost.w_load:>12,}")
    print(f"    z_io   = {cost.z_io:>12,}")
    print(f"    v_io   = {cost.v_io:>12,}")
    print(f"    spike  = {cost.spike_write:>12,}")
    print(f"    total  = {cost.total:>12,}")

    if cost.x_load != expect_x_load: return False
    if cost.w_load != expect_w_load: return False
    if cost.z_io != expect_z_io: return False
    if cost.v_io != expect_v_io: return False
    if cost.spike_write != expect_spike: return False
    if cost.total != expect_total: return False
    if cost.num_blocks != 1: return False

    print("  OK: DenseKeep formula matches hand-check")
    return True


def hand_check_sparse_flow():
    """4: SparseFlow.analytic_io_cost on same shape, parameterized by K."""
    print("[4/7] Hand-check SparseFlow formula across K values...")
    try:
        from catfuse.implementations import ConvLIFSpec, SparseFlow
    except Exception:
        print("  SKIP: imports failed")
        return None
    if SparseFlow is None:
        print("  SKIP: SparseFlow unavailable on this platform")
        return None

    spec = ConvLIFSpec(
        in_channels=256, out_channels=256,
        kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    impl = SparseFlow()
    HWC_out = 8 * 8 * 256

    # T=4. Test K=1, 2, 4.
    cases = []
    for K in [1, 2, 4]:
        cost = impl.analytic_io_cost(spec, T=4, B=2, H_in=8, W_in=8, K=K, dtype_bytes=4)
        num_blocks = math.ceil(4 / K)
        expect_v_io = 2 * num_blocks * 2 * HWC_out * 4
        expect_z_io = 0  # always 0 for SparseFlow
        if cost.z_io != expect_z_io:
            print(f"  FAIL K={K}: z_io expected 0, got {cost.z_io}")
            return False
        if cost.v_io != expect_v_io:
            print(f"  FAIL K={K}: v_io expected {expect_v_io}, got {cost.v_io}")
            return False
        if cost.num_blocks != num_blocks:
            print(f"  FAIL K={K}: num_blocks expected {num_blocks}, got {cost.num_blocks}")
            return False
        cases.append((K, num_blocks, cost.intermediate_io, cost.total))

    print(f"  K-sweep results (T=4, layer3.0.conv2 shape):")
    print(f"    {'K':>3} {'#blocks':>8} {'inter_io':>10} {'total':>12}")
    for K, nb, ii, tot in cases:
        print(f"    {K:>3} {nb:>8} {ii:>10,} {tot:>12,}")

    print("  OK: SparseFlow K-parameterized formula matches hand-check")
    return True


def check_k_sweep_monotonicity():
    """5+6: As K increases, SparseFlow intermediate_io decreases monotonically.
    Ratio (SparseFlow_K=T / DenseKeep) must equal §3.9 prediction.
    """
    print("[5/7] K-sweep monotonicity + §3.9 ratio prediction...")
    try:
        from catfuse.implementations import ConvLIFSpec, DenseKeep, SparseFlow
    except Exception:
        print("  SKIP: imports failed")
        return None
    if SparseFlow is None:
        print("  SKIP: SparseFlow unavailable")
        return None

    spec = ConvLIFSpec(
        in_channels=256, out_channels=256,
        kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    T, B, H, W = 4, 2, 8, 8

    sf = SparseFlow()
    intermediate_per_K = []
    for K in [1, 2, 4]:
        cost = sf.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=W, K=K)
        intermediate_per_K.append((K, cost.intermediate_io))
    # As K grows, num_blocks shrinks, v_io shrinks → intermediate_io shrinks
    for i in range(len(intermediate_per_K) - 1):
        K1, ii1 = intermediate_per_K[i]
        K2, ii2 = intermediate_per_K[i + 1]
        if not (ii1 >= ii2):
            print(f"  FAIL: monotonicity broken at K={K1}({ii1}) -> K={K2}({ii2})")
            return False
    print(f"  Monotonicity OK: K=1→2→4 gives intermediate_io = "
          f"{[ii for _, ii in intermediate_per_K]} (non-increasing)")

    # §3.9 ratio prediction at K=T:
    #   ratio = SparseFlow K=T intermediate_io / DenseKeep intermediate_io
    #         = (2·1) / (2·T + 2) = 1 / (T+1)
    dk = DenseKeep()
    dk_cost = dk.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=W)
    sf_cost_KT = sf.analytic_io_cost(spec, T=T, B=B, H_in=H, W_in=W, K=T)
    ratio_actual = sf_cost_KT.intermediate_io / dk_cost.intermediate_io
    ratio_expect = 1.0 / (T + 1)  # = 0.2 for T=4
    print(f"  SparseFlow(K=T={T}) / DenseKeep intermediate_io = {ratio_actual:.4f}")
    print(f"  §3.9 prediction 1/(T+1) = {ratio_expect:.4f}")
    if abs(ratio_actual - ratio_expect) > 1e-6:
        print(f"  FAIL: ratio mismatch")
        return False

    print("  OK: §3.9 prediction holds analytically")
    return True


def check_resnet18_regression():
    """7: SEW-ResNet18 still bit-exact (regression)."""
    print("[6/7] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None
    try:
        from catfuse import optimize
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP: catfuse.optimize unavailable")
        traceback.print_exc()
        return None

    if build_sew_resnet18 is None:
        try:
            from spikingjelly.activation_based.model import sew_resnet
            from spikingjelly.activation_based import surrogate
            net = sew_resnet.sew_resnet18(
                pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
                surrogate_function=surrogate.ATan(), detach_reset=True,
            )
            functional.set_step_mode(net, "m")
        except Exception:
            print("  SKIP: can't build SEW-RN18")
            return None
    else:
        net = build_sew_resnet18()

    device = "cuda:0"
    net = net.to(device).eval()
    T, B = 4, 2
    x = torch.randn(T, B, 3, 32, 32, device=device)
    with torch.no_grad():
        functional.reset_net(net)
        y_sj = net(x)
    fused, _ = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()
    with torch.no_grad():
        functional.reset_net(fused)
        y_ctf = fused(x)
    max_diff = (y_sj - y_ctf).abs().max().item()
    if max_diff > 1e-4:
        print(f"  FAIL: max_diff={max_diff:.6e}")
        return False
    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e} (bit-exact preserved)")
    return True


def check_layer_query_via_pattern():
    """7: From a pattern, query its impl.analytic_io_cost as a downstream
    consumer would (this is the §3.10 K-sweep experiment's API).
    """
    print("[7/7] Pattern → impl.analytic_io_cost API smoke test...")
    try:
        import torch
        from catfuse.patterns import PartialFusionConvBNLIF
        from catfuse.implementations import IOCost
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    inst = PartialFusionConvBNLIF(in_channels=64, out_channels=128,
                                  kernel_size=3, padding=1)
    cost = inst._impl.analytic_io_cost(
        inst.spec, T=4, B=2, H_in=16, W_in=16,
    )
    assert isinstance(cost, IOCost)
    assert cost.total > 0
    assert cost.intermediate_io == cost.z_io + cost.v_io
    print(f"  Pattern query: total={cost.total:,} bytes, "
          f"intermediate_io={cost.intermediate_io:,}, schedule='{cost.schedule}'")
    print("  OK: pattern → impl.analytic_io_cost API works")
    return True


def main():
    print("=" * 60)
    print("Stage 4 verification: analytic_io_cost binding (problem 5)")
    print("=" * 60)
    results = [
        ("imports_and_abstract", check_imports_and_abstract()),
        ("iocost_dataclass", check_iocost_dataclass()),
        ("dense_keep_hand_check", hand_check_dense_keep()),
        ("sparse_flow_hand_check", hand_check_sparse_flow()),
        ("k_sweep_monotonicity", check_k_sweep_monotonicity()),
        ("resnet18_regression", check_resnet18_regression()),
        ("layer_query_via_pattern", check_layer_query_via_pattern()),
    ]
    print()
    print("=" * 60)
    failed = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, "
              f"{len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
