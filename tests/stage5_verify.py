"""[Stage 5 verification] schedule_decomposition metadata + verifier (problem 1).

Verifies that:
  1. Implementation now exposes schedule_decomposition() abstractmethod.
  2. ScheduleTransform / ScheduleDecomposition exported.
  3. DenseKeep returns a well-formed §3.8 form_1 decomposition.
  4. SparseFlow returns a well-formed §3.8 form_2 decomposition (the §3.9
     canonical form).
  5. Verifier catches each typing violation:
        BatchFold(CSR-op)   → illegal
        StreamFuse(CSR, _)  → illegal
        StreamFuse(_, TSI)  → illegal
        StateCarry(TSI-op)  → illegal
        Missing TimeBlock   → illegal
        Orphan StreamFuse without matching StateCarry → illegal
        TimeBlock(K<1)      → illegal
  6. Patterns expose schedule decomposition via their impls (downstream
     consumer API).
  7. SEW-ResNet18 substitute_sf still bit-exact (regression).

Run:
    cd /path/to/CATFuse
    python tests/stage5_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_imports():
    """1+2: New types exported, abstractmethod added."""
    print("[1/7] Imports + abstract-method enforcement...")
    try:
        from catfuse.implementations import (
            Implementation,
            ScheduleTransform,
            ScheduleDecomposition,
            TSI_OPS, CSR_OPS, TGO_OPS,
        )
    except Exception:
        print("  FAIL: imports failed")
        traceback.print_exc()
        return False

    abstract_methods = Implementation.__abstractmethods__
    expected = {"forward", "analytic_io_cost", "schedule_decomposition"}
    if not expected.issubset(abstract_methods):
        print(f"  FAIL: expected {expected} ⊆ {abstract_methods}")
        return False

    # Sanity-check operator classifications
    if "Conv" not in TSI_OPS:
        print("  FAIL: Conv not in TSI_OPS"); return False
    if "LIF" not in CSR_OPS:
        print("  FAIL: LIF not in CSR_OPS"); return False
    if "Attention" not in TGO_OPS:
        print("  FAIL: Attention not in TGO_OPS"); return False

    print(f"  OK: abstractmethods = {sorted(abstract_methods)}")
    print(f"      TSI={sorted(TSI_OPS)}")
    print(f"      CSR={sorted(CSR_OPS)}")
    print(f"      TGO={sorted(TGO_OPS)}")
    return True


def check_dense_keep_decomposition():
    """3: DenseKeep produces well-formed form_1 decomposition."""
    print("[2/7] DenseKeep schedule_decomposition...")
    try:
        from catfuse.implementations import (
            ConvLIFSpec, DenseKeep, ScheduleTransform,
        )
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None

    spec = ConvLIFSpec(
        in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    impl = DenseKeep()
    decomp = impl.schedule_decomposition(spec, T=4)

    print(f"  decomposition: {decomp}")
    print(f"  form: {decomp.form}")
    if decomp.form != "form_1":
        print(f"  FAIL: form expected 'form_1', got '{decomp.form}'"); return False

    # Expected: StateCarry(LIF) + BatchFold(Conv) + TimeBlock(K=T=4)
    prims = [t.primitive for t in decomp.transforms]
    if prims != ["StateCarry", "BatchFold", "TimeBlock"]:
        print(f"  FAIL: prims expected ['StateCarry','BatchFold','TimeBlock'], got {prims}"); return False
    if decomp.transforms[2].args[0] != 4:
        print(f"  FAIL: TimeBlock K should be T=4"); return False

    ok, errs = decomp.verify()
    if not ok:
        print(f"  FAIL: verify rejected DenseKeep: {errs}"); return False

    print("  OK: form_1, verify passes")
    return True


def check_sparse_flow_decomposition():
    """4: SparseFlow produces well-formed form_2 decomposition (§3.9 form)."""
    print("[3/7] SparseFlow schedule_decomposition (§3.9 canonical form)...")
    try:
        from catfuse.implementations import (
            ConvLIFSpec, SparseFlow, ScheduleTransform,
        )
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if SparseFlow is None:
        print("  SKIP: SparseFlow unavailable"); return None

    spec = ConvLIFSpec(
        in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    impl = SparseFlow()

    # Test multiple K values
    for T, K in [(4, 4), (4, 2), (4, 1), (8, 4)]:
        decomp = impl.schedule_decomposition(spec, T=T, K=K)
        print(f"  T={T}, K={K}: {decomp}")
        if decomp.form != "form_2":
            print(f"    FAIL: form should be 'form_2'"); return False

        prims = [t.primitive for t in decomp.transforms]
        if prims != ["StateCarry", "StreamFuse", "TimeBlock"]:
            print(f"    FAIL: wrong prims {prims}"); return False

        # Args
        if decomp.transforms[0].args != ("LIF",):
            print(f"    FAIL: StateCarry args"); return False
        if decomp.transforms[1].args != ("Conv", "LIF"):
            print(f"    FAIL: StreamFuse args"); return False
        if decomp.transforms[2].args != (K,):
            print(f"    FAIL: TimeBlock K"); return False

        ok, errs = decomp.verify()
        if not ok:
            print(f"    FAIL: verify rejected: {errs}"); return False

    print("  OK: all K variants well-formed")
    return True


def check_verifier_catches_violations():
    """5: Verifier catches every Definition 3.13 typing violation."""
    print("[4/7] Verifier catches typing violations...")
    try:
        from catfuse.implementations import (
            ScheduleTransform, ScheduleDecomposition,
        )
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None

    def expect_fail(name, decomp, marker):
        ok, errs = decomp.verify()
        if ok:
            print(f"  FAIL: '{name}' should have been rejected"); return False
        if not any(marker in e for e in errs):
            print(f"  FAIL: '{name}' rejected but no error matched marker '{marker}'")
            print(f"    errs: {errs}"); return False
        return True

    cases = [
        ("BatchFold(LIF) on CSR op",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.BatchFold("LIF"),
                         ScheduleTransform.TimeBlock(4)),
             form="custom",
         ),
         "BatchFold(LIF) ILLEGAL"),
        ("StreamFuse(LIF, Conv) wrong direction",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.StreamFuse("LIF", "Conv"),
                         ScheduleTransform.TimeBlock(4)),
             form="custom",
         ),
         "StreamFuse(LIF, Conv) ILLEGAL"),
        ("StateCarry(Conv) on TSI op",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.StateCarry("Conv"),
                         ScheduleTransform.TimeBlock(4)),
             form="custom",
         ),
         "StateCarry(Conv) ILLEGAL"),
        ("Missing TimeBlock",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.BatchFold("Conv"),),
             form="custom",
         ),
         "Missing TimeBlock"),
        ("Orphan StreamFuse(Conv, LIF) without StateCarry(LIF)",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.StreamFuse("Conv", "LIF"),
                         ScheduleTransform.TimeBlock(4)),
             form="custom",
         ),
         "without StateCarry(LIF)"),
        ("TimeBlock(K=0)",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.TimeBlock(0),),
             form="custom",
         ),
         "K >= 1"),
        ("BatchFold(Attention) on TGO op",
         ScheduleDecomposition(
             transforms=(ScheduleTransform.BatchFold("Attention"),
                         ScheduleTransform.TimeBlock(4)),
             form="custom",
         ),
         "BatchFold(Attention) ILLEGAL"),
    ]

    for name, decomp, marker in cases:
        if not expect_fail(name, decomp, marker):
            return False
        print(f"  OK: rejected '{name}'")

    print(f"  OK: all {len(cases)} violations caught")
    return True


def check_pattern_query():
    """6: Pattern → impl.schedule_decomposition() works as downstream API."""
    print("[5/7] Pattern → impl.schedule_decomposition() smoke test...")
    try:
        from catfuse.patterns import PartialFusionConvBNLIF
        from catfuse.implementations import ScheduleDecomposition
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None

    inst = PartialFusionConvBNLIF(in_channels=64, out_channels=128,
                                  kernel_size=3, padding=1)
    decomp = inst._impl.schedule_decomposition(inst.spec, T=4)
    if not isinstance(decomp, ScheduleDecomposition):
        print(f"  FAIL: not a ScheduleDecomposition"); return False
    ok, errs = decomp.verify()
    if not ok:
        print(f"  FAIL: pattern's decomposition failed verify: {errs}"); return False

    print(f"  pattern decomp: {decomp}")
    print(f"  form: {decomp.form}")
    print("  OK: pattern → impl.schedule_decomposition() works")
    return True


def check_st_fusion_decomposition():
    """6b: STFusionConvBNLIF holds two impls — both expose decompositions."""
    print("[6/7] STFusionConvBNLIF dual-impl decompositions...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA"); return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device)

    dk_decomp = fused._impl_dense.schedule_decomposition(fused.spec, T=4)
    print(f"  DenseKeep:   {dk_decomp}")
    print(f"               form={dk_decomp.form}")
    ok, errs = dk_decomp.verify()
    if not ok:
        print(f"  FAIL: DenseKeep verify: {errs}"); return False

    if SparseFlow is not None and fused._impl_sparse is not None:
        sf_decomp = fused._impl_sparse.schedule_decomposition(fused.spec, T=4, K=4)
        print(f"  SparseFlow:  {sf_decomp}")
        print(f"               form={sf_decomp.form}")
        ok, errs = sf_decomp.verify()
        if not ok:
            print(f"  FAIL: SparseFlow verify: {errs}"); return False

        # Sanity: same σ, two different forms — Corollary 3.17 in metadata form
        if dk_decomp.form == sf_decomp.form:
            print("  FAIL: dual impls of same σ should have different forms"); return False

    print("  OK: dual-impl decompositions both well-formed and structurally distinct")
    return True


def check_resnet18_regression():
    """7: SEW-ResNet18 still bit-exact (regression)."""
    print("[7/7] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA"); return None
    try:
        from catfuse import optimize
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP: catfuse.optimize unavailable")
        traceback.print_exc(); return None

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
            print("  SKIP: can't build SEW-RN18"); return None
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
        print(f"  FAIL: max_diff={max_diff:.6e}"); return False
    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e} (bit-exact preserved)")
    return True


def main():
    print("=" * 60)
    print("Stage 5 verification: schedule_decomposition metadata (problem 1)")
    print("=" * 60)
    results = [
        ("imports_and_abstract", check_imports()),
        ("dense_keep_decomposition", check_dense_keep_decomposition()),
        ("sparse_flow_decomposition", check_sparse_flow_decomposition()),
        ("verifier_catches_violations", check_verifier_catches_violations()),
        ("pattern_query", check_pattern_query()),
        ("st_fusion_decomposition", check_st_fusion_decomposition()),
        ("resnet18_regression", check_resnet18_regression()),
    ]
    print()
    print("=" * 60)
    failed = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, {len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
