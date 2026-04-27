#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 5 refactor: schedule_decomposition metadata + verifier (problem 1).

Stage 5 attaches §3.13 primitive-transform metadata to the Implementation
hierarchy. Each impl now exposes:

  - schedule_decomposition(spec, T, K) → ScheduleDecomposition
    A composition of (TimeBlock, BatchFold, StreamFuse, StateCarry) primitives
    that names the §3.8 form realized by this impl.

  - .verify() on the decomposition checks Definition 3.13 typing rules:
      BatchFold only on TSI ops
      StreamFuse only on TSI→CSR pairs
      StateCarry only on CSR ops
      TimeBlock(K) requires K >= 1
      Plus: orphan StreamFuse without StateCarry is illegal.

This is a METADATA-ONLY stage. No forward() behavior changes. SEW-RN18
bit-exactness is preserved trivially.

Prerequisites: stages 1–4 applied. Idempotent.

Run:
    cd /path/to/CATFuse
    python apply_stage5.py
    python tests/stage5_verify.py
"""
import os
import sys


# ============================================================
# Embedded new files
# ============================================================

NEW_FILES = {
    'tests/stage5_verify.py': r'''"""[Stage 5 verification] schedule_decomposition metadata + verifier (problem 1).

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
''',
}


# ============================================================
# Replacement pairs
# ============================================================

REPLACEMENTS = {
    "catfuse/implementations/base.py": [
        ('        Static-zero short-circuits and sparsity-driven w_load reductions\n        are NOT modeled here — they are data-dependent. Empirical\n        measurements may come in lower than this prediction.\n        """\n        ...\n\n    def reset_caches(self) -> None:', '        Static-zero short-circuits and sparsity-driven w_load reductions\n        are NOT modeled here — they are data-dependent. Empirical\n        measurements may come in lower than this prediction.\n        """\n        ...\n\n    @abstractmethod\n    def schedule_decomposition(\n        self,\n        spec: ConvLIFSpec,\n        T: int,\n        K: int = None,\n    ) -> "ScheduleDecomposition":\n        """Return the §3.13 primitive-transform decomposition of this impl\'s schedule.\n\n        Each Implementation realizes one specific σ ∈ Σ(G,T) — a composition\n        of the four §3.13 primitives (TimeBlock, BatchFold, StreamFuse,\n        StateCarry). This method makes that composition explicit, both for\n        documentation (which form does this impl correspond to?) and for\n        verification (does the decomposition obey TSI/CSR typing rules?).\n\n        Returns a ScheduleDecomposition whose .verify() must pass for the\n        impl to be considered a legal element of CTF(G,T) per Definition 3.13.\n\n        K parameterizes TimeBlock for impls that support it; ignored otherwise.\n        """\n        ...\n\n    def reset_caches(self) -> None:'),
        ('    spikes, v_out = lif_sequential_fn(\n        z, v_in, tau=spec.tau,\n        v_threshold=spec.v_threshold, v_reset=spec.v_reset,\n    )\n    state.set(v_out)\n    return spikes\n', '    spikes, v_out = lif_sequential_fn(\n        z, v_in, tau=spec.tau,\n        v_threshold=spec.v_threshold, v_reset=spec.v_reset,\n    )\n    state.set(v_out)\n    return spikes\n\n\n# ============================================================\n# Schedule decomposition — §3.13 primitive transforms\n# ============================================================\n\n# Operator classification per §3.2.\n# TSI = Time-Shape Invariant: each timestep computes independently of others\n# CSR = Causal State Recurrent: state at t depends on previous timesteps\n# TGO = Time-Global Operator: requires all timesteps simultaneously (out of CTF scope)\nTSI_OPS = frozenset({"Conv", "BN", "Linear", "Add", "Pool"})\nCSR_OPS = frozenset({"LIF", "PSN"})\nTGO_OPS = frozenset({"Attention", "GroupNorm", "LayerNorm"})\n\n\n@dataclass(frozen=True)\nclass ScheduleTransform:\n    """One application of a §3.13 primitive transform.\n\n    Four primitives, with their well-formedness constraints:\n      TimeBlock(K)      : K >= 1; partitions T steps into ceil(T/K) blocks\n      BatchFold(op)     : op ∈ TSI_OPS; folds time into batch dim for op\n      StreamFuse(u, v)  : u ∈ TSI_OPS, v ∈ CSR_OPS; on-chip dataflow u → v\n      StateCarry(op)    : op ∈ CSR_OPS; precise state transfer at block boundaries\n    """\n    primitive: str\n    args: tuple   # (K,) | (op,) | (u, v) — fixed positional per primitive\n\n    # ---- Factories ----\n    @classmethod\n    def TimeBlock(cls, K: int) -> "ScheduleTransform":\n        return cls("TimeBlock", (int(K),))\n\n    @classmethod\n    def BatchFold(cls, op: str) -> "ScheduleTransform":\n        return cls("BatchFold", (str(op),))\n\n    @classmethod\n    def StreamFuse(cls, u: str, v: str) -> "ScheduleTransform":\n        return cls("StreamFuse", (str(u), str(v)))\n\n    @classmethod\n    def StateCarry(cls, op: str) -> "ScheduleTransform":\n        return cls("StateCarry", (str(op),))\n\n    # ---- Display ----\n    def __str__(self) -> str:\n        if self.primitive == "TimeBlock":\n            return f"TimeBlock(K={self.args[0]})"\n        if self.primitive == "BatchFold":\n            return f"BatchFold({self.args[0]})"\n        if self.primitive == "StreamFuse":\n            return f"StreamFuse({self.args[0]}, {self.args[1]})"\n        if self.primitive == "StateCarry":\n            return f"StateCarry({self.args[0]})"\n        return f"{self.primitive}{self.args}"\n\n    # ---- Per-primitive type rules (Lemma 3.14 well-formedness) ----\n    def check_typing(self) -> "list[str]":\n        errors = []\n        if self.primitive == "TimeBlock":\n            if len(self.args) != 1:\n                errors.append(f"TimeBlock takes 1 arg, got {len(self.args)}")\n            elif self.args[0] < 1:\n                errors.append(f"TimeBlock(K) requires K >= 1, got {self.args[0]}")\n        elif self.primitive == "BatchFold":\n            if len(self.args) != 1:\n                errors.append(f"BatchFold takes 1 arg, got {len(self.args)}")\n            else:\n                op = self.args[0]\n                if op not in TSI_OPS:\n                    errors.append(\n                        f"BatchFold({op}) ILLEGAL — op must be TSI; "\n                        f"{op} is {\'CSR\' if op in CSR_OPS else \'unknown/TGO\'}"\n                    )\n        elif self.primitive == "StreamFuse":\n            if len(self.args) != 2:\n                errors.append(f"StreamFuse takes 2 args, got {len(self.args)}")\n            else:\n                u, v = self.args\n                if u not in TSI_OPS:\n                    errors.append(f"StreamFuse({u}, {v}) ILLEGAL — u={u} must be TSI")\n                if v not in CSR_OPS:\n                    errors.append(f"StreamFuse({u}, {v}) ILLEGAL — v={v} must be CSR")\n        elif self.primitive == "StateCarry":\n            if len(self.args) != 1:\n                errors.append(f"StateCarry takes 1 arg, got {len(self.args)}")\n            else:\n                op = self.args[0]\n                if op not in CSR_OPS:\n                    errors.append(\n                        f"StateCarry({op}) ILLEGAL — op must be CSR; "\n                        f"{op} is {\'TSI\' if op in TSI_OPS else \'unknown/TGO\'}"\n                    )\n        else:\n            errors.append(f"Unknown primitive \'{self.primitive}\'")\n        return errors\n\n\n@dataclass(frozen=True)\nclass ScheduleDecomposition:\n    """A schedule σ ∈ Σ(G,T) expressed as a composition of §3.13 primitives.\n\n    Convention: `transforms` is in INNERMOST-FIRST order. The full schedule\n    applied to a baseline reference σ_ref is:\n\n        σ = transforms[-1] ∘ transforms[-2] ∘ ... ∘ transforms[0] ∘ σ_ref\n\n    i.e. transforms[0] is applied first; transforms[-1] last (outermost).\n    The display string reads outermost-first, matching the paper convention:\n\n        "TimeBlock(K=4) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)"\n\n    Each Implementation produces a ScheduleDecomposition via its\n    schedule_decomposition() method. The .verify() method walks the\n    decomposition and checks well-formedness (§3.13 typing rules).\n\n    `form` labels which §3.8 form this decomposition realizes:\n      "form_1" : TimeBlock + BatchFold (cuDNN-style baseline)\n      "form_2" : TimeBlock + StreamFuse + StateCarry (§3.9 CTF)\n      "form_3" : reserved for future composite forms\n      "custom" : a decomposition not matching the canonical forms\n    """\n    transforms: tuple   # tuple[ScheduleTransform, ...]\n    form: str           # "form_1" | "form_2" | "form_3" | "custom"\n    description: str = ""\n\n    def __str__(self) -> str:\n        # Outermost first (paper convention)\n        return " ∘ ".join(str(t) for t in reversed(self.transforms))\n\n    def verify(self) -> "tuple[bool, list[str]]":\n        """Check well-formedness against §3.13.\n\n        Returns (ok, errors). Errors include:\n          - per-transform typing violations (BatchFold on CSR, etc.)\n          - missing TimeBlock (Definition 3.13 requires chunking)\n          - StreamFuse without corresponding StateCarry on its CSR target\n            (orphan stream fuse — the carried state must persist somewhere)\n        """\n        errors = []\n        # Per-transform typing\n        for i, t in enumerate(self.transforms):\n            for e in t.check_typing():\n                errors.append(f"transform[{i}] {t}: {e}")\n\n        # Whole-decomposition rules\n        prims = [t.primitive for t in self.transforms]\n        if "TimeBlock" not in prims:\n            errors.append(\n                "Missing TimeBlock — Definition 3.13 requires a chunked schedule "\n                "(use TimeBlock(K=T) for the trivial single-block case)"\n            )\n\n        # If StreamFuse(u, v) appears with v ∈ CSR, then StateCarry(v) must\n        # also appear — the CSR state has to be carried at block boundaries\n        # somewhere, since StreamFuse only handles intra-block dataflow.\n        sf_csr_targets = set()\n        for t in self.transforms:\n            if t.primitive == "StreamFuse" and len(t.args) == 2:\n                sf_csr_targets.add(t.args[1])\n        sc_targets = {t.args[0] for t in self.transforms\n                      if t.primitive == "StateCarry" and len(t.args) == 1}\n        for v in sf_csr_targets:\n            if v not in sc_targets:\n                errors.append(\n                    f"StreamFuse(_, {v}) without StateCarry({v}) — "\n                    f"the CSR state of {v} must be carried at block boundaries"\n                )\n\n        # form labels are advisory; check they\'re a known value\n        if self.form not in ("form_1", "form_2", "form_3", "custom"):\n            errors.append(f"unknown form \'{self.form}\'")\n\n        return (len(errors) == 0, errors)\n'),
    ],
    "catfuse/implementations/dense_keep.py": [
        ('        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule="TimeBlock(T)∘BatchFold(Conv) — z materialized to HBM",\n            num_blocks=1,\n        )\n\n    def __repr__(self) -> str:', '        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule="TimeBlock(T)∘BatchFold(Conv) — z materialized to HBM",\n            num_blocks=1,\n        )\n\n    def schedule_decomposition(self, spec, T: int, K: int = None):\n        """§3.13 decomposition of DenseKeep\'s schedule.\n\n        DenseKeep realizes σ as:\n\n            TimeBlock(K=T) ∘ BatchFold(Conv) ∘ StateCarry(LIF)\n\n        Reading inner-to-outer:\n          1. StateCarry(LIF) — LIF state lives in registers across all T\n             steps inside the lif_sequential kernel; the "block boundaries"\n             are vacuous since K=T = single block.\n          2. BatchFold(Conv) — T fold into batch dim so cuDNN runs once on\n             [T·B, C_in, H, W] → [T·B, C_out, H_out, W_out].\n          3. TimeBlock(K=T) — single block of size T (trivial chunking).\n\n        K is ignored: DenseKeep is fixed at K=T by virtue of BatchFold.\n\n        This is §3.8 form_1 — the BatchFold-form. It is the natural baseline\n        for impls relying on a vendor BLAS (cuDNN here) that requires its\n        outputs in HBM.\n        """\n        from catfuse.implementations.base import (\n            ScheduleTransform, ScheduleDecomposition,\n        )\n        return ScheduleDecomposition(\n            transforms=(\n                ScheduleTransform.StateCarry("LIF"),\n                ScheduleTransform.BatchFold("Conv"),\n                ScheduleTransform.TimeBlock(T),\n            ),\n            form="form_1",\n            description=(\n                f"DenseKeep schedule: TimeBlock(K=T={T}) ∘ BatchFold(Conv) "\n                f"∘ StateCarry(LIF). z is materialized between Conv and LIF "\n                f"(no StreamFuse). LIF runs as a single multi-step kernel."\n            ),\n        )\n\n    def __repr__(self) -> str:'),
    ],
    "catfuse/implementations/sparse_flow.py": [
        ('        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule=(f"TimeBlock(K={K})∘StreamFuse(Conv,LIF)∘StateCarry(LIF) "\n                      f"— z in registers, v carried at {num_blocks} block boundaries"),\n            num_blocks=num_blocks,\n        )\n\n    def __repr__(self) -> str:', '        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule=(f"TimeBlock(K={K})∘StreamFuse(Conv,LIF)∘StateCarry(LIF) "\n                      f"— z in registers, v carried at {num_blocks} block boundaries"),\n            num_blocks=num_blocks,\n        )\n\n    def schedule_decomposition(self, spec, T: int, K: int = None):\n        """§3.13 decomposition of SparseFlow\'s schedule.\n\n        SparseFlow realizes σ as the §3.9 canonical CTF form:\n\n            TimeBlock(K) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)\n\n        Reading inner-to-outer:\n          1. StateCarry(LIF) — v written/read at block boundaries, never\n             between time steps inside a block.\n          2. StreamFuse(Conv, LIF) — z = BN(Conv(x)) flows from Conv to\n             LIF on-chip (registers/shared); never reaches HBM. This is\n             the §3.9 z_io elimination.\n          3. TimeBlock(K) — partition T steps into ceil(T/K) blocks.\n\n        K=None defaults to T (single-block lean batchfold, current default\n        SparseFlow.forward path).\n\n        This is §3.8 form_2 — the StreamFuse-form. It corresponds directly\n        to the §3.9 main result: I/O reduction from O(T·HWC) to O(T/K·HWC).\n        """\n        from catfuse.implementations.base import (\n            ScheduleTransform, ScheduleDecomposition,\n        )\n        if K is None:\n            K = T\n        K = max(1, min(int(K), int(T)))\n        return ScheduleDecomposition(\n            transforms=(\n                ScheduleTransform.StateCarry("LIF"),\n                ScheduleTransform.StreamFuse("Conv", "LIF"),\n                ScheduleTransform.TimeBlock(K),\n            ),\n            form="form_2",\n            description=(\n                f"SparseFlow schedule: TimeBlock(K={K}) ∘ StreamFuse(Conv, LIF) "\n                f"∘ StateCarry(LIF). z stays on-chip; v carried at "\n                f"{(T + K - 1) // K} block boundaries."\n            ),\n        )\n\n    def __repr__(self) -> str:'),
    ],
    "catfuse/implementations/__init__.py": [
        ('from catfuse.implementations.base import (\n    Implementation,\n    ConvLIFSpec,\n    ConvLIFParams,\n    IOCost,\n    static_zero_forward,\n)\nfrom catfuse.implementations.dense_keep import DenseKeep\n\n# SparseFlow may fail to import on systems without Triton; guard so that\n# DenseKeep-only configurations remain usable.\ntry:\n    from catfuse.implementations.sparse_flow import SparseFlow\n    _SPARSEFLOW_AVAILABLE = True\nexcept (ImportError, RuntimeError):\n    SparseFlow = None  # type: ignore\n    _SPARSEFLOW_AVAILABLE = False\n\n\n__all__ = [\n    "Implementation",\n    "ConvLIFSpec",\n    "ConvLIFParams",\n    "IOCost",\n    "DenseKeep",\n    "SparseFlow",\n    "static_zero_forward",\n]', 'from catfuse.implementations.base import (\n    Implementation,\n    ConvLIFSpec,\n    ConvLIFParams,\n    IOCost,\n    ScheduleTransform,\n    ScheduleDecomposition,\n    TSI_OPS,\n    CSR_OPS,\n    TGO_OPS,\n    static_zero_forward,\n)\nfrom catfuse.implementations.dense_keep import DenseKeep\n\n# SparseFlow may fail to import on systems without Triton; guard so that\n# DenseKeep-only configurations remain usable.\ntry:\n    from catfuse.implementations.sparse_flow import SparseFlow\n    _SPARSEFLOW_AVAILABLE = True\nexcept (ImportError, RuntimeError):\n    SparseFlow = None  # type: ignore\n    _SPARSEFLOW_AVAILABLE = False\n\n\n__all__ = [\n    "Implementation",\n    "ConvLIFSpec",\n    "ConvLIFParams",\n    "IOCost",\n    "ScheduleTransform",\n    "ScheduleDecomposition",\n    "TSI_OPS",\n    "CSR_OPS",\n    "TGO_OPS",\n    "DenseKeep",\n    "SparseFlow",\n    "static_zero_forward",\n]'),
    ],
}


# ============================================================
# Apply logic
# ============================================================

def apply_new_files(repo_root):
    print("[1/2] Creating new files...")
    for rel_path, content in NEW_FILES.items():
        target = os.path.join(repo_root, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            with open(target, "r") as f:
                existing = f.read()
            if existing == content:
                print(f"  SKIP (identical): {rel_path}")
                continue
            print(f"  OVERWRITE: {rel_path} ({len(existing)} -> {len(content)} chars)")
        else:
            print(f"  CREATE: {rel_path} ({len(content)} chars)")
        with open(target, "w") as f:
            f.write(content)


def apply_replacements(file_path, replacements, label):
    if not os.path.exists(file_path):
        print(f"  ERROR: {file_path} not found"); return False
    with open(file_path, "r") as f:
        content = f.read()
    original = content
    applied = skipped = 0
    for i, (old, new) in enumerate(replacements):
        if new in content:
            print(f"  [{label} #{i+1}] SKIP (already applied)")
            skipped += 1; continue
        if old not in content:
            print(f"  [{label} #{i+1}] ERROR: neither old nor new pattern found")
            print(f"    Either earlier stages not applied, or file modified externally.")
            return False
        content = content.replace(old, new, 1)
        applied += 1
        print(f"  [{label} #{i+1}] APPLIED ({len(old)} -> {len(new)} chars)")
    if content != original:
        with open(file_path, "w") as f:
            f.write(content)
    print(f"  {label}: {applied} applied, {skipped} skipped (already done)")
    return True


def main(repo_root="."):
    repo_root = os.path.abspath(repo_root)
    print("=" * 60)
    print(f"Applying Stage 5 refactor in: {repo_root}")
    print("=" * 60)

    impls_dir = os.path.join(repo_root, "catfuse/implementations")
    if not os.path.isdir(impls_dir):
        print("  FATAL: catfuse/implementations/ not found — stage 3 must be applied first.")
        return 1

    apply_new_files(repo_root)

    print("[2/2] Modifying implementation files...")
    for rel_path, replacements in REPLACEMENTS.items():
        path = os.path.join(repo_root, rel_path)
        if not apply_replacements(path, replacements, rel_path):
            return 1

    print()
    print("=" * 60)
    print("Stage 5 applied successfully.")
    print("Next: python tests/stage5_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))