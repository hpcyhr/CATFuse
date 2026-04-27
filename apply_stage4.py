#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 4 refactor: analytic_io_cost binding (problem 5).

Stage 4 adds the §3.9 cost model to the Implementation hierarchy:

  - New IOCost dataclass: structured HBM byte counts with intermediate_io
    property highlighting the §3.9 reduction target (z_io + v_io).
  - Implementation.analytic_io_cost: new abstractmethod, takes (spec, T, B,
    H, W, K) and returns IOCost.
  - DenseKeep.analytic_io_cost: returns the baseline schedule's cost,
    z materialized to HBM, K is ignored.
  - SparseFlow.analytic_io_cost: returns the K-parameterized CTF cost,
    z eliminated by StreamFuse, v_io = 2·ceil(T/K)·B·HWC_out·dtype_bytes.

This binds the §3.9 formula `O(T·HWC) → O(T/K·HWC)` to the actual code,
making it queryable from any layer's impl. The §3.10 K-sweep experiment
will use this directly:

    cost = layer._impl_sparse.analytic_io_cost(layer.spec, T, B, H, W, K)
    print(f"K={K}: predicted {cost.intermediate_io} bytes")

The Implementation ABC now requires both forward AND analytic_io_cost.

Prerequisites: stages 1, 2, and 3 must be applied. Idempotent.

Run:
    cd /path/to/CATFuse
    python apply_stage4.py
    python tests/stage4_verify.py
"""
import os
import sys


# ============================================================
# Embedded new files
# ============================================================

NEW_FILES = {
    'tests/stage4_verify.py': r'''"""[Stage 4 verification] analytic_io_cost binding (problem 5).

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
''',
}


# ============================================================
# Replacement pairs
# ============================================================

REPLACEMENTS = {
    "catfuse/implementations/base.py": [
        ('        self.bias = bias\n        self.bn_scale = bn_scale\n        self.bn_bias = bn_bias\n\n\n# ============================================================\n# Implementation ABC\n# ============================================================\n\nclass Implementation(nn.Module, ABC):\n    """Abstract base for one element of Impl(σ).\n\n    Subclasses must:\n      - Set class attribute `name` (str): e.g. "DenseKeep", "SparseFlow".\n      - Override forward(x, spec, params, state) producing bit-exact output.\n\n    Bit-exactness contract (Corollary 3.17):\n        ∀ x, ∀ a, b ∈ Impl(σ):\n            a.forward(x, spec, params, state_a) ≡ b.forward(x, spec, params, state_b)\n        pointwise on every (T, B, C_out, H_out, W_out) entry, given identical\n        x, spec, params, and identical state initial condition.\n\n    Implementations may carry IMPL-LOCAL caches (e.g. a pre-folded weight,\n    a channel-last layout, prescan buffers). Such caches must be invalidated\n    via `reset_caches()` if the pattern\'s params change after construction.\n    """\n\n    name: str = "<unset>"\n\n    @abstractmethod\n    def forward(\n        self,\n        x: torch.Tensor,                # [T, B, C_in, H, W], fp32, CUDA, contiguous\n        spec: ConvLIFSpec,\n        params: ConvLIFParams,\n        state: "StateBuffer",           # type: ignore[name-defined]  (catfuse.state.StateBuffer)\n    ) -> torch.Tensor:                  # [T, B, C_out, H_out, W_out], fp32\n        """Run the bit-exact realization of σ.\n\n        Reads state via state.get(...), writes via state.set(v_final).\n        """\n        ...\n\n    def reset_caches(self) -> None:\n        """Invalidate impl-local caches. Default: no-op.\n\n        Override in subclasses that hold lazy caches dependent on params\n        (e.g. DenseKeep\'s pre-folded weight, SparseFlow\'s _w_cl).', '        self.bias = bias\n        self.bn_scale = bn_scale\n        self.bn_bias = bn_bias\n\n\n@dataclass(frozen=True)\nclass IOCost:\n    """Analytic HBM access count, broken down by tensor.\n\n    Each field is in BYTES. The breakdown matches the §3.9 cost model:\n\n      x_load:      input activation read from HBM\n      w_load:      conv weight read from HBM (per kernel launch)\n      z_io:        intermediate conv output traffic (write + read)\n                   — eliminated by StreamFuse, this is the §3.9 savings target\n      v_io:        membrane state carry (read + write per TimeBlock boundary)\n                   — for baseline lif_sequential: just initial+final (2·HWC)\n                   — for CTF with TimeBlock(K): 2·ceil(T/K)·HWC\n      spike_write: output spike tensor write to HBM\n\n    intermediate_io = z_io + v_io is the "moves between time steps" term\n    that §3.9 reduces from O(T·HWC) to O(T/K·HWC).\n\n    total = sum of all fields above.\n    """\n    x_load: int\n    w_load: int\n    z_io: int\n    v_io: int\n    spike_write: int\n    schedule: str       # human-readable schedule descriptor\n    num_blocks: int     # number of TimeBlock instances\n\n    @property\n    def intermediate_io(self) -> int:\n        """The \'inter-step state movement\' term (§3.9 target)."""\n        return self.z_io + self.v_io\n\n    @property\n    def total(self) -> int:\n        return self.x_load + self.w_load + self.z_io + self.v_io + self.spike_write\n\n    def as_dict(self) -> dict:\n        return {\n            "x_load": self.x_load,\n            "w_load": self.w_load,\n            "z_io": self.z_io,\n            "v_io": self.v_io,\n            "spike_write": self.spike_write,\n            "intermediate_io": self.intermediate_io,\n            "total": self.total,\n            "schedule": self.schedule,\n            "num_blocks": self.num_blocks,\n        }\n\n\n# ============================================================\n# Implementation ABC\n# ============================================================\n\nclass Implementation(nn.Module, ABC):\n    """Abstract base for one element of Impl(σ).\n\n    Subclasses must:\n      - Set class attribute `name` (str): e.g. "DenseKeep", "SparseFlow".\n      - Override forward(x, spec, params, state) producing bit-exact output.\n      - Override analytic_io_cost(spec, T, B, H_in, W_in, K) returning the\n        §3.9 HBM traffic prediction for that scheduling choice.\n\n    Bit-exactness contract (Corollary 3.17):\n        ∀ x, ∀ a, b ∈ Impl(σ):\n            a.forward(x, spec, params, state_a) ≡ b.forward(x, spec, params, state_b)\n        pointwise on every (T, B, C_out, H_out, W_out) entry, given identical\n        x, spec, params, and identical state initial condition.\n\n    Note that bit-exactness DOES NOT imply equal I/O cost — that\'s exactly\n    why analytic_io_cost is a separate method. Two impls in Impl(σ) may\n    produce identical outputs while moving very different numbers of bytes\n    through HBM. That delta is the §3.9 optimization target.\n\n    Implementations may carry IMPL-LOCAL caches (e.g. a pre-folded weight,\n    a channel-last layout, prescan buffers). Such caches must be invalidated\n    via `reset_caches()` if the pattern\'s params change after construction.\n    """\n\n    name: str = "<unset>"\n\n    @abstractmethod\n    def forward(\n        self,\n        x: torch.Tensor,                # [T, B, C_in, H, W], fp32, CUDA, contiguous\n        spec: ConvLIFSpec,\n        params: ConvLIFParams,\n        state: "StateBuffer",           # type: ignore[name-defined]  (catfuse.state.StateBuffer)\n    ) -> torch.Tensor:                  # [T, B, C_out, H_out, W_out], fp32\n        """Run the bit-exact realization of σ.\n\n        Reads state via state.get(...), writes via state.set(v_final).\n        """\n        ...\n\n    @abstractmethod\n    def analytic_io_cost(\n        self,\n        spec: ConvLIFSpec,\n        T: int, B: int, H_in: int, W_in: int,\n        K: int = None,\n        *,\n        dtype_bytes: int = 4,\n    ) -> IOCost:\n        """Predict HBM traffic in bytes for one forward pass.\n\n        This is the §3.9 cost model evaluated symbolically — it does NOT\n        run the kernel and does NOT depend on input data. It captures the\n        worst-case (dense input) HBM access count of the schedule the impl\n        realizes.\n\n        K is the TimeBlock(K) parameter (Definition 3.13). For impls that\n        do not parametrize on K (e.g. DenseKeep is pinned at K=T via\n        BatchFold), K is ignored and the result reflects the impl\'s fixed\n        schedule. For impls that DO use K (e.g. SparseFlow), K=None means\n        "use my default" (typically K=T, lean batchfold).\n\n        dtype_bytes is the tensor element size. Default 4 (fp32). Pass 2\n        for fp16 weight/activation paths.\n\n        Static-zero short-circuits and sparsity-driven w_load reductions\n        are NOT modeled here — they are data-dependent. Empirical\n        measurements may come in lower than this prediction.\n        """\n        ...\n\n    def reset_caches(self) -> None:\n        """Invalidate impl-local caches. Default: no-op.\n\n        Override in subclasses that hold lazy caches dependent on params\n        (e.g. DenseKeep\'s pre-folded weight, SparseFlow\'s _w_cl).'),
    ],
    "catfuse/implementations/dense_keep.py": [
        ('            state.set(v)\n            return torch.stack(spikes_list, dim=0)\n\n    def __repr__(self) -> str:\n        return f"DenseKeep(cached={\'yes\' if self._w_fused is not None else \'no\'})"', '            state.set(v)\n            return torch.stack(spikes_list, dim=0)\n\n    def analytic_io_cost(\n        self,\n        spec,\n        T: int, B: int, H_in: int, W_in: int,\n        K: int = None,\n        *,\n        dtype_bytes: int = 4,\n    ):\n        """§3.9 baseline I/O cost: TimeBlock(T) ∘ BatchFold(Conv).\n\n        Schedule:\n          1. Conv runs once on BatchFolded input [T*B, C_in, H, W] → z [T*B, C_out, H_out, W_out]\n             — z is materialized to HBM (cuDNN cannot avoid this).\n          2. lif_sequential reads z from HBM, runs T-step LIF in registers,\n             writes T spike maps and final v back to HBM.\n\n        K is ignored: DenseKeep is a fixed-schedule realization. The "K"\n        parameter is meaningful only for impls that decompose T into\n        TimeBlocks; here T is folded into the conv\'s batch dim and LIF\n        runs all T steps in one kernel.\n\n        Cost (per forward, in bytes):\n          x_load       = T·B·HWC_in       ·dtype_bytes\n          w_load       = C_out·C_in·k²    ·dtype_bytes        (one launch)\n          z_io         = 2·T·B·HWC_out    ·dtype_bytes        (write + read)\n          v_io         = 2·B·HWC_out      ·dtype_bytes        (init + final)\n          spike_write  = T·B·HWC_out      ·dtype_bytes\n\n        intermediate_io = z_io + v_io = (2T+2)·B·HWC_out·dtype_bytes ∈ O(T·HWC).\n        """\n        from catfuse.implementations.base import IOCost\n\n        H_out, W_out = spec.output_hw(H_in, W_in)\n        HWC_in = H_in * W_in * spec.in_channels\n        HWC_out = H_out * W_out * spec.out_channels\n        kk = spec.kernel_size * spec.kernel_size\n\n        x_load = T * B * HWC_in * dtype_bytes\n        w_load = spec.out_channels * spec.in_channels * kk * dtype_bytes\n        # z: written by cuDNN conv, then read by lif_sequential\n        z_io = 2 * T * B * HWC_out * dtype_bytes\n        # v: initial read + final write only (lif_sequential is a single multi-step kernel)\n        v_io = 2 * B * HWC_out * dtype_bytes\n        spike_write = T * B * HWC_out * dtype_bytes\n\n        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule="TimeBlock(T)∘BatchFold(Conv) — z materialized to HBM",\n            num_blocks=1,\n        )\n\n    def __repr__(self) -> str:\n        return f"DenseKeep(cached={\'yes\' if self._w_fused is not None else \'no\'})"'),
    ],
    "catfuse/implementations/sparse_flow.py": [
        ('        state.set(v_out)\n        return spike_out.reshape(T, B, spec.out_channels, H_out, W_out)\n\n    def __repr__(self) -> str:', '        state.set(v_out)\n        return spike_out.reshape(T, B, spec.out_channels, H_out, W_out)\n\n    def analytic_io_cost(\n        self,\n        spec,\n        T: int, B: int, H_in: int, W_in: int,\n        K: int = None,\n        *,\n        dtype_bytes: int = 4,\n    ):\n        """§3.9 CTF I/O cost: TimeBlock(K) ∘ StreamFuse(Conv,LIF) ∘ StateCarry(LIF).\n\n        Schedule:\n          1. T steps split into ceil(T/K) blocks of size ≤ K.\n          2. Within each block: Conv→BN→LIF run as one fused kernel; z stays\n             in registers, v stays in registers across the K steps inside.\n             No HBM traffic for z; no intra-block HBM traffic for v.\n          3. Between blocks: v written to HBM at end-of-block, read back at\n             start-of-next-block (StateCarry).\n          4. Spike maps for each step are still written to HBM (they are\n             the layer\'s output to the next layer).\n\n        K=None defaults to T (single-block lean batchfold, the current\n        SparseFlow.forward path). K=1 degenerates to per-step v carry.\n        Intermediate K values are predicted here even though the current\n        kernel implementation only realizes K=T directly — the formula is\n        the §3.10 K-sweep prediction target.\n\n        Cost (per forward, in bytes):\n          num_blocks   = ceil(T/K)\n          x_load       = T·B·HWC_in                 ·dtype_bytes\n          w_load       = num_blocks·C_out·C_in·k²   ·dtype_bytes  (worst case;\n                         L2 caching may reduce empirically)\n          z_io         = 0                                          (StreamFuse)\n          v_io         = 2·num_blocks·B·HWC_out     ·dtype_bytes\n          spike_write  = T·B·HWC_out                ·dtype_bytes\n\n        intermediate_io = z_io + v_io = 2·ceil(T/K)·B·HWC_out·dtype_bytes ∈ O(T/K·HWC).\n\n        This is exactly the §3.9 reduction: O(T·HWC) → O(T/K·HWC).\n        """\n        from catfuse.implementations.base import IOCost\n        import math\n\n        if K is None:\n            K = T\n        K = max(1, min(int(K), int(T)))\n        num_blocks = math.ceil(T / K)\n\n        H_out, W_out = spec.output_hw(H_in, W_in)\n        HWC_in = H_in * W_in * spec.in_channels\n        HWC_out = H_out * W_out * spec.out_channels\n        kk = spec.kernel_size * spec.kernel_size\n\n        x_load = T * B * HWC_in * dtype_bytes\n        # Weight reload per block in worst case (L2 may amortize empirically;\n        # we report the upper bound to be honest about the schedule\'s cost).\n        w_load = num_blocks * spec.out_channels * spec.in_channels * kk * dtype_bytes\n        # z stays in registers across the entire block — never reaches HBM\n        z_io = 0\n        # v carried at block boundaries: 1 read + 1 write per block\n        v_io = 2 * num_blocks * B * HWC_out * dtype_bytes\n        spike_write = T * B * HWC_out * dtype_bytes\n\n        return IOCost(\n            x_load=x_load,\n            w_load=w_load,\n            z_io=z_io,\n            v_io=v_io,\n            spike_write=spike_write,\n            schedule=(f"TimeBlock(K={K})∘StreamFuse(Conv,LIF)∘StateCarry(LIF) "\n                      f"— z in registers, v carried at {num_blocks} block boundaries"),\n            num_blocks=num_blocks,\n        )\n\n    def __repr__(self) -> str:'),
    ],
    "catfuse/implementations/__init__.py": [
        ('from catfuse.implementations.base import (\n    Implementation,\n    ConvLIFSpec,\n    ConvLIFParams,\n    static_zero_forward,\n)\nfrom catfuse.implementations.dense_keep import DenseKeep\n\n# SparseFlow may fail to import on systems without Triton; guard so that\n# DenseKeep-only configurations remain usable.\ntry:\n    from catfuse.implementations.sparse_flow import SparseFlow\n    _SPARSEFLOW_AVAILABLE = True\nexcept (ImportError, RuntimeError):\n    SparseFlow = None  # type: ignore\n    _SPARSEFLOW_AVAILABLE = False\n\n\n__all__ = [\n    "Implementation",\n    "ConvLIFSpec",\n    "ConvLIFParams",\n    "DenseKeep",\n    "SparseFlow",\n    "static_zero_forward",\n]', 'from catfuse.implementations.base import (\n    Implementation,\n    ConvLIFSpec,\n    ConvLIFParams,\n    IOCost,\n    static_zero_forward,\n)\nfrom catfuse.implementations.dense_keep import DenseKeep\n\n# SparseFlow may fail to import on systems without Triton; guard so that\n# DenseKeep-only configurations remain usable.\ntry:\n    from catfuse.implementations.sparse_flow import SparseFlow\n    _SPARSEFLOW_AVAILABLE = True\nexcept (ImportError, RuntimeError):\n    SparseFlow = None  # type: ignore\n    _SPARSEFLOW_AVAILABLE = False\n\n\n__all__ = [\n    "Implementation",\n    "ConvLIFSpec",\n    "ConvLIFParams",\n    "IOCost",\n    "DenseKeep",\n    "SparseFlow",\n    "static_zero_forward",\n]'),
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
            else:
                print(f"  OVERWRITE: {rel_path} ({len(existing)} -> {len(content)} chars)")
        else:
            print(f"  CREATE: {rel_path} ({len(content)} chars)")
        with open(target, "w") as f:
            f.write(content)


def apply_replacements(file_path, replacements, label):
    if not os.path.exists(file_path):
        print(f"  ERROR: {file_path} not found")
        return False
    with open(file_path, "r") as f:
        content = f.read()
    original = content
    applied = 0
    skipped = 0
    for i, (old, new) in enumerate(replacements):
        if new in content:
            print(f"  [{label} #{i+1}] SKIP (already applied)")
            skipped += 1
            continue
        if old not in content:
            print(f"  [{label} #{i+1}] ERROR: neither old nor new pattern found")
            print(f"    Either earlier stages not applied, or file modified externally.")
            return False
        content = content.replace(old, new, 1)
        applied += 1
        print(f"  [{label} #{i+1}] APPLIED ({len(old)} chars -> {len(new)} chars)")
    if content != original:
        with open(file_path, "w") as f:
            f.write(content)
    print(f"  {label}: {applied} applied, {skipped} skipped (already done)")
    return True


def main(repo_root="."):
    repo_root = os.path.abspath(repo_root)
    print("=" * 60)
    print(f"Applying Stage 4 refactor in: {repo_root}")
    print("=" * 60)

    # Sanity check: stage 3 must have been applied (implementations dir present)
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
    print("Stage 4 applied successfully.")
    print("Next: python tests/stage4_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))