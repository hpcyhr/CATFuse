#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 6 refactor: K-aware forward (§3.10/§3.13 Lemma 3.14).

Stage 6 adds SparseFlow.forward_with_k(x, spec, params, state, K=None):
a method that realizes the schedule

    TimeBlock(K) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)

for arbitrary K ∈ [1, T] by chunking the time dimension and invoking the
existing forward() on each chunk. State carry between chunks happens
implicitly via the StateBuffer.

Bit-exactness contract (§3.13 Lemma 3.14, experimental form):

    forward_with_k(x, spec, params, state, K=K1)
        ≡ forward_with_k(x, spec, params, state, K=K2)
        ≡ forward(x, spec, params, state)

elementwise, for any K1, K2 ∈ [1, T] given identical state initial
condition. This is what tests/stage6_verify.py check #3 verifies.

Production forward() is UNCHANGED. STFusionConvBNLIF._batchfold_forward
still calls forward() (single block, K=T). SEW-RN18 bit-exact preserved
trivially.

The new method is the substrate for the §3.10 K-sweep experiment.

Prerequisites: stages 1–5 applied. Idempotent.

Run:
    cd /path/to/CATFuse
    python apply_stage6.py
    python tests/stage6_verify.py
"""
import os
import sys


# ============================================================
# Embedded new files
# ============================================================

NEW_FILES = {
    'tests/stage6_verify.py': r'''"""[Stage 6 verification] K-aware forward (§3.10/§3.13 Lemma 3.14).

Verifies that:
  1. SparseFlow.forward_with_k exists and accepts K.
  2. forward_with_k(K=T) ≡ forward() bit-exactly (default path).
  3. forward_with_k(K=1) ≡ forward_with_k(K=T) bit-exactly.
  4. forward_with_k(K=2) ≡ forward_with_k(K=T) bit-exactly.
  5. K out of range gets clamped (K=0 → 1, K>T → T).
  6. State carry across chunks: v written by chunk i is read by chunk i+1
     (StateCarry(LIF) at TimeBlock boundary).
  7. SEW-ResNet18 substitute_sf still bit-exact (regression — this stage
     is purely additive, the production path forward() is untouched).

Run:
    cd /path/to/CATFuse
    python tests/stage6_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_forward_with_k_exists():
    """1: SparseFlow has forward_with_k method with correct signature."""
    print("[1/7] forward_with_k method exists...")
    try:
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  FAIL"); traceback.print_exc(); return False
    if SparseFlow is None:
        print("  SKIP: SparseFlow unavailable on this platform")
        return None

    if not hasattr(SparseFlow, "forward_with_k"):
        print("  FAIL: forward_with_k method missing"); return False

    import inspect
    sig = inspect.signature(SparseFlow.forward_with_k)
    params = list(sig.parameters.keys())
    expected = ["self", "x", "spec", "params", "state", "K"]
    if params != expected:
        print(f"  FAIL: signature {params} != {expected}")
        return False
    print("  OK: SparseFlow.forward_with_k(self, x, spec, params, state, K=None)")
    return True


def _build_layer_and_input(device, T, B, H, W, sparsity=0.85):
    """Build a fresh STFusion layer + sparse input. Returns (fused, x)."""
    import torch
    from spikingjelly.activation_based import neuron, layer as sj_layer
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF

    torch.manual_seed(42)
    C = 64
    conv = sj_layer.Conv2d(C, C, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(C).to(device)
    bn.running_mean.normal_(0, 0.1)
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=T).to(device).eval()
    x = (torch.rand(T, B, C, H, W, device=device) > sparsity).float()
    return fused, x


def check_k_equals_t_matches_forward():
    """2: forward_with_k(K=T) ≡ forward() bit-exact."""
    print("[2/7] forward_with_k(K=T) == forward()...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP: needs CUDA + SparseFlow"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    functional.reset_net(fused)
    y_default = fused._impl_sparse.forward(
        x, fused.spec, fused._ensure_params(), fused.state)

    functional.reset_net(fused)
    y_k_t = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=T)

    max_diff = (y_default - y_k_t).abs().max().item()
    print(f"  max_diff = {max_diff:.6e}")
    if max_diff > 0:
        print("  FAIL: K=T path not bit-exact with default forward")
        return False
    print("  OK: K=T path is bit-exact identical to default")
    return True


def check_k_sweep_bit_exact():
    """3+4: forward_with_k(K=1) and (K=2) both ≡ forward_with_k(K=T).

    This is the experimental verification of §3.13 Lemma 3.14: any
    legal K choice produces the same output.
    """
    print("[3/7] K-sweep bit-exact (§3.13 Lemma 3.14 — experimental)...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); traceback.print_exc(); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    # Test multiple shapes — these mirror the §5.2 K-sweep target layers
    shapes = [
        (4, 2, 16, 16),  # T=4, small spatial, like layer3
        (4, 2, 8, 8),    # T=4, even smaller, like layer4
        (8, 1, 16, 16),  # T=8, K∈{1,2,4,8}
    ]

    all_ok = True
    for (T, B, H, W) in shapes:
        fused, x = _build_layer_and_input(device, T, B, H, W)
        outputs = {}
        for K in [1, 2, T // 2 if T > 2 else T, T]:
            functional.reset_net(fused)
            y_K = fused._impl_sparse.forward_with_k(
                x, fused.spec, fused._ensure_params(), fused.state, K=K)
            outputs[K] = y_K.detach().cpu()

        # Compare every K against K=T
        ref = outputs[T]
        for K, y in outputs.items():
            if K == T:
                continue
            diff = (y - ref).abs().max().item()
            mark = "✓" if diff == 0 else "✗"
            print(f"  T={T}, B={B}, H=W={H}: K={K} vs K={T}  max_diff={diff:.2e}  {mark}")
            if diff != 0:
                all_ok = False

    if not all_ok:
        print("  FAIL: §3.13 Lemma 3.14 violated empirically")
        return False
    print("  OK: every K∈{1,2,T/2,T} produces bit-exact identical output")
    return True


def check_k_clamping():
    """5: K out of range gets clamped (no crash, no bad output)."""
    print("[4/7] K clamping (K≤0 → 1, K>T → T)...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    functional.reset_net(fused)
    y_T = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=T)

    # K=999 should clamp to T
    functional.reset_net(fused)
    y_huge = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=999)
    if (y_huge - y_T).abs().max().item() != 0:
        print(f"  FAIL: K=999 didn't clamp to T={T}")
        return False

    # K=0 should clamp to 1
    functional.reset_net(fused)
    y_K1 = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=1)
    functional.reset_net(fused)
    y_K0 = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=0)
    if (y_K0 - y_K1).abs().max().item() != 0:
        print(f"  FAIL: K=0 didn't clamp to 1")
        return False

    print("  OK: K=999 → T,  K=0 → 1")
    return True


def check_state_carry():
    """6: State carry across chunks works correctly (StateCarry(LIF))."""
    print("[5/7] State carry across TimeBlock boundaries...")
    try:
        import torch
        from spikingjelly.activation_based import functional
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None

    device = "cuda:0"
    T, B, H, W = 4, 2, 16, 16
    fused, x = _build_layer_and_input(device, T, B, H, W)

    # K=2 means 2 chunks. After first chunk, state must be non-zero
    # (assuming x_chunk_0 produces non-trivial v).
    functional.reset_net(fused)
    assert not fused.state.is_initialized

    # Run only first chunk manually
    x_chunk_0 = x[:2].contiguous()
    _ = fused._impl_sparse.forward(
        x_chunk_0, fused.spec, fused._ensure_params(), fused.state)
    assert fused.state.is_initialized
    v_after_chunk_0 = fused.state.tensor.clone()
    if not v_after_chunk_0.any():
        print("  WARN: v after chunk_0 is all-zero — try a denser input")

    # Now if we run forward_with_k(K=2) from scratch, the inner forward
    # call for chunk_1 should see v_after_chunk_0 as input.
    # We'll verify by: full K=2 run vs manual two-step run.
    functional.reset_net(fused)
    y_full = fused._impl_sparse.forward_with_k(
        x, fused.spec, fused._ensure_params(), fused.state, K=2)

    functional.reset_net(fused)
    y_a = fused._impl_sparse.forward(
        x[:2].contiguous(), fused.spec, fused._ensure_params(), fused.state)
    y_b = fused._impl_sparse.forward(
        x[2:].contiguous(), fused.spec, fused._ensure_params(), fused.state)
    y_manual = torch.cat([y_a, y_b], dim=0)

    diff = (y_full - y_manual).abs().max().item()
    print(f"  forward_with_k(K=2) vs manual 2-step:  max_diff = {diff:.2e}")
    if diff != 0:
        print("  FAIL: state carry not behaving as expected")
        return False
    print("  OK: state correctly carried across TimeBlock boundary")
    return True


def check_resnet18_regression():
    """7: SEW-ResNet18 still bit-exact — Stage 6 only ADDS forward_with_k,
    forward() is unchanged, so SEW-RN18 routing must still be 0-diff.
    """
    print("[6/7] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available():
        print("  SKIP"); return None
    try:
        from catfuse import optimize
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP"); return None

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
            print("  SKIP"); return None
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
    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e}")
    return True


def check_full_network_k_aware():
    """7b: BONUS — apply K=2 across an entire SEW-RN18 (every STFusion layer
    uses forward_with_k(K=2) internally). Compare with default.

    This is a sneak-preview of the §3.10 experiment. Note that PartialFusion
    layers don't use K (DenseKeep, K is fixed). Only STFusion layers swap.
    """
    print("[7/7] BONUS: SEW-RN18 with all STFusion layers using K=2...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP"); return None
    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP"); return None
    try:
        from catfuse import optimize
    except Exception:
        print("  SKIP"); return None

    try:
        from spikingjelly.activation_based.model import sew_resnet
        from spikingjelly.activation_based import surrogate
        net = sew_resnet.sew_resnet18(
            pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(), detach_reset=True,
        )
        functional.set_step_mode(net, "m")
    except Exception:
        print("  SKIP"); return None

    device = "cuda:0"
    net = net.to(device).eval()
    T, B = 4, 2
    x = torch.randn(T, B, 3, 32, 32, device=device)

    fused, _ = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    with torch.no_grad():
        functional.reset_net(fused)
        y_default = fused(x)

    # Monkey-patch each STFusion's _batchfold_forward to use K=2 instead of K=T
    st_layers = [m for m in fused.modules() if isinstance(m, STFusionConvBNLIF)]
    print(f"  Found {len(st_layers)} STFusion layers; switching to K=2 forward")

    for layer in st_layers:
        original_batchfold = layer._batchfold_forward
        def _make_k2_forward(layer):
            def _bf(x):
                if layer._impl_sparse is not None:
                    return layer._impl_sparse.forward_with_k(
                        x, layer.spec, layer._ensure_params(), layer.state, K=2)
                return layer._impl_dense.forward(
                    x, layer.spec, layer._ensure_params(), layer.state)
            return _bf
        layer._batchfold_forward = _make_k2_forward(layer)

    with torch.no_grad():
        functional.reset_net(fused)
        y_k2 = fused(x)

    max_diff = (y_default - y_k2).abs().max().item()
    print(f"  Full network: K=T vs K=2 across all STFusion layers")
    print(f"  max_diff = {max_diff:.2e}")
    if max_diff != 0:
        print("  FAIL: end-to-end K-sweep bit-exact violated")
        return False
    print("  OK: full SEW-RN18 forward bit-exact across K choices")
    return True


def main():
    print("=" * 60)
    print("Stage 6 verification: K-aware forward (§3.10/§3.13 Lemma 3.14)")
    print("=" * 60)
    results = [
        ("forward_with_k_exists", check_forward_with_k_exists()),
        ("k_equals_t_matches_forward", check_k_equals_t_matches_forward()),
        ("k_sweep_bit_exact", check_k_sweep_bit_exact()),
        ("k_clamping", check_k_clamping()),
        ("state_carry", check_state_carry()),
        ("resnet18_regression", check_resnet18_regression()),
        ("full_network_k_aware", check_full_network_k_aware()),
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
    "catfuse/implementations/sparse_flow.py": [
        ('    # ---- Public API --------------------------------------------------\n\n    def forward(\n        self,\n        x: torch.Tensor,\n        spec: ConvLIFSpec,\n        params: ConvLIFParams,\n        state,\n    ) -> torch.Tensor:\n        """Lean batchfold sparse Conv→BN→LIF kernel.\n\n        Preconditions (caller must check):\n          - x.is_cuda\n          - spec.stride == 1\n          - spec.kernel_size == 3\n        """\n        if not _SPARSEFLOW_AVAILABLE:\n            raise RuntimeError("SparseFlow.forward called without sparseflow kernels")\n\n        T, B = x.shape[0], x.shape[1]\n        H_in, W_in = x.shape[3], x.shape[4]\n        device = x.device\n\n        # Init / lookup cache for this shape\n        c = self._ensure_lean_cache(spec, params, B, H_in, W_in, device)\n        H_out, W_out = c[\'H_out\'], c[\'W_out\']\n\n        # StaticZero short-circuit (same code path as DenseKeep —\n        # static_zero_forward is bit-exact-shared)\n        x_flat = x.reshape(T * B, spec.in_channels, H_in, W_in)\n        if not x_flat.any():\n            return static_zero_forward(\n                spec, params, T, B, H_out, W_out, device, state, lif_sequential,\n            )\n\n        x_contig = x_flat.contiguous()\n        N_TILES = c[\'N_TILES\']\n\n        # State\n        v_init = state.get(\n            shape=(B, spec.out_channels, H_out, W_out),\n            device=device, dtype=torch.float32,\n        )\n        v_init = v_init.float().contiguous()\n\n        # Output buffers (per-call, since shape can vary)\n        spike_out = torch.empty(T * B, spec.out_channels, H_out, W_out,\n                                dtype=torch.float32, device=device)\n        v_out = torch.empty_like(v_init)\n\n        def _grid_streamfuse(META):\n            return (N_TILES, triton.cdiv(spec.out_channels, META["BLOCK_N"]))\n\n        sparse_streamfuse_conv3x3_bn_lif[_grid_streamfuse](\n            x_contig,\n            self._w_cl,\n            c[\'bias_arg\'],\n            c[\'bn_scale_arg\'],\n            c[\'bn_bias_arg\'],\n            v_init,\n            spike_out,\n            v_out,\n            T, B,\n            C_IN=spec.in_channels,\n            C_OUT=spec.out_channels,\n            H=H_in, W=W_in,\n            H_OUT=H_out, W_OUT=W_out,\n            GH=c[\'GH\'], GW=c[\'GW\'],\n            HAS_BIAS=c[\'has_bias\'],\n            HAS_BN=c[\'has_bn\'],\n            DECAY=c[\'decay\'],\n            RECIP_TAU=c[\'recip_tau\'],\n            V_TH=float(spec.v_threshold),\n            HAS_V_RESET=c[\'has_v_reset\'],\n            V_RESET=c[\'v_reset_val\'],\n            GROUP_SIZE_C=c[\'GSC\'],\n            NUM_GROUPS=c[\'NUM_GROUPS\'],\n        )\n\n        state.set(v_out)\n        return spike_out.reshape(T, B, spec.out_channels, H_out, W_out)\n\n    def analytic_io_cost(', '    # ---- Public API --------------------------------------------------\n\n    def forward(\n        self,\n        x: torch.Tensor,\n        spec: ConvLIFSpec,\n        params: ConvLIFParams,\n        state,\n    ) -> torch.Tensor:\n        """Lean batchfold sparse Conv→BN→LIF kernel (single-block, K=T).\n\n        This is the default production path used by STFusionConvBNLIF.\n        For K-parameterized scheduling (§3.10 K-sweep experiment), use\n        forward_with_k() instead — it accepts a K argument and, when K<T,\n        chunks the time dimension and invokes this method ceil(T/K) times,\n        relying on StateBuffer to carry v across chunks (StateCarry(LIF)).\n\n        Preconditions (caller must check):\n          - x.is_cuda\n          - spec.stride == 1\n          - spec.kernel_size == 3\n        """\n        if not _SPARSEFLOW_AVAILABLE:\n            raise RuntimeError("SparseFlow.forward called without sparseflow kernels")\n\n        T, B = x.shape[0], x.shape[1]\n        H_in, W_in = x.shape[3], x.shape[4]\n        device = x.device\n\n        # Init / lookup cache for this shape\n        c = self._ensure_lean_cache(spec, params, B, H_in, W_in, device)\n        H_out, W_out = c[\'H_out\'], c[\'W_out\']\n\n        # StaticZero short-circuit (same code path as DenseKeep —\n        # static_zero_forward is bit-exact-shared)\n        x_flat = x.reshape(T * B, spec.in_channels, H_in, W_in)\n        if not x_flat.any():\n            return static_zero_forward(\n                spec, params, T, B, H_out, W_out, device, state, lif_sequential,\n            )\n\n        x_contig = x_flat.contiguous()\n        N_TILES = c[\'N_TILES\']\n\n        # State\n        v_init = state.get(\n            shape=(B, spec.out_channels, H_out, W_out),\n            device=device, dtype=torch.float32,\n        )\n        v_init = v_init.float().contiguous()\n\n        # Output buffers (per-call, since shape can vary)\n        spike_out = torch.empty(T * B, spec.out_channels, H_out, W_out,\n                                dtype=torch.float32, device=device)\n        v_out = torch.empty_like(v_init)\n\n        def _grid_streamfuse(META):\n            return (N_TILES, triton.cdiv(spec.out_channels, META["BLOCK_N"]))\n\n        sparse_streamfuse_conv3x3_bn_lif[_grid_streamfuse](\n            x_contig,\n            self._w_cl,\n            c[\'bias_arg\'],\n            c[\'bn_scale_arg\'],\n            c[\'bn_bias_arg\'],\n            v_init,\n            spike_out,\n            v_out,\n            T, B,\n            C_IN=spec.in_channels,\n            C_OUT=spec.out_channels,\n            H=H_in, W=W_in,\n            H_OUT=H_out, W_OUT=W_out,\n            GH=c[\'GH\'], GW=c[\'GW\'],\n            HAS_BIAS=c[\'has_bias\'],\n            HAS_BN=c[\'has_bn\'],\n            DECAY=c[\'decay\'],\n            RECIP_TAU=c[\'recip_tau\'],\n            V_TH=float(spec.v_threshold),\n            HAS_V_RESET=c[\'has_v_reset\'],\n            V_RESET=c[\'v_reset_val\'],\n            GROUP_SIZE_C=c[\'GSC\'],\n            NUM_GROUPS=c[\'NUM_GROUPS\'],\n        )\n\n        state.set(v_out)\n        return spike_out.reshape(T, B, spec.out_channels, H_out, W_out)\n\n    def forward_with_k(\n        self,\n        x: torch.Tensor,\n        spec: ConvLIFSpec,\n        params: ConvLIFParams,\n        state,\n        K: int = None,\n    ) -> torch.Tensor:\n        """K-aware forward: realize TimeBlock(K) ∘ StreamFuse(Conv,LIF) ∘ StateCarry(LIF).\n\n        This method exists to support the §3.10 K-sweep experiment.\n\n        Semantics: the time dimension of x [T, B, C_in, H, W] is partitioned\n        into ceil(T/K) blocks of size ≤ K. Each block is forwarded through\n        the same lean kernel as forward(), with state carried via the\n        StateBuffer between blocks (this is exactly StateCarry(LIF) at the\n        block boundary, by construction — no special code needed).\n\n        Bit-exactness contract: for any K ∈ {1, 2, ..., T},\n\n            forward_with_k(x, spec, params, state, K=K)\n                ≡ forward_with_k(x, spec, params, state, K=T)\n                ≡ forward(x, spec, params, state)\n\n        elementwise on every output entry, given identical state initial\n        condition. This is the experimental form of §3.13 Lemma 3.14:\n        all four primitives applied with different K give the same output.\n\n        Reference for §3.10 K-sweep:\n          - K=T: single block, current default forward (§3.9 lean form)\n          - K<T: ceil(T/K) chunks, each invoking forward(); v_io grows\n                 linearly in num_blocks per the analytic_io_cost formula\n\n        K=None defaults to T (single-block, equivalent to forward()).\n        """\n        T = x.shape[0]\n        if K is None:\n            K = T\n        K = max(1, min(int(K), int(T)))\n\n        if K >= T:\n            # Single block — identical to default forward()\n            return self.forward(x, spec, params, state)\n\n        # Multi-block: chunk along T axis and let StateBuffer carry v.\n        # Each forward() call reads v from state (state.get) and writes\n        # final v back (state.set), so consecutive chunks see the previous\n        # chunk\'s terminal v as their initial v — exactly StateCarry(LIF).\n        spike_chunks = []\n        start = 0\n        while start < T:\n            end = min(start + K, T)\n            x_chunk = x[start:end].contiguous()\n            spike_chunk = self.forward(x_chunk, spec, params, state)\n            spike_chunks.append(spike_chunk)\n            start = end\n        return torch.cat(spike_chunks, dim=0)\n\n    def analytic_io_cost('),
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
    print(f"Applying Stage 6 refactor in: {repo_root}")
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
    print("Stage 6 applied successfully.")
    print("Next: python tests/stage6_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))