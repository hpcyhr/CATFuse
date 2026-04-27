"""catfuse.implementations.sparse_flow — SparseFlow implementation of σ.

SparseFlow realizes Conv→BN→LIF as a single fused multi-step Triton kernel
that exploits input sparsity. Used by Runtime EGD when input sparsity > 0.7.

Bit-exact equivalence with DenseKeep (Corollary 3.17) is empirically verified
in tests/parity_histogram.py — both implementations produce max_diff = 0
across all CATFuse-supported networks (SEW-RN18 verified post-stage-2).

This module currently supports only the LEAN BATCHFOLD path (the multi-step
sparse_streamfuse_conv3x3_bn_lif kernel). The legacy per-step path
(sparse_fused_conv_bn_lif_forward, used as a fallback in pre-stage-3 code)
remains in catfuse.sparseflow.ops.st_fusion_conv_bn_lif.STFusionConvBNLIF
for backward-compatibility but is not exposed here. Constraints under which
SparseFlow.forward is callable:
  - x.is_cuda
  - spec.kernel_size == 3
  - spec.stride == 1
  - the lean Triton kernels are compilable on this device

The pattern (STFusionConvBNLIF) is responsible for checking these and
falling back to DenseKeep otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from catfuse.implementations.base import (
    ConvLIFSpec, ConvLIFParams, Implementation, static_zero_forward,
)


# Lean kernel imports — guarded so that import of this module does not fail
# on environments where Triton is unavailable (DenseKeep still works).
try:
    import triton
    from catfuse.sparseflow.fused_conv_bn_lif_kernel import (
        fused_bm_conv3x3_bn_lif_8x8,
        fused_bm_conv3x3_bn_lif_8x16,
    )
    from catfuse.sparseflow.sparse_conv2d_kernel import (
        _select_tile_sizes, choose_group_size,
    )
    from catfuse.sparseflow.streamfuse_kernel import (
        sparse_streamfuse_conv3x3_bn_lif,
    )
    from catfuse.sparseflow.lif_seq_kernel import lif_sequential
    _SPARSEFLOW_AVAILABLE = True
except ImportError:
    _SPARSEFLOW_AVAILABLE = False


class SparseFlow(Implementation):
    """Multi-step sparse fused Conv→BN→LIF Triton kernel.

    Caches:
      _w_cl: channel-last fp16 weight layout used by the sparse kernel
      _lean_cache: per-(B, H_in, W_in) precomputed shapes, BN args, kernel
                   selection, prescan buffers (initialized lazily on first
                   forward via _init_lean_cache).

    Caches are invalidated by reset_caches() when params change.
    """

    name = "SparseFlow"

    def __init__(self):
        super().__init__()
        if not _SPARSEFLOW_AVAILABLE:
            raise RuntimeError(
                "SparseFlow Implementation requires catfuse.sparseflow "
                "kernels (Triton); they failed to import."
            )
        # Channel-last fp16 weight (set by _ensure_w_cl)
        self._w_cl: Optional[torch.Tensor] = None
        self._w_cl_key: Optional[int] = None
        # Lean cache keyed by (B, H_in, W_in, device-index)
        self._lean_cache: Dict[str, Any] = {}
        self._lean_cache_key: Optional[tuple] = None

    def reset_caches(self) -> None:
        self._w_cl = None
        self._w_cl_key = None
        self._lean_cache = {}
        self._lean_cache_key = None

    # ---- Internal cache management -----------------------------------

    def _ensure_w_cl(self, params: ConvLIFParams) -> None:
        """Build channel-last fp16 weight layout for the sparse kernel."""
        key = id(params.weight)
        if self._w_cl is not None and self._w_cl_key == key:
            return
        # weight: [C_out, C_in, kH, kW] -> [C_out, kH, kW, C_in] fp16
        self._w_cl = params.weight.data.half().permute(0, 2, 3, 1).contiguous()
        self._w_cl_key = key

    def _ensure_lean_cache(
        self,
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        B: int,
        H_in: int,
        W_in: int,
        device,
    ) -> Dict[str, Any]:
        """Initialize / look up the lean kernel cache for given input shape.

        Cache invalidation: keyed by (B, H_in, W_in, device index, params
        identity). Mismatch triggers re-init.
        """
        # Note: device.index can be None for CPU; only CUDA expected here.
        dev_idx = device.index if device.index is not None else -1
        key = (
            B, H_in, W_in, dev_idx,
            id(params.weight), id(params.bias),
            id(params.bn_scale), id(params.bn_bias),
        )
        if self._lean_cache_key == key:
            return self._lean_cache

        H_out, W_out = spec.output_hw(H_in, W_in)
        BH, BW = _select_tile_sizes(H_out, W_out)
        GH = triton.cdiv(H_out, BH)
        GW = triton.cdiv(W_out, BW)
        N_TILES = B * GH * GW
        GSC = choose_group_size(spec.in_channels)
        NUM_GROUPS = triton.cdiv(spec.in_channels, GSC)

        # LIF constants
        decay = 1.0 - 1.0 / spec.tau
        recip_tau = 1.0 / spec.tau
        has_v_reset = spec.v_reset is not None
        v_reset_val = 0.0 if spec.v_reset is None else float(spec.v_reset)

        # BN args (kernel takes them as fp32 contiguous; placeholder if absent)
        if spec.has_bn:
            bn_scale_arg = params.bn_scale.float().contiguous()
            bn_bias_arg = params.bn_bias.float().contiguous()
        else:
            bn_scale_arg = torch.empty(1, dtype=torch.float32, device=device)
            bn_bias_arg = torch.empty(1, dtype=torch.float32, device=device)

        # Conv bias arg
        if spec.has_conv_bias and params.bias is not None:
            bias_arg = params.bias.detach().float().contiguous()
        else:
            bias_arg = torch.empty(1, dtype=torch.float32, device=device)

        # Per-step kernel selection (used when STFusion calls
        # _per_step_forward via SparseFlow; not used in lean batchfold path).
        kernel = fused_bm_conv3x3_bn_lif_8x16 if BW == 16 else fused_bm_conv3x3_bn_lif_8x8

        def _grid_per_step(META):
            return (N_TILES, triton.cdiv(spec.out_channels, META["BLOCK_N"]))

        # Pre-allocate buffers
        self._ensure_w_cl(params)
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
        spike_buf = torch.empty(B, spec.out_channels, H_out, W_out,
                                dtype=torch.float32, device=device)
        v_next_buf = torch.empty_like(spike_buf)

        self._lean_cache = {
            'H_out': H_out, 'W_out': W_out,
            'BH': BH, 'BW': BW, 'GH': GH, 'GW': GW,
            'N_TILES': N_TILES, 'GSC': GSC, 'NUM_GROUPS': NUM_GROUPS,
            'decay': decay, 'recip_tau': recip_tau,
            'has_v_reset': has_v_reset, 'v_reset_val': v_reset_val,
            'has_bias': spec.has_conv_bias, 'has_bn': spec.has_bn,
            'bias_arg': bias_arg,
            'bn_scale_arg': bn_scale_arg, 'bn_bias_arg': bn_bias_arg,
            'kernel': kernel, 'grid': _grid_per_step,
            'ag_mask_buf': ag_mask_buf, 'tile_class_buf': tile_class_buf,
            'spike_buf': spike_buf, 'v_next_buf': v_next_buf,
        }
        self._lean_cache_key = key
        return self._lean_cache

    # ---- Public API --------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        state,
    ) -> torch.Tensor:
        """Lean batchfold sparse Conv→BN→LIF kernel (single-block, K=T).

        This is the default production path used by STFusionConvBNLIF.
        For K-parameterized scheduling (§3.10 K-sweep experiment), use
        forward_with_k() instead — it accepts a K argument and, when K<T,
        chunks the time dimension and invokes this method ceil(T/K) times,
        relying on StateBuffer to carry v across chunks (StateCarry(LIF)).

        Preconditions (caller must check):
          - x.is_cuda
          - spec.stride == 1
          - spec.kernel_size == 3
        """
        if not _SPARSEFLOW_AVAILABLE:
            raise RuntimeError("SparseFlow.forward called without sparseflow kernels")

        T, B = x.shape[0], x.shape[1]
        H_in, W_in = x.shape[3], x.shape[4]
        device = x.device

        # Init / lookup cache for this shape
        c = self._ensure_lean_cache(spec, params, B, H_in, W_in, device)
        H_out, W_out = c['H_out'], c['W_out']

        # StaticZero short-circuit (same code path as DenseKeep —
        # static_zero_forward is bit-exact-shared)
        x_flat = x.reshape(T * B, spec.in_channels, H_in, W_in)
        if not x_flat.any():
            return static_zero_forward(
                spec, params, T, B, H_out, W_out, device, state, lif_sequential,
            )

        x_contig = x_flat.contiguous()
        N_TILES = c['N_TILES']

        # State
        v_init = state.get(
            shape=(B, spec.out_channels, H_out, W_out),
            device=device, dtype=torch.float32,
        )
        v_init = v_init.float().contiguous()

        # Output buffers (per-call, since shape can vary)
        spike_out = torch.empty(T * B, spec.out_channels, H_out, W_out,
                                dtype=torch.float32, device=device)
        v_out = torch.empty_like(v_init)

        def _grid_streamfuse(META):
            return (N_TILES, triton.cdiv(spec.out_channels, META["BLOCK_N"]))

        sparse_streamfuse_conv3x3_bn_lif[_grid_streamfuse](
            x_contig,
            self._w_cl,
            c['bias_arg'],
            c['bn_scale_arg'],
            c['bn_bias_arg'],
            v_init,
            spike_out,
            v_out,
            T, B,
            C_IN=spec.in_channels,
            C_OUT=spec.out_channels,
            H=H_in, W=W_in,
            H_OUT=H_out, W_OUT=W_out,
            GH=c['GH'], GW=c['GW'],
            HAS_BIAS=c['has_bias'],
            HAS_BN=c['has_bn'],
            DECAY=c['decay'],
            RECIP_TAU=c['recip_tau'],
            V_TH=float(spec.v_threshold),
            HAS_V_RESET=c['has_v_reset'],
            V_RESET=c['v_reset_val'],
            GROUP_SIZE_C=c['GSC'],
            NUM_GROUPS=c['NUM_GROUPS'],
        )

        state.set(v_out)
        return spike_out.reshape(T, B, spec.out_channels, H_out, W_out)

    def forward_with_k(
        self,
        x: torch.Tensor,
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        state,
        K: int = None,
    ) -> torch.Tensor:
        """K-aware forward: realize TimeBlock(K) ∘ StreamFuse(Conv,LIF) ∘ StateCarry(LIF).

        This method exists to support the §3.10 K-sweep experiment.

        Semantics: the time dimension of x [T, B, C_in, H, W] is partitioned
        into ceil(T/K) blocks of size ≤ K. Each block is forwarded through
        the same lean kernel as forward(), with state carried via the
        StateBuffer between blocks (this is exactly StateCarry(LIF) at the
        block boundary, by construction — no special code needed).

        Bit-exactness contract: for any K ∈ {1, 2, ..., T},

            forward_with_k(x, spec, params, state, K=K)
                ≡ forward_with_k(x, spec, params, state, K=T)
                ≡ forward(x, spec, params, state)

        elementwise on every output entry, given identical state initial
        condition. This is the experimental form of §3.13 Lemma 3.14:
        all four primitives applied with different K give the same output.

        Reference for §3.10 K-sweep:
          - K=T: single block, current default forward (§3.9 lean form)
          - K<T: ceil(T/K) chunks, each invoking forward(); v_io grows
                 linearly in num_blocks per the analytic_io_cost formula

        K=None defaults to T (single-block, equivalent to forward()).
        """
        T = x.shape[0]
        if K is None:
            K = T
        K = max(1, min(int(K), int(T)))

        if K >= T:
            # Single block — identical to default forward()
            return self.forward(x, spec, params, state)

        # Multi-block: chunk along T axis and let StateBuffer carry v.
        # Each forward() call reads v from state (state.get) and writes
        # final v back (state.set), so consecutive chunks see the previous
        # chunk's terminal v as their initial v — exactly StateCarry(LIF).
        spike_chunks = []
        start = 0
        while start < T:
            end = min(start + K, T)
            x_chunk = x[start:end].contiguous()
            spike_chunk = self.forward(x_chunk, spec, params, state)
            spike_chunks.append(spike_chunk)
            start = end
        return torch.cat(spike_chunks, dim=0)

    def analytic_io_cost(
        self,
        spec,
        T: int, B: int, H_in: int, W_in: int,
        K: int = None,
        *,
        dtype_bytes: int = 4,
    ):
        """§3.9 CTF I/O cost: TimeBlock(K) ∘ StreamFuse(Conv,LIF) ∘ StateCarry(LIF).

        Schedule:
          1. T steps split into ceil(T/K) blocks of size ≤ K.
          2. Within each block: Conv→BN→LIF run as one fused kernel; z stays
             in registers, v stays in registers across the K steps inside.
             No HBM traffic for z; no intra-block HBM traffic for v.
          3. Between blocks: v written to HBM at end-of-block, read back at
             start-of-next-block (StateCarry).
          4. Spike maps for each step are still written to HBM (they are
             the layer's output to the next layer).

        K=None defaults to T (single-block lean batchfold, the current
        SparseFlow.forward path). K=1 degenerates to per-step v carry.
        Intermediate K values are predicted here even though the current
        kernel implementation only realizes K=T directly — the formula is
        the §3.10 K-sweep prediction target.

        Cost (per forward, in bytes):
          num_blocks   = ceil(T/K)
          x_load       = T·B·HWC_in                 ·dtype_bytes
          w_load       = num_blocks·C_out·C_in·k²   ·dtype_bytes  (worst case;
                         L2 caching may reduce empirically)
          z_io         = 0                                          (StreamFuse)
          v_io         = 2·num_blocks·B·HWC_out     ·dtype_bytes
          spike_write  = T·B·HWC_out                ·dtype_bytes

        intermediate_io = z_io + v_io = 2·ceil(T/K)·B·HWC_out·dtype_bytes ∈ O(T/K·HWC).

        This is exactly the §3.9 reduction: O(T·HWC) → O(T/K·HWC).
        """
        from catfuse.implementations.base import IOCost
        import math

        if K is None:
            K = T
        K = max(1, min(int(K), int(T)))
        num_blocks = math.ceil(T / K)

        H_out, W_out = spec.output_hw(H_in, W_in)
        HWC_in = H_in * W_in * spec.in_channels
        HWC_out = H_out * W_out * spec.out_channels
        kk = spec.kernel_size * spec.kernel_size

        x_load = T * B * HWC_in * dtype_bytes
        # Weight reload per block in worst case (L2 may amortize empirically;
        # we report the upper bound to be honest about the schedule's cost).
        w_load = num_blocks * spec.out_channels * spec.in_channels * kk * dtype_bytes
        # z stays in registers across the entire block — never reaches HBM
        z_io = 0
        # v carried at block boundaries: 1 read + 1 write per block
        v_io = 2 * num_blocks * B * HWC_out * dtype_bytes
        spike_write = T * B * HWC_out * dtype_bytes

        return IOCost(
            x_load=x_load,
            w_load=w_load,
            z_io=z_io,
            v_io=v_io,
            spike_write=spike_write,
            schedule=(f"TimeBlock(K={K})∘StreamFuse(Conv,LIF)∘StateCarry(LIF) "
                      f"— z in registers, v carried at {num_blocks} block boundaries"),
            num_blocks=num_blocks,
        )

    def schedule_decomposition(self, spec, T: int, K: int = None):
        """§3.13 decomposition of SparseFlow's schedule.

        SparseFlow realizes σ as the §3.9 canonical CTF form:

            TimeBlock(K) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)

        Reading inner-to-outer:
          1. StateCarry(LIF) — v written/read at block boundaries, never
             between time steps inside a block.
          2. StreamFuse(Conv, LIF) — z = BN(Conv(x)) flows from Conv to
             LIF on-chip (registers/shared); never reaches HBM. This is
             the §3.9 z_io elimination.
          3. TimeBlock(K) — partition T steps into ceil(T/K) blocks.

        K=None defaults to T (single-block lean batchfold, current default
        SparseFlow.forward path).

        This is §3.8 form_2 — the StreamFuse-form. It corresponds directly
        to the §3.9 main result: I/O reduction from O(T·HWC) to O(T/K·HWC).
        """
        from catfuse.implementations.base import (
            ScheduleTransform, ScheduleDecomposition,
        )
        if K is None:
            K = T
        K = max(1, min(int(K), int(T)))
        return ScheduleDecomposition(
            transforms=(
                ScheduleTransform.StateCarry("LIF"),
                ScheduleTransform.StreamFuse("Conv", "LIF"),
                ScheduleTransform.TimeBlock(K),
            ),
            form="form_2",
            description=(
                f"SparseFlow schedule: TimeBlock(K={K}) ∘ StreamFuse(Conv, LIF) "
                f"∘ StateCarry(LIF). z stays on-chip; v carried at "
                f"{(T + K - 1) // K} block boundaries."
            ),
        )

    def __repr__(self) -> str:
        if not _SPARSEFLOW_AVAILABLE:
            return "SparseFlow(unavailable)"
        return (f"SparseFlow(w_cl={'yes' if self._w_cl is not None else 'no'}, "
                f"cached_shape={self._lean_cache_key})")
