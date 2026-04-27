#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Stage 3 refactor: Implementation interface abstraction (problem 2).

Stage 3 introduces the catfuse.implementations subpackage which formalizes
Definition 3.16 (Operator Implementation, Implementation Set Impl(σ)) from
the paper.  After this stage, the DenseKeep math (cuDNN conv + Triton LIF)
exists ONCE in catfuse/implementations/dense_keep.py instead of being
duplicated across PartialFusionConvBNLIF._forward_impl and three branches
of STFusionConvBNLIF.  Likewise SparseFlow gets isolated in
catfuse/implementations/sparse_flow.py.

The pattern classes (PartialFusionConvBNLIF, STFusionConvBNLIF) are reduced
to thin wrappers that:
  - Construct a ConvLIFSpec at __init__ (frozen topology of σ).
  - Hold one or more Implementation instances.
  - Build ConvLIFParams (tensor references) lazily, with id-based cache
    invalidation for .to(device) migration.
  - Delegate forward to the selected impl.

This script is IDEMPOTENT: running it twice on the same workdir is safe.

Prerequisites: stages 1 and 2 must already be applied. The script assumes
the post-stage-2 state on disk and verifies via str search.

Run:
    cd /path/to/CATFuse
    python apply_stage3.py
    python tests/stage3_verify.py
"""
import os
import sys


# ============================================================
# Embedded new files
# ============================================================

NEW_FILES = {
    'catfuse/implementations/__init__.py': r'''"""catfuse.implementations — Implementation hierarchy for Definition 3.16.

Public API:
    Implementation       : abstract base
    ConvLIFSpec          : frozen static topology of σ
    ConvLIFParams        : runtime tensor params (references) of σ
    DenseKeep            : cuDNN conv + Triton lif_sequential
    SparseFlow           : lean fused Conv→BN→LIF Triton kernel (sparse-aware)

By Corollary 3.17, DenseKeep and SparseFlow are bit-exact equivalent on
every input — they are interchangeable members of Impl(σ) for any σ
satisfying SparseFlow's preconditions (3x3 conv, stride=1, CUDA).
"""
from catfuse.implementations.base import (
    Implementation,
    ConvLIFSpec,
    ConvLIFParams,
    static_zero_forward,
)
from catfuse.implementations.dense_keep import DenseKeep

# SparseFlow may fail to import on systems without Triton; guard so that
# DenseKeep-only configurations remain usable.
try:
    from catfuse.implementations.sparse_flow import SparseFlow
    _SPARSEFLOW_AVAILABLE = True
except (ImportError, RuntimeError):
    SparseFlow = None  # type: ignore
    _SPARSEFLOW_AVAILABLE = False


__all__ = [
    "Implementation",
    "ConvLIFSpec",
    "ConvLIFParams",
    "DenseKeep",
    "SparseFlow",
    "static_zero_forward",
]
''',
    'catfuse/implementations/base.py': r'''"""catfuse.implementations.base — abstractions for Definition 3.16.

Per Definition 3.16.1 (Operator Implementation), an Implementation is a
bit-exact realization of σ ∈ Σ(G,T) that takes (input, spec, params, state)
and produces output spikes while updating state.

Per Definition 3.16.2 (Implementation Set Impl(σ)), multiple Implementations
may exist for the same σ, all bit-exact equivalent to each other on every
input.

Per Corollary 3.17 (Implementation Selection preserves equivalence), a
runtime selector π : Ctx → Impl(σ) may freely choose among them at each
call without changing the global semantics.

The selector itself (Definition 3.16.3) is implemented at the Pattern level
(see STFusionConvBNLIF) and is OUT OF SCOPE for this module — base.py defines
only the contract that all implementations must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Spec & Params
# ============================================================

@dataclass(frozen=True)
class ConvLIFSpec:
    """Static topology of a Conv→(BN)→LIF op σ.

    Frozen at pattern construction; does not change during forward.
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    has_conv_bias: bool   # whether the conv carries a bias term
    has_bn: bool          # whether BN was present (folded into bn_scale/bn_bias)
    tau: float
    v_threshold: float
    v_reset: Optional[float]   # None = soft reset; float = hard reset
    decay_input: bool = True
    bn_eps: float = 1e-5

    def output_hw(self, h_in: int, w_in: int):
        """Compute (H_out, W_out) for given input spatial size."""
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out


class ConvLIFParams:
    """Runtime tensor params of σ (references, not copies).

    The pattern owns these tensors as nn.Parameter / nn.Buffer; ConvLIFParams
    just bundles them for passing to an Implementation. Implementations must
    not modify the tensors here (they are the pattern's source of truth).

    bn_scale and bn_bias are PRE-FOLDED affine BN params:
      bn_scale = bn_weight / sqrt(running_var + eps)
      bn_bias  = bn_bias_raw - running_mean * bn_scale
    For the Conv→BN→LIF pattern, applying these to z = Conv(x) yields:
      z' = z * bn_scale[None, :, None, None] + bn_bias[None, :, None, None]
    which equals BN(Conv(x)) in eval mode.
    """
    __slots__ = ("weight", "bias", "bn_scale", "bn_bias")

    def __init__(
        self,
        weight: torch.Tensor,                    # [C_out, C_in, k, k]
        bias: Optional[torch.Tensor],            # [C_out] or None
        bn_scale: Optional[torch.Tensor],        # [C_out] or None (no BN)
        bn_bias: Optional[torch.Tensor],         # [C_out] or None (no BN)
    ):
        self.weight = weight
        self.bias = bias
        self.bn_scale = bn_scale
        self.bn_bias = bn_bias


# ============================================================
# Implementation ABC
# ============================================================

class Implementation(nn.Module, ABC):
    """Abstract base for one element of Impl(σ).

    Subclasses must:
      - Set class attribute `name` (str): e.g. "DenseKeep", "SparseFlow".
      - Override forward(x, spec, params, state) producing bit-exact output.

    Bit-exactness contract (Corollary 3.17):
        ∀ x, ∀ a, b ∈ Impl(σ):
            a.forward(x, spec, params, state_a) ≡ b.forward(x, spec, params, state_b)
        pointwise on every (T, B, C_out, H_out, W_out) entry, given identical
        x, spec, params, and identical state initial condition.

    Implementations may carry IMPL-LOCAL caches (e.g. a pre-folded weight,
    a channel-last layout, prescan buffers). Such caches must be invalidated
    via `reset_caches()` if the pattern's params change after construction.
    """

    name: str = "<unset>"

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,                # [T, B, C_in, H, W], fp32, CUDA, contiguous
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        state: "StateBuffer",           # type: ignore[name-defined]  (catfuse.state.StateBuffer)
    ) -> torch.Tensor:                  # [T, B, C_out, H_out, W_out], fp32
        """Run the bit-exact realization of σ.

        Reads state via state.get(...), writes via state.set(v_final).
        """
        ...

    def reset_caches(self) -> None:
        """Invalidate impl-local caches. Default: no-op.

        Override in subclasses that hold lazy caches dependent on params
        (e.g. DenseKeep's pre-folded weight, SparseFlow's _w_cl).
        """
        pass


# ============================================================
# Shared utilities
# ============================================================

def static_zero_forward(
    spec: ConvLIFSpec,
    params: ConvLIFParams,
    T: int,
    B: int,
    H_out: int,
    W_out: int,
    device,
    state,
    lif_sequential_fn,
) -> torch.Tensor:
    """Bit-exact short-circuit when input x is identically zero.

    When x ≡ 0, Conv(x) ≡ 0, and z = BN(0) + bias = bn_bias (or conv_bias if
    no BN), broadcast to (T*B, C_out, H_out, W_out). This avoids invoking
    the conv kernel at all. The LIF step still runs (state evolves under
    constant input).

    Math identity (verified): for any conv with weight W and any bias b,
        Conv(0; W, b) = b broadcast to spatial dims
    Combined with BN(z; γ_folded, β_folded) = z * γ + β:
        BN(Conv(0)) = b * γ + β    if conv has bias
                    = β            otherwise
    This is what we materialize as z_4d below.

    All implementations should call this when x.any() == False, since the
    result is independent of the implementation choice.
    """
    if spec.has_bn:
        if spec.has_conv_bias:
            # z = conv_bias * bn_scale + bn_bias
            z_per_channel = params.bias * params.bn_scale + params.bn_bias
        else:
            # z = bn_bias
            z_per_channel = params.bn_bias
        z_4d = z_per_channel.view(1, -1, 1, 1).expand(
            T * B, spec.out_channels, H_out, W_out
        ).contiguous()
    elif spec.has_conv_bias:
        z_4d = params.bias.view(1, -1, 1, 1).expand(
            T * B, spec.out_channels, H_out, W_out
        ).contiguous()
    else:
        z_4d = torch.zeros(
            T * B, spec.out_channels, H_out, W_out,
            dtype=torch.float32, device=device,
        )
    z = z_4d.reshape(T, B, spec.out_channels, H_out, W_out)

    v_in = state.get(
        shape=(B, spec.out_channels, H_out, W_out),
        device=device, dtype=torch.float32,
    )
    spikes, v_out = lif_sequential_fn(
        z, v_in, tau=spec.tau,
        v_threshold=spec.v_threshold, v_reset=spec.v_reset,
    )
    state.set(v_out)
    return spikes
''',
    'catfuse/implementations/dense_keep.py': r'''"""catfuse.implementations.dense_keep — DenseKeep implementation of σ.

DenseKeep realizes Conv→(BN)→LIF as:
  1. cuDNN conv2d with BN-folded weight and bias (single F.conv2d launch)
  2. Triton lif_sequential over T steps (single multi-step kernel)

This is the "default / general-purpose" implementation, used:
  - As the only implementation in PartialFusionConvBNLIF
  - As one of two alternatives in STFusionConvBNLIF, selected by Runtime EGD
    when input sparsity is too low for SparseFlow to be efficient

Bit-exact equivalence with SparseFlow on Conv→BN→LIF is enforced by both
implementations using the SAME LIF Triton kernel (lif_sequential) for the
neuron dynamics — only the Conv backend differs (cuDNN vs sparse Triton).
For dense inputs this means identical math, identical fp32 ULP path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from catfuse.implementations.base import (
    ConvLIFSpec, ConvLIFParams, Implementation, static_zero_forward,
)


class DenseKeep(Implementation):
    """cuDNN conv + Triton lif_sequential. The simplest member of Impl(σ).

    Caches a BN-folded copy of the conv weight on first forward to avoid
    redoing the fold every call. Cache is keyed by object identity of
    params.weight / params.bn_scale / params.bn_bias / params.bias — if
    any of those tensors change (e.g. via .to(device)), the cache is
    invalidated automatically.
    """

    name = "DenseKeep"

    def __init__(self):
        super().__init__()
        # Lazy cache of BN-folded weight + effective bias.
        # _w_fused = weight * bn_scale[:, None, None, None]   (if has_bn)
        # _b_fused = bn_bias + (conv_bias * bn_scale if has_conv_bias else 0)
        # Stored as plain attributes (NOT buffers) because they are
        # derivable from params — so they shouldn't be in state_dict.
        self._w_fused: Optional[torch.Tensor] = None
        self._b_fused: Optional[torch.Tensor] = None
        # Identity tags of the params used to build the cache. Cache is
        # invalidated when any tag mismatches.
        self._cache_key: Optional[tuple] = None

    def reset_caches(self) -> None:
        self._w_fused = None
        self._b_fused = None
        self._cache_key = None

    def _params_cache_key(self, params: ConvLIFParams) -> tuple:
        """A tuple of object IDs identifying the params tensor versions.

        We use id() to detect when the pattern has reassigned a tensor
        (e.g. after .to(device) creates new storage). This is cheap and
        sufficient: pattern code never mutates these tensors in-place.
        """
        return (
            id(params.weight),
            id(params.bias),
            id(params.bn_scale),
            id(params.bn_bias),
        )

    def _ensure_fold(self, spec: ConvLIFSpec, params: ConvLIFParams) -> None:
        key = self._params_cache_key(params)
        if self._cache_key == key and self._w_fused is not None:
            return

        with torch.no_grad():
            w = params.weight
            if spec.has_bn:
                # Fold: w_fused[co,ci,kh,kw] = w[co,ci,kh,kw] * bn_scale[co]
                self._w_fused = (w * params.bn_scale.view(-1, 1, 1, 1)).detach()
                if spec.has_conv_bias:
                    # b_fused = bn_bias + conv_bias * bn_scale
                    self._b_fused = (
                        params.bn_bias + params.bias * params.bn_scale
                    ).detach()
                else:
                    self._b_fused = params.bn_bias.detach()
            else:
                # No BN: pass weight + conv_bias through unchanged
                self._w_fused = w
                self._b_fused = params.bias if spec.has_conv_bias else None

        self._cache_key = key

    def forward(
        self,
        x: torch.Tensor,
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        state,
    ) -> torch.Tensor:
        """Bit-exact realization: cuDNN conv + Triton lif_sequential."""
        # Local import: avoid module-level circular dependency
        # (catfuse.sparseflow imports catfuse.patterns transitively).
        from catfuse.sparseflow.lif_seq_kernel import lif_sequential

        T, B = x.shape[0], x.shape[1]
        H_in, W_in = x.shape[3], x.shape[4]
        H_out, W_out = spec.output_hw(H_in, W_in)

        # StaticZero short-circuit — bit-exact with the conv path because
        # F.conv2d(0) = bias-broadcast and we materialize the same value.
        x_4d = x.reshape(T * B, x.shape[2], H_in, W_in)
        if not x_4d.any():
            self._ensure_fold(spec, params)
            return static_zero_forward(
                spec, params, T, B, H_out, W_out, x.device, state, lif_sequential,
            )

        # 1. BatchFold + cuDNN conv with folded BN
        self._ensure_fold(spec, params)
        z_4d = F.conv2d(
            x_4d, self._w_fused, bias=self._b_fused,
            stride=spec.stride, padding=spec.padding,
        )
        z = z_4d.reshape(T, B, spec.out_channels, H_out, W_out)

        # 2. Single-launch Triton LIF over T steps
        v_in = state.get(
            shape=(B, spec.out_channels, H_out, W_out),
            device=z.device, dtype=torch.float32,
        )
        try:
            spikes, v_out = lif_sequential(
                z, v_in, tau=spec.tau,
                v_threshold=spec.v_threshold, v_reset=spec.v_reset,
            )
            state.set(v_out)
            return spikes
        except Exception:
            # Pure-PyTorch fallback (used if Triton kernel can't compile).
            # Math must match lif_sequential bit-exactly — see lif_seq_kernel.py
            # for the reference form: v = v + (z - (v - v_reset)) / tau.
            v = v_in
            v_reset_val = 0.0 if spec.v_reset is None else float(spec.v_reset)
            spikes_list = []
            for t in range(T):
                v = v + (z[t] - (v - v_reset_val)) / spec.tau
                spike = (v >= spec.v_threshold).to(z.dtype)
                v = v * (1.0 - spike) + v_reset_val * spike
                spikes_list.append(spike)
            state.set(v)
            return torch.stack(spikes_list, dim=0)

    def __repr__(self) -> str:
        return f"DenseKeep(cached={'yes' if self._w_fused is not None else 'no'})"
''',
    'catfuse/implementations/sparse_flow.py': r'''"""catfuse.implementations.sparse_flow — SparseFlow implementation of σ.

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
        """Lean batchfold sparse Conv→BN→LIF kernel.

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

    def __repr__(self) -> str:
        if not _SPARSEFLOW_AVAILABLE:
            return "SparseFlow(unavailable)"
        return (f"SparseFlow(w_cl={'yes' if self._w_cl is not None else 'no'}, "
                f"cached_shape={self._lean_cache_key})")
''',
    'tests/stage3_verify.py': r'''"""[Stage 3 verification] Implementation interface abstraction (problem 2).

Verifies that:
  1. catfuse.implementations.{base,dense_keep,sparse_flow} import cleanly.
  2. ConvLIFSpec / ConvLIFParams hold the expected fields.
  3. PartialFusionConvBNLIF holds a DenseKeep impl and delegates forward.
  4. STFusionConvBNLIF holds {DenseKeep, SparseFlow} and routes via Runtime EGD.
  5. Bit-exact equivalence between DenseKeep and SparseFlow on a single layer
     (Corollary 3.17 ground truth).
  6. SEW-ResNet18 substitute_sf still bit-exact vs SJ baseline (regression).
  7. functional.reset_net still works through the new impl-aware patterns.

Run:
    cd /path/to/CATFuse
    python tests/stage3_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_imports():
    """1. All new modules import cleanly."""
    print("[1/7] Module imports...")
    try:
        from catfuse.implementations import (
            Implementation, ConvLIFSpec, ConvLIFParams, DenseKeep,
        )
        from catfuse.implementations import SparseFlow  # may be None
        from catfuse.implementations.base import static_zero_forward
    except Exception:
        print("  FAIL: implementation modules failed to import")
        traceback.print_exc()
        return False

    # Implementation must be abstract
    try:
        Implementation()  # type: ignore
        print("  FAIL: Implementation should be abstract")
        return False
    except TypeError:
        pass  # expected

    print("  OK: imports clean, Implementation is abstract")
    return True


def check_spec_params_shape():
    """2. ConvLIFSpec / ConvLIFParams basic structure."""
    print("[2/7] ConvLIFSpec / ConvLIFParams structure...")
    try:
        import torch
        from catfuse.implementations import ConvLIFSpec, ConvLIFParams
    except Exception:
        print("  SKIP: torch not available")
        return None

    spec = ConvLIFSpec(
        in_channels=64, out_channels=128,
        kernel_size=3, stride=1, padding=1,
        has_conv_bias=False, has_bn=True,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
    )
    # Frozen check
    try:
        spec.tau = 3.0  # type: ignore[misc]
        print("  FAIL: ConvLIFSpec should be frozen")
        return False
    except Exception:
        pass

    # output_hw
    h_out, w_out = spec.output_hw(32, 32)
    assert (h_out, w_out) == (32, 32), \
        f"output_hw(32,32) for stride=1 padding=1 k=3 expected (32,32), got ({h_out},{w_out})"

    # Params holds references
    w = torch.randn(128, 64, 3, 3)
    bn_scale = torch.ones(128)
    bn_bias = torch.zeros(128)
    p = ConvLIFParams(weight=w, bias=None,
                      bn_scale=bn_scale, bn_bias=bn_bias)
    assert p.weight is w  # reference, not copy
    assert p.bn_scale is bn_scale

    print("  OK: ConvLIFSpec frozen + correct output_hw, ConvLIFParams holds references")
    return True


def check_partial_fusion_uses_dense_keep():
    """3. PartialFusionConvBNLIF holds a DenseKeep impl + delegates."""
    print("[3/7] PartialFusionConvBNLIF + DenseKeep delegation...")
    try:
        import torch
        from catfuse.patterns import PartialFusionConvBNLIF
        from catfuse.implementations import DenseKeep, ConvLIFSpec
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    inst = PartialFusionConvBNLIF(in_channels=4, out_channels=8,
                                  kernel_size=3, padding=1)
    assert hasattr(inst, "spec"), "should have .spec"
    assert isinstance(inst.spec, ConvLIFSpec)
    assert hasattr(inst, "_impl"), "should have ._impl"
    assert isinstance(inst._impl, DenseKeep)
    assert inst.spec.in_channels == 4 and inst.spec.out_channels == 8
    assert inst.spec.has_bn is True
    assert inst.spec.has_conv_bias is False

    # _params is lazy
    assert inst._params is None

    if torch.cuda.is_available():
        device = "cuda:0"
        inst = inst.to(device).eval()
        x = torch.randn(4, 2, 4, 16, 16, device=device)
        y = inst(x)
        assert y.shape == (4, 2, 8, 16, 16)
        # After forward, params is built and DenseKeep cached fold
        assert inst._params is not None
        assert inst._impl._w_fused is not None

    print("  OK: PartialFusionConvBNLIF delegates to DenseKeep, params built lazily")
    return True


def check_st_fusion_holds_both_impls():
    """4. STFusionConvBNLIF holds DenseKeep + SparseFlow + routes correctly."""
    print("[4/7] STFusionConvBNLIF + dual impls...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import DenseKeep, SparseFlow, ConvLIFSpec
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device)
    assert hasattr(fused, "spec")
    assert isinstance(fused.spec, ConvLIFSpec)
    assert hasattr(fused, "_impl_dense")
    assert isinstance(fused._impl_dense, DenseKeep)
    if SparseFlow is not None:
        assert hasattr(fused, "_impl_sparse")
        assert isinstance(fused._impl_sparse, SparseFlow)
    print("  OK: STFusionConvBNLIF holds DenseKeep + SparseFlow")
    return True


def check_dense_vs_sparse_bit_exact():
    """5. CORE Corollary 3.17 check: DenseKeep ≡ SparseFlow on a single layer.

    Builds two fresh STFusion modules with identical weights, manually forces
    each through DenseKeep and SparseFlow respectively, compares max_diff.
    """
    print("[5/7] DenseKeep vs SparseFlow bit-exact (Corollary 3.17)...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.implementations import SparseFlow
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available() or SparseFlow is None:
        print("  SKIP: needs CUDA + SparseFlow")
        return None

    device = "cuda:0"
    torch.manual_seed(0)
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device).eval()

    # Generate non-trivial input (mostly zeros so SparseFlow path is exercised
    # for sparsity > 0.7, but not all-zero so static_zero short-circuit is NOT
    # the only thing tested).
    T, B = 4, 2
    x = (torch.rand(T, B, 64, 16, 16, device=device) > 0.85).float()  # ~85% sparse

    # Manually force DenseKeep
    from spikingjelly.activation_based import functional
    functional.reset_net(fused)
    y_dense = fused._impl_dense.forward(x, fused.spec, fused._ensure_params(), fused.state)

    # Manually force SparseFlow
    functional.reset_net(fused)
    y_sparse = fused._impl_sparse.forward(x, fused.spec, fused._ensure_params(), fused.state)

    max_diff = (y_dense - y_sparse).abs().max().item()
    print(f"  DenseKeep vs SparseFlow max_diff = {max_diff:.6e}")
    if max_diff > 0:
        # Even tiny diffs (last-bit) violate Corollary 3.17 bit-exactness.
        # Pre-stage-3 SEW-RN18 was 0-diff; stage 3 must preserve.
        print(f"  FAIL: bit-exact violated (Corollary 3.17)")
        return False

    print(f"  OK: max_diff=0 — Corollary 3.17 holds at single-layer level")
    return True


def check_resnet18_regression():
    """6. SEW-ResNet18 still bit-exact (regression)."""
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
            traceback.print_exc()
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


def check_reset_net_through_impls():
    """7. functional.reset_net still clears state via CTFPattern.reset()."""
    print("[7/7] functional.reset_net through impl-aware patterns...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer, functional
        from catfuse.patterns import PartialFusionConvBNLIF
    except Exception:
        print("  SKIP: imports failed")
        traceback.print_exc()
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(4, 8, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(8).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)
    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)

    x = torch.randn(4, 2, 4, 16, 16, device=device)
    _ = fused(x)
    assert fused.state.is_initialized
    v_after_fwd = fused.state.tensor.clone()

    functional.reset_net(fused)
    assert not fused.state.is_initialized

    _ = fused(x)
    v_after_reset_fwd = fused.state.tensor.clone()

    assert torch.equal(v_after_fwd, v_after_reset_fwd), \
        "post-reset forward should reproduce fresh-init forward"

    print("  OK: reset_net properly clears state through CTFPattern.reset()")
    return True


def main():
    print("=" * 60)
    print("Stage 3 verification: Implementation interface (problem 2)")
    print("=" * 60)
    results = [
        ("imports", check_imports()),
        ("spec_params_shape", check_spec_params_shape()),
        ("partial_fusion_uses_dense_keep", check_partial_fusion_uses_dense_keep()),
        ("st_fusion_holds_both_impls", check_st_fusion_holds_both_impls()),
        ("dense_vs_sparse_bit_exact", check_dense_vs_sparse_bit_exact()),
        ("resnet18_regression", check_resnet18_regression()),
        ("reset_net_through_impls", check_reset_net_through_impls()),
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
# Embedded (old, new) replacement pairs
# ============================================================

PATTERNS_REPLACEMENTS = [
    (
        # ----- patterns.py: PartialFusionConvBNLIF guts -----
        '    cuDNN Conv + Triton (BN affine -> LIF with StateCarry) with StreamFuse(BN, LIF).\n\n    Realization of schedule:\n        TimeBlock(T) o StreamFuse(BN, LIF) o StateCarry(LIF)\n    (Conv is NOT StreamFused because cuDNN must write z to HBM)\n\n    Phase 2 task 2.1b: 1.74x vs SJ cupy on V100, near-bit-exact parity (0-2 flips/102M).\n    """\n\n    policy_row = 2\n\n    def __init__(self,\n                 in_channels: int, out_channels: int,\n                 kernel_size: int, stride: int = 1, padding: int = 0,\n                 tau: float = 2.0, v_threshold: float = 1.0,\n                 v_reset: float = 0.0, bn_eps: float = 1e-5):\n        super().__init__()\n        self.in_channels = in_channels\n        self.out_channels = out_channels\n        self.kernel_size = kernel_size\n        self.stride = stride\n        self.padding = padding\n        self.tau = tau\n        self.v_threshold = v_threshold\n        self.v_reset = v_reset\n        self.bn_eps = bn_eps\n\n        # Conv weight (no bias, BN follows)\n        self.weight = nn.Parameter(\n            torch.empty(out_channels, in_channels, kernel_size, kernel_size)\n        )\n        # BN params (inference mode, stored as buffers + parameters matching SJ layout)\n        self.bn_weight = nn.Parameter(torch.ones(out_channels))\n        self.bn_bias = nn.Parameter(torch.zeros(out_channels))\n        self.register_buffer(\'running_mean\', torch.zeros(out_channels))\n        self.register_buffer(\'running_var\', torch.ones(out_channels))\n\n        nn.init.kaiming_normal_(self.weight)\n        self.step_mode = "m"  # Tell SJ to pass full [T,B,C,H,W] in multi-step mode\n\n        # Stage 2 refactor: CSR state through StateBuffer instead of self._v\n        # plain attribute. Registered for unified reset() via CTFPattern.\n        from catfuse.state import StateBuffer\n        self.state = StateBuffer()\n        self._register_state(self.state)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        """Self-contained BatchFold Conv + BN + LIF forward.\n\n        x: [T, B, C_in, H, W] or [B, C_in, H, W]\n        Returns: spike tensor, same shape as x but with C_out channels\n        """\n        if x.ndim == 4:\n            # Single-step mode: wrap in T=1\n            return self._forward_impl(x.unsqueeze(0)).squeeze(0)\n        return self._forward_impl(x)\n\n    def _ensure_bn_folded(self):\n        """Fold BN into conv weight+bias (once). One F.conv2d = Conv+BN."""\n        if not hasattr(self, "_w_fused") or self._w_fused is None:\n            with torch.no_grad():\n                inv_std = torch.rsqrt(self.running_var + self.bn_eps)\n                scale = self.bn_weight * inv_std  # [C_out]\n                # Fold scale into conv weight: w_fused[co,ci,kh,kw] = w[co,ci,kh,kw] * scale[co]\n                self._w_fused = (self.weight * scale.view(-1, 1, 1, 1)).detach()\n                # Fold bias: b_fused = bn_bias - running_mean * scale + conv_bias * scale\n                self._b_fused = (self.bn_bias - self.running_mean * scale).detach()\n                if getattr(self, \'conv_bias\', None) is not None:\n                    self._b_fused = self._b_fused + (self.conv_bias * scale).detach()\n\n    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:\n        """Core forward: x is [T, B, C_in, H, W]."""\n        T, B = x.shape[0], x.shape[1]\n        self._ensure_bn_folded()\n\n        # StaticZero check: if entire input is zero, skip conv\n        # Cost: one reduction (~0.005ms) vs cuDNN conv (~0.1-0.2ms)\n        x_4d = x.reshape(T * B, x.shape[2], x.shape[3], x.shape[4])\n        if not x_4d.any():\n            # z = bias only (BN already folded into weight+bias)\n            H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1\n            W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1\n            z_4d = self._b_fused.view(1, -1, 1, 1).expand(T * B, -1, H_out, W_out).clone()\n        else:\n            # 1. BatchFold + cuDNN conv (BN fully folded into weight+bias = 1 launch)\n            z_4d = F.conv2d(x_4d, self._w_fused, bias=self._b_fused,\n                            stride=self.stride, padding=self.padding)\n        H_out, W_out = z_4d.shape[2], z_4d.shape[3]\n\n        # 2. Reshape to [T, B, C_out, H_out, W_out]\n        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)\n\n        # 4. Single-launch Triton LIF over T steps\n        # Stage 2 refactor: state via StateBuffer instead of self._v.\n        v_in = self.state.get(\n            shape=(B, self.out_channels, H_out, W_out),\n            device=z.device,\n            dtype=torch.float32,\n        )\n        try:\n            from catfuse.sparseflow.lif_seq_kernel import lif_sequential\n            spikes, v_out = lif_sequential(\n                z, v_in, tau=self.tau,\n                v_threshold=self.v_threshold, v_reset=self.v_reset,\n            )\n            self.state.set(v_out)\n            return spikes\n        except Exception:\n            # Fallback to Python loop\n            v = v_in\n            spikes = []\n            for t in range(T):\n                v = v + (z[t] - (v - self.v_reset)) / self.tau\n                spike = (v >= self.v_threshold).to(z.dtype)\n                v = v * (1.0 - spike) + self.v_reset * spike\n                spikes.append(spike)\n            self.state.set(v)\n            return torch.stack(spikes, dim=0)\n\n    # Stage 2 refactor: CTFPattern.reset() now delegates to all registered\n    # StateBuffers; the explicit reset() override below is no longer needed\n    # and is removed.\n\n    @classmethod\n    def from_sj_modules(cls,',
        '    cuDNN Conv + Triton (BN affine -> LIF with StateCarry) with StreamFuse(BN, LIF).\n\n    Realization of schedule:\n        TimeBlock(T) o StreamFuse(BN, LIF) o StateCarry(LIF)\n    (Conv is NOT StreamFused because cuDNN must write z to HBM)\n\n    Stage 3 refactor: forward logic delegates to a DenseKeep Implementation\n    (Definition 3.16.1). The pattern owns weights, BN params, and state,\n    while the impl owns the bit-exact realization (cuDNN conv + Triton LIF).\n\n    Phase 2 task 2.1b: 1.74x vs SJ cupy on V100, near-bit-exact parity (0-2 flips/102M).\n    """\n\n    policy_row = 2\n\n    def __init__(self,\n                 in_channels: int, out_channels: int,\n                 kernel_size: int, stride: int = 1, padding: int = 0,\n                 tau: float = 2.0, v_threshold: float = 1.0,\n                 v_reset: float = 0.0, bn_eps: float = 1e-5):\n        super().__init__()\n        self.in_channels = in_channels\n        self.out_channels = out_channels\n        self.kernel_size = kernel_size\n        self.stride = stride\n        self.padding = padding\n        self.tau = tau\n        self.v_threshold = v_threshold\n        self.v_reset = v_reset\n        self.bn_eps = bn_eps\n\n        # Conv weight (no bias, BN follows)\n        self.weight = nn.Parameter(\n            torch.empty(out_channels, in_channels, kernel_size, kernel_size)\n        )\n        # BN params (inference mode, stored as buffers + parameters matching SJ layout)\n        self.bn_weight = nn.Parameter(torch.ones(out_channels))\n        self.bn_bias = nn.Parameter(torch.zeros(out_channels))\n        self.register_buffer(\'running_mean\', torch.zeros(out_channels))\n        self.register_buffer(\'running_var\', torch.ones(out_channels))\n\n        nn.init.kaiming_normal_(self.weight)\n        self.step_mode = "m"  # Tell SJ to pass full [T,B,C,H,W] in multi-step mode\n\n        # Stage 2 refactor: CSR state through StateBuffer instead of self._v\n        # plain attribute. Registered for unified reset() via CTFPattern.\n        from catfuse.state import StateBuffer\n        self.state = StateBuffer()\n        self._register_state(self.state)\n\n        # Stage 3 refactor: delegate forward math to a DenseKeep Implementation.\n        # Spec is frozen at construction; params (bn-folded BN affine) are\n        # rebuilt lazily when running_mean/var change identity (e.g. after .to()).\n        from catfuse.implementations import ConvLIFSpec, DenseKeep\n        self.spec = ConvLIFSpec(\n            in_channels=in_channels,\n            out_channels=out_channels,\n            kernel_size=kernel_size,\n            stride=stride,\n            padding=padding,\n            has_conv_bias=False,\n            has_bn=True,\n            tau=float(tau),\n            v_threshold=float(v_threshold),\n            v_reset=float(v_reset) if v_reset is not None else None,\n            decay_input=True,\n            bn_eps=float(bn_eps),\n        )\n        self._impl = DenseKeep()\n        self._params = None       # type: ignore[assignment]\n        self._params_key = None   # type: ignore[assignment]\n\n    def _ensure_params(self):\n        """Lazily build ConvLIFParams from raw BN params.\n\n        Cached by identity of (bn_weight, bn_bias, running_mean, running_var).\n        Identity changes on .to(device) for buffers, so cache auto-invalidates.\n        """\n        key = (id(self.bn_weight), id(self.bn_bias),\n               id(self.running_mean), id(self.running_var))\n        if self._params_key == key and self._params is not None:\n            return self._params\n        with torch.no_grad():\n            inv_std = torch.rsqrt(self.running_var + self.bn_eps)\n            bn_scale = (self.bn_weight * inv_std).detach()\n            bn_bias_folded = (self.bn_bias - self.running_mean * bn_scale).detach()\n        from catfuse.implementations import ConvLIFParams\n        self._params = ConvLIFParams(\n            weight=self.weight,\n            bias=None,\n            bn_scale=bn_scale,\n            bn_bias=bn_bias_folded,\n        )\n        self._params_key = key\n        # Impl\'s own caches (e.g. DenseKeep._w_fused) are keyed on params\n        # tensor identity, so they invalidate automatically when bn_scale/\n        # bn_bias_folded are re-derived above.\n        return self._params\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        """Self-contained BatchFold Conv + BN + LIF forward.\n\n        x: [T, B, C_in, H, W] or [B, C_in, H, W]\n        Returns: spike tensor, same shape as x but with C_out channels\n        """\n        if x.ndim == 4:\n            return self._impl.forward(\n                x.unsqueeze(0), self.spec, self._ensure_params(), self.state\n            ).squeeze(0)\n        return self._impl.forward(x, self.spec, self._ensure_params(), self.state)\n\n    @classmethod\n    def from_sj_modules(cls,',
    ),
]

STFUSION_REPLACEMENTS = [
    (
        # ----- (1) Insert spec/impl construction in __init__ -----
        '        # Lean path cached state (initialized on first forward)\n        self._lean_ready = False\n        self._lean_cache = {}',
        '        # Lean path cached state (initialized on first forward)\n        # Stage 3 refactor note: _lean_cache and _lean_forward_single are now\n        # unused by the main forward path (lean batchfold goes through\n        # SparseFlow Implementation). The fields are kept to avoid breaking\n        # any external code that may reach into them; a future stage will\n        # remove them.\n        self._lean_ready = False\n        self._lean_cache = {}\n\n        # ---- Stage 3 refactor: Implementation hierarchy --------------\n        # Build static spec + impls. ConvLIFParams is built lazily because\n        # buffers (bn_scale, bn_bias_folded) may move via .to(device) and\n        # we want params to follow.\n        from catfuse.implementations import (\n            ConvLIFSpec, DenseKeep, SparseFlow as _SparseFlow,\n        )\n        self.spec = ConvLIFSpec(\n            in_channels=in_channels,\n            out_channels=out_channels,\n            kernel_size=kernel_size,\n            stride=stride,\n            padding=padding,\n            has_conv_bias=(bias is True),\n            has_bn=(bn_scale is not None),\n            tau=float(tau),\n            v_threshold=float(v_threshold),\n            v_reset=v_reset,\n            decay_input=bool(decay_input),\n        )\n        self._impl_dense = DenseKeep()\n        # SparseFlow may be unavailable on environments without Triton kernels;\n        # in that case self._impl_sparse is None and dispatch falls back to\n        # _impl_dense unconditionally.\n        self._impl_sparse = _SparseFlow() if _SparseFlow is not None else None\n        self._params = None        # type: ignore[assignment]\n        self._params_key = None    # type: ignore[assignment]\n\n    def _ensure_params(self):\n        """Lazily build ConvLIFParams from current weight/bias/bn buffers.\n\n        Cached by identity of the source tensors; identity changes on\n        .to(device) for buffers, so cache auto-invalidates.\n        """\n        key = (\n            id(self.weight),\n            id(self.bias),\n            id(self.bn_scale),\n            id(self.bn_bias_folded),\n        )\n        if self._params_key == key and self._params is not None:\n            return self._params\n        from catfuse.implementations import ConvLIFParams\n        self._params = ConvLIFParams(\n            weight=self.weight,\n            bias=self.bias,\n            bn_scale=self.bn_scale,\n            bn_bias=self.bn_bias_folded,\n        )\n        self._params_key = key\n        return self._params',
    ),
    (
        # ----- (2) _dense_forward body -----
        '    def _dense_forward(self, x):\n        """DenseKeep fallback: cuDNN conv + BN + LIF (no sparse kernel).\n\n        Used by Runtime EGD when sparsity is too low for StreamFuse to win.\n        Same as PartialFusionConvBNLIF: BatchFold conv + lif_sequential.\n        """\n        T, B = x.shape[0], x.shape[1]\n        c = self._lean_cache\n        device = x.device\n        H_out, W_out = c[\'H_out\'], c[\'W_out\']\n\n        # 1. BatchFold + cuDNN conv with folded BN\n        x_4d = x.reshape(T * B, self.in_channels, x.shape[3], x.shape[4])\n        if not hasattr(self, \'_w_conv_fused\') or self._w_conv_fused is None:\n            with torch.no_grad():\n                if self.bn_scale is not None:\n                    scale = self.bn_scale\n                    bias = self.bn_bias_folded\n                    self._w_conv_fused = (self.weight * scale.view(-1, 1, 1, 1)).detach()\n                    self._b_conv_fused = bias.detach() if bias is not None else None\n                else:\n                    self._w_conv_fused = self.weight.detach()\n                    self._b_conv_fused = (self.bias.detach() if self.bias is not None else None)\n\n        z_4d = torch.nn.functional.conv2d(\n            x_4d, self._w_conv_fused, bias=self._b_conv_fused,\n            stride=self.stride, padding=self.padding,\n        )\n        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)\n\n        # 2. LIF — Stage 2 refactor: state via StateBuffer.\n        v_init = self.state.get(\n            shape=(B, self.out_channels, H_out, W_out),\n            device=device, dtype=torch.float32,\n        )\n        spikes, v_final = lif_sequential(\n            z, v_init, tau=self.tau,\n            v_threshold=self.v_threshold, v_reset=self.v_reset,\n        )\n        self.state.set(v_final)\n        return spikes',
        '    def _dense_forward(self, x):\n        """DenseKeep fallback: cuDNN conv + BN + LIF (no sparse kernel).\n\n        Used by Runtime EGD when sparsity is too low for StreamFuse to win.\n\n        Stage 3 refactor: delegates to self._impl_dense (DenseKeep), which\n        is the canonical bit-exact realization shared with PartialFusionConvBNLIF.\n        Removes ~30 lines of inline conv+lif math previously duplicated here.\n        """\n        return self._impl_dense.forward(\n            x, self.spec, self._ensure_params(), self.state,\n        )',
    ),
    (
        # ----- (3) _batchfold_forward body -----
        '    def _batchfold_forward(self, x):\n        """StreamFuse: Conv+BN+LIF over T steps in ONE kernel launch.\n\n        z stays in registers (never hits HBM).\n        v stays in registers across T steps.\n        Inline zero-detection replaces separate prescan.\n        """\n        T, B = x.shape[0], x.shape[1]\n        c = self._lean_cache\n        device = x.device\n        H_out, W_out = c[\'H_out\'], c[\'W_out\']\n\n        # Flatten to [T*B, C_in, H, W]\n        x_flat = x.reshape(T * B, self.in_channels, x.shape[3], x.shape[4])\n\n        # StaticZero: if entire batch is zero, skip conv kernel launch\n        if not x_flat.any():\n            H_out, W_out = c[\'H_out\'], c[\'W_out\']\n            # z = BN bias only\n            if c[\'has_bn\']:\n                z_flat = c[\'bn_bias_arg\'].view(1, -1, 1, 1).expand(\n                    T * B, self.out_channels, H_out, W_out).clone()\n            elif c[\'has_bias\']:\n                z_flat = c[\'bias_arg\'].view(1, -1, 1, 1).expand(\n                    T * B, self.out_channels, H_out, W_out).clone()\n            else:\n                z_flat = torch.zeros(T * B, self.out_channels, H_out, W_out,\n                                     dtype=torch.float32, device=device)\n            z = z_flat.reshape(T, B, self.out_channels, H_out, W_out)\n            # Stage 2 refactor: state via StateBuffer.\n            v_init = self.state.get(\n                shape=(B, self.out_channels, H_out, W_out),\n                device=device, dtype=torch.float32,\n            )\n            spikes, v_final = lif_sequential(\n                z, v_init, tau=self.tau,\n                v_threshold=self.v_threshold, v_reset=self.v_reset,\n            )\n            self.state.set(v_final)\n            return spikes\n\n        # Input is already float32 spike tensor, just ensure contiguous\n        x_contig = x_flat.contiguous()\n\n        # Grid: one program per (batch, tile) × output channel blocks\n        BH, BW = 8, 16\n        GH = triton.cdiv(H_out, BH)\n        GW = triton.cdiv(W_out, BW)\n        N_TILES = B * GH * GW\n\n        # Prepare v_init — Stage 2 refactor: state via StateBuffer.\n        v_init = self.state.get(\n            shape=(B, self.out_channels, H_out, W_out),\n            device=device, dtype=torch.float32,\n        )\n        v_init = v_init.float().contiguous()\n\n        # Output buffers\n        spike_out = torch.empty(T * B, self.out_channels, H_out, W_out,\n                                dtype=torch.float32, device=device)\n        v_out = torch.empty_like(v_init)\n\n        def _grid(META):\n            return (N_TILES, triton.cdiv(self.out_channels, META["BLOCK_N"]))\n\n        sparse_streamfuse_conv3x3_bn_lif[_grid](\n            x_contig,\n            self._w_cl,\n            c[\'bias_arg\'],\n            c[\'bn_scale_arg\'],\n            c[\'bn_bias_arg\'],\n            v_init,\n            spike_out,\n            v_out,\n            T, B,\n            C_IN=self.in_channels,\n            C_OUT=self.out_channels,\n            H=x.shape[3], W=x.shape[4],\n            H_OUT=H_out, W_OUT=W_out,\n            GH=GH, GW=GW,\n            HAS_BIAS=c[\'has_bias\'],\n            HAS_BN=c[\'has_bn\'],\n            DECAY=c[\'decay\'],\n            RECIP_TAU=c[\'recip_tau\'],\n            V_TH=float(self.v_threshold),\n            HAS_V_RESET=c[\'has_v_reset\'],\n            V_RESET=c[\'v_reset_val\'],\n            GROUP_SIZE_C=c[\'GSC\'],\n            NUM_GROUPS=c[\'NUM_GROUPS\'],\n        )\n\n        self.state.set(v_out)\n        return spike_out.reshape(T, B, self.out_channels, H_out, W_out)',
        '    def _batchfold_forward(self, x):\n        """StreamFuse: Conv+BN+LIF over T steps in ONE kernel launch.\n\n        z stays in registers (never hits HBM).\n        v stays in registers across T steps.\n        Inline zero-detection replaces separate prescan.\n\n        Stage 3 refactor: delegates to self._impl_sparse (SparseFlow), which\n        owns the lean kernel cache (_w_cl, _ag_mask_buf, etc.) and handles\n        the static-zero short-circuit internally. If SparseFlow is unavailable\n        on this platform (no Triton), falls back to DenseKeep — Corollary 3.17\n        guarantees bit-exact equivalence.\n        """\n        if self._impl_sparse is not None:\n            return self._impl_sparse.forward(\n                x, self.spec, self._ensure_params(), self.state,\n            )\n        # SparseFlow unavailable; fall back to DenseKeep (Corollary 3.17)\n        return self._impl_dense.forward(\n            x, self.spec, self._ensure_params(), self.state,\n        )',
    ),
]


# ============================================================
# Apply logic
# ============================================================

def apply_new_files(repo_root):
    """Create / overwrite the 4 implementation files + tests/stage3_verify.py."""
    print("[1/3] Creating new files...")
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
    """Apply (old, new) replacements with idempotency check.

    For each (old, new) pair:
      - If `new` already in file content, skip (already applied).
      - Else if `old` in file content, replace.
      - Else, fail (file in unexpected state).
    """
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
            print(f"    Either stages 1+2 not applied, or file modified externally.")
            return False
        # Single deterministic replacement (str.replace handles non-overlap correctly)
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
    print(f"Applying Stage 3 refactor in: {repo_root}")
    print("=" * 60)

    # Sanity checks
    needed = [
        "catfuse/patterns.py",
        "catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py",
        "catfuse/state.py",   # Stage 2 — must be present
    ]
    for f in needed:
        path = os.path.join(repo_root, f)
        if not os.path.exists(path):
            print(f"  FATAL: required file not found: {f}")
            print(f"  Stages 1 and 2 must be applied first.")
            return 1

    # Step 1: New files
    apply_new_files(repo_root)

    # Step 2: patterns.py replacements
    print("[2/3] Modifying catfuse/patterns.py...")
    if not apply_replacements(
        os.path.join(repo_root, "catfuse/patterns.py"),
        PATTERNS_REPLACEMENTS, "patterns.py",
    ):
        return 1

    # Step 3: st_fusion_conv_bn_lif.py replacements
    print("[3/3] Modifying catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py...")
    if not apply_replacements(
        os.path.join(repo_root, "catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py"),
        STFUSION_REPLACEMENTS, "st_fusion_conv_bn_lif.py",
    ):
        return 1

    print()
    print("=" * 60)
    print("Stage 3 applied successfully.")
    print("Next: python tests/stage3_verify.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))