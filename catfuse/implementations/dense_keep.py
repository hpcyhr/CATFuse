"""catfuse.implementations.dense_keep — DenseKeep implementation of σ.

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

    def analytic_io_cost(
        self,
        spec,
        T: int, B: int, H_in: int, W_in: int,
        K: int = None,
        *,
        dtype_bytes: int = 4,
    ):
        """§3.9 baseline I/O cost: TimeBlock(T) ∘ BatchFold(Conv).

        Schedule:
          1. Conv runs once on BatchFolded input [T*B, C_in, H, W] → z [T*B, C_out, H_out, W_out]
             — z is materialized to HBM (cuDNN cannot avoid this).
          2. lif_sequential reads z from HBM, runs T-step LIF in registers,
             writes T spike maps and final v back to HBM.

        K is ignored: DenseKeep is a fixed-schedule realization. The "K"
        parameter is meaningful only for impls that decompose T into
        TimeBlocks; here T is folded into the conv's batch dim and LIF
        runs all T steps in one kernel.

        Cost (per forward, in bytes):
          x_load       = T·B·HWC_in       ·dtype_bytes
          w_load       = C_out·C_in·k²    ·dtype_bytes        (one launch)
          z_io         = 2·T·B·HWC_out    ·dtype_bytes        (write + read)
          v_io         = 2·B·HWC_out      ·dtype_bytes        (init + final)
          spike_write  = T·B·HWC_out      ·dtype_bytes

        intermediate_io = z_io + v_io = (2T+2)·B·HWC_out·dtype_bytes ∈ O(T·HWC).
        """
        from catfuse.implementations.base import IOCost

        H_out, W_out = spec.output_hw(H_in, W_in)
        HWC_in = H_in * W_in * spec.in_channels
        HWC_out = H_out * W_out * spec.out_channels
        kk = spec.kernel_size * spec.kernel_size

        x_load = T * B * HWC_in * dtype_bytes
        w_load = spec.out_channels * spec.in_channels * kk * dtype_bytes
        # z: written by cuDNN conv, then read by lif_sequential
        z_io = 2 * T * B * HWC_out * dtype_bytes
        # v: initial read + final write only (lif_sequential is a single multi-step kernel)
        v_io = 2 * B * HWC_out * dtype_bytes
        spike_write = T * B * HWC_out * dtype_bytes

        return IOCost(
            x_load=x_load,
            w_load=w_load,
            z_io=z_io,
            v_io=v_io,
            spike_write=spike_write,
            schedule="TimeBlock(T)∘BatchFold(Conv) — z materialized to HBM",
            num_blocks=1,
        )

    def schedule_decomposition(self, spec, T: int, K: int = None):
        """§3.13 decomposition of DenseKeep's schedule.

        DenseKeep realizes σ as:

            TimeBlock(K=T) ∘ BatchFold(Conv) ∘ StateCarry(LIF)

        Reading inner-to-outer:
          1. StateCarry(LIF) — LIF state lives in registers across all T
             steps inside the lif_sequential kernel; the "block boundaries"
             are vacuous since K=T = single block.
          2. BatchFold(Conv) — T fold into batch dim so cuDNN runs once on
             [T·B, C_in, H, W] → [T·B, C_out, H_out, W_out].
          3. TimeBlock(K=T) — single block of size T (trivial chunking).

        K is ignored: DenseKeep is fixed at K=T by virtue of BatchFold.

        This is §3.8 form_1 — the BatchFold-form. It is the natural baseline
        for impls relying on a vendor BLAS (cuDNN here) that requires its
        outputs in HBM.
        """
        from catfuse.implementations.base import (
            ScheduleTransform, ScheduleDecomposition,
        )
        return ScheduleDecomposition(
            transforms=(
                ScheduleTransform.StateCarry("LIF"),
                ScheduleTransform.BatchFold("Conv"),
                ScheduleTransform.TimeBlock(T),
            ),
            form="form_1",
            description=(
                f"DenseKeep schedule: TimeBlock(K=T={T}) ∘ BatchFold(Conv) "
                f"∘ StateCarry(LIF). z is materialized between Conv and LIF "
                f"(no StreamFuse). LIF runs as a single multi-step kernel."
            ),
        )

    def __repr__(self) -> str:
        return f"DenseKeep(cached={'yes' if self._w_fused is not None else 'no'})"
