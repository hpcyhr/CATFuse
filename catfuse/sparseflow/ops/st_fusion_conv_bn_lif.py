"""
STFusionConvBNLIF — Spatio-Temporal Fused Conv + BN + LIF module.

This is the core CATFuse-SF integration piece. It combines:
  - SparseFlow spatial backend: prescan + sparse conv (or dense fallback)
  - CATFuse temporal structure: BN affine fused into kernel epilogue
  - CATFuse StateCarry: membrane potential v chains across TimeBlock(K) steps

Forward path for [T, B, C_in, H, W] input:
  for each TimeBlock of K steps:
    v = read from state (start of block)
    for t in block:
      prescan(spike_t) → bitmask
      sparse_conv(spike_t, bitmask) + BN affine → z_t (on-chip)
      LIF(z_t, v) → spike_t, v_new
      v = v_new  (carry in Python)
    state = v  (end of block, persists to next forward call)

Three spatial paths (selected per-invocation by SparseFlow dispatch):
  - StaticZero: input all-zero → z = bias → BN → LIF (v still decays!)
  - Sparse: prescan + grouped sparse conv → BN → LIF
  - DenseKeep: cuDNN dense conv fallback → BN → LIF

Usage:
  module = STFusionConvBNLIF.from_sj_modules(conv2d, bn2d, lif_node, K=4)
  spike_out = module(spike_input)  # [T, B, C_out, H, W]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.activation_based import base as sj_base
    _SJ_MEMORY_MODULE = sj_base.MemoryModule
except ImportError:
    _SJ_MEMORY_MODULE = nn.Module

try:
    from catfuse.sparseflow.fused_conv_bn_lif_kernel import (
        sparse_fused_conv_bn_lif_forward,
        fused_bm_conv3x3_bn_lif_8x8,
        fused_bm_conv3x3_bn_lif_8x16,
    )
    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False

try:
    from catfuse.sparseflow.fast_prescan_v2 import fast_spike_prescan_2d_v2
    from catfuse.sparseflow.sparse_conv2d_kernel import _select_tile_sizes, choose_group_size
    import triton
    _LEAN_AVAILABLE = True
    from catfuse.sparseflow.lif_seq_kernel import lif_sequential
    from catfuse.sparseflow.streamfuse_kernel import sparse_streamfuse_conv3x3_bn_lif
except ImportError:
    _LEAN_AVAILABLE = False


def _fold_bn_params(bn: nn.BatchNorm2d):
    """Precompute BN affine parameters for inference.

    BN in eval mode: y = (x - mean) / sqrt(var + eps) * weight + bias
    Folded: y = x * scale + offset
      where scale = weight / sqrt(var + eps)
            offset = bias - mean * scale
    """
    assert not bn.training, "BN must be in eval mode for folding"
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight  # may be None if affine=False
    beta = bn.bias

    inv_std = 1.0 / torch.sqrt(var + eps)
    scale = inv_std if gamma is None else gamma * inv_std
    offset = -mean * scale if beta is None else beta - mean * scale
    return scale.detach(), offset.detach()


class STFusionConvBNLIF(_SJ_MEMORY_MODULE):
    """Spatio-temporal fused Conv→BN→LIF with SparseFlow spatial backend.

    Attributes:
        K: TimeBlock size (number of steps per block)
        weight: Conv2d weight [C_out, C_in, kH, kW]
        bias: Conv2d bias [C_out] or None
        bn_scale: folded BN scale [C_out]
        bn_bias: folded BN bias [C_out]
        tau, v_threshold, v_reset: LIF parameters
    """

    policy_row = "STFusionConvBNLIF"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        # BN params (precomputed)
        bn_scale: Optional[torch.Tensor] = None,
        bn_bias: Optional[torch.Tensor] = None,
        # LIF params
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        decay_input: bool = True,
        # TimeBlock
        K: int = 4,
        # SparseFlow
        threshold: float = 1e-6,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.K = K

        # LIF params
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.decay_input = decay_input
        self.threshold = threshold

        # Conv weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # BN folded params (not nn.Parameter — frozen at construction time)
        if bn_scale is not None:
            self.register_buffer('bn_scale', bn_scale.clone())
            self.register_buffer('bn_bias_folded', bn_bias.clone())
        else:
            self.register_buffer('bn_scale', None)
            self.register_buffer('bn_bias_folded', None)

        # Precomputed weight layout for SparseFlow kernel (channel-last for 3x3)
        self._w_cl: Optional[torch.Tensor] = None

        # Membrane potential state
        self.register_memory('v', 0.0)

        # Prescan buffer (reusable across steps)
        self._ag_mask_buf: Optional[torch.Tensor] = None
        self._tile_class_buf: Optional[torch.Tensor] = None

        # Lean path cached state (initialized on first forward)
        self._lean_ready = False
        self._lean_cache = {}

    def _ensure_w_cl(self):
        """Precompute channel-last weight layout for SparseFlow kernel."""
        if self._w_cl is None and self.kernel_size == 3:
            self._w_cl = self.weight.data.half().permute(0, 2, 3, 1).contiguous()

    def _ensure_ag_mask_buf(self, N_TILES: int, device):
        """Ensure prescan buffer is allocated and large enough."""
        if self._ag_mask_buf is None or self._ag_mask_buf.numel() < N_TILES:
            self._ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
        if self._tile_class_buf is None or self._tile_class_buf.numel() < N_TILES:
            self._tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

    def _init_lean_cache(self, B: int, H_in: int, W_in: int, device):
        """Initialize all cached state for the lean kernel path."""
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        BH, BW = _select_tile_sizes(H_out, W_out)
        GH = triton.cdiv(H_out, BH)
        GW = triton.cdiv(W_out, BW)
        N_TILES = B * GH * GW
        GSC = choose_group_size(self.in_channels)
        NUM_GROUPS = triton.cdiv(self.in_channels, GSC)

        # LIF constants
        decay = 1.0 - 1.0 / self.tau
        recip_tau = 1.0 / self.tau
        has_v_reset = self.v_reset is not None
        v_reset_val = 0.0 if self.v_reset is None else float(self.v_reset)

        # BN args
        has_bn = self.bn_scale is not None
        bn_scale_arg = (
            self.bn_scale.float().contiguous() if has_bn
            else torch.empty(1, dtype=torch.float32, device=device)
        )
        bn_bias_arg = (
            self.bn_bias_folded.float().contiguous() if has_bn
            else torch.empty(1, dtype=torch.float32, device=device)
        )

        # Bias arg
        has_bias = self.bias is not None
        bias_arg = (
            self.bias.detach().float().contiguous() if has_bias
            else torch.empty(1, dtype=torch.float32, device=device)
        )

        # Kernel selection
        kernel = fused_bm_conv3x3_bn_lif_8x16 if BW == 16 else fused_bm_conv3x3_bn_lif_8x8

        def _grid(META):
            return (N_TILES, triton.cdiv(self.out_channels, META["BLOCK_N"]))

        # Pre-allocate buffers
        self._ensure_w_cl()
        self._ensure_ag_mask_buf(N_TILES, device)

        # Pre-allocate per-step reusable buffers
        spike_buf = torch.empty(B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
        v_next_buf = torch.empty_like(spike_buf)

        self._lean_cache = {
            'spike_buf': spike_buf, 'v_next_buf': v_next_buf,
            'H_out': H_out, 'W_out': W_out,
            'BH': BH, 'BW': BW, 'GH': GH, 'GW': GW,
            'N_TILES': N_TILES, 'GSC': GSC, 'NUM_GROUPS': NUM_GROUPS,
            'decay': decay, 'recip_tau': recip_tau,
            'has_v_reset': has_v_reset, 'v_reset_val': v_reset_val,
            'has_bias': has_bias, 'has_bn': has_bn,
            'bias_arg': bias_arg,
            'bn_scale_arg': bn_scale_arg, 'bn_bias_arg': bn_bias_arg,
            'kernel': kernel, 'grid': _grid,
        }
        self._lean_ready = True

    def _lean_forward_single(self, x_t, v_prev):
        """Lean forward: fast_prescan_v2 + direct kernel launch.

        All buffers pre-allocated in _init_lean_cache to avoid per-step overhead.
        """
        c = self._lean_cache
        B = x_t.shape[0]

        # 1. Fast prescan on NCHW input (no permute, no dtype conversion)
        fast_spike_prescan_2d_v2(
            x_t, c['H_out'], c['W_out'],
            kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            block_h=c['BH'], block_w=c['BW'],
            group_size_c=c['GSC'],
            ag_mask_out=self._ag_mask_buf,
            tile_class_out=self._tile_class_buf,
        )

        # 2. Convert to fp16 — reuse buffer if possible
        if x_t.dtype == torch.float16:
            x_f16 = x_t
        else:
            if not hasattr(self, "_x_f16_buf") or self._x_f16_buf is None or self._x_f16_buf.shape != x_t.shape:
                self._x_f16_buf = torch.empty_like(x_t, dtype=torch.float16)
            self._x_f16_buf.copy_(x_t)  # in-place copy, no alloc
            x_f16 = self._x_f16_buf

        # 3. Use pre-allocated output buffers
        spike_out = c['spike_buf']
        v_next = c['v_next_buf']
        # v_prev should already be fp32 contiguous from previous step
        v_prev_f32 = v_prev if (v_prev.dtype == torch.float32 and v_prev.is_contiguous()) else v_prev.float().contiguous()

        # 4. Direct kernel launch
        c['kernel'][c['grid']](
            x_f16,
            self._w_cl,
            c['bias_arg'],
            c['bn_scale_arg'],
            c['bn_bias_arg'],
            self._ag_mask_buf,
            v_prev_f32,
            spike_out,
            v_next,
            B,
            self.in_channels, self.out_channels,
            c['H_out'], c['W_out'], c['GH'], c['GW'],
            HAS_BIAS=c['has_bias'],
            HAS_BN=c['has_bn'],
            DECAY=c['decay'],
            RECIP_TAU=c['recip_tau'],
            V_TH=float(self.v_threshold),
            HAS_V_RESET=c['has_v_reset'],
            V_RESET=c['v_reset_val'],
            GROUP_SIZE_C=c['GSC'],
            NUM_GROUPS=c['NUM_GROUPS'],
        )

        return spike_out, v_next

    @classmethod
    def from_sj_modules(
        cls,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        lif,  # spikingjelly LIFNode
        K: int = 4,
        threshold: float = 1e-6,
    ) -> 'STFusionConvBNLIF':
        """Build from SpikingJelly Conv2d + BatchNorm2d + LIFNode.

        Args:
            conv: nn.Conv2d (or sj layer.Conv2d)
            bn: nn.BatchNorm2d (must be in eval mode)
            lif: SpikingJelly LIFNode with tau, v_threshold, v_reset attributes
            K: TimeBlock size
            threshold: SparseFlow prescan activity threshold
        """
        bn.eval()
        bn_scale, bn_bias_val = _fold_bn_params(bn)

        # Extract LIF params
        tau = float(getattr(lif, 'tau', 2.0))
        v_threshold = float(getattr(lif, 'v_threshold', 1.0))
        v_reset_raw = getattr(lif, 'v_reset', 0.0)
        v_reset = None if v_reset_raw is None else float(v_reset_raw)
        decay_input = bool(getattr(lif, 'decay_input', True))

        ksize = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        stride_val = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        padding_val = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding

        module = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=ksize,
            stride=stride_val,
            padding=padding_val,
            bias=conv.bias is not None,
            bn_scale=bn_scale,
            bn_bias=bn_bias_val,
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=decay_input,
            K=K,
            threshold=threshold,
        )

        # Copy conv weights
        module.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            module.bias.data.copy_(conv.bias.data)

        return module

    def _fallback_forward_single(self, x_t, v_prev):
        """Pure PyTorch fallback for one timestep (no Triton)."""
        z = F.conv2d(x_t, self.weight, self.bias,
                     stride=self.stride, padding=self.padding)
        if self.bn_scale is not None:
            z = z * self.bn_scale.view(1, -1, 1, 1) + self.bn_bias_folded.view(1, -1, 1, 1)

        vp = v_prev.to(dtype=z.dtype)
        v_reset_val = 0.0 if self.v_reset is None else float(self.v_reset)
        if self.decay_input:
            vt = vp + (z - (vp - v_reset_val)) / self.tau
        else:
            vt = vp - (vp - v_reset_val) / self.tau + z

        spike = (vt >= self.v_threshold).to(dtype=vt.dtype)
        if self.v_reset is None:
            v_next = vt - spike * self.v_threshold
        else:
            v_next = torch.where(spike.bool(),
                                 torch.full_like(vt, float(self.v_reset)), vt)
        return spike, v_next


    def _dense_forward(self, x):
        """DenseKeep fallback: cuDNN conv + BN + LIF (no sparse kernel).

        Used by Runtime EGD when sparsity is too low for StreamFuse to win.
        Same as PartialFusionConvBNLIF: BatchFold conv + lif_sequential.
        """
        T, B = x.shape[0], x.shape[1]
        c = self._lean_cache
        device = x.device
        H_out, W_out = c['H_out'], c['W_out']

        # 1. BatchFold + cuDNN conv with folded BN
        x_4d = x.reshape(T * B, self.in_channels, x.shape[3], x.shape[4])
        if not hasattr(self, '_w_conv_fused') or self._w_conv_fused is None:
            with torch.no_grad():
                if self.bn_scale is not None:
                    scale = self.bn_scale
                    bias = self.bn_bias_folded
                    self._w_conv_fused = (self.weight * scale.view(-1, 1, 1, 1)).detach()
                    self._b_conv_fused = bias.detach() if bias is not None else None
                else:
                    self._w_conv_fused = self.weight.detach()
                    self._b_conv_fused = (self.bias.detach() if self.bias is not None else None)

        z_4d = torch.nn.functional.conv2d(
            x_4d, self._w_conv_fused, bias=self._b_conv_fused,
            stride=self.stride, padding=self.padding,
        )
        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)

        # 2. LIF
        v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
            B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
        spikes, v_final = lif_sequential(
            z, v_init, tau=self.tau,
            v_threshold=self.v_threshold, v_reset=self.v_reset,
        )
        self.v = v_final
        return spikes

    def _batchfold_forward(self, x):
        """StreamFuse: Conv+BN+LIF over T steps in ONE kernel launch.

        z stays in registers (never hits HBM).
        v stays in registers across T steps.
        Inline zero-detection replaces separate prescan.
        """
        T, B = x.shape[0], x.shape[1]
        c = self._lean_cache
        device = x.device
        H_out, W_out = c['H_out'], c['W_out']

        # Flatten to [T*B, C_in, H, W]
        x_flat = x.reshape(T * B, self.in_channels, x.shape[3], x.shape[4])

        # StaticZero: if entire batch is zero, skip conv kernel launch
        if not x_flat.any():
            H_out, W_out = c['H_out'], c['W_out']
            # z = BN bias only
            if c['has_bn']:
                z_flat = c['bn_bias_arg'].view(1, -1, 1, 1).expand(
                    T * B, self.out_channels, H_out, W_out).clone()
            elif c['has_bias']:
                z_flat = c['bias_arg'].view(1, -1, 1, 1).expand(
                    T * B, self.out_channels, H_out, W_out).clone()
            else:
                z_flat = torch.zeros(T * B, self.out_channels, H_out, W_out,
                                     dtype=torch.float32, device=device)
            z = z_flat.reshape(T, B, self.out_channels, H_out, W_out)
            v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
                B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
            spikes, v_final = lif_sequential(
                z, v_init, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self.v = v_final
            return spikes

        # Input is already float32 spike tensor, just ensure contiguous
        x_contig = x_flat.contiguous()

        # Grid: one program per (batch, tile) × output channel blocks
        BH, BW = 8, 16
        GH = triton.cdiv(H_out, BH)
        GW = triton.cdiv(W_out, BW)
        N_TILES = B * GH * GW

        # Prepare v_init
        v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
            B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
        v_init = v_init.float().contiguous()

        # Output buffers
        spike_out = torch.empty(T * B, self.out_channels, H_out, W_out,
                                dtype=torch.float32, device=device)
        v_out = torch.empty_like(v_init)

        def _grid(META):
            return (N_TILES, triton.cdiv(self.out_channels, META["BLOCK_N"]))

        sparse_streamfuse_conv3x3_bn_lif[_grid](
            x_contig,
            self._w_cl,
            c['bias_arg'],
            c['bn_scale_arg'],
            c['bn_bias_arg'],
            v_init,
            spike_out,
            v_out,
            T, B,
            C_IN=self.in_channels,
            C_OUT=self.out_channels,
            H=x.shape[3], W=x.shape[4],
            H_OUT=H_out, W_OUT=W_out,
            GH=GH, GW=GW,
            HAS_BIAS=c['has_bias'],
            HAS_BN=c['has_bn'],
            DECAY=c['decay'],
            RECIP_TAU=c['recip_tau'],
            V_TH=float(self.v_threshold),
            HAS_V_RESET=c['has_v_reset'],
            V_RESET=c['v_reset_val'],
            GROUP_SIZE_C=c['GSC'],
            NUM_GROUPS=c['NUM_GROUPS'],
        )

        self.v = v_out
        return spike_out.reshape(T, B, self.out_channels, H_out, W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with TimeBlock(K) structure.

        Uses lean path (fast_prescan_v2 + direct kernel) when available,
        falls back to sparse_fused_conv_bn_lif_forward or pure PyTorch.
        """
        T = x.shape[0]
        B = x.shape[1]
        device = x.device
        K = min(self.K, T)

        # Initialize v if needed
        if not isinstance(self.v, torch.Tensor) or self.v.shape == ():
            H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1
            self.v = torch.zeros(B, self.out_channels, H_out, W_out,
                                 dtype=torch.float32, device=device)

        # Choose execution path
        use_lean = (
            _LEAN_AVAILABLE and _KERNEL_AVAILABLE
            and x.is_cuda and self.stride == 1 and self.kernel_size == 3
        )

        if use_lean and not self._lean_ready:
            self._init_lean_cache(B, x.shape[3], x.shape[4], device)

        if use_lean:
            # Runtime EGD: check sparsity to choose backend
            x_flat = x.reshape(-1)
            nnz = x_flat.count_nonzero().item()
            sparsity = 1.0 - nnz / x_flat.numel()

            if nnz == 0:
                # StaticZero path (already in _batchfold_forward)
                return self._batchfold_forward(x)
            elif sparsity > 0.7:
                # High sparsity: StreamFuse wins
                return self._batchfold_forward(x)
            else:
                # Low sparsity: cuDNN DenseKeep wins
                return self._dense_forward(x)

        if not use_lean:
            self._ensure_w_cl()

        spike_list = []
        v = self.v

        for block_start in range(0, T, K):
            block_end = min(block_start + K, T)

            for t in range(block_start, block_end):
                x_t = x[t]

                if _KERNEL_AVAILABLE and x_t.is_cuda and self.stride == 1:
                    spike_t, v, _ = sparse_fused_conv_bn_lif_forward(
                        x=x_t, v_prev=v, weight=self.weight, bias=self.bias,
                        bn_scale=self.bn_scale, bn_bias=self.bn_bias_folded,
                        tau=self.tau, v_threshold=self.v_threshold,
                        v_reset=self.v_reset, decay_input=self.decay_input,
                        kernel_size=self.kernel_size, threshold=self.threshold,
                        w_cl=self._w_cl, ag_mask_buf=self._ag_mask_buf,
                    )
                else:
                    spike_t, v = self._fallback_forward_single(x_t, v)

                spike_list.append(spike_t)

        self.v = v
        return torch.stack(spike_list, dim=0)

    def extra_repr(self):
        s = (f'{self.in_channels}, {self.out_channels}, '
             f'kernel_size={self.kernel_size}, K={self.K}')
        if self.bn_scale is not None:
            s += ', bn=folded'
        s += f', tau={self.tau}, v_th={self.v_threshold}'
        return s
