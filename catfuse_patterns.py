"""
catfuse.patterns — CTF pattern library (Phase 3 task 3.1 skeleton)

Unified nn.Module wrappers for all fused patterns from Phase 0 and Phase 2.
Each pattern class provides:

  - `forward(x)` accepting `[T, B, ...]` input in SJ multi-step convention
  - `from_sj_modules(...)` classmethod for construction from SJ original modules
    (for Phase 3 task 3.2 substitute mechanism to copy weights directly)
  - `.policy_row` class attribute documenting which policy table row this
    pattern implements

Each pattern class wraps one specific Triton kernel or cuDNN+Triton combination.
The substitution mechanism (`catfuse.substitute`) scans SJ modules, matches
them against these patterns via topological inspection, and replaces the
matched subgraphs with instances of these classes.

Current patterns (from Phase 2 policy table v1):

  Row  Pattern name                  Policy
  ---  ----------------------------  -------------------------------------
  1    PartialFusionConvLIF          cuDNN Conv + Triton LIF (compute-bound)
  2    PartialFusionConvBNLIF        cuDNN Conv + Triton BN+LIF (compute-bound)
  3    FusedConv1x1LIF               full Triton fusion (compute-light)       [TBD Phase 4]
  4    FusedLinearLIF                full Triton fusion                       [from P0 linear_lif_k_sweep_v3]
  5    FusedAddLIF                   full Triton fusion                       [from P0 add_lif_k_sweep]
  6    FusedAddBNLIF                 full Triton fusion                       [from P2 task 2.2b + P0 add_bn_lif_min]
  7    FusedAvgPoolLIF               full Triton fusion                       [from P0]
  8    PartialFusionConvLIF (s=2)    cuDNN Conv s=2 + Triton LIF              [same class as row 1]

This file is the SKELETON — it defines the class hierarchy and API shape
but leaves `from_sj_modules` bodies TODO until we inspect real SJ network
module structures (Phase 3 task 3.0 prerequisite).
"""

from __future__ import annotations

import os
import sys

# Auto-resolve benchmarks/ path for Phase 0/2 kernel imports.
# This file lives at /data/yhr/CATFuse/catfuse_patterns.py and needs to
# import kernels from /data/yhr/CATFuse/benchmarks/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS_DIR = os.path.join(_THIS_DIR, 'benchmarks')
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import SJ's MemoryModule so our CTFPattern is a proper subclass.
# This makes SJ's functional.reset_net recognize CTF patterns and NOT emit warnings.
from spikingjelly.activation_based.base import MemoryModule as _SJMemoryModule


# ============================================================
# Base class
# ============================================================

class CTFPattern(_SJMemoryModule):
    """
    Base class for all CTF fused patterns.

    Subclasses must:
      - Override `forward(x)` to implement the fused kernel call
      - Override `from_sj_modules(...)` classmethod for construction from SJ modules
      - Set `policy_row` class attribute (int, matches policy table row)

    Inherits from SJ's MemoryModule so that functional.reset_net recognizes
    CTF patterns as proper SJ-compatible modules and does not emit warnings.
    CTF patterns carry state only inside Triton kernel register files
    per-call, so reset_net is a no-op for us.
    """

    policy_row: Optional[int] = None

    def __init__(self):
        super().__init__()
        # MemoryModule.__init__ already sets up _memories and _memories_rv
        # as OrderedDicts, so we don't need to redeclare them.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, ...] tensor in SJ multi-step convention (fp32, CUDA, contiguous)
        Returns:
            spike output of same or transformed shape
        """
        raise NotImplementedError

    @classmethod
    def from_sj_modules(cls, *sj_modules, **kwargs):
        """
        Build a CTF pattern instance by copying weights/parameters from
        the corresponding SJ original modules.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset — no-op for CTF patterns (state is per-call in kernel registers).
        Override the MemoryModule reset to avoid iterating over empty memories.
        """
        pass

    def single_step_forward(self, x):
        """
        SJ MemoryModule expects step_mode logic. CTF patterns are always
        multi-step (operate on [T, B, ...] input), so single_step is just
        a pass-through to forward.
        """
        return self.forward(x)

    def multi_step_forward(self, x_seq):
        """Multi-step forward is just our forward."""
        return self.forward(x_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, ...] tensor in SJ multi-step convention (fp32, CUDA, contiguous)
        Returns:
            spike output of same or transformed shape
        """
        raise NotImplementedError

    @classmethod
    def from_sj_modules(cls, *sj_modules, **kwargs):
        """
        Build a CTF pattern instance by copying weights/parameters from
        the corresponding SJ original modules (e.g., sj_conv, sj_bn, sj_lif).

        Subclass implementations vary by pattern type. Parameters vary by
        pattern — see each subclass's docstring.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset any internal state. For CTF patterns, state is carried inside
        the kernel's register file per-call, so there's no persistent state
        to reset at the module level. Override in subclasses that maintain
        persistent buffers.
        """
        pass


# ============================================================
# Row 1: PartialFusionConvLIF
# ============================================================

class PartialFusionConvLIF(CTFPattern):
    """
    cuDNN Conv + Triton LIF with StateCarry (K=T single block).

    Realization of schedule:
        TimeBlock(T) o StateCarry(LIF)
    (no StreamFuse because cuDNN Conv must materialize z to HBM)

    Phase 2 task 2.1a: 1.61x vs SJ cupy on V100, bit-exact parity.
    """

    policy_row = 1

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        # Conv weight as Parameter, no bias (assumes BN follows; if not, accept
        # bias via from_sj_modules)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import here to avoid circular import at module load
        from partial_fusion_conv_lif import partial_fusion_conv_lif
        return partial_fusion_conv_lif(
            x, self.weight,
            tau=self.tau, v_th=self.v_threshold, v_reset=self.v_reset,
            stride=self.stride, padding=self.padding,
        )

    @classmethod
    def from_sj_modules(cls,
                        sj_conv,    # SJ layer.Conv2d
                        sj_lif,     # SJ neuron.LIFNode
                        ):
        """
        Build from SJ Conv2d + LIFNode pair.

        TODO: fill in once SJ module inspection confirms parameter names.
        Expected SJ attribute names:
          sj_conv.in_channels, sj_conv.out_channels, sj_conv.kernel_size,
          sj_conv.stride, sj_conv.padding, sj_conv.weight
          sj_lif.tau, sj_lif.v_threshold, sj_lif.v_reset
        """
        # Extract conv params
        ks = sj_conv.kernel_size
        ks = ks[0] if isinstance(ks, tuple) else ks
        stride = sj_conv.stride
        stride = stride[0] if isinstance(stride, tuple) else stride
        padding = sj_conv.padding
        padding = padding[0] if isinstance(padding, tuple) else padding

        # Extract LIF params
        tau = float(sj_lif.tau) if hasattr(sj_lif, 'tau') else 2.0
        v_th = float(sj_lif.v_threshold)
        v_reset = float(sj_lif.v_reset) if sj_lif.v_reset is not None else 0.0
        assert v_reset == 0.0, "PartialFusionConvLIF only supports v_reset=0 hard reset"

        inst = cls(
            in_channels=sj_conv.in_channels,
            out_channels=sj_conv.out_channels,
            kernel_size=ks, stride=stride, padding=padding,
            tau=tau, v_threshold=v_th, v_reset=v_reset,
        )
        # Copy weight
        inst.weight.data.copy_(sj_conv.weight.data)
        return inst


# ============================================================
# Row 2: PartialFusionConvBNLIF
# ============================================================

class PartialFusionConvBNLIF(CTFPattern):
    """
    cuDNN Conv + Triton (BN affine -> LIF with StateCarry) with StreamFuse(BN, LIF).

    Realization of schedule:
        TimeBlock(T) o StreamFuse(BN, LIF) o StateCarry(LIF)
    (Conv is NOT StreamFused because cuDNN must write z to HBM)

    Phase 2 task 2.1b: 1.74x vs SJ cupy on V100, near-bit-exact parity (0-2 flips/102M).
    """

    policy_row = 2

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, bn_eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.bn_eps = bn_eps

        # Conv weight (no bias, BN follows)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        # BN params (inference mode, stored as buffers + parameters matching SJ layout)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from partial_fusion_conv_bn_lif import partial_fusion_conv_bn_lif
        return partial_fusion_conv_bn_lif(
            x, self.weight,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            eps=self.bn_eps,
            tau=self.tau, v_th=self.v_threshold, v_reset=self.v_reset,
            stride=self.stride, padding=self.padding,
        )

    @classmethod
    def from_sj_modules(cls,
                        sj_conv,   # SJ layer.Conv2d
                        sj_bn,     # SJ layer.BatchNorm2d
                        sj_lif,    # SJ neuron.LIFNode
                        ):
        """
        Build from SJ Conv2d + BatchNorm2d + LIFNode triple.
        """
        ks = sj_conv.kernel_size
        ks = ks[0] if isinstance(ks, tuple) else ks
        stride = sj_conv.stride
        stride = stride[0] if isinstance(stride, tuple) else stride
        padding = sj_conv.padding
        padding = padding[0] if isinstance(padding, tuple) else padding

        tau = float(sj_lif.tau) if hasattr(sj_lif, 'tau') else 2.0
        v_th = float(sj_lif.v_threshold)
        v_reset = float(sj_lif.v_reset) if sj_lif.v_reset is not None else 0.0
        assert v_reset == 0.0

        inst = cls(
            in_channels=sj_conv.in_channels,
            out_channels=sj_conv.out_channels,
            kernel_size=ks, stride=stride, padding=padding,
            tau=tau, v_threshold=v_th, v_reset=v_reset,
            bn_eps=sj_bn.eps,
        )
        # Copy conv weight
        inst.weight.data.copy_(sj_conv.weight.data)
        # Copy BN params (SJ BN has gamma, beta, running_mean, running_var)
        inst.bn_weight.data.copy_(sj_bn.weight.data)
        inst.bn_bias.data.copy_(sj_bn.bias.data)
        inst.running_mean.data.copy_(sj_bn.running_mean.data)
        inst.running_var.data.copy_(sj_bn.running_var.data)
        return inst


# ============================================================
# Row 9: PartialFusionConvBNAddLIF (for spiking_resnet BasicBlock second half)
# ============================================================

class PartialFusionConvBNAddLIF(CTFPattern):
    """
    cuDNN Conv + Triton (BN -> Add -> LIF with StateCarry), StreamFuse(BN, Add, LIF).

    Supports the standard spiking_resnet BasicBlock second-half sequence:
        out = conv2(x); bn2; out += identity; sn2

    The `identity` tensor is pre-computed by the caller (either raw block input
    if no downsample, or downsample(x) if spatial reduction).

    Realization of schedule:
        TimeBlock(T) o StreamFuse(BN, Add, LIF) o StateCarry(LIF)
    (Conv is NOT StreamFused because cuDNN must write z to HBM.)

    Phase 3 task 3.2c (new pattern, policy row 9).
    """

    policy_row = 9

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, bn_eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.bn_eps = bn_eps

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input [T, B, C_in, H, W]
            identity: pre-computed identity [T, B, C_out, H_out, W_out]
                (caller's responsibility — either raw x or downsample(x))

        Returns:
            spike [T, B, C_out, H_out, W_out]
        """
        from partial_fusion_conv_bn_add_lif import partial_fusion_conv_bn_add_lif
        return partial_fusion_conv_bn_add_lif(
            x, identity, self.weight,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            eps=self.bn_eps,
            tau=self.tau, v_th=self.v_threshold, v_reset=self.v_reset,
            stride=self.stride, padding=self.padding,
        )

    @classmethod
    def from_sj_modules(cls, sj_conv, sj_bn, sj_lif):
        ks = sj_conv.kernel_size
        ks = ks[0] if isinstance(ks, tuple) else ks
        stride = sj_conv.stride
        stride = stride[0] if isinstance(stride, tuple) else stride
        padding = sj_conv.padding
        padding = padding[0] if isinstance(padding, tuple) else padding

        tau = float(sj_lif.tau) if hasattr(sj_lif, 'tau') else 2.0
        v_th = float(sj_lif.v_threshold)
        v_reset = float(sj_lif.v_reset) if sj_lif.v_reset is not None else 0.0
        assert v_reset == 0.0

        inst = cls(
            in_channels=sj_conv.in_channels,
            out_channels=sj_conv.out_channels,
            kernel_size=ks, stride=stride, padding=padding,
            tau=tau, v_threshold=v_th, v_reset=v_reset,
            bn_eps=sj_bn.eps,
        )
        inst.weight.data.copy_(sj_conv.weight.data)
        inst.bn_weight.data.copy_(sj_bn.weight.data)
        inst.bn_bias.data.copy_(sj_bn.bias.data)
        inst.running_mean.data.copy_(sj_bn.running_mean.data)
        inst.running_var.data.copy_(sj_bn.running_var.data)
        return inst


# ============================================================
# Row 4: FusedLinearLIF (placeholder — to be wired to P0 linear_lif_k_sweep_v3)
# ============================================================

class FusedLinearLIF(CTFPattern):
    """
    Full Triton fusion: Linear -> LIF, K=T single block.

    Realization of schedule:
        TimeBlock(T) o BatchFold(Linear) o StreamFuse(Linear, LIF) o StateCarry(LIF)

    Phase 1: 1.29x vs SJ cupy, 5.18x vs SJ torch.
    """

    policy_row = 4

    def __init__(self,
                 in_features: int, out_features: int,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        # Weight in Phase 0 convention [I, O] (not PyTorch [O, I])
        self.weight_io = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight_io.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from linear_lif_k_sweep_v3 import run_fusion_k
        T = x.shape[0]
        return run_fusion_k(
            x, self.weight_io, K=T,
            tau=self.tau, v_th=self.v_threshold, v_reset=self.v_reset,
        )

    @classmethod
    def from_sj_modules(cls,
                        sj_linear,   # SJ layer.Linear
                        sj_lif,      # SJ neuron.LIFNode
                        ):
        tau = float(sj_lif.tau) if hasattr(sj_lif, 'tau') else 2.0
        v_th = float(sj_lif.v_threshold)
        v_reset = float(sj_lif.v_reset) if sj_lif.v_reset is not None else 0.0
        assert v_reset == 0.0

        inst = cls(
            in_features=sj_linear.in_features,
            out_features=sj_linear.out_features,
            tau=tau, v_threshold=v_th, v_reset=v_reset,
        )
        # PyTorch Linear weight is [O, I], transpose to [I, O] for Phase 0 kernel
        inst.weight_io.data.copy_(sj_linear.weight.data.T)
        return inst


# ============================================================
# Row 6: FusedAddBNLIF
# ============================================================

class FusedAddBNLIF(CTFPattern):
    """
    Full Triton fusion: Add -> BN -> LIF, K=T single block.

    Phase 0: 2.17x self-fusion, Phase 2 task 2.2b: 9.34x over torch-per-step.
    """

    policy_row = 6

    def __init__(self, num_features: int,
                 tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, bn_eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.bn_eps = bn_eps

        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Two-input forward: a and x are both [T, B, C, H, W], get added,
        then BN + LIF in one fused kernel.
        """
        # Precompute fused affine
        inv_std = torch.rsqrt(self.running_var + self.bn_eps)
        fused_scale = (self.bn_weight * inv_std).contiguous()
        fused_bias = (self.bn_bias - self.running_mean * fused_scale).contiguous()

        from add_bn_lif_min import run_fusion as p0_add_bn_lif_fusion
        return p0_add_bn_lif_fusion(
            a, x, fused_scale, fused_bias,
            tau=self.tau, v_th=self.v_threshold, v_reset=self.v_reset,
        )

    @classmethod
    def from_sj_modules(cls, sj_bn, sj_lif):
        """
        Build from SJ BatchNorm2d + LIFNode. Add is not a module in SJ — it's
        either an inline `a + x` in forward or a residual-branch connection.
        This classmethod is called when the substitute mechanism has
        identified that two activations feed into sj_bn and then into sj_lif.
        """
        tau = float(sj_lif.tau) if hasattr(sj_lif, 'tau') else 2.0
        v_th = float(sj_lif.v_threshold)
        v_reset = float(sj_lif.v_reset) if sj_lif.v_reset is not None else 0.0
        assert v_reset == 0.0

        inst = cls(
            num_features=sj_bn.num_features,
            tau=tau, v_threshold=v_th, v_reset=v_reset,
            bn_eps=sj_bn.eps,
        )
        inst.bn_weight.data.copy_(sj_bn.weight.data)
        inst.bn_bias.data.copy_(sj_bn.bias.data)
        inst.running_mean.data.copy_(sj_bn.running_mean.data)
        inst.running_var.data.copy_(sj_bn.running_var.data)
        return inst


# ============================================================
# CTFSEWBasicBlock — for SEW-ResNet BasicBlock replacement
# ============================================================

class CTFSEWBasicBlock(CTFPattern):
    """
    CTF-fused replacement for SEW-ResNet18's BasicBlock.

    Original SJ SEWBasicBlock.forward:
        identity = x
        out = conv1(x); bn1; sn1
        out = conv2(out); bn2; sn2
        if downsample:
            identity = downsample_sn(downsample(x))
        out = sew_function(identity, out, cnf)
        return out

    where sew_function(x, y, cnf='ADD') = x + y.

    Key insight: all three sn* sites (sn1, sn2, downsample_sn) are preceded
    by Conv+BN, making them all valid `PartialFusionConvBNLIF` patterns.
    The sew_function add is a POST-LIF spike-level add, which is just a
    torch `+` on two spike tensors — no new pattern needed.
    """

    def __init__(self, fused_first, fused_second, fused_downsample, cnf):
        super().__init__()
        self.fused_first = fused_first
        self.fused_second = fused_second
        self.fused_downsample = fused_downsample  # may be None
        self.cnf = cnf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fused_first(x)
        out = self.fused_second(out)
        if self.fused_downsample is not None:
            identity = self.fused_downsample(x)
        else:
            identity = x
        # sew_function
        if self.cnf == 'ADD':
            return identity + out
        elif self.cnf == 'AND':
            return identity * out
        elif self.cnf == 'IAND':
            return identity * (1.0 - out)
        else:
            raise NotImplementedError(f"Unknown cnf: {self.cnf}")

    @classmethod
    def from_sj_block(cls, sj_block):
        """
        Build from an SJ SEWResNet BasicBlock.
        """
        fused_first = PartialFusionConvBNLIF.from_sj_modules(
            sj_block.conv1, sj_block.bn1, sj_block.sn1
        )
        fused_second = PartialFusionConvBNLIF.from_sj_modules(
            sj_block.conv2, sj_block.bn2, sj_block.sn2
        )
        fused_downsample = None
        if sj_block.downsample is not None:
            # downsample is Sequential(Conv2d, BatchNorm2d) + downsample_sn is the LIFNode
            ds_conv = sj_block.downsample[0]
            ds_bn = sj_block.downsample[1]
            ds_sn = sj_block.downsample_sn
            fused_downsample = PartialFusionConvBNLIF.from_sj_modules(
                ds_conv, ds_bn, ds_sn
            )
        return cls(fused_first, fused_second, fused_downsample, sj_block.cnf)


# ============================================================
# CTFSpikingBasicBlock — for standard spiking_resnet BasicBlock replacement
# ============================================================

class CTFSpikingBasicBlock(CTFPattern):
    """
    CTF-fused replacement for standard spiking_resnet18's BasicBlock.

    Original SJ spiking_resnet BasicBlock.forward:
        identity = x
        out = conv1(x); bn1; sn1
        out = conv2(out); bn2
        if downsample: identity = downsample(x)   # Sequential(Conv2d, BN2d), NO LIF
        out += identity                            # Add BEFORE LIF
        out = sn2(out)                             # LIF LAST

    Pattern decomposition:
      - `conv1+bn1+sn1` = PartialFusionConvBNLIF (row 2)
      - `conv2+bn2+(+identity)+sn2` = PartialFusionConvBNAddLIF (row 9)
      - downsample (if present) = Conv+BN, NO LIF — we use a vanilla PyTorch
        path (F.conv2d + batch_norm fold) because it's only hit 3 times total
        in ResNet18 and not worth a dedicated fused kernel

    Difference from CTFSEWBasicBlock:
      - SEW has Add AFTER LIF (spike-level add of binary tensors)
      - Standard ResNet has Add BEFORE LIF (continuous add, then LIF)
      - This forces the Add into the fused kernel → PartialFusionConvBNAddLIF
      - SEW's downsample has a LIFNode (downsample_sn), this one does not
    """

    def __init__(self, fused_first, fused_second_add,
                 downsample_conv_weight, downsample_bn_scale, downsample_bn_bias,
                 downsample_stride, downsample_padding):
        """
        Args:
            fused_first: PartialFusionConvBNLIF for conv1+bn1+sn1
            fused_second_add: PartialFusionConvBNAddLIF for conv2+bn2+Add+sn2
            downsample_*: optional pre-computed downsample Conv+BN (None if no downsample)
        """
        super().__init__()
        self.fused_first = fused_first
        self.fused_second_add = fused_second_add
        # downsample params stored as buffers + parameters (not a sub-nn.Module, to
        # keep the computation inline without extra module overhead)
        if downsample_conv_weight is not None:
            # Register downsample weight as a parameter so it moves with .to(device)
            self.has_downsample = True
            self.downsample_stride = downsample_stride
            self.downsample_padding = downsample_padding
            self.downsample_weight = nn.Parameter(downsample_conv_weight.clone())
            # BN fold for downsample (pre-computed once, static)
            self.register_buffer('downsample_bn_scale', downsample_bn_scale.clone())
            self.register_buffer('downsample_bn_bias', downsample_bn_bias.clone())
        else:
            self.has_downsample = False

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsample Conv+BN fold, no LIF. Used for identity path."""
        import torch.nn.functional as F
        T, B, C_in, H, W = x.shape
        x_4d = x.view(T * B, C_in, H, W)
        z_4d = F.conv2d(x_4d, self.downsample_weight, bias=None,
                        stride=self.downsample_stride,
                        padding=self.downsample_padding)
        C_out, H_out, W_out = z_4d.shape[1], z_4d.shape[2], z_4d.shape[3]
        z = z_4d.view(T, B, C_out, H_out, W_out)
        # Apply BN fold: z_bn = z * scale + bias  (broadcasting over T, B, H, W)
        scale = self.downsample_bn_scale.view(1, 1, -1, 1, 1)
        bias = self.downsample_bn_bias.view(1, 1, -1, 1, 1)
        return (z * scale + bias).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First half: conv1 + bn1 + sn1
        out = self.fused_first(x)
        # Identity path
        if self.has_downsample:
            identity = self._downsample(x)
        else:
            identity = x
        # Second half: conv2 + bn2 + Add(identity) + sn2 in one fused kernel
        out = self.fused_second_add(out, identity)
        return out

    @classmethod
    def from_sj_block(cls, sj_block):
        """
        Build from an SJ spiking_resnet BasicBlock.

        The block has attributes:
            conv1, bn1, sn1 (LIF)
            conv2, bn2, sn2 (LIF)
            downsample (None or Sequential(Conv2d, BN2d), NO LIF)
        """
        fused_first = PartialFusionConvBNLIF.from_sj_modules(
            sj_block.conv1, sj_block.bn1, sj_block.sn1
        )
        fused_second_add = PartialFusionConvBNAddLIF.from_sj_modules(
            sj_block.conv2, sj_block.bn2, sj_block.sn2
        )
        if sj_block.downsample is not None:
            ds_conv = sj_block.downsample[0]
            ds_bn = sj_block.downsample[1]
            # Pre-compute BN fold for downsample (one-time, static in inference)
            inv_std = torch.rsqrt(ds_bn.running_var + ds_bn.eps)
            ds_scale = ds_bn.weight * inv_std
            ds_bias = ds_bn.bias - ds_bn.running_mean * ds_scale
            ds_stride = ds_conv.stride
            ds_stride = ds_stride[0] if isinstance(ds_stride, tuple) else ds_stride
            ds_padding = ds_conv.padding
            ds_padding = ds_padding[0] if isinstance(ds_padding, tuple) else ds_padding
            return cls(
                fused_first, fused_second_add,
                downsample_conv_weight=ds_conv.weight.data,
                downsample_bn_scale=ds_scale.detach(),
                downsample_bn_bias=ds_bias.detach(),
                downsample_stride=ds_stride,
                downsample_padding=ds_padding,
            )
        else:
            return cls(fused_first, fused_second_add, None, None, None, 0, 0)



# ============================================================
# Registry — for substitute mechanism to enumerate available patterns
# ============================================================

PATTERN_REGISTRY = {
    'PartialFusionConvLIF':      PartialFusionConvLIF,
    'PartialFusionConvBNLIF':    PartialFusionConvBNLIF,
    'PartialFusionConvBNAddLIF': PartialFusionConvBNAddLIF,
    'FusedLinearLIF':            FusedLinearLIF,
    'FusedAddBNLIF':             FusedAddBNLIF,
    'CTFSEWBasicBlock':          CTFSEWBasicBlock,
    'CTFSpikingBasicBlock':      CTFSpikingBasicBlock,
    # TODO row 3: FusedConv1x1LIF
    # TODO row 5: FusedAddLIF
    # TODO row 7: FusedAvgPoolLIF
}


__all__ = [
    'CTFPattern',
    'PartialFusionConvLIF',
    'PartialFusionConvBNLIF',
    'PartialFusionConvBNAddLIF',
    'FusedLinearLIF',
    'FusedAddBNLIF',
    'CTFSEWBasicBlock',
    'CTFSpikingBasicBlock',
    'PATTERN_REGISTRY',
]
