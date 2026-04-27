"""CATFuse temporal fused kernels (Triton).

Public kernel API used by `catfuse.patterns`:
  - partial_fusion_conv_lif       (from partial_fusion_conv_lif_impl)
  - partial_fusion_conv_bn_lif    (from partial_fusion_conv_bn_lif_impl)
  - partial_fusion_conv_bn_add_lif (from partial_fusion_conv_bn_add_lif_impl)

Note: prior to Stage 1 of the refactor (problem 7 in the design audit), these
implementations lived in `benchmarks/` and were imported via a sys.path hack.
They have now been moved here and are importable via the normal package path.
"""

from catfuse.kernels.partial_fusion_conv_lif_impl import (
    partial_fusion_conv_lif,
)
from catfuse.kernels.partial_fusion_conv_bn_lif_impl import (
    partial_fusion_conv_bn_lif,
)
from catfuse.kernels.partial_fusion_conv_bn_add_lif_impl import (
    partial_fusion_conv_bn_add_lif,
)

__all__ = [
    "partial_fusion_conv_lif",
    "partial_fusion_conv_bn_lif",
    "partial_fusion_conv_bn_add_lif",
]