"""catfuse.implementations — Implementation hierarchy for Definition 3.16.

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
    IOCost,
    ScheduleTransform,
    ScheduleDecomposition,
    TSI_OPS,
    CSR_OPS,
    TGO_OPS,
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
    "IOCost",
    "ScheduleTransform",
    "ScheduleDecomposition",
    "TSI_OPS",
    "CSR_OPS",
    "TGO_OPS",
    "DenseKeep",
    "SparseFlow",
    "static_zero_forward",
]
