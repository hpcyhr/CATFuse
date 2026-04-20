"""
SparseFlow spatial sparse backend.

Provides prescan-based block-sparse Conv2d execution and three-path
dispatch (StaticZero / Sparse / DenseKeep) for SNN spike inputs.

Integrated into CATFuse as the spatial execution backend.
"""
