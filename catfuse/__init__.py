"""
CATFuse-SF: Certified Spatio-Temporal Fusion for SNN Inference
"""

from catfuse.substitute import substitute as _substitute
from catfuse.substitute import substitute_sf as _substitute_sf


def optimize(model, device='cuda:0', policy=None, T=16, use_sparseflow=True):
    """One-line entry: substitute SJ model with CATFuse-SF fused patterns."""
    if use_sparseflow:
        return _substitute_sf(model, T=T)
    return _substitute(model)


__version__ = "0.2.0"
