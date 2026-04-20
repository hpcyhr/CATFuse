"""
Lean linear launcher: bypass sparse_linear_forward's Python overhead.

For linear with small N (B=8-128), the prescan is trivially cheap
with PyTorch ops. The bottleneck is the Python wrapper, not prescan.
"""
import torch
import triton
import triton.language as tl
from typing import Optional

TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2


def _pytorch_linear_prescan(x, block_m, group_size_c, num_groups):
    """Fast PyTorch prescan for linear (no Triton kernel needed).

    For B=8, this is <0.01ms (1 tile, trivial work).
    """
    N, C_IN = x.shape
    device = x.device
    n_tiles = triton.cdiv(N, block_m)

    # Pad N to multiple of block_m
    if N % block_m != 0:
        x_pad = torch.zeros(n_tiles * block_m, C_IN, dtype=x.dtype, device=device)
        x_pad[:N] = x
    else:
        x_pad = x

    # Reshape to [n_tiles, block_m, num_groups, group_size_c]
    c_pad = num_groups * group_size_c
    if C_IN < c_pad:
        x_pad = torch.nn.functional.pad(x_pad, (0, c_pad - C_IN))

    x_tiled = x_pad.reshape(n_tiles, block_m, num_groups, group_size_c)

    # Per-tile, per-group: any nonzero?
    group_active = (x_tiled.abs().amax(dim=(1, 3)) > 0)  # [n_tiles, num_groups]

    # Pack to bitmask
    bits = torch.arange(num_groups, device=device, dtype=torch.int32)
    ag_mask = (group_active.int() << bits[None, :]).sum(dim=1).to(torch.int32)

    # Tile class
    any_active = group_active.any(dim=1)
    all_active = group_active.all(dim=1)
    tile_class = torch.full((n_tiles,), TILE_SPARSE, dtype=torch.int32, device=device)
    tile_class[~any_active] = TILE_ZERO
    tile_class[all_active] = TILE_DENSEISH

    return ag_mask, tile_class


def lean_sparse_linear(
    x, w_t, bias,
    block_m=16, group_size_c=16,
    ag_mask_buf=None, tile_class_buf=None, y_buf=None,
):
    """Lean sparse linear: PyTorch prescan + direct kernel launch."""
    from catfuse.sparseflow.sparse_linear_kernel import sparse_linear_grouped_kernel

    N, C_IN = x.shape
    C_OUT = w_t.shape[1]
    device = x.device

    n_tiles = triton.cdiv(N, block_m)
    num_groups = triton.cdiv(C_IN, group_size_c)
    # dense_k and BLOCK_N provided by autotune

    # Ensure fp16
    x_f16 = x if x.dtype == torch.float16 else x.half()

    # 1. Fast PyTorch prescan
    ag_mask, tile_class = _pytorch_linear_prescan(x_f16, block_m, group_size_c, num_groups)

    # 2. Output buffer
    if y_buf is None or y_buf.shape != (N, C_OUT):
        y_buf = torch.empty(N, C_OUT, dtype=torch.float32, device=device)

    # 3. Direct kernel launch
    has_bias = bias is not None
    bias_arg = bias.float().contiguous() if has_bias else torch.empty(1, dtype=torch.float32, device=device)
    tile_ids_dummy = torch.empty(1, dtype=torch.int32, device=device)

    def _grid(META):
        return (n_tiles, triton.cdiv(C_OUT, META["BLOCK_N"]))


    sparse_linear_grouped_kernel[_grid](
        x_f16, w_t, bias_arg,
        tile_class, ag_mask, tile_ids_dummy, y_buf,
        N,
        C_IN=C_IN, C_OUT=C_OUT,
        N_TILES_KEY=n_tiles,
        BLOCK_M_KEY=block_m,
        GROUP_SIZE_C_KEY=group_size_c,
        BLOCK_M=block_m,
        GROUP_SIZE_C=group_size_c,
        NUM_GROUPS=num_groups,
        HAS_BIAS=has_bias,
        USE_TILE_IDS=False,
    )

    return y_buf


class LeanSparseLinearCache:
    """Pre-compute and cache state for repeated lean_sparse_linear calls."""

    def __init__(self, weight, bias, block_m=16, group_size_c=16):
        self.w_t = weight.t().half().contiguous()
        self.bias = bias
        self.block_m = block_m
        self.group_size_c = group_size_c
        self.y_buf = None

    def __call__(self, x):
        y = lean_sparse_linear(
            x, self.w_t, self.bias,
            block_m=self.block_m,
            group_size_c=self.group_size_c,
            y_buf=self.y_buf,
        )
        if self.y_buf is None:
            self.y_buf = torch.empty_like(y)
        return y
