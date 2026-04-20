"""
Fast Triton prescan for SNN spike tensors.

Replaces the 6-step PyTorch prescan pipeline (0.43ms) with a single
Triton kernel launch (~0.02-0.05ms target).

Key insight: SNN spikes are binary (0 or 1). We don't need abs(), amax(),
or max_pool for receptive field expansion. A single kernel can directly
scan each tile's receptive field and output per-group bitmask.

Algorithm (one program per tile):
  1. Compute this tile's receptive field bounding box in input space
  2. For each channel group:
     - Scan all (ih, iw) positions in the RF
     - Load GROUP_SIZE channels at each position
     - If any is nonzero → mark group active, skip remaining positions
  3. Pack active groups into uint32 bitmask
  4. Classify tile: ZERO / SPARSE / DENSEISH
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# Tile classification constants (must match sparse_helpers.py)
TILE_ZERO = 0
TILE_SPARSE = 1
TILE_DENSEISH = 2


@triton.jit
def _spike_prescan_2d_kernel(
    # Input spike tensor [N, C, H, W] in NCHW layout
    x_ptr,
    # Outputs
    ag_mask_ptr,       # [N_TILES] uint32 per-tile bitmask
    tile_class_ptr,    # [N_TILES] int32 per-tile class
    # Tensor dims
    N_val,
    C: tl.constexpr,
    H,
    W,
    HW,                # = H * W (precomputed)
    CHW,               # = C * H * W (precomputed)
    # Output spatial dims (after conv)
    H_OUT,
    W_OUT,
    # Tile grid dims
    GH,
    GW,
    # Conv params
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    # Prescan params
    KS: tl.constexpr,          # kernel size (1 or 3)
    GROUP_SIZE: tl.constexpr,  # channels per group
    NUM_GROUPS: tl.constexpr,  # total groups
    BLOCK_H: tl.constexpr,    # tile spatial height
    BLOCK_W: tl.constexpr,    # tile spatial width
):
    tile_id = tl.program_id(0)

    # Decode tile → (n, gh, gw)
    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    # Base pointer for this batch element
    x_base = n_idx * CHW

    # Receptive field bounding box in input space
    # Output pixel (oh, ow) reads input at (oh*STRIDE - PAD + kh, ow*STRIDE - PAD + kw)
    # for kh in [0, KS), kw in [0, KS)
    # Tile covers output rows [gh_idx*BLOCK_H, gh_idx*BLOCK_H + BLOCK_H)
    # So RF starts at ih = gh_idx * BLOCK_H * STRIDE - PAD
    # and ends at ih = (gh_idx * BLOCK_H + BLOCK_H - 1) * STRIDE - PAD + KS - 1
    ih_start = gh_idx * BLOCK_H * STRIDE - PAD
    iw_start = gw_idx * BLOCK_W * STRIDE - PAD

    RF_H: tl.constexpr = BLOCK_H * STRIDE + KS - 1
    RF_W: tl.constexpr = BLOCK_W * STRIDE + KS - 1

    # Channel offsets within a group [GROUP_SIZE]
    c_local = tl.arange(0, GROUP_SIZE)

    ag_mask = 0
    active_count = 0

    for g in range(NUM_GROUPS):
        c_start = g * GROUP_SIZE
        c_offs = c_start + c_local       # [GROUP_SIZE]
        c_valid = c_offs < C              # mask for channels beyond C

        found = 0  # scalar flag: 1 if this group has any active spike

        # Scan the receptive field
        for rh in tl.static_range(RF_H):
            ih = ih_start + rh
            h_valid = (ih >= 0) & (ih < H)

            for rw in tl.static_range(RF_W):
                iw = iw_start + rw
                hw_valid = h_valid & (iw >= 0) & (iw < W)

                # Only load if: position valid, channel valid, not yet found
                load_mask = hw_valid & c_valid & (found == 0)

                # Address: x[n, c, ih, iw] = x_ptr[n*CHW + c*HW + ih*W + iw]
                addrs = x_base + c_offs * HW + ih * W + iw
                vals = tl.load(x_ptr + addrs, mask=load_mask, other=0.0)

                # Check if any channel in this group has a spike
                has_spike = tl.sum(vals) > 0.0
                found = tl.where(has_spike, 1, found)

        # Set bit g in bitmask
        ag_mask = ag_mask | (found << g)
        active_count = active_count + found

    # Classify tile
    tile_class = tl.where(
        ag_mask == 0, TILE_ZERO,
        tl.where(active_count == NUM_GROUPS, TILE_DENSEISH, TILE_SPARSE)
    )

    tl.store(ag_mask_ptr + tile_id, ag_mask)
    tl.store(tile_class_ptr + tile_id, tile_class)


def fast_spike_prescan_2d(
    x: torch.Tensor,           # [N, C, H, W] NCHW float32 spike tensor
    H_OUT: int,
    W_OUT: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    block_h: int = 8,
    block_w: int = 16,
    group_size_c: int = 16,
    threshold: float = 1e-6,   # ignored for binary spikes, kept for API compat
    ag_mask_out: Optional[torch.Tensor] = None,
    tile_class_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel prescan for binary spike tensors.

    Replaces the 6-step PyTorch pipeline with one Triton kernel launch.

    Args:
        x: [N, C, H, W] spike tensor (NCHW, float32, values 0.0 or 1.0)
        H_OUT, W_OUT: output spatial dims after conv
        kernel_size: conv kernel size (1 or 3)
        stride: conv stride
        padding: conv padding
        block_h, block_w: tile spatial dimensions
        group_size_c: channels per group for bitmask
        threshold: activity threshold (ignored for binary spikes)
        ag_mask_out: optional pre-allocated [N_TILES] int32 buffer
        tile_class_out: optional pre-allocated [N_TILES] int32 buffer

    Returns:
        (tile_class, ag_mask) — both [N_TILES] int32 tensors on device
    """
    N, C, H, W = x.shape
    device = x.device

    GH = triton.cdiv(H_OUT, block_h)
    GW = triton.cdiv(W_OUT, block_w)
    N_TILES = N * GH * GW
    NUM_GROUPS = triton.cdiv(C, group_size_c)

    # Allocate or reuse output buffers
    if ag_mask_out is None or ag_mask_out.numel() < N_TILES:
        ag_mask_out = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_out is None or tile_class_out.numel() < N_TILES:
        tile_class_out = torch.empty(N_TILES, dtype=torch.int32, device=device)

    # Ensure contiguous NCHW layout
    x_contig = x.contiguous()

    # Launch: one program per tile
    grid = (N_TILES,)

    _spike_prescan_2d_kernel[grid](
        x_contig,
        ag_mask_out,
        tile_class_out,
        N,
        C,
        H, W,
        H * W,       # HW
        C * H * W,   # CHW
        H_OUT, W_OUT,
        GH, GW,
        STRIDE=stride,
        PAD=padding,
        KS=kernel_size,
        GROUP_SIZE=group_size_c,
        NUM_GROUPS=NUM_GROUPS,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
    )

    return tile_class_out[:N_TILES], ag_mask_out[:N_TILES]


# ============================================================
# Drop-in replacement for _build_two_stage_metadata
# ============================================================

def fast_build_metadata(
    x: torch.Tensor,       # [N, C, H, W] NCHW (not channels-last, not fp16)
    N: int, C_IN: int,
    H_IN: int, W_IN: int,
    H_OUT: int, W_OUT: int,
    BH: int, BW: int,
    GH: int, GW: int,
    kernel_size: int,
    stride: int,
    padding: int,
    threshold: float,
    ag_mask_buf: torch.Tensor,
    tile_class_buf: Optional[torch.Tensor],
):
    """Drop-in replacement for sparse_conv2d_kernel._build_two_stage_metadata.

    Key differences from the original:
      - Accepts NCHW float32 input directly (no fp16 conversion, no permute)
      - Single Triton kernel launch instead of 6 PyTorch ops
      - Optimized for binary spike inputs
    """
    from catfuse.sparseflow.sparse_conv2d_kernel import choose_group_size

    group_size_c = choose_group_size(C_IN)

    # Handle fp16 input (convert to fp32 for the prescan)
    if x.dtype == torch.float16:
        x = x.float()

    # Handle channels-last input (convert to NCHW)
    # The original pipeline expected channels-last, we expect NCHW
    if x.ndim == 4:
        # Check if it's already NCHW or NHWC
        if x.shape[1] == C_IN:
            x_nchw = x  # already NCHW
        else:
            # Assume NHWC, permute to NCHW
            x_nchw = x.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")

    tile_class, ag_mask = fast_spike_prescan_2d(
        x_nchw,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        block_h=BH,
        block_w=BW,
        group_size_c=group_size_c,
        threshold=threshold,
        ag_mask_out=ag_mask_buf,
        tile_class_out=tile_class_buf,
    )

    return tile_class, ag_mask