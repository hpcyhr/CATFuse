"""
Lean conv launchers: bypass sparse_conv*_forward Python overhead.

Supports:
  - Conv2d 3x3 stride=1 (existing kernels)
  - Conv2d 3x3 stride=2 (existing kernels)
  - Conv2d 1x1 stride=1 (existing kernels)
  - Conv1d 3 stride=1 (existing kernel)

Common pattern:
  1. fast prescan (Triton or PyTorch, depending on cost)
  2. direct kernel launch (no wrapper overhead)
  3. pre-allocated buffers
"""
import torch
import triton
from typing import Optional, Tuple

from catfuse.sparseflow.sparse_conv2d_kernel import (
    sparse_conv1x1_nhwc_kernel_8x8,
    sparse_conv1x1_nhwc_kernel_8x16,
    sparse_conv3x3s1_nhwc_kernel_8x8,
    sparse_conv3x3s1_nhwc_kernel_8x16,
    _select_tile_sizes,
    choose_group_size,
    _build_two_stage_metadata,
)
from catfuse.sparseflow.fast_prescan_v2 import fast_spike_prescan_2d_v2


def lean_conv2d(
    x: torch.Tensor,          # [N, C_IN, H, W] float spike
    w_cl: torch.Tensor,       # channel-last weight, fp16
    bias: Optional[torch.Tensor],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    # cached state
    ag_mask_buf: Optional[torch.Tensor] = None,
    tile_class_buf: Optional[torch.Tensor] = None,
    y_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lean conv2d: fast prescan + direct kernel launch.
    
    Supports 3x3/s1, 3x3/s2, 1x1/s1.
    """
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = w_cl.shape[0]
    device = x.device

    H_OUT = (H_IN + 2 * padding - kernel_size) // stride + 1
    W_OUT = (W_IN + 2 * padding - kernel_size) // stride + 1

    BH, BW = _select_tile_sizes(H_OUT, W_OUT)
    GH = triton.cdiv(H_OUT, BH)
    GW = triton.cdiv(W_OUT, BW)
    N_TILES = N * GH * GW
    GSC = choose_group_size(C_IN)
    NUM_GROUPS = triton.cdiv(C_IN, GSC)
    ALL_ONES = (1 << NUM_GROUPS) - 1
    DENSE_K = min(GSC * 2, 64)
    if DENSE_K < 16:
        DENSE_K = 16

    # Ensure buffers
    if ag_mask_buf is None or ag_mask_buf.numel() < N_TILES:
        ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_class_buf is None or tile_class_buf.numel() < N_TILES:
        tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if y_buf is None or y_buf.shape != (N, C_OUT, H_OUT, W_OUT):
        y_buf = torch.empty(N, C_OUT, H_OUT, W_OUT, dtype=torch.float32, device=device)

    # 1. Prescan
    fast_spike_prescan_2d_v2(
        x, H_OUT, W_OUT,
        kernel_size=kernel_size, stride=stride, padding=padding,
        block_h=BH, block_w=BW, group_size_c=GSC,
        ag_mask_out=ag_mask_buf, tile_class_out=tile_class_buf,
    )

    # 2. Convert to fp16 NHWC for kernel
    x_f16 = x.half()
    x_nhwc = x_f16.permute(0, 2, 3, 1).contiguous()

    has_bias = bias is not None
    bias_arg = bias.float().contiguous() if has_bias else torch.empty(1, dtype=torch.float32, device=device)
    tile_ids_dummy = torch.empty(1, dtype=torch.int32, device=device)

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META["BLOCK_N"]))

    # 3. Select and launch kernel
    if kernel_size == 1:
        kernel = sparse_conv1x1_nhwc_kernel_8x16 if BW == 16 else sparse_conv1x1_nhwc_kernel_8x8
        kernel[_grid](
            x_nhwc, w_cl, bias_arg,
            ag_mask_buf, tile_ids_dummy, y_buf, N,
            C_IN=C_IN, C_OUT=C_OUT,
            H_IN=H_IN, W_IN=W_IN,
            H_OUT=H_OUT, W_OUT=W_OUT,
            GH=GH, GW=GW,
            HAS_BIAS=has_bias,
            GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
            ALL_ONES_MASK=ALL_ONES,
            DENSE_K=DENSE_K,
            USE_TILE_IDS=False,
        )
    elif kernel_size == 3:
        kernel = sparse_conv3x3s1_nhwc_kernel_8x16 if BW == 16 else sparse_conv3x3s1_nhwc_kernel_8x8
        kernel[_grid](
            x_nhwc, w_cl, bias_arg,
            ag_mask_buf, tile_ids_dummy, y_buf, N,
            C_IN=C_IN, C_OUT=C_OUT,
            H_IN=H_IN, W_IN=W_IN,
            H_OUT=H_OUT, W_OUT=W_OUT,
            GH=GH, GW=GW,
            HAS_BIAS=has_bias,
            GROUP_SIZE_C=GSC, NUM_GROUPS=NUM_GROUPS,
            ALL_ONES_MASK=ALL_ONES,
            DENSE_K=DENSE_K,
            USE_TILE_IDS=False,
        )

    return y_buf


def lean_conv1d(
    x: torch.Tensor,          # [N, C_IN, L] float spike
    w: torch.Tensor,           # [C_OUT, C_IN, K] weight
    bias: Optional[torch.Tensor],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    """Lean conv1d: PyTorch prescan + direct kernel launch.
    
    For conv1d, the shapes are small enough that a simple
    approach works: check sparsity, if >threshold use sparse kernel,
    else use F.conv1d.
    """
    import torch.nn.functional as F
    from catfuse.sparseflow.sparse_conv1d_kernel import sparse_conv1d_forward

    N, C_IN, L = x.shape
    
    # For small tensors, F.conv1d is faster
    if N * C_IN * L < 65536:
        return F.conv1d(x, w, bias, stride=stride, padding=padding)
    
    # Check sparsity
    sparsity = 1.0 - x.count_nonzero().item() / x.numel()
    if sparsity < 0.8:
        return F.conv1d(x, w, bias, stride=stride, padding=padding)

    # Use sparse kernel with launch_all_tiles to avoid host sync
    x_f16 = x.half().contiguous()
    result = sparse_conv1d_forward(
        x_f16, w, bias,
        kernel_size=kernel_size, stride=stride, padding=padding,
        threshold=1e-6, launch_all_tiles=True,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


class LeanConv2dCache:
    """Pre-compute and cache state for repeated lean_conv2d calls."""

    def __init__(self, weight, bias, kernel_size=3, stride=1, padding=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Pre-compute weight in channel-last fp16
        if kernel_size == 1:
            self.w_cl = weight.half().permute(0, 2, 3, 1).contiguous()
        elif kernel_size == 3:
            self.w_cl = weight.half().permute(0, 2, 3, 1).contiguous()
        
        self.ag_mask_buf = None
        self.tile_class_buf = None
        self.y_buf = None

    def __call__(self, x):
        return lean_conv2d(
            x, self.w_cl, self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ag_mask_buf=self.ag_mask_buf,
            tile_class_buf=self.tile_class_buf,
            y_buf=self.y_buf,
        )
