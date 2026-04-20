"""
Sparse StreamFuse kernel: Conv3x3+BN+LIF over T steps in ONE launch.

Key properties:
  - z (conv+BN output) stays in registers → never hits HBM
  - v (membrane potential) stays in registers across T steps
  - Inline zero-detection replaces separate prescan launch
  - Each program = one (tile, output_channel_block) for all T steps

I/O per tile per step:
  Read:  x[t] tiles (only active groups), w tiles (cached across t)
  Write: spike[t] tiles
  NO z read/write, v only loaded once (t=0) and written once (t=T-1)
"""

import torch
import triton
import triton.language as tl
from triton import Config


def _make_sf_configs():
    cfgs = []
    for bn in [64, 128]:
        for nw in [4, 8]:
            cfgs.append(Config({"BLOCK_N": bn}, num_warps=nw, num_stages=1))
    return cfgs

_SF_CONFIGS = _make_sf_configs()


@triton.autotune(configs=_SF_CONFIGS, key=["C_IN", "C_OUT", "H_OUT", "W_OUT"])
@triton.jit
def sparse_streamfuse_conv3x3_bn_lif(
    # Input: [T*B, C_IN, H, W] in NCHW (all T steps flattened with B)
    x_ptr,
    # Conv weights: [C_OUT, 3, 3, C_IN] channel-last
    w_cl_ptr,
    # Bias
    bias_ptr,
    # BN params
    bn_scale_ptr,
    bn_bias_ptr,
    # LIF state
    v_init_ptr,     # [B, C_OUT, H_OUT, W_OUT] — initial membrane potential
    # Outputs
    spike_ptr,      # [T*B, C_OUT, H_OUT, W_OUT] — output spikes (flat)
    v_final_ptr,    # [B, C_OUT, H_OUT, W_OUT] — final membrane potential
    # Dims
    T_val,          # number of timesteps
    B_val,          # batch size
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    # Flags
    HAS_BIAS: tl.constexpr,
    HAS_BN: tl.constexpr,
    # LIF params
    DECAY: tl.constexpr,
    RECIP_TAU: tl.constexpr,
    V_TH: tl.constexpr,
    HAS_V_RESET: tl.constexpr,
    V_RESET: tl.constexpr,
    # Sparse params
    GROUP_SIZE_C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    # Block dims
    BLOCK_N: tl.constexpr,
):
    BLOCK_H: tl.constexpr = 8
    BLOCK_W: tl.constexpr = 16
    BLOCK_M: tl.constexpr = BLOCK_H * BLOCK_W  # 128

    # Each program: one (batch_idx, gh, gw) tile × one output channel block
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = B_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH  # batch index within B

    # Output channel offsets
    offs_n = pid_cout * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    # Spatial offsets within tile
    offs_m = tl.arange(0, BLOCK_M)
    out_h = gh_idx * BLOCK_H + offs_m // BLOCK_W
    out_w = gw_idx * BLOCK_W + offs_m % BLOCK_W
    m_mask = (out_h < H_OUT) & (out_w < W_OUT)
    om = m_mask[:, None] & n_mask[None, :]

    # Strides
    HW: tl.constexpr = H * W
    HW_OUT: tl.constexpr = H_OUT * W_OUT
    C_IN_HW = C_IN * HW
    W_CS: tl.constexpr = C_IN
    W_KH: tl.constexpr = 3 * C_IN
    W_CO: tl.constexpr = 9 * C_IN

    # Output address template (within one timestep's slice)
    oa_base = (n_idx * C_OUT + offs_n[None, :]) * HW_OUT + out_h[:, None] * W_OUT + out_w[:, None]

    # BN params (constant across T, load once)
    if HAS_BN:
        bn_s = tl.load(bn_scale_ptr + offs_n, mask=n_mask, other=1.0)
        bn_b = tl.load(bn_bias_ptr + offs_n, mask=n_mask, other=0.0)

    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    # Load initial v [B, C_OUT, H_OUT, W_OUT]
    v = tl.load(v_init_ptr + oa_base, mask=om, other=0.0)

    # ================================================================
    # Main loop over T timesteps
    # ================================================================
    for t in range(T_val):
        # x base for this timestep: x[(t * B + n_idx), :, :, :]
        x_batch_offset = (t * B_val + n_idx) * C_IN_HW

        # Sparse conv3x3: inline zero-group detection
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)
                hw_ok = m_mask & h_ok & w_ok

                w_off = kh * W_KH + kw * W_CS

                for g in range(NUM_GROUPS):
                    cs = g * GROUP_SIZE_C
                    offs_k = cs + tl.arange(0, GROUP_SIZE_C)
                    k_m = offs_k < C_IN

                    # Load x tile: x[tb, c, ih, iw]
                    x_addrs = x_batch_offset + offs_k[None, :] * HW + safe_h[:, None] * W + safe_w[:, None]
                    x_tile = tl.load(x_ptr + x_addrs,
                                     mask=k_m[None, :] & hw_ok[:, None],
                                     other=0.0)

                    # Inline zero-detection: skip dot product if all zeros
                    has_nonzero = tl.sum(x_tile) > 0.0
                    if has_nonzero:
                        x_f16 = x_tile.to(tl.float16)
                        w_addrs = offs_n[None, :] * W_CO + w_off + offs_k[:, None]
                        w_tile = tl.load(w_cl_ptr + w_addrs,
                                         mask=k_m[:, None] & n_mask[None, :],
                                         other=0.0).to(tl.float16)
                        acc += tl.dot(x_f16, w_tile)

        # BN affine (in registers, z never touches HBM)
        if HAS_BIAS:
            acc += bias_val[None, :]
        if HAS_BN:
            acc = acc * bn_s[None, :] + bn_b[None, :]

        # LIF dynamics (v in registers, carried across steps)
        v = v * DECAY + acc * RECIP_TAU + V_RESET * RECIP_TAU
        sp = (v >= V_TH).to(tl.float32)
        if HAS_V_RESET:
            v = v * (1.0 - sp) + V_RESET * sp
        else:
            v = v - sp * V_TH

        # Store spike for this timestep
        spike_offset = t * B_val * C_OUT * HW_OUT
        tl.store(spike_ptr + spike_offset + oa_base, sp, mask=om)

    # Store final v
    tl.store(v_final_ptr + oa_base, v, mask=om)
