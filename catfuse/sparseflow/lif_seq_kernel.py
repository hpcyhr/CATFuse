"""
Sequential LIF kernel: processes z[T, B, C, H, W] → spike[T, B, C, H, W]
with causal v dependency, in a single Triton kernel launch.

Each program handles one (b, c, h, w) position across all T steps.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lif_seq_kernel(
    z_ptr,         # [T, B, C, H, W] input (conv+BN output)
    v_in_ptr,      # [B, C, H, W] initial membrane potential
    spike_ptr,     # [T, B, C, H, W] output spikes
    v_out_ptr,     # [B, C, H, W] final membrane potential
    T: tl.constexpr,
    BCHW,          # B * C * H * W (stride for T dim)
    DECAY: tl.constexpr,
    RECIP_TAU: tl.constexpr,
    V_TH: tl.constexpr,
    V_RESET: tl.constexpr,
    HAS_V_RESET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < BCHW

    # Load initial v
    v = tl.load(v_in_ptr + offs, mask=mask, other=0.0)

    # Sequential LIF over T steps
    for t in tl.static_range(T):
        z = tl.load(z_ptr + t * BCHW + offs, mask=mask, other=0.0)
        v = v * DECAY + z * RECIP_TAU + V_RESET * RECIP_TAU
        spike = (v >= V_TH).to(tl.float32)

        if HAS_V_RESET:
            v = v * (1.0 - spike) + V_RESET * spike
        else:
            v = v - spike * V_TH

        tl.store(spike_ptr + t * BCHW + offs, spike, mask=mask)

    # Store final v
    tl.store(v_out_ptr + offs, v, mask=mask)


def lif_sequential(
    z: torch.Tensor,       # [T, B, C, H, W]
    v_init: torch.Tensor,  # [B, C, H, W]
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> tuple:
    """Single-launch LIF over T steps.

    Returns:
        (spikes, v_final) — spikes is [T, B, C, H, W], v_final is [B, C, H, W]
    """
    T = z.shape[0]
    BCHW = z[0].numel()
    device = z.device

    spike_out = torch.empty_like(z)
    v_out = torch.empty_like(v_init)
    v_in = v_init.float().contiguous()
    z_contig = z.float().contiguous()

    decay = 1.0 - 1.0 / tau
    recip_tau = 1.0 / tau
    has_v_reset = v_reset is not None
    v_reset_val = 0.0 if v_reset is None else float(v_reset)

    BLOCK = 1024
    grid = (triton.cdiv(BCHW, BLOCK),)

    _lif_seq_kernel[grid](
        z_contig, v_in, spike_out, v_out,
        T=T, BCHW=BCHW,
        DECAY=decay, RECIP_TAU=recip_tau,
        V_TH=v_threshold,
        V_RESET=v_reset_val,
        HAS_V_RESET=has_v_reset,
        BLOCK=BLOCK,
    )

    return spike_out, v_out
