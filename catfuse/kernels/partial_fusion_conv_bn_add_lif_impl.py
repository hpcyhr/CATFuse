"""
Phase 3 task 3.2c / Policy row 9 — PartialFusionConvBNAddLIF

Supports the standard spiking_resnet BasicBlock's second half:
    out = conv2(x_prev_spike)
    out = bn2(out)
    out += identity              # Add BEFORE LIF
    out = sn2(out)               # LIF LAST

where `identity` is:
  - The raw block input `x` if no downsample (shape matches conv2 output)
  - `self.downsample(x)` (Sequential(Conv2d, BatchNorm2d)) if spatial reduction
    is needed. The downsample path does NOT have a LIF — identity is continuous.

Mathematics (in inference mode):
    z = conv2(x_in)
    z_bn = (z - running_mean) / sqrt(running_var + eps) * gamma + beta
         = z * fused_scale + fused_bias
    z_add = z_bn + identity
    v = v * (1 - 1/tau) + z_add * (1/tau)
    spike = (v >= v_th).float()
    v = v * (1 - spike)

Schedule (formal):
    TimeBlock(T) o StreamFuse(BN, Add, LIF) o StateCarry(LIF)

  - Conv2 stays as cuDNN (TSI, not fused with BN/Add/LIF)
  - BN affine + Add + LIF fused in one Triton kernel, StreamFuse
  - LIF state v carried in register across all T time steps

Analytic HBM ratio (per time step, vs reference):
  Reference: Conv writes z, BN reads z + writes z_bn, Add reads z_bn + identity
             + writes z_add, LIF reads z_add + v_{t-1} + writes v_t + s_t
             = 9 T |step| total

  PartialFusionConvBNAddLIF:
    (1) Conv writes z (cuDNN unavoidable)                     1 T |step|
    (2) Triton kernel reads z                                 1 T |step|
    (3) Triton kernel reads identity                          1 T |step|
    (4) BN affine + Add fused inside kernel (no HBM)          0
    (5) LIF state v in register                               2 T/K |step|
    (6) Triton kernel writes s                                1 T |step|
  Total: (4 + 2/K) T |step|

  Ratio = (4 + 2/K) / 9
  K=T=16: 4.125/9 = 0.458 (saves 54.2%)
  K->inf: 4/9 = 0.444 (saves 55.6%)

Parity expectation:
  - vs SJ torch (Conv2->BN->Add->LIF eager): bit-exact or small number of v_th
    boundary flips. BN affine and Add are linear operations, FMA-order reordering
    gives at most 1 ULP per element, which only flips spikes at exact v_th
    boundaries.

Design decisions (locked 2026-04-15 morning):
  - Based directly on PartialFusionConvBNLIF (task 2.1b) kernel structure
  - Identity tensor expected in same [T, B, C_out, H_out, W_out] shape as z
  - Caller responsible for computing identity (either raw x reshape or
    downsample(x) result). This keeps the fused kernel simple.
  - v_reset=0 hard reset only (consistent with other CTF patterns)
  - Padding, stride passed through to F.conv2d
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Triton kernel: fused (BN affine -> Add identity -> LIF -> StateCarry)
# ============================================================

@triton.jit
def _bn_add_lif_state_carry_kernel(
    z_ptr,              # [T, N]  (cuDNN conv output)
    identity_ptr,       # [T, N]  (pre-computed identity tensor)
    s_ptr,              # [T, N]  (spike output)
    fused_scale_ptr,    # [C]
    fused_bias_ptr,     # [C]
    T,
    N,                  # = B * C * H * W per time step
    C,
    HW,                 # = H * W
    tau_inv,
    one_minus_tau_inv,
    v_th,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Compute channel index for each element in the block.
    c_idx = (offs_n // HW) % C

    # Load fused affine for these channels.
    scale = tl.load(fused_scale_ptr + c_idx, mask=mask_n, other=1.0)
    bias = tl.load(fused_bias_ptr + c_idx, mask=mask_n, other=0.0)

    # v starts at 0
    v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        z = tl.load(z_ptr + t * N + offs_n, mask=mask_n, other=0.0)
        identity = tl.load(identity_ptr + t * N + offs_n, mask=mask_n, other=0.0)

        # BN affine (fused, in register)
        z_bn = z * scale + bias

        # Add identity (fused, in register)
        z_add = z_bn + identity

        # LIF update with decay_input=True
        v = v * one_minus_tau_inv + z_add * tau_inv

        # Spike + hard reset
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike)

        tl.store(s_ptr + t * N + offs_n, spike, mask=mask_n)


def _launch_bn_add_lif_state_carry(
    z: torch.Tensor,            # [T, B, C, H, W]
    identity: torch.Tensor,     # [T, B, C, H, W] — same shape as z
    fused_scale: torch.Tensor,  # [C]
    fused_bias: torch.Tensor,   # [C]
    tau: float,
    v_th: float,
) -> torch.Tensor:
    assert z.is_cuda and z.dtype == torch.float32 and z.is_contiguous()
    assert identity.is_cuda and identity.dtype == torch.float32
    assert identity.is_contiguous()
    assert z.shape == identity.shape, f"z {z.shape} vs identity {identity.shape}"
    assert fused_scale.is_contiguous() and fused_bias.is_contiguous()
    assert fused_scale.dtype == torch.float32 and fused_bias.dtype == torch.float32

    T, B, C, H, W = z.shape
    HW = H * W
    N = B * C * H * W
    s = torch.empty_like(z)

    BLOCK_N = 1024
    grid = (triton.cdiv(N, BLOCK_N),)

    _bn_add_lif_state_carry_kernel[grid](
        z, identity, s,
        fused_scale, fused_bias,
        T, N, C, HW,
        1.0 / tau,
        1.0 - 1.0 / tau,
        v_th,
        BLOCK_N=BLOCK_N,
    )
    return s


# ============================================================
# BN affine precomputation (same as task 2.1b)
# ============================================================

def _precompute_fused_affine(
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_std = torch.rsqrt(running_var + eps)
    fused_scale = bn_weight * inv_std
    fused_bias = bn_bias - running_mean * fused_scale
    return fused_scale.contiguous(), fused_bias.contiguous()


# ============================================================
# Public API: partial_fusion_conv_bn_add_lif
# ============================================================

def partial_fusion_conv_bn_add_lif(
    x: torch.Tensor,            # [T, B, C_in, H, W]
    identity: torch.Tensor,     # [T, B, C_out, H_out, W_out] — already computed
    w_conv: torch.Tensor,       # [C_out, C_in, k, k]
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    tau: float = 2.0,
    v_th: float = 1.0,
    v_reset: float = 0.0,
    stride: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    """
    Partial fusion Conv -> BN -> Add(identity) -> LIF.

    The identity tensor must be pre-computed by the caller (either the raw
    block input x, or the downsample(x) result). Its shape must match the
    conv2 output after BN.

    Args:
        x: input [T, B, C_in, H, W]
        identity: pre-computed identity [T, B, C_out, H_out, W_out]
        w_conv: conv weight [C_out, C_in, k, k]
        bn_weight, bn_bias, running_mean, running_var: BN params
        eps, tau, v_th, v_reset: BN + LIF params
        stride, padding: conv params

    Returns:
        s: spike output [T, B, C_out, H_out, W_out]
    """
    assert x.ndim == 5
    assert identity.ndim == 5
    assert w_conv.ndim == 4
    assert v_reset == 0.0, "only v_reset=0 hard reset supported"

    T, B, C_in, H, W_spatial = x.shape
    C_out = w_conv.shape[0]

    # Step 1: cuDNN conv
    x_4d = x.view(T * B, C_in, H, W_spatial)
    z_4d = F.conv2d(x_4d, w_conv, bias=None, stride=stride, padding=padding)
    H_out, W_out = z_4d.shape[2], z_4d.shape[3]
    z = z_4d.view(T, B, C_out, H_out, W_out).contiguous()

    # Check identity shape matches z shape
    assert identity.shape == z.shape, \
        f"identity shape {identity.shape} must match z shape {z.shape}"

    identity_c = identity.contiguous() if not identity.is_contiguous() else identity

    # Step 2: precompute fused BN affine
    fused_scale, fused_bias = _precompute_fused_affine(
        bn_weight, bn_bias, running_mean, running_var, eps
    )

    # Step 3: Triton kernel — BN + Add + LIF + StateCarry
    s = _launch_bn_add_lif_state_carry(
        z, identity_c, fused_scale, fused_bias, tau=tau, v_th=v_th,
    )

    return s


# ============================================================
# Smoke test
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--B', type=int, default=32)
    parser.add_argument('--C', type=int, default=128)
    parser.add_argument('--H', type=int, default=28)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v_th', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    # Build input + weights. identity has same shape as conv output = same C, H, W as input (stride=1).
    x = (torch.randn(args.T, args.B, args.C, args.H, args.H,
                     device=device, dtype=torch.float32) * 0.5).contiguous()
    identity = (torch.randn(args.T, args.B, args.C, args.H, args.H,
                            device=device, dtype=torch.float32) * 0.5).contiguous()
    w_conv = (torch.randn(args.C, args.C, 3, 3,
                          device=device, dtype=torch.float32) * 0.3).contiguous()

    bn_weight = (1.0 + torch.randn(args.C, device=device, dtype=torch.float32) * 0.1).contiguous()
    bn_bias = (torch.randn(args.C, device=device, dtype=torch.float32) * 0.1).contiguous()
    running_mean = (torch.randn(args.C, device=device, dtype=torch.float32) * 0.05).contiguous()
    running_var = (1.0 + torch.randn(args.C, device=device, dtype=torch.float32) * 0.05).abs().contiguous()
    eps = 1e-5

    # Run fused
    s_hybrid = partial_fusion_conv_bn_add_lif(
        x, identity, w_conv,
        bn_weight, bn_bias, running_mean, running_var, eps=eps,
        tau=args.tau, v_th=args.v_th, v_reset=0.0,
        stride=1, padding=1,
    )
    print(f"[smoke] input:    {list(x.shape)}")
    print(f"[smoke] identity: {list(identity.shape)}")
    print(f"[smoke] output:   {list(s_hybrid.shape)}")
    print(f"[smoke] spike_rate: {s_hybrid.mean().item():.4f}")

    # Build SJ reference: Conv -> BN -> Add -> LIF (manual Add because SJ
    # doesn't have a built-in multi-step Add+LIF in one Sequential)
    from spikingjelly.activation_based import layer, neuron, functional
    conv_ref = layer.Conv2d(args.C, args.C, kernel_size=3, stride=1,
                            padding=1, bias=False).to(device)
    bn_ref = layer.BatchNorm2d(args.C).to(device)
    lif_ref = neuron.LIFNode(tau=args.tau, v_threshold=args.v_th, v_reset=0.0,
                             step_mode='m').to(device)
    functional.set_step_mode(conv_ref, 'm')
    functional.set_step_mode(bn_ref, 'm')
    functional.set_step_mode(lif_ref, 'm')

    conv_ref.weight.data.copy_(w_conv)
    bn_ref.eval()
    bn_ref.weight.data.copy_(bn_weight)
    bn_ref.bias.data.copy_(bn_bias)
    bn_ref.running_mean.data.copy_(running_mean)
    bn_ref.running_var.data.copy_(running_var)
    bn_ref.eps = eps

    with torch.no_grad():
        z = conv_ref(x)
        z_bn = bn_ref(z)
        z_add = z_bn + identity
        functional.reset_net(lif_ref)
        s_sj = lif_ref(z_add)

    # Parity
    spike_match = (s_hybrid == s_sj).float().mean().item() * 100
    spike_flips = (s_hybrid != s_sj).sum().item()
    total = s_hybrid.numel()
    print(f"\n[parity] vs SJ torch Conv->BN->Add->LIF:")
    print(f"  spike_match: {spike_match:.6f}%")
    print(f"  spike_flips: {spike_flips} / {total}")
    print(f"  spike_rate_hybrid: {s_hybrid.mean().item():.6f}")
    print(f"  spike_rate_sj:     {s_sj.mean().item():.6f}")

    if spike_match > 99.9:
        print(f"  PARITY: PASS")
    else:
        print(f"  PARITY: FAIL (match = {spike_match:.4f}%)")
