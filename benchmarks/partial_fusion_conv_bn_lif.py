"""
Phase 2 Task 2.1b — PartialFusionConvBNLIF

System core module #2: cuDNN conv (via F.conv2d) + Triton-fused (BN -> LIF)
with StreamFuse between BN and LIF + StateCarry on LIF.

Motivation:
  Same as task 2.1a (compute-bound Conv must delegate to cuDNN), with the
  additional StreamFuse(BN, LIF) optimization: BN's affine transform is
  applied to z inside the LIF kernel, so z_bn never materializes to HBM.
  In the reference execution, BN reads z, writes z_bn, then LIF reads z_bn —
  this is 3 |step| of BN-related HBM traffic per time step. We eliminate the
  z_bn write+read pair (2 |step| per step) by folding BN affine into the LIF
  kernel.

Affine fusion:
  Standard inference-mode BN:
    z_bn = (z - running_mean) / sqrt(running_var + eps) * weight + bias
  We precompute outside the kernel:
    fused_scale = weight / sqrt(running_var + eps)        [C]
    fused_bias  = bias - running_mean * fused_scale       [C]
  Then inside the kernel:
    z_bn = z * fused_scale[c] + fused_bias[c]
  This is the standard BN-fold optimization for inference, mathematically
  equivalent to the original formulation in fp32 (single FMA instead of
  subtract+divide+multiply+add).

Schedule (formal):
  TimeBlock(T) o StreamFuse(BN, LIF) o StateCarry(LIF)
  - Conv stays as a separate cuDNN call (TSI, not fused with BN/LIF)
  - BN -> LIF is StreamFused: z_bn never materializes
  - LIF state v carried in register across all T time steps within one program

Analytic HBM ratio:
  Reference (per time step, includes BN HBM):
    (1) Conv writes z                                          1 |step|
    (2) BN reads z                                             1 |step|
    (3) BN writes z_bn                                         1 |step|
    (4) LIF reads z_bn                                         1 |step|
    (5) LIF reads v_{t-1}                                      1 |step|
    (6) LIF writes v_t                                         1 |step|
    (7) LIF writes s_t                                         1 |step|
  Total: 7 T |step|

  PartialFusionConvBNLIF:
    (1) Conv writes z (cuDNN unavoidable)                      1 T |step|
    (2) Triton kernel reads z                                  1 T |step|
    (3) BN affine fused inside Triton kernel (no HBM)          0
    (4) LIF state v in register, only at K-block boundaries    2 T/K |step|
    (5) Triton kernel writes s                                 1 T |step|
  Total: (3 + 2/K) T |step|

  Ratio = (3 + 2/K) / 7
  K=T=16: 3.125 / 7 = 0.446 (saves 55.4%)
  K=4:    3.5 / 7   = 0.500 (saves 50.0%)
  K->inf: 3 / 7     = 0.429 (saves 57.1%)

  Compare to PartialFusionConvLIF (no BN): ratio = (3 + 2/K)/5 = 0.625 at K=T
  Adding BN to the chain INCREASES the relative savings because BN's HBM
  contribution (z_bn write+read) is now eliminated by StreamFuse.

  Compare to full Triton ConvBNLIF (Phase 0 data): ratio 0.3439 (saves 65.6%)
  Full Triton saves more HBM but loses ~5x in wall-clock on V100 (Phase 0 V100
  speedup 0.608x).

Parity expectation:
  - vs SJ torch (BN -> LIF in eager mode): expect bit-exact OR a small number
    of single-spike v_th boundary flips (10s out of 100M+), depending on
    whether SJ's BN inference path produces identical FMA order to our fused
    affine (z * fused_scale + fused_bias). In fp32 these differ at most by
    one ULP, which only flips spikes at v_th boundaries.

Design decisions (locked 2026-04-14 evening):
  - F.conv2d (not nn.Conv2d module)
  - Internal fused affine precomputation (option a)
  - Forward accepts raw BN params (option 2): bn_weight, bn_bias,
    running_mean, running_var, eps — caller copies directly from sj_bn module
  - decay_input=True only
  - K=T single block (K-sweep deferred to task 2.2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Triton kernel: fused (BN affine -> LIF -> StateCarry)
# ============================================================
#
# Input:  z [T, B, C_out, H_out, W_out]  (cuDNN conv output, in HBM)
#         fused_scale [C_out]
#         fused_bias  [C_out]
# Output: s [T, B, C_out, H_out, W_out]
#
# Layout: we flatten the per-step tensor to [N] where N = B * C_out * H * W,
# but we need to know which channel each element belongs to for the affine.
# Channel index for element n: c = (n // (H*W)) % C_out
# We pass H*W and C_out to the kernel so it can compute c via integer division.
#
# This is a simple flat layout. A more cache-friendly layout would keep
# channels contiguous and load fused_scale[c] once per BLOCK_N if we tile by
# channel; for task 2.1b we use the simplest correct layout, optimization
# deferred.
# ============================================================

@triton.jit
def _bn_lif_state_carry_kernel(
    z_ptr,              # [T, N]
    s_ptr,              # [T, N]
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
    # n = b * C * HW + c * HW + h * W + w
    # c = (n // HW) % C
    c_idx = (offs_n // HW) % C

    # Load fused affine for these channels.
    scale = tl.load(fused_scale_ptr + c_idx, mask=mask_n, other=1.0)
    bias = tl.load(fused_bias_ptr + c_idx, mask=mask_n, other=0.0)

    # v starts at 0 (SJ default initial membrane potential)
    v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        z = tl.load(z_ptr + t * N + offs_n, mask=mask_n, other=0.0)

        # BN affine (fused, in register)
        z_bn = z * scale + bias

        # LIF update with decay_input=True
        v = v * one_minus_tau_inv + z_bn * tau_inv

        # Spike + hard reset (v_reset=0 assumed)
        spike = (v >= v_th).to(tl.float32)
        v = v * (1.0 - spike)

        tl.store(s_ptr + t * N + offs_n, spike, mask=mask_n)


def _launch_bn_lif_state_carry(
    z: torch.Tensor,         # [T, B, C, H, W]
    fused_scale: torch.Tensor,  # [C]
    fused_bias: torch.Tensor,   # [C]
    tau: float,
    v_th: float,
) -> torch.Tensor:
    assert z.is_cuda and z.dtype == torch.float32 and z.is_contiguous()
    assert fused_scale.is_contiguous() and fused_bias.is_contiguous()
    assert fused_scale.dtype == torch.float32 and fused_bias.dtype == torch.float32

    T, B, C, H, W = z.shape
    HW = H * W
    N = B * C * H * W
    s = torch.empty_like(z)

    BLOCK_N = 1024
    grid = (triton.cdiv(N, BLOCK_N),)

    _bn_lif_state_carry_kernel[grid](
        z, s,
        fused_scale, fused_bias,
        T, N, C, HW,
        1.0 / tau,
        1.0 - 1.0 / tau,
        v_th,
        BLOCK_N=BLOCK_N,
    )
    return s


# ============================================================
# BN affine precomputation (inference mode, fused)
# ============================================================

def _precompute_fused_affine(
    bn_weight: torch.Tensor,    # gamma [C]
    bn_bias: torch.Tensor,      # beta [C]
    running_mean: torch.Tensor, # [C]
    running_var: torch.Tensor,  # [C]
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard BN inference fold:
      z_bn = (z - mean) / sqrt(var + eps) * gamma + beta
           = z * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps))
           = z * fused_scale + fused_bias
    """
    inv_std = torch.rsqrt(running_var + eps)
    fused_scale = bn_weight * inv_std
    fused_bias = bn_bias - running_mean * fused_scale
    return fused_scale.contiguous(), fused_bias.contiguous()


# ============================================================
# Public API: partial_fusion_conv_bn_lif
# ============================================================

def partial_fusion_conv_bn_lif(
    x: torch.Tensor,            # [T, B, C_in, H, W]
    w_conv: torch.Tensor,       # [C_out, C_in, k, k]
    bn_weight: torch.Tensor,    # gamma [C_out]
    bn_bias: torch.Tensor,      # beta [C_out]
    running_mean: torch.Tensor, # [C_out]
    running_var: torch.Tensor,  # [C_out]
    eps: float = 1e-5,
    tau: float = 2.0,
    v_th: float = 1.0,
    v_reset: float = 0.0,
    stride: int = 1,
    padding: int = 1,
    K_block: int | None = None,  # K=T always for task 2.1b
) -> torch.Tensor:
    """
    Partial fusion Conv -> BN -> LIF: cuDNN conv + Triton (BN -> LIF) with
    StreamFuse and StateCarry.

    Args:
        x: input [T, B, C_in, H, W], fp32, contiguous, CUDA
        w_conv: conv weight [C_out, C_in, k, k]
        bn_weight, bn_bias: BN gamma/beta [C_out]
        running_mean, running_var: BN running stats [C_out]
        eps: BN epsilon
        tau, v_th, v_reset: LIF parameters (decay_input=True)
        stride, padding: conv2d parameters
        K_block: time block size, currently ignored (K=T always)

    Returns:
        s: spike output [T, B, C_out, H_out, W_out]
    """
    assert x.ndim == 5
    assert w_conv.ndim == 4
    assert v_reset == 0.0, "task 2.1b: only v_reset=0 hard reset supported"

    T, B, C_in, H, W_spatial = x.shape
    C_out = w_conv.shape[0]

    # Step 1: cuDNN conv via F.conv2d
    x_4d = x.view(T * B, C_in, H, W_spatial)
    z_4d = F.conv2d(x_4d, w_conv, bias=None, stride=stride, padding=padding)
    H_out, W_out = z_4d.shape[2], z_4d.shape[3]
    z = z_4d.view(T, B, C_out, H_out, W_out).contiguous()

    # Step 2: precompute fused BN affine
    fused_scale, fused_bias = _precompute_fused_affine(
        bn_weight, bn_bias, running_mean, running_var, eps
    )

    # Step 3: Triton kernel — BN affine + LIF + StateCarry, all fused
    s = _launch_bn_lif_state_carry(z, fused_scale, fused_bias, tau=tau, v_th=v_th)

    return s


# ============================================================
# Analytic HBM traffic ratio
# ============================================================

def analytic_hbm_ratio_conv_bn_lif(K: int) -> float:
    """
    PartialFusionConvBNLIF HBM ratio vs reference (7T |step|).

    Reference: 7T |step| (Conv writes z, BN reads/writes z_bn, LIF reads z_bn,
                          v read+write, s write)
    Hybrid:    (3 + 2/K) T |step|

    K=T=16: 3.125/7 = 0.446 (saves 55.4%)
    K=4:    3.5/7   = 0.500 (saves 50.0%)
    """
    return (3.0 + 2.0 / K) / 7.0


# ============================================================
# Self-test / smoke test
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--B', type=int, default=64)
    parser.add_argument('--C_in', type=int, default=128)
    parser.add_argument('--C_out', type=int, default=128)
    parser.add_argument('--H', type=int, default=28)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--v_th', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    # Build input + weights
    x = (torch.randn(args.T, args.B, args.C_in, args.H, args.H,
                     device=device, dtype=torch.float32) * 0.3).contiguous()
    w_conv = (torch.randn(args.C_out, args.C_in, args.k, args.k,
                          device=device, dtype=torch.float32) * 0.3).contiguous()

    # Realistic-ish BN parameters for inference
    bn_weight = (1.0 + torch.randn(args.C_out, device=device, dtype=torch.float32) * 0.1).contiguous()
    bn_bias = (torch.randn(args.C_out, device=device, dtype=torch.float32) * 0.1).contiguous()
    running_mean = (torch.randn(args.C_out, device=device, dtype=torch.float32) * 0.05).contiguous()
    running_var = (1.0 + torch.randn(args.C_out, device=device, dtype=torch.float32) * 0.05).abs().contiguous()
    eps = 1e-5

    # Run PartialFusionConvBNLIF
    s_hybrid = partial_fusion_conv_bn_lif(
        x, w_conv,
        bn_weight, bn_bias, running_mean, running_var, eps=eps,
        tau=args.tau, v_th=args.v_th, v_reset=0.0,
        stride=args.stride, padding=args.padding,
    )
    print(f"[smoke] input shape:  {list(x.shape)}")
    print(f"[smoke] output shape: {list(s_hybrid.shape)}")
    print(f"[smoke] spike rate:   {s_hybrid.mean().item():.4f}")

    # Build SJ reference: layer.Conv2d + layer.BatchNorm2d + LIFNode
    from spikingjelly.activation_based import layer, neuron, functional
    net = torch.nn.Sequential(
        layer.Conv2d(args.C_in, args.C_out, kernel_size=args.k,
                     stride=args.stride, padding=args.padding, bias=False),
        layer.BatchNorm2d(args.C_out),
        neuron.LIFNode(tau=args.tau, v_threshold=args.v_th, v_reset=0.0,
                       step_mode='m'),
    )
    functional.set_step_mode(net, 'm')
    functional.set_backend(net, 'torch')
    net = net.to(device)
    net[0].weight.data.copy_(w_conv)
    # Inject the same BN parameters into SJ's BN module
    bn_module = net[1]
    bn_module.eval()
    bn_module.weight.data.copy_(bn_weight)
    bn_module.bias.data.copy_(bn_bias)
    bn_module.running_mean.data.copy_(running_mean)
    bn_module.running_var.data.copy_(running_var)
    bn_module.eps = eps

    with torch.no_grad():
        functional.reset_net(net)
        s_sj = net(x)

    # Parity report
    spike_match = (s_hybrid == s_sj).float().mean().item() * 100
    spike_flips = (s_hybrid != s_sj).sum().item()
    total_spikes = s_hybrid.numel()
    print(f"\n[parity] vs SJ torch Conv->BN->LIF:")
    print(f"  spike_match:       {spike_match:.6f}%")
    print(f"  spike_flips:       {spike_flips} / {total_spikes}")
    print(f"  spike_rate_hybrid: {s_hybrid.mean().item():.6f}")
    print(f"  spike_rate_sj:     {s_sj.mean().item():.6f}")

    if spike_match > 99.9:
        print(f"  PARITY: PASS (spike_match > 99.9%)")
    else:
        print(f"  PARITY: FAIL (spike_match = {spike_match:.4f}%)")

    # Analytic HBM ratios
    print(f"\n[hbm] analytic hybrid ratio at K=T={args.T}: "
          f"{analytic_hbm_ratio_conv_bn_lif(args.T):.4f}")
    print(f"[hbm] (for comparison) PartialFusionConvLIF K=T: "
          f"{(3 + 2/args.T) / 5:.4f}")
    print(f"[hbm] (for comparison) Phase 0 full Triton ConvBNLIF: 0.3439")