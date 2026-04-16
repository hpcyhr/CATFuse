"""
Phase 2 Task 2.1a — PartialFusionConvLIF

System core module #1: cuDNN conv (via F.conv2d) + Triton-fused LIF with StateCarry.

Motivation (Path 1 / system framing):
  Phase 1 showed that fusing conv into a Triton kernel loses to SJ torch's cuDNN
  path by ~5x on V100 sm_70. Full Triton fusion is the wrong realization for the
  compute-bound Conv regime. The CTF framework allows any valid schedule in
  Sigma(G, T); delegating the Conv TSI to cuDNN while keeping the LIF CSR in a
  Triton kernel with StateCarry is still a valid schedule — it is:

     TimeBlock(K) o StateCarry(LIF)

  without StreamFuse. The intermediate z_t DOES go through HBM (cuDNN writes
  it, Triton reads it once), but we save:
    - the LIF-side k_launch per time step (T launches collapsed to 1)
    - the LIF-side read of v_{t-1} from HBM (carried in registers across steps
      within a block, only written at block boundaries)
    - the LIF-side write of v_t from HBM (same reason)

Analytic HBM ratio for this hybrid vs reference (5T |step|):
  - Ref path: z write, z read (LIF), v_{t-1} read, v_t write, s_t write = 5T
  - Hybrid:   z write (cuDNN), z read (Triton LIF), 2(T/K) for v boundary, T for s
               = T + T + 2T/K + T = 3T + 2T/K
  - ratio = (3 + 2/K) / 5
  - K=1: 5/5 = 1.00  (no savings from TimeBlock, but still save LIF launches)
  - K=2: 4/5 = 0.80
  - K=4: 3.5/5 = 0.70
  - K=8: 3.25/5 = 0.65
  - K=T=16: 3.125/5 = 0.625
  - K->inf: 3/5 = 0.60

Compare to full Triton fusion: ratio = (1 + 2/K)/5, K=4 gives 0.30.
Hybrid saves less HBM (0.70 vs 0.30 at K=4) but wins wall-clock because cuDNN
conv is ~5x faster per FLOP than Triton implicit-GEMM on V100.

Semantic preservation:
  This realization is TimeBlock(K) o StateCarry(LIF) applied to the reference
  schedule. By Lemma 3.14 (basic transforms preserve Sigma), the resulting
  schedule is in Sigma(G, T), and by Theorem 3.9 is semantically equivalent to
  the reference execution. Parity against SJ torch is expected to be
  spike_match > 99.9% (single-spike ULP flips possible at v_th boundaries due
  to cuDNN algorithm choice differences in the conv path).

Design decisions (locked 2026-04-14 evening):
  - F.conv2d (not nn.Conv2d module) — weights injected externally for Phase 3 substitution
  - decay_input=True only (SJ default); other variants added later as needed
  - Parity target: SJ torch backend Conv->LIF
  - Wall-clock baselines: SJ torch, SJ cupy, Phase 0 FusedConvLIF

Usage:
    x = torch.randn(T, B, C_in, H, W, device='cuda')
    w = torch.randn(C_out, C_in, k, k, device='cuda')
    y = partial_fusion_conv_lif(
        x, w,
        tau=2.0, v_th=1.0, v_reset=0.0,
        stride=1, padding=1, K_block=4,
    )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Triton kernel: LIF with StateCarry over a time-block
# ============================================================
#
# Input:  z [T, B, C_out, H_out, W_out]  (cuDNN conv output, already in HBM)
# Output: s [T, B, C_out, H_out, W_out]  (spikes)
#
# We launch a 1D grid over the flat spatial*channel*batch index, and each
# program instance iterates through all T time steps, carrying v in a register.
# This is the simplest realization of TimeBlock(K=T) o StateCarry(LIF) — a
# single block covering the whole time dimension.
#
# For K < T (multi-block) we could launch T/K blocks along a separate dim and
# reload v at block boundaries. We support K as a parameter but for task 2.1a
# we always use K=T (single block, simplest case). K-sweep deferred to Phase 2
# task 2.2 (ablation).
#
# decay_input=True  =>  v_new = v + (z - v) / tau  =  v * (1 - 1/tau) + z / tau
# hard reset with v_reset=0  =>  v <- v * (1 - s) + v_reset * s = v * (1 - s)
# ============================================================

@triton.jit
def _lif_state_carry_kernel(
    z_ptr,          # input [T, N] flattened over (B, C_out, H_out, W_out)
    s_ptr,          # output [T, N] flattened
    T,              # int: number of time steps
    N,              # int: flattened spatial*channel*batch size
    tau_inv,        # float: 1.0 / tau
    one_minus_tau_inv,  # float: 1.0 - 1.0 / tau  (decay_input=True form)
    v_th,           # float
    v_reset,        # float (not used when we hard-reset to 0, but kept for generality)
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # v starts at 0 (SJ default initial membrane potential)
    v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate across the time dimension within this program instance.
    # Each iteration reads z_t from HBM, computes s_t, keeps v in register.
    for t in range(T):
        z = tl.load(z_ptr + t * N + offs_n, mask=mask_n, other=0.0)

        # LIF update: v = v * (1 - 1/tau) + z * (1/tau)   (decay_input=True)
        v = v * one_minus_tau_inv + z * tau_inv

        # Spike generation: s = 1 if v >= v_th else 0
        spike = (v >= v_th).to(tl.float32)

        # Hard reset: v <- v * (1 - s)   (v_reset=0 assumed)
        v = v * (1.0 - spike)

        tl.store(s_ptr + t * N + offs_n, spike, mask=mask_n)


def _launch_lif_state_carry(
    z: torch.Tensor,
    tau: float,
    v_th: float,
    v_reset: float,
) -> torch.Tensor:
    """
    Launch the Triton LIF kernel on z of shape [T, B, C_out, H_out, W_out].
    Returns s of the same shape.
    """
    assert z.is_cuda and z.dtype == torch.float32 and z.is_contiguous()
    assert v_reset == 0.0, "task 2.1a: only v_reset=0 hard reset supported"

    T = z.shape[0]
    N = z[0].numel()  # flattened size per time step
    s = torch.empty_like(z)

    BLOCK_N = 1024
    grid = (triton.cdiv(N, BLOCK_N),)

    _lif_state_carry_kernel[grid](
        z, s,
        T, N,
        1.0 / tau,
        1.0 - 1.0 / tau,
        v_th,
        v_reset,
        BLOCK_N=BLOCK_N,
    )
    return s


# ============================================================
# Public API: partial_fusion_conv_lif
# ============================================================

def partial_fusion_conv_lif(
    x: torch.Tensor,        # [T, B, C_in, H, W]
    w: torch.Tensor,        # [C_out, C_in, k, k]
    tau: float = 2.0,
    v_th: float = 1.0,
    v_reset: float = 0.0,
    stride: int = 1,
    padding: int = 1,
    K_block: int | None = None,  # unused in task 2.1a (K=T always); accepted for API stability
) -> torch.Tensor:
    """
    Partial fusion Conv->LIF: cuDNN conv + Triton LIF with StateCarry.

    Args:
        x: input [T, B, C_in, H, W], fp32, contiguous, CUDA
        w: weight [C_out, C_in, k, k], fp32, contiguous, CUDA
        tau, v_th, v_reset: LIF parameters (decay_input=True)
        stride, padding: Conv2d parameters (dilation=1, groups=1 assumed)
        K_block: time block size; task 2.1a uses K=T always (single block)

    Returns:
        s: spike output [T, B, C_out, H_out, W_out], fp32

    Semantic schedule: TimeBlock(T) o StateCarry(LIF) applied to the reference
    Conv->LIF schedule. Legal under Sigma(G, T) by Lemma 3.14.
    """
    assert x.ndim == 5, f"x must be [T, B, C, H, W], got shape {x.shape}"
    assert w.ndim == 4, f"w must be [C_out, C_in, k, k], got shape {w.shape}"
    assert x.is_cuda and x.dtype == torch.float32
    assert w.is_cuda and w.dtype == torch.float32
    assert x.is_contiguous() and w.is_contiguous()

    T, B, C_in, H, W_spatial = x.shape
    C_out = w.shape[0]

    # ----- Step 1: cuDNN conv via F.conv2d -----
    # Reshape x: [T, B, C, H, W] -> [T*B, C, H, W]
    x_4d = x.view(T * B, C_in, H, W_spatial)

    # cuDNN conv2d. No bias (policy: conv is followed by BN or LIF, bias folded elsewhere).
    z_4d = F.conv2d(x_4d, w, bias=None, stride=stride, padding=padding)

    # Reshape back: [T*B, C_out, H_out, W_out] -> [T, B, C_out, H_out, W_out]
    H_out, W_out = z_4d.shape[2], z_4d.shape[3]
    z = z_4d.view(T, B, C_out, H_out, W_out).contiguous()

    # ----- Step 2: Triton LIF + StateCarry (K=T single block) -----
    s = _launch_lif_state_carry(z, tau=tau, v_th=v_th, v_reset=v_reset)

    return s


# ============================================================
# Analytic HBM traffic estimator
# ============================================================

def analytic_hbm_ratio(K: int) -> float:
    """
    Hybrid (cuDNN conv + Triton LIF/StateCarry) HBM ratio vs reference.

    Reference: 5T |step|  (z write, z read, v read, v write, s write per step)
    Hybrid:    (3 + 2/K) T |step|  (z write, z read, 2T/K v boundary, s write)

    Task 2.1a always uses K=T, so K=T=16 => ratio = (3 + 2/16)/5 = 0.625
    """
    return (3.0 + 2.0 / K) / 5.0


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

    # Build input
    x = (torch.randn(args.T, args.B, args.C_in, args.H, args.H,
                     device=device, dtype=torch.float32) * 0.3).contiguous()
    w = (torch.randn(args.C_out, args.C_in, args.k, args.k,
                     device=device, dtype=torch.float32) * 0.3).contiguous()

    # Run PartialFusionConvLIF
    s_hybrid = partial_fusion_conv_lif(
        x, w,
        tau=args.tau, v_th=args.v_th, v_reset=0.0,
        stride=args.stride, padding=args.padding,
    )
    print(f"[smoke] input shape:  {list(x.shape)}")
    print(f"[smoke] output shape: {list(s_hybrid.shape)}")
    print(f"[smoke] spike rate:   {s_hybrid.mean().item():.4f}")
    print(f"[smoke] dtype:        {s_hybrid.dtype}")

    # Run SJ torch reference for parity
    from spikingjelly.activation_based import layer, neuron, functional
    net = torch.nn.Sequential(
        layer.Conv2d(args.C_in, args.C_out, kernel_size=args.k,
                     stride=args.stride, padding=args.padding, bias=False),
        neuron.LIFNode(tau=args.tau, v_threshold=args.v_th, v_reset=0.0,
                       step_mode='m'),
    )
    functional.set_step_mode(net, 'm')
    functional.set_backend(net, 'torch')
    net = net.to(device)
    net[0].weight.data.copy_(w)

    with torch.no_grad():
        functional.reset_net(net)
        s_sj = net(x)

    # Parity
    spike_match = (s_hybrid == s_sj).float().mean().item() * 100
    spike_flips = (s_hybrid != s_sj).sum().item()
    total_spikes = s_hybrid.numel()
    print(f"\n[parity] vs SJ torch Conv->LIF:")
    print(f"  spike_match:      {spike_match:.6f}%")
    print(f"  spike_flips:      {spike_flips} / {total_spikes}")
    print(f"  spike_rate_hybrid:{s_hybrid.mean().item():.6f}")
    print(f"  spike_rate_sj:    {s_sj.mean().item():.6f}")

    if spike_match > 99.9:
        print(f"  PARITY: PASS (spike_match > 99.9%)")
    else:
        print(f"  PARITY: FAIL (spike_match = {spike_match:.4f}%)")

    # Analytic HBM ratio
    print(f"\n[hbm] analytic hybrid ratio at K=T={args.T}: "
          f"{analytic_hbm_ratio(args.T):.4f}")
    print(f"[hbm] analytic hybrid ratio at K=4:        "
          f"{analytic_hbm_ratio(4):.4f}")
    print(f"[hbm] (for comparison) full-Triton K=4:    "
          f"{(1 + 2/4) / 5:.4f}")