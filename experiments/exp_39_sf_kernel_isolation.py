"""§3.6 framework debug — isolate SparseFlow kernel bug.

Loads the saved layer3.1.conv1 state from /tmp/sf_bug/, runs THREE
implementations on the SAME (input, weights):

  A. PyTorch reference: per-step Conv → BN-fold → multi-step LIF (SJ-equivalent)
  B. DenseKeep impl:    BatchFold conv (cuDNN) + lif_sequential (Triton)
  C. SparseFlow impl:   sparse_streamfuse_conv3x3_bn_lif (Triton, the buggy one)

Compares pairwise. The pair that DIFFERS isolates the bug:
  - A == B != C: bug is in SparseFlow kernel only
  - A != B != C: bug is broader, possibly in lif_sequential or shared
  - A != B == C: A is wrong (test artifact)
  - all match:  test setup wrong

Then drills into intermediate state of SF: dump v values at each timestep
to find the first timestep where SF's v diverges from reference v. Once
we know which timestep + which channel + which spatial position, we can
read SF kernel source and trace the bug.

Run:
    python experiments/exp_39_sf_kernel_isolation.py
"""
from __future__ import annotations

import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F


DEVICE = "cuda:0"
STATE_PATH = "/tmp/sf_bug/layer3_1_conv1_state.pt"


def reference_pytorch_forward(x, weight, bn_scale, bn_bias_folded, bias,
                               tau, v_threshold, v_reset, decay_input,
                               kernel_size, padding, stride):
    """Pure PyTorch reference implementation matching SJ's LIFNode m mode.

    LIF dynamics (SJ default, decay_input=True):
        v_charge = v_prev + (z - v_prev) / tau    if decay_input=True
        v_charge = v_prev * (1 - 1/tau) + z       if decay_input=False
        spike = (v_charge >= v_th).float()
        v_next = v_charge * (1 - spike) + v_reset * spike   if v_reset is not None
        v_next = (v_charge - v_th) * spike + v_charge * (1 - spike)  otherwise (soft reset)

    Note: SJ's LIFNode formula is:
        h = v + (x - (v - v_reset)) / tau     [decay_input=True, v_reset given]
        spike = surrogate(h - v_th)
        v = h * (1 - spike) + v_reset * spike
    """
    T, B, C_in, H, W = x.shape
    C_out = weight.shape[0]

    # 1. Conv per timestep
    x_4d = x.reshape(T * B, C_in, H, W)
    z_4d = F.conv2d(x_4d, weight, bias=bias, stride=stride, padding=padding)
    H_out, W_out = z_4d.shape[-2], z_4d.shape[-1]
    z = z_4d.reshape(T, B, C_out, H_out, W_out)

    # 2. BN fold (already pre-applied in our case — bn_scale * (z + bias))
    if bn_scale is not None:
        z = z * bn_scale.view(1, 1, -1, 1, 1) + bn_bias_folded.view(1, 1, -1, 1, 1)

    # 3. LIF over T steps
    v = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=z.dtype)
    spikes = []
    for t in range(T):
        # SJ LIFNode forward (matching neuron.LIFNode, hard reset, decay_input=True)
        if decay_input:
            if v_reset is not None:
                # h = v + (z - (v - v_reset)) / tau
                v_charge = v + (z[t] - (v - v_reset)) / tau
            else:
                v_charge = v + (z[t] - v) / tau
        else:
            if v_reset is not None:
                # h = v - (v - v_reset)/tau + z = v*(1 - 1/tau) + v_reset/tau + z
                v_charge = v * (1.0 - 1.0/tau) + v_reset / tau + z[t]
            else:
                v_charge = v * (1.0 - 1.0/tau) + z[t]

        spike = (v_charge >= v_threshold).to(z.dtype)

        if v_reset is not None:
            # hard reset
            v = v_charge * (1.0 - spike) + v_reset * spike
        else:
            # soft reset
            v = v_charge - spike * v_threshold

        spikes.append(spike)

    return torch.stack(spikes, dim=0), v


def main():
    if not os.path.exists(STATE_PATH):
        print(f"FATAL: {STATE_PATH} not found — run the save-state script first.")
        return 1

    print("=" * 96)
    print("SparseFlow kernel isolation test")
    print("=" * 96)

    state = torch.load(STATE_PATH, map_location=DEVICE)
    x = state['input'].to(DEVICE)
    weight = state['weight'].to(DEVICE)
    bn_scale = state['bn_scale'].to(DEVICE) if state['bn_scale'] is not None else None
    bn_bias = state['bn_bias_folded'].to(DEVICE) if state['bn_bias_folded'] is not None else None
    bias = state['bias'].to(DEVICE) if state['bias'] is not None else None
    tau = state['tau']
    v_threshold = state['v_threshold']
    v_reset = state['v_reset']
    decay_input = state['decay_input']
    T, B = state['T'], state['B']
    print(f"Loaded state: x.shape={tuple(x.shape)}, weight.shape={tuple(weight.shape)}")
    print(f"  tau={tau}  v_threshold={v_threshold}  v_reset={v_reset}  decay_input={decay_input}")
    print(f"  bn_scale range=[{bn_scale.min():.4f}, {bn_scale.max():.4f}]")

    # ============================================================
    # A. PyTorch reference
    # ============================================================
    print()
    print("A. PyTorch reference (per-step, hard-reset)")
    print("─" * 96)
    spike_ref, v_ref_final = reference_pytorch_forward(
        x, weight, bn_scale, bn_bias, bias,
        tau, v_threshold, v_reset, decay_input,
        kernel_size=state['kernel_size'],
        padding=state['padding'],
        stride=state['stride'],
    )
    print(f"  spike_ref shape={tuple(spike_ref.shape)}  "
          f"sum={spike_ref.sum().item():.0f}  "
          f"sparsity={1 - spike_ref.mean().item():.4f}")

    # ============================================================
    # B. DenseKeep
    # ============================================================
    print()
    print("B. DenseKeep impl")
    print("─" * 96)
    from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
    from spikingjelly.activation_based import (
        functional, neuron, layer as sj_layer
    )
    import torch.nn as nn

    # Build clean STFusion via from_sj_modules
    conv = sj_layer.Conv2d(state['in_channels'], state['out_channels'],
                           state['kernel_size'], padding=state['padding'],
                           stride=state['stride'], bias=False)
    conv.weight.data.copy_(state['weight'])
    bn = sj_layer.BatchNorm2d(state['out_channels'])
    bn.eval()
    bn.weight.data.fill_(1.0); bn.bias.data.fill_(0.0)
    bn.running_mean.fill_(0.0); bn.running_var.fill_(1.0); bn.eps = 0.0
    lif_node = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset,
                              step_mode='m')
    fused_dk = STFusionConvBNLIF.from_sj_modules(conv, bn, lif_node, K=state['K'])
    if state['bn_scale'] is not None:
        fused_dk.bn_scale.copy_(state['bn_scale'])
        fused_dk.bn_bias_folded.copy_(state['bn_bias_folded'])
    fused_dk = fused_dk.to(DEVICE).eval()
    fused_dk._impl_sparse = None  # force DK
    functional.reset_net(fused_dk)
    with torch.no_grad():
        spike_dk = fused_dk(x)
    print(f"  spike_dk shape={tuple(spike_dk.shape)}  "
          f"sum={spike_dk.sum().item():.0f}  "
          f"sparsity={1 - spike_dk.mean().item():.4f}")

    # ============================================================
    # C. SparseFlow
    # ============================================================
    print()
    print("C. SparseFlow impl (the buggy one)")
    print("─" * 96)
    fused_sf = STFusionConvBNLIF.from_sj_modules(conv, bn, lif_node, K=state['K'])
    if state['bn_scale'] is not None:
        fused_sf.bn_scale.copy_(state['bn_scale'])
        fused_sf.bn_bias_folded.copy_(state['bn_bias_folded'])
    fused_sf = fused_sf.to(DEVICE).eval()
    functional.reset_net(fused_sf)
    with torch.no_grad():
        spike_sf = fused_sf(x)
    print(f"  spike_sf shape={tuple(spike_sf.shape)}  "
          f"sum={spike_sf.sum().item():.0f}  "
          f"sparsity={1 - spike_sf.mean().item():.4f}")

    # ============================================================
    # Pairwise comparison
    # ============================================================
    print()
    print("=" * 96)
    print("Pairwise comparison")
    print("=" * 96)
    diff_AB = (spike_ref - spike_dk).abs().max().item()
    diff_AC = (spike_ref - spike_sf).abs().max().item()
    diff_BC = (spike_dk - spike_sf).abs().max().item()
    print(f"  A (Reference) vs B (DenseKeep):  max_diff = {diff_AB:.4e}")
    print(f"  A (Reference) vs C (SparseFlow): max_diff = {diff_AC:.4e}")
    print(f"  B (DenseKeep) vs C (SparseFlow): max_diff = {diff_BC:.4e}")

    # ============================================================
    # Per-timestep diff (find first time SF diverges from reference)
    # ============================================================
    print()
    print("Per-timestep divergence (SF vs Reference)")
    print("─" * 96)
    for t in range(T):
        diff_t = (spike_sf[t] - spike_ref[t]).abs().max().item()
        n_diff = (spike_sf[t] != spike_ref[t]).sum().item()
        if diff_t > 0:
            # Find one position
            mask = (spike_sf[t] != spike_ref[t])
            idx = mask.nonzero()[0].tolist()
            ref_val = spike_ref[t][tuple(idx)].item()
            sf_val = spike_sf[t][tuple(idx)].item()
            print(f"  t={t}: max_diff={diff_t:.4e}, n_diff={n_diff}, "
                  f"first_pos={idx} ref={ref_val} sf={sf_val}")
        else:
            print(f"  t={t}: bit-exact")

    # ============================================================
    # Per-channel diff
    # ============================================================
    print()
    print("Per-channel divergence count (SF vs Reference)")
    print("─" * 96)
    diff_per_ch = (spike_sf != spike_ref).sum(dim=(0, 1, 3, 4))  # [C_out]
    n_buggy_ch = (diff_per_ch > 0).sum().item()
    print(f"  {n_buggy_ch} / {diff_per_ch.numel()} output channels have divergent spikes")
    if n_buggy_ch > 0:
        # Top 5 buggy channels
        top = torch.topk(diff_per_ch, k=min(5, diff_per_ch.numel()))
        print(f"  Top diverging channels:")
        for ch, count in zip(top.indices.tolist(), top.values.tolist()):
            print(f"    ch={ch}: {count} divergent positions  "
                  f"bn_scale={bn_scale[ch].item():.4f}  "
                  f"bn_bias={bn_bias[ch].item():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())