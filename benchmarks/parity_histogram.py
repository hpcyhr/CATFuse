"""
Parity evidence: mismatch-neuron distribution histogram for CATFuse paper.

Shows that spike-flip mismatches between CATFuse and SJ reference are
concentrated at neurons whose pre-spike membrane potential v is within
O(1e-7) of the firing threshold v_th.

Method:
  1. Run isolated Conv→BN→LIF chain in SJ and CATFuse at matching init.
  2. Extract per-neuron pre-spike membrane potential v and output spike s.
  3. Compute mismatches (spike_SJ != spike_CTF).
  4. Histogram |v - v_th| for mismatch neurons vs all-neurons.
  5. Save as figs/parity_histogram.pdf/png.

Usage:
    /data_priv/dagongcheng/snn118/bin/python parity_histogram.py
"""
import copy
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikingjelly.activation_based import functional, neuron, surrogate, layer as sjlayer

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from catfuse_patterns import PartialFusionConvBNLIF


def main():
    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Config: moderately-sized Conv→BN→LIF to produce many neurons + flips
    T, B, C, H = 16, 8, 64, 56
    v_th = 0.1
    tau = 2.0

    torch.manual_seed(42)

    # Build SJ reference chain
    sj_chain = nn.Sequential(
        sjlayer.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False),
        sjlayer.BatchNorm2d(C),
        neuron.LIFNode(tau=tau, v_threshold=v_th,
                       surrogate_function=surrogate.Sigmoid(),
                       detach_reset=True,
                       store_v_seq=True),   # <-- keep v sequence
    )
    functional.set_step_mode(sj_chain, 'm')

    # Random BN stats so activations are realistic
    bn = sj_chain[1]
    bn.running_mean.data.normal_(0, 0.1)
    bn.running_var.data.uniform_(0.5, 1.5)
    bn.eval()
    sj_chain = sj_chain.to(device)

    # Build CATFuse fused version from the same modules
    conv, bn_mod, lif = sj_chain[0], sj_chain[1], sj_chain[2]
    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn_mod, lif).to(device)

    # Forward pass
    x = torch.randn(T, B, C, H, H, device=device)

    with torch.no_grad():
        functional.reset_net(sj_chain)
        s_sj = sj_chain(x)
        # Grab v sequence from the SJ LIF (shape [T, B, C, H, W], pre-spike potential)
        v_sj = lif.v_seq.detach()   # this is v_t *after* update, pre-spike threshold check
        functional.reset_net(sj_chain)

        functional.reset_net(fused)
        s_ctf = fused(x)
        functional.reset_net(fused)

    # Mismatch mask (per-neuron per-time)
    mismatch = (s_sj != s_ctf)
    n_mismatch = int(mismatch.sum().item())
    n_total = s_sj.numel()

    # |v - v_th| for all neurons, and for mismatches
    dist = (v_sj - v_th).abs().flatten().cpu().numpy()
    dist_mismatch = (v_sj - v_th).abs()[mismatch].flatten().cpu().numpy()

    print(f"\nShape: [T, B, C, H, W] = [{T}, {B}, {C}, {H}, {H}]")
    print(f"Total neurons * T steps: {n_total}")
    print(f"Mismatched spikes:       {n_mismatch}")
    print(f"Mismatch rate:           {n_mismatch/n_total:.2e}")

    if n_mismatch == 0:
        print(f"\nNOTE: zero mismatches (bit-exact). Cannot generate distribution.")
        print(f"This is actually the 'good' outcome for SEW-like patterns.")
        # Generate a small synthetic histogram to demonstrate layout anyway?
        # No — just report and exit.
        return

    print(f"\nMax |v - v_th|:           {dist.max():.4e}")
    print(f"Median |v - v_th| (all):  {np.median(dist):.4e}")
    print(f"Median |v - v_th| (mism): {np.median(dist_mismatch):.4e}")
    print(f"Mismatches within 1e-6:   "
          f"{(dist_mismatch < 1e-6).sum()} / {n_mismatch} "
          f"({100*(dist_mismatch < 1e-6).sum()/n_mismatch:.1f}%)")
    print(f"Mismatches within 1e-5:   "
          f"{(dist_mismatch < 1e-5).sum()} / {n_mismatch} "
          f"({100*(dist_mismatch < 1e-5).sum()/n_mismatch:.1f}%)")

    # Plot
    os.makedirs('figs', exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Log-scale bins from 1e-10 to max
    bins = np.logspace(-10, max(-3, np.log10(dist.max() + 1e-12)), 40)

    # Normalize each to fraction to fit on same y-axis
    ax.hist(dist, bins=bins, alpha=0.4, label='All neurons',
            color='gray', density=True)
    ax.hist(dist_mismatch, bins=bins, alpha=0.75,
            label=f'Mismatch neurons (N={n_mismatch})',
            color='#d62728', density=True)

    ax.set_xscale('log')
    ax.set_xlabel(r'$|v - v_{\mathrm{th}}|$ (pre-spike distance to threshold)',
                  fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout(pad=0.3)
    plt.savefig('figs/parity_histogram.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('figs/parity_histogram.png', dpi=150,
                bbox_inches='tight', pad_inches=0.02)
    print(f"\nSaved figs/parity_histogram.pdf + figs/parity_histogram.png")


if __name__ == '__main__':
    main()