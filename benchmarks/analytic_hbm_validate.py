"""
Analytical HBM I/O derivation for CATFuse paper §V — verify (2+2/K)/5 formula.

Runs on the server. Uses SEW-ResNet-18 layer1[0]'s first Conv→BN→LIF chain
as a worked example at T=16, B=8, 64ch @ 56x56.

Methodology:
  1. ANALYTIC: count HBM transfers per the §III-G ledger for reference vs hybrid.
  2. EMPIRIC: use torch.cuda.max_memory_allocated() + reset_peak to measure
     peak intermediate tensor memory for one forward pass of the isolated chain.
  3. Cross-validate: peak memory reflects how many (T,B,C,H,W) tensors are
     simultaneously in HBM; ratio should track (2+2/K)/5.

Usage:
    /data_priv/dagongcheng/snn118/bin/python analytic_hbm_validate.py
"""
import copy
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron, surrogate, layer as sjlayer

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from catfuse_patterns import PartialFusionConvBNLIF


def build_sj_chain(T, B, C_in, C_out, H_in):
    """Build a single SpikingJelly Conv→BN→LIF chain in multi-step mode."""
    chain = nn.Sequential(
        sjlayer.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1, bias=False),
        sjlayer.BatchNorm2d(C_out),
        neuron.LIFNode(tau=2.0, v_threshold=0.1,
                       surrogate_function=surrogate.Sigmoid(),
                       detach_reset=True),
    )
    functional.set_step_mode(chain, 'm')
    # Random BN stats so activations are non-trivial
    bn = chain[1]
    bn.running_mean.data.normal_(0, 0.1)
    bn.running_var.data.uniform_(0.5, 1.5)
    bn.eval()
    return chain


def analytic_hbm(T, B, C, H, W, K, bytes_per_elt=4):
    """
    HBM transfer ledger per paper §III-G, extended to Conv→BN→LIF.

    Reference (SJ multi-step) per time step, per chain:
      1. write z (conv output)
      2. read z (BN input)
      3. write z' (BN output)
      4. read z' (LIF input)
      5. read v_{t-1}
      6. write v_t
      7. write s_t
    = 7 tensor-sized transfers per step (excluding weight reads).

    Hybrid CATFuse (PartialFusionConvBNLIF) per time step:
      1. write z (cuDNN conv output)   — cuDNN path
      2. read z   (fused Triton kernel input, L2-cold count)
      3. write s (fused kernel output)
      (v stays in registers across K steps; per block boundary: read v_start, write v_end)
    = 3 tensor-sized transfers per step + 2/K per step averaged.

    Simplified by L2-hot assumption (as in §III-G):
      Reference: 5 per step (paper's count, ignoring BN extra read/write)
      Hybrid:    2 per step + 2/K averaged (paper's formula)
      Ratio = (2 + 2/K) / 5

    This function reports BOTH the strict 7-count and the paper's 5-count.
    """
    step_bytes = B * C * H * W * bytes_per_elt
    # Conservative count (with BN explicitly tracked):
    ref_strict = 7 * T * step_bytes                           # 7 per step
    ctf_strict = 3 * T * step_bytes + 2 * (T // K) * step_bytes  # 3+2/K per step
    # Paper's formula count (L2-hot for BN passthrough):
    ref_paper = 5 * T * step_bytes
    ctf_paper = 2 * T * step_bytes + 2 * (T // K) * step_bytes  # (2+2/K)/5

    return {
        'step_bytes': step_bytes,
        'ref_strict_MB': ref_strict / 1e6,
        'ctf_strict_MB': ctf_strict / 1e6,
        'ref_paper_MB':  ref_paper / 1e6,
        'ctf_paper_MB':  ctf_paper / 1e6,
        'ratio_strict':  ctf_strict / ref_strict,
        'ratio_paper':   ctf_paper / ref_paper,
        'predicted_paper_formula': (2 + 2/K) / 5,
    }


@torch.no_grad()
def measure_peak_memory(chain, x, device):
    """Measure peak CUDA memory for one forward pass."""
    functional.reset_net(chain)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    pre = torch.cuda.memory_allocated(device)
    _ = chain(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    functional.reset_net(chain)
    return peak - pre


def main():
    device = torch.device('cuda:0')
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Representative SEW-RN18 layer1 shape
    T, B, C, H = 16, 8, 64, 56
    K = T  # CATFuse uses K=T single-block StateCarry

    print(f"\n=== Worked example: Conv→BN→LIF chain, T={T}, B={B}, C={C}, "
          f"H=W={H}, K={K} ===\n")

    # Analytical counts
    a = analytic_hbm(T, B, C, H, H, K)
    print(f"--- Analytical HBM transfers (per one ConvBN-LIF chain) ---")
    print(f"  Per-step tensor size  : {a['step_bytes']/1e6:.3f} MB")
    print()
    print(f"  Strict count (7 per step ref, 3+2/K per step CTF):")
    print(f"    Reference             : {a['ref_strict_MB']:8.2f} MB")
    print(f"    CATFuse (hybrid)      : {a['ctf_strict_MB']:8.2f} MB")
    print(f"    Ratio CTF/Ref         : {a['ratio_strict']:.3f}")
    print()
    print(f"  Paper §III-G count (5 per step ref, 2+2/K per step CTF):")
    print(f"    Reference             : {a['ref_paper_MB']:8.2f} MB")
    print(f"    CATFuse (hybrid)      : {a['ctf_paper_MB']:8.2f} MB")
    print(f"    Ratio CTF/Ref         : {a['ratio_paper']:.3f}")
    print(f"    Paper formula (2+2/K)/5 = {a['predicted_paper_formula']:.3f}")
    print()

    # Empirical peak memory
    print(f"--- Empirical peak CUDA memory (PyTorch allocator, isolated chain) ---")
    sj_chain = build_sj_chain(T, B, C, C, H).to(device)

    # Build CATFuse version via substitute on the same chain
    ctf_chain_base = copy.deepcopy(sj_chain)
    # Manually substitute the Conv→BN→LIF triplet with PartialFusionConvBNLIF
    conv, bn, lif = ctf_chain_base[0], ctf_chain_base[1], ctf_chain_base[2]
    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)
    # Use the fused module as the whole chain (it computes Conv+BN+LIF end-to-end)

    x = torch.randn(T, B, C, H, H, device=device)

    sj_peak = measure_peak_memory(sj_chain, x, device)
    ctf_peak = measure_peak_memory(fused, x, device)

    print(f"  SJ chain peak mem     : {sj_peak/1e6:8.2f} MB")
    print(f"  CATFuse chain peak mem: {ctf_peak/1e6:8.2f} MB")
    print(f"  Ratio CTF/SJ          : {ctf_peak/sj_peak:.3f}")
    print()

    # Summary
    print(f"--- Cross-validation summary ---")
    print(f"  Paper formula (2+2/K)/5 @ K={K}:   {a['predicted_paper_formula']:.3f}")
    print(f"  Strict analytical ratio         :  {a['ratio_strict']:.3f}")
    print(f"  Empirical peak-memory ratio     :  {ctf_peak/sj_peak:.3f}")
    print()
    print(f"  Note: peak memory tracks intermediate tensor allocation, which is")
    print(f"  a proxy for HBM materialization but not identical to HBM bytes")
    print(f"  moved (operator buffers can be reused, kernels may cache in L2).")
    print(f"  The empirical ratio is expected to bracket the two analytical")
    print(f"  predictions and to be closer to the strict count because PyTorch's")
    print(f"  allocator materializes every intermediate in the sequential path.")


if __name__ == '__main__':
    main()