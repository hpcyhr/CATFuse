"""experiments/_helpers.py — shared model construction.

Centralizes SEW-ResNet18 building so that experiment scripts use the
SAME architecture as the trained checkpoint.

The training script `training/phaseC1_train_sew_cifar10.py` uses the
following CIFAR10 adaptations:

  - conv1: 3x3 stride=1 (NOT default 7x7 stride=2)
  - maxpool: Identity (NOT default 3x3 maxpool)
  - surrogate: Sigmoid (NOT ATan)
  - tau=2.0, v_threshold=1.0, cnf='ADD', detach_reset=True
  - step_mode='m'

Earlier experiment scripts built SEW-RN18 with default (ImageNet) stem
and ATan surrogate, then load_state_dict(strict=False). The mismatch
silently dropped conv1.weight from the checkpoint (shape mismatch
64x3x7x7 vs 64x3x3x3), leaving conv1 randomly initialized. The result
was a degenerate forward where all LIF outputs after layer1 were zero
on every input — measurements based on this were comparing zero spike
chains, not real network behavior.

Use build_sew_rn18_cifar10() in any experiment that loads the CIFAR10
checkpoint.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import (
    functional, neuron, surrogate, layer as sj_layer,
)
from spikingjelly.activation_based.model import sew_resnet


_DEFAULT_CKPT = "checkpoints/sew_resnet18_cifar10_lif_best.pth"


def unwrap_ckpt(state):
    """Unwrap a torch.load result to get the actual model state_dict.

    Handles common wrapping patterns from various training frameworks:
      - {'model_state_dict': sd, ...}      (phaseC1 — current ckpt)
      - {'state_dict': sd, ...}            (PyTorch Lightning, etc.)
      - {'net': sd, ...}                   (some SJ-style scripts)
      - {'model': sd}                      (torchvision-style)
      - sd                                 (raw state_dict)

    The phaseC1 training script saves with key 'model_state_dict' along
    with epoch/optimizer/test_acc fields. Earlier helpers only checked
    'net' and 'model', so this dict was returned as-is and the actual
    weights inside 'model_state_dict' were never reached, causing
    load_state_dict(strict=False) to silently drop ALL keys.

    Returns the unwrapped state_dict, or the input unchanged if no known
    wrapper key is found.
    """
    if not isinstance(state, dict):
        return state
    for key in ("model_state_dict", "state_dict", "net", "model"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def build_sew_rn18_cifar10(
    repo_root: str,
    device: str = "cuda:0",
    ckpt_path: str = None,
) -> Tuple[nn.Module, str]:
    """Build SEW-ResNet18 with CIFAR10 stem adaption matching the trained
    checkpoint, and load weights from disk if available.

    Returns (model, source_note).

    Architecture matches training/phaseC1_train_sew_cifar10.py exactly:
      - conv1: 3x3 stride=1 padding=1 bias=False  (replaces default 7x7 s=2)
      - maxpool: Identity                          (replaces default 3x3 maxpool)
      - surrogate.Sigmoid()                        (matches training)
      - tau=2.0, v_threshold=1.0, cnf='ADD'
      - step_mode='m'

    Args:
        repo_root: project root containing checkpoints/.
        device: target CUDA device.
        ckpt_path: explicit ckpt path; defaults to <repo_root>/checkpoints/...

    Note: model is returned in .eval() mode and on `device`.
    """
    net = sew_resnet.sew_resnet18(
        pretrained=False, num_classes=10, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        tau=2.0,
        v_threshold=1.0,
    )

    # CIFAR10 stem adaption — MUST be done BEFORE load_state_dict so that
    # conv1.weight has the correct shape (64x3x3x3) to match the ckpt.
    # We use sj_layer.Conv2d (multi-step aware), not nn.Conv2d, to match
    # the rest of the network's step_mode semantics.
    net.conv1 = sj_layer.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False,
    )
    net.maxpool = nn.Identity()

    functional.set_step_mode(net, "m")

    note = "synthetic weights"
    ckpt = ckpt_path or os.path.join(repo_root, _DEFAULT_CKPT)
    if os.path.exists(ckpt):
        try:
            raw = torch.load(ckpt, map_location="cpu")
            state = unwrap_ckpt(raw)
            missing, unexpected = net.load_state_dict(state, strict=False)
            # If conv1 still missing, ckpt is wrong shape — surface that.
            conv1_keys = [k for k in missing
                          if k.startswith("conv1.")]
            if conv1_keys:
                note = (f"ckpt shape mismatch (missing {conv1_keys}); "
                        f"check whether ckpt was trained with cifar10_stem")
            else:
                # Include ckpt metadata when available
                meta_bits = []
                if isinstance(raw, dict):
                    if "epoch" in raw:
                        meta_bits.append(f"epoch={raw['epoch']}")
                    if "test_acc" in raw:
                        meta_bits.append(f"test_acc={raw['test_acc']:.4f}")
                meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""
                note = f"loaded from {ckpt}{meta_str}"
                if missing:
                    note += f"  (note: {len(missing)} keys still missing)"
                if unexpected:
                    note += f"  (note: {len(unexpected)} unexpected keys)"
        except Exception as e:
            note = f"ckpt load failed: {e}"

    net = net.to(device).eval()
    return net, note


def quick_spike_sanity_check(net, device: str = "cuda:0", T: int = 4, B: int = 2):
    """Verify the network actually produces non-zero spikes on a real input.

    Returns a dict with diagnostics for stdout.
    """
    from spikingjelly.activation_based import functional, neuron

    torch.manual_seed(0)
    # Real-ish CIFAR10 image scale: pixel values in roughly [0, 1].
    x = torch.rand(T, B, 3, 32, 32, device=device)

    # Trace LIF outputs across the network
    spike_traces = {}
    handles = []

    def _hook(name):
        def fn(mod, inp, out):
            spike_traces[name] = (out.shape, out.detach().abs().mean().item(),
                                  (out != 0).float().mean().item())
        return fn

    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            handles.append(m.register_forward_hook(_hook(name)))

    with torch.no_grad():
        functional.reset_net(net)
        y = net(x)

    for h in handles:
        h.remove()

    # Summarize
    n_total = len(spike_traces)
    n_alive = sum(1 for (_, _, nnz) in spike_traces.values() if nnz > 0)

    return {
        "y_max_abs": y.detach().abs().max().item(),
        "y_unique_count": int(y.detach().reshape(-1).unique().numel()),
        "lif_total": n_total,
        "lif_with_spikes": n_alive,
        "spike_traces": spike_traces,
        "first_dead_layer": next(
            (name for name, (_, _, nnz) in spike_traces.items() if nnz == 0),
            None,
        ),
    }