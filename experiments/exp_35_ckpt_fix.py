"""§5.3 fix verification — load ckpt with corrected unwrap and check
whether spike chain is now alive.

The ckpt at checkpoints/sew_resnet18_cifar10_lif_best.pth is structured:
    {
      'epoch': 2,
      'model_state_dict': <122 keys>,
      'optimizer_state_dict': ...,
      'test_acc': 0.631,
      'args': ...,
    }

Earlier scripts only unwrapped 'net' or 'model' keys, so 'model_state_dict'
was never reached and load_state_dict(strict=False) silently dropped
102/122 keys. This script:

  1. Unwraps correctly (checks 'model_state_dict' too)
  2. Verifies key alignment — reports missing/unexpected after correct unwrap
  3. Runs forward on RIGHT-stem network with three input distributions
  4. Reports per-LIF-layer spike rate

Note: ckpt was saved at epoch=2 with test_acc=63.1% — this is far from a
fully-trained model (training script targets 88-92% over 200 epochs).
We use this ckpt as-is for §5.3, but the framing in the paper should
acknowledge that wall-clock numbers come from a partially-trained
network. SNN spike patterns at 63% acc are still meaningful; they just
won't match a fully-converged model's distribution.

Run:
    cd /path/to/CATFuse
    python experiments/exp_35_ckpt_fix.py
"""
from __future__ import annotations

import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from spikingjelly.activation_based import (
    functional, neuron, surrogate, layer as sj_layer,
)
from spikingjelly.activation_based.model import sew_resnet


DEVICE = "cuda:0"
T = 4
B = 2
CKPT_PATH = os.path.join(_REPO_ROOT, "checkpoints",
                         "sew_resnet18_cifar10_lif_best.pth")


def unwrap_ckpt(state):
    """Unwrap a torch.load result to get the model state_dict.

    Handles common wrapping patterns:
      - {'net': sd}                        (some SJ scripts)
      - {'model': sd}                      (torchvision-style)
      - {'model_state_dict': sd, ...}      (phaseC1 training script)
      - {'state_dict': sd, ...}            (PyTorch Lightning, etc.)
      - sd                                 (raw state_dict)
    """
    if not isinstance(state, dict):
        return state
    for key in ("model_state_dict", "state_dict", "net", "model"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def build_right_stem_net():
    net = sew_resnet.sew_resnet18(
        pretrained=False, num_classes=10, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        tau=2.0, v_threshold=1.0,
    )
    net.conv1 = sj_layer.Conv2d(3, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)
    net.maxpool = nn.Identity()
    functional.set_step_mode(net, "m")
    return net


def trace_spikes(net, x):
    rates = {}
    handles = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            def _make(n):
                def fn(_mod, _inp, out):
                    rates[n] = (out.detach() != 0).float().mean().item()
                return fn
            handles.append(m.register_forward_hook(_make(name)))
    with torch.no_grad():
        functional.reset_net(net)
        y = net(x)
    for h in handles:
        h.remove()
    return rates, y


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 fix verification — corrected ckpt unwrap")
    print("=" * 96)

    if not os.path.exists(CKPT_PATH):
        print(f"FATAL: ckpt not found at {CKPT_PATH}")
        return 1

    raw_state = torch.load(CKPT_PATH, map_location="cpu")

    # ============================================================
    # Step 1: report ckpt metadata
    # ============================================================
    print()
    print("Step 1 — ckpt metadata")
    print("─" * 96)
    if isinstance(raw_state, dict):
        if "epoch" in raw_state:
            print(f"  epoch:    {raw_state['epoch']}")
        if "test_acc" in raw_state:
            acc = raw_state['test_acc']
            print(f"  test_acc: {acc:.4f} ({acc*100:.2f}%)")
        if "args" in raw_state and isinstance(raw_state["args"], dict):
            args = raw_state["args"]
            print(f"  training args (selected):")
            for k in ("epochs", "T", "lr", "batch_size",
                     "cifar10_stem", "cifar10-stem"):
                if k in args:
                    print(f"    {k}: {args[k]}")
        print(f"  top-level keys: {list(raw_state.keys())}")

    # ============================================================
    # Step 2: unwrap and verify key alignment
    # ============================================================
    print()
    print("Step 2 — unwrap + key alignment")
    print("─" * 96)

    state = unwrap_ckpt(raw_state)
    print(f"  unwrapped state has {len(state)} keys")
    print(f"  first 5: {list(state.keys())[:5]}")

    net = build_right_stem_net().to(DEVICE).eval()
    net_keys = set(net.state_dict().keys())
    state_keys = set(state.keys())

    missing = sorted(net_keys - state_keys)
    unexpected = sorted(state_keys - net_keys)

    print(f"  net has {len(net_keys)} keys, ckpt has {len(state_keys)} keys")
    print(f"  in both:        {len(net_keys & state_keys)}")
    print(f"  missing (net not in ckpt): {len(missing)}")
    print(f"  unexpected (ckpt not in net): {len(unexpected)}")

    if missing:
        print(f"  sample missing:")
        for k in missing[:8]:
            print(f"    {k}")
    if unexpected:
        print(f"  sample unexpected:")
        for k in unexpected[:8]:
            print(f"    {k}")

    # ============================================================
    # Step 3: load and check shape compatibility
    # ============================================================
    print()
    print("Step 3 — load_state_dict")
    print("─" * 96)

    result = net.load_state_dict(state, strict=False)
    print(f"  load_state_dict missing:    {len(result.missing_keys)}")
    print(f"  load_state_dict unexpected: {len(result.unexpected_keys)}")

    # Specifically check conv1
    conv1_shape = net.conv1.weight.shape
    print(f"  conv1.weight shape after load: {tuple(conv1_shape)}")
    if "conv1.weight" in state:
        ckpt_conv1_shape = state["conv1.weight"].shape
        print(f"  conv1.weight shape in ckpt:    {tuple(ckpt_conv1_shape)}")
        if conv1_shape == ckpt_conv1_shape:
            print(f"  ✓ conv1 shapes match")
        else:
            print(f"  ✗ conv1 shape mismatch — load_state_dict skipped this key")

    # ============================================================
    # Step 4: spike trace on three input variants
    # ============================================================
    print()
    print("Step 4 — spike pattern on three input distributions")
    print("─" * 96)

    inputs = {
        "random_unit":      torch.rand(T, B, 3, 32, 32, device=DEVICE),
        "random_normal":    torch.randn(T, B, 3, 32, 32, device=DEVICE),
        "cifar_normalized": 0.25 * torch.randn(T, B, 3, 32, 32, device=DEVICE),
    }

    for inp_name, x in inputs.items():
        torch.manual_seed(0)  # for fairness
        rates, y = trace_spikes(net, x)
        n_total = len(rates)
        n_alive = sum(1 for r in rates.values() if r > 0)
        alive_rates = [r for r in rates.values() if r > 0]

        print()
        print(f"  Input: {inp_name}  (mean={x.mean():.4f} std={x.std():.4f})")
        print(f"    Output: y.max_abs={y.detach().abs().max().item():.4f}")
        print(f"    LIF: {n_alive}/{n_total} firing")
        if alive_rates:
            print(f"    Alive layer spike rates: "
                  f"min={min(alive_rates)*100:.2f}%, "
                  f"mean={sum(alive_rates)/len(alive_rates)*100:.2f}%, "
                  f"max={max(alive_rates)*100:.2f}%")
            # Show every layer
            print(f"    Per-layer:")
            for name in sorted(rates.keys()):
                r = rates[name]
                marker = " ← DEAD" if r == 0 else ""
                print(f"      {name:<35} {r*100:>6.2f}%{marker}")
        else:
            print(f"    All layers dead. ckpt may be corrupt or unwrap still wrong.")

    # ============================================================
    # Diagnosis
    # ============================================================
    print()
    print("=" * 96)
    print("Diagnosis")
    print("=" * 96)

    # Use cifar_normalized result as reference for trained-model behavior
    rates_ref, _ = trace_spikes(net, inputs["cifar_normalized"])
    n_alive = sum(1 for r in rates_ref.values() if r > 0)
    n_total = len(rates_ref)

    if len(missing) <= 1 and "conv1.weight" not in missing:
        # Includes case where missing is ['conv1.weight'] (which shouldn't
        # be missing — we replaced conv1 with right shape) or 0 missing.
        print(f"  ✓ ckpt unwrap fixed: only {len(missing)} key(s) missing in")
        print(f"    net (ideally 0). conv1.weight is NOT in missing list.")
    else:
        print(f"  ✗ ckpt unwrap STILL incomplete: {len(missing)} keys missing.")
        print(f"    Investigate the key-naming mismatch with the sample list above.")

    if n_alive == n_total:
        print(f"  ✓ All {n_total} LIF layers fire on cifar_normalized input.")
        print(f"    The all-zero chain bug from exp_31/33 was the ckpt unwrap.")
        print(f"    Note: ckpt is at epoch=2 acc=63.1% — partially trained.")
        print(f"    Spike patterns are valid but not those of a fully-converged model.")
        print()
        print(f"  Next: re-run exp_31_end_to_end with fixed ckpt unwrap and update")
        print(f"        _helpers.build_sew_rn18_cifar10 to use unwrap_ckpt.")
    elif n_alive >= n_total * 0.7:
        print(f"  ⚠ Most layers fire ({n_alive}/{n_total}) but some are dead.")
        print(f"    Could be partial training (epoch=2) — some layers haven't")
        print(f"    learned to fire yet. Acceptable for §5.3 but report carefully.")
    elif n_alive > 0:
        print(f"  ⚠ Only {n_alive}/{n_total} layers fire on cifar_normalized.")
        print(f"    Significant dead chain — ckpt may be too early-stage to be")
        print(f"    representative. Consider continuing training to higher acc.")
    else:
        print(f"  ✗ Even with corrected unwrap, all layers dead.")
        print(f"    Bug is NOT the unwrap. Check forward path / BN running stats /")
        print(f"    set_step_mode.")

    return 0


if __name__ == "__main__":
    sys.exit(main())