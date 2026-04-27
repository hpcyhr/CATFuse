"""§5.3 diagnostic — verify the stem-fix hypothesis.

Hypothesis: the all-zero spike chain in exp_31/33 is caused by a stem
architecture mismatch between the experiment scripts (default ImageNet
stem: 7x7 stride=2 + 3x3 maxpool) and the trained checkpoint
(CIFAR10 stem: 3x3 stride=1 + Identity maxpool). load_state_dict(strict=False)
silently dropped conv1.weight due to shape mismatch (64x3x7x7 vs 64x3x3x3),
leaving conv1 randomly initialized.

This script tests the hypothesis by running side-by-side:

  A. WRONG stem (default ImageNet, what exp_31/33 used)
  B. RIGHT stem (CIFAR10 adaption, matching ckpt)

For each, we report:
  - whether ckpt loaded fully (no missing/unexpected keys)
  - per-LIF-layer spike rate on a real CIFAR10-distribution input
  - first dead layer (if any)

Three input distributions are tested in each variant to rule out the
"input scale problem" alternative:
  - random_unit:   torch.rand in [0, 1]              (uniform)
  - random_normal: torch.randn (mean=0 std=1)        (gaussian, unnormalized)
  - cifar_normalized: gaussian scaled to typical CIFAR10 normalization
                      (mean=0, std≈0.25)             (matches training)

Expected outcomes:

  - VARIANT A + any input → all-zero spike chain (existing observation)
  - VARIANT B + cifar_normalized input → typical SNN spike rates
    (5-30% per layer, monotonically decreasing through depth, NEVER all-zero)

If VARIANT B still gives all-zero spikes, the bug is NOT the stem;
it's the ckpt itself or something else, and we need a deeper investigation
before any experiment data is trustworthy.

Run:
    cd /path/to/CATFuse
    python experiments/exp_34_stem_fix_diagnostic.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

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


def build_with_stem(stem_kind: str) -> Tuple[nn.Module, list, list]:
    """Build SEW-RN18 with either WRONG (default ImageNet) stem or RIGHT
    (CIFAR10) stem; load checkpoint with strict=False and return missing/unexpected keys.

    stem_kind: "wrong" | "right"
    """
    # Match training surrogate (Sigmoid) for both variants — the surrogate
    # only affects backward, but consistency is good.
    net = sew_resnet.sew_resnet18(
        pretrained=False, num_classes=10, cnf="ADD",
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
        tau=2.0,
        v_threshold=1.0,
    )

    if stem_kind == "right":
        # CIFAR10 stem: matches phaseC1 training
        net.conv1 = sj_layer.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
        )
        net.maxpool = nn.Identity()
    elif stem_kind == "wrong":
        # Keep default ImageNet stem — what exp_31/33 used
        pass
    else:
        raise ValueError(stem_kind)

    functional.set_step_mode(net, "m")

    missing, unexpected = [], []
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(state, dict):
            if "net" in state:
                state = state["net"]
            elif "model" in state:
                state = state["model"]
        result = net.load_state_dict(state, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)

    net = net.to(DEVICE).eval()
    return net, missing, unexpected


def make_input(kind: str) -> torch.Tensor:
    """Generate a [T, B, 3, 32, 32] input tensor with specified distribution."""
    torch.manual_seed(0)
    if kind == "random_unit":
        return torch.rand(T, B, 3, 32, 32, device=DEVICE)
    elif kind == "random_normal":
        return torch.randn(T, B, 3, 32, 32, device=DEVICE)
    elif kind == "cifar_normalized":
        # Mimic CIFAR10 normalization: mean ~ [0.49, 0.48, 0.45], std ~ [0.25]
        # We just use mean=0 std=0.25 which matches the post-normalization
        # distribution typical for trained CIFAR10 networks.
        return 0.25 * torch.randn(T, B, 3, 32, 32, device=DEVICE)
    else:
        raise ValueError(kind)


def trace_spike_rates(net) -> Dict[str, float]:
    """Hook all LIFNodes; return {layer_name: spike_rate}."""
    rates = {}

    def _hook(name):
        def fn(_mod, _inp, out):
            t = out.detach()
            rates[name] = (t != 0).float().mean().item()
        return fn

    handles = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            handles.append(m.register_forward_hook(_hook(name)))

    return rates, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def run_one(stem_kind: str, input_kind: str):
    print(f"\n{'─' * 96}")
    print(f"  stem={stem_kind}, input={input_kind}")
    print(f"{'─' * 96}")

    net, missing, unexpected = build_with_stem(stem_kind)
    conv1_missing = [k for k in missing if k.startswith("conv1.")]
    print(f"    ckpt: {len(missing)} missing keys (conv1 missing: "
          f"{len(conv1_missing)}), {len(unexpected)} unexpected keys")
    if conv1_missing:
        print(f"      → conv1 keys missing: {conv1_missing}")

    x = make_input(input_kind)
    print(f"    input shape={tuple(x.shape)}  mean={x.mean():.4f}  std={x.std():.4f}")

    rates, handles = trace_spike_rates(net)
    with torch.no_grad():
        functional.reset_net(net)
        y = net(x)
    remove_hooks(handles)

    n_total = len(rates)
    n_alive = sum(1 for r in rates.values() if r > 0)
    n_dead = n_total - n_alive

    print(f"    output: y.max_abs={y.detach().abs().max().item():.4f}  "
          f"y.unique={y.detach().reshape(-1).unique().numel()}")
    print(f"    LIF layers: {n_total} total, {n_alive} firing, {n_dead} fully dead")

    # Print first few + summary
    print(f"\n    {'layer':<35} {'spike_rate':>12}")
    print(f"    {'-'*35} {'-'*12}")
    for name in sorted(rates.keys()):
        r = rates[name]
        marker = " ← DEAD" if r == 0 else ""
        print(f"    {name:<35} {r*100:>10.2f}%{marker}")

    return {
        "stem": stem_kind,
        "input": input_kind,
        "n_dead_lif": n_dead,
        "n_total_lif": n_total,
        "first_dead": next((n for n in sorted(rates.keys()) if rates[n] == 0), None),
        "all_alive_rates": [rates[n] for n in sorted(rates.keys()) if rates[n] > 0],
    }


def main():
    if not torch.cuda.is_available():
        print("Requires CUDA")
        return 1

    print("=" * 96)
    print("§5.3 diagnostic — stem-fix hypothesis test")
    print("=" * 96)
    print()
    print("  Comparing WRONG-stem (ImageNet 7x7 s=2, what exp_31/33 used)")
    print("  vs RIGHT-stem (CIFAR10 3x3 s=1 + Identity maxpool, matches")
    print("  phaseC1 training).")
    print()
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  T={T}  B={B}")
    print()

    if not os.path.exists(CKPT_PATH):
        print(f"  WARNING: checkpoint not found at {CKPT_PATH}")
        print("  Diagnostic still runs (with random init), but conclusions about")
        print("  ckpt-vs-arch mismatch will be unavailable.")
        print()

    results = []
    for stem in ("wrong", "right"):
        for inp in ("random_unit", "random_normal", "cifar_normalized"):
            r = run_one(stem, inp)
            results.append(r)

    # ============================================================
    # Diagnosis
    # ============================================================
    print()
    print("=" * 96)
    print("Summary table")
    print("=" * 96)
    print(f"  {'stem':<8} {'input':<22} {'dead/total':>12} {'mean alive rate':>20}")
    print(f"  {'-'*8} {'-'*22} {'-'*12} {'-'*20}")
    for r in results:
        n_alive = r["n_total_lif"] - r["n_dead_lif"]
        if n_alive > 0:
            mean_rate = sum(r["all_alive_rates"]) / n_alive
            mean_str = f"{mean_rate*100:.2f}%"
        else:
            mean_str = "(none alive)"
        print(f"  {r['stem']:<8} {r['input']:<22} "
              f"{r['n_dead_lif']:>4}/{r['n_total_lif']:<6} {mean_str:>20}")

    print()
    print("=" * 96)
    print("Diagnosis")
    print("=" * 96)

    wrong_results = [r for r in results if r["stem"] == "wrong"]
    right_results = [r for r in results if r["stem"] == "right"]

    wrong_all_dead = all(r["n_dead_lif"] >= r["n_total_lif"] - 1 for r in wrong_results)
    right_any_alive = any(r["n_dead_lif"] < r["n_total_lif"] - 1 for r in right_results)
    right_all_alive = all(r["n_dead_lif"] == 0 for r in right_results)

    if wrong_all_dead and right_all_alive:
        print("  STRONG CONFIRMATION of stem-mismatch hypothesis:")
        print("    - WRONG-stem network: dead LIF chain on all input variants")
        print("    - RIGHT-stem network: every LIF layer has non-zero firing")
        print()
        print("  → exp_31's 2x speedup over SJ-eager is ENTIRELY a StaticZero")
        print("    short-circuit artifact and must be redone with RIGHT-stem.")
        print("  → exp_30 single-layer K-sweep is UNAFFECTED (uses synthetic")
        print("    sparse input, doesn't depend on full-network propagation).")
        print("  → Stage 1-8 SEW-RN18 verify regressions are technically valid")
        print("    (bit-exact zero == zero) but should be re-run on the fixed")
        print("    network for stronger evidence.")
    elif wrong_all_dead and right_any_alive:
        print("  PARTIAL CONFIRMATION:")
        print("    - WRONG-stem network: dead chain (consistent with hypothesis)")
        print("    - RIGHT-stem network: ALIVE on some inputs but not others")
        print()
        print("  → Stem mismatch is one cause; input distribution matters too.")
        print("    Look at which input variant of RIGHT-stem produces firing")
        print("    and use that distribution for exp_31 redo.")
    elif not wrong_all_dead:
        print("  HYPOTHESIS UNDERMINED:")
        print("    - WRONG-stem network sometimes fires; stem mismatch is NOT")
        print("      the unique cause of the dead-chain observation in exp_31/33.")
        print()
        print("  → Investigate: maybe the eval-mode input distribution in")
        print("    exp_31/33 happened to be unfortunate, or there's a separate")
        print("    state-leak across runs.")
    else:
        print("  HYPOTHESIS REFUTED:")
        print("    - RIGHT-stem network ALSO produces dead chains")
        print()
        print("  → The bug is NOT the stem. Possibilities:")
        print("    (1) ckpt itself is broken — re-train from phaseC1 or check")
        print("        training logs for accuracy/loss")
        print("    (2) input distribution issue — try realistic CIFAR10 image")
        print("    (3) eval-mode BN parameters wrong — check running_mean/var")
        print("    (4) functional.reset_net / set_step_mode issue")
        print("  This blocks ALL §5.3 work until resolved.")

    # Per-layer alive table for the RIGHT+cifar_normalized case (most
    # representative of real deployment)
    rep = next((r for r in results
                if r["stem"] == "right" and r["input"] == "cifar_normalized"),
               None)
    if rep is not None and rep["n_dead_lif"] < rep["n_total_lif"] - 1:
        n_alive = rep["n_total_lif"] - rep["n_dead_lif"]
        rates = rep["all_alive_rates"]
        print()
        print(f"  Reference SNN profile (RIGHT-stem, CIFAR-distrib input):")
        print(f"    {n_alive}/{rep['n_total_lif']} LIF layers firing")
        print(f"    spike rate range: {min(rates)*100:.2f}% - {max(rates)*100:.2f}%")
        print(f"    mean spike rate:  {sum(rates)/len(rates)*100:.2f}%")
        print(f"  → This is the input distribution and architecture to use for")
        print(f"    exp_31/exp_32 re-runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())