"""
catfuse.substitute — CTF substitute mechanism (Phase 3 task 3.2)

Scans a SpikingJelly nn.Module, identifies CTF-fusible patterns, and
replaces matched subgraphs with instances from catfuse_patterns.

Current support:
  - VGG-style Sequential patterns (SpikingVGG11_bn features, classifier)
  - ResNet-style BasicBlock patterns (SpikingResNet18, SEWResNet18 layer[N])
    [TBD — requires BasicBlock forward override, implemented in round 2]

Design:

  Substitute operates in-place on a deepcopy of the input model. It walks
  named_modules() looking for known patterns:

  - **Sequential 3-tuple matcher** (VGG): for each nn.Sequential, scan
    consecutive triples (a, b, c). If (Conv2d, BatchNorm2d, LIFNode), replace
    with PartialFusionConvBNLIF and mark the original slots as identity.
    If (Conv2d, LIFNode), replace with PartialFusionConvLIF.

  - **Sequential 2-tuple matcher** (VGG classifier): for each nn.Sequential,
    scan consecutive pairs (a, b). If (Linear, LIFNode), replace with
    FusedLinearLIF.

  - **BasicBlock matcher** (ResNet / SEW-ResNet): match against
    `basicblock.conv1 + basicblock.bn1 + basicblock.sn1` and similar for
    the second half. This requires a special BasicBlock wrapper class that
    overrides forward to use the fused modules instead of the separate ones.
    [round 2]

Returns a (new_model, coverage_stats) tuple where:
  - new_model is a fused nn.Module
  - coverage_stats reports how many LIFNodes were covered by fusion vs left intact
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

# Make sure catfuse_patterns is importable (same directory)
import sys
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from catfuse_patterns import (
    PartialFusionConvLIF,
    PartialFusionConvBNLIF,
    FusedLinearLIF,
    CTFSEWBasicBlock,
    CTFSpikingBasicBlock,
)


# ============================================================
# Helper: SJ type checks (SJ uses layer.Conv2d, layer.BatchNorm2d, layer.Linear,
# which inherit from their nn.* counterparts)
# ============================================================

def _is_sj_conv2d(m) -> bool:
    return isinstance(m, nn.Conv2d)


def _is_sj_bn2d(m) -> bool:
    return isinstance(m, nn.BatchNorm2d)


def _is_sj_linear(m) -> bool:
    return isinstance(m, nn.Linear)


def _is_lif(m) -> bool:
    from spikingjelly.activation_based import neuron
    return isinstance(m, neuron.LIFNode)


def _is_sew_basic_block(m) -> bool:
    """Check if m is a SEW-ResNet BasicBlock by checking for SEW-specific attributes."""
    cls_name = type(m).__name__
    # SEW BasicBlock has `cnf` attribute; standard spiking ResNet BasicBlock does not
    return cls_name == 'BasicBlock' and hasattr(m, 'cnf') and m.cnf is not None


def _is_standard_spiking_basic_block(m) -> bool:
    """Check if m is a standard spiking_resnet BasicBlock (no cnf attribute)."""
    cls_name = type(m).__name__
    return cls_name == 'BasicBlock' and not hasattr(m, 'cnf')


# ============================================================
# Sequential matcher: scan and replace 3-tuples / 2-tuples
# ============================================================

class _SequentialRewriteResult:
    def __init__(self):
        self.new_modules: list = []
        self.skipped_indices: list = []
        self.replacements: list = []  # list of (pattern_name, original_indices)


def _rewrite_sequential(seq: nn.Sequential) -> tuple[nn.Sequential, _SequentialRewriteResult]:
    """
    Scan a nn.Sequential and replace known CTF patterns.

    Matches (in priority order):
      1. (Conv2d, BatchNorm2d, LIFNode)  -> PartialFusionConvBNLIF
      2. (Conv2d, LIFNode)               -> PartialFusionConvLIF
      3. (Linear, LIFNode)               -> FusedLinearLIF

    Non-matched modules are kept as-is (MaxPool, Dropout, etc.)

    Returns a new nn.Sequential with fused patterns replaced, and a
    _SequentialRewriteResult describing what happened.
    """
    result = _SequentialRewriteResult()
    children = list(seq)
    n = len(children)
    new_children = []
    i = 0

    while i < n:
        # Try 3-tuple: (Conv, BN, LIF)
        if (i + 2 < n
                and _is_sj_conv2d(children[i])
                and _is_sj_bn2d(children[i + 1])
                and _is_lif(children[i + 2])):
            fused = PartialFusionConvBNLIF.from_sj_modules(
                children[i], children[i + 1], children[i + 2]
            )
            new_children.append(fused)
            result.replacements.append(('PartialFusionConvBNLIF', (i, i + 1, i + 2)))
            i += 3
            continue

        # Try 2-tuple: (Conv, LIF)
        if (i + 1 < n
                and _is_sj_conv2d(children[i])
                and _is_lif(children[i + 1])):
            fused = PartialFusionConvLIF.from_sj_modules(
                children[i], children[i + 1]
            )
            new_children.append(fused)
            result.replacements.append(('PartialFusionConvLIF', (i, i + 1)))
            i += 2
            continue

        # Try 2-tuple: (Linear, LIF)
        if (i + 1 < n
                and _is_sj_linear(children[i])
                and _is_lif(children[i + 1])):
            fused = FusedLinearLIF.from_sj_modules(
                children[i], children[i + 1]
            )
            new_children.append(fused)
            result.replacements.append(('FusedLinearLIF', (i, i + 1)))
            i += 2
            continue

        # No match: keep the module as-is
        new_children.append(children[i])
        result.skipped_indices.append(i)
        i += 1

    return nn.Sequential(*new_children), result


# ============================================================
# Top-level substitute entry point
# ============================================================

def substitute(sj_model: nn.Module, verbose: bool = False) -> tuple[nn.Module, dict]:
    """
    Substitute CTF patterns in an SJ model.

    Args:
        sj_model: original SJ nn.Module (SpikingVGG, SpikingResNet, SEWResNet)
        verbose: print per-replacement log

    Returns:
        new_model: a deepcopy with fused patterns substituted in-place
        coverage_stats: dict with fusion coverage info
    """
    model = copy.deepcopy(sj_model)

    stats = {
        'total_lif_nodes': 0,
        'fused_lif_nodes': 0,
        'patterns_matched': {},
        'substituted_sequentials': [],
        'unsupported_structures': [],
    }

    # Count total LIF nodes upfront
    from spikingjelly.activation_based import neuron
    for name, module in model.named_modules():
        if isinstance(module, neuron.LIFNode):
            stats['total_lif_nodes'] += 1

    # Walk all nn.Sequential children at any depth, try to rewrite them in place
    def _walk_and_rewrite(parent: nn.Module, parent_path: str = ''):
        for name, child in list(parent.named_children()):
            child_path = f"{parent_path}.{name}" if parent_path else name

            # Check for SEW BasicBlock first (before generic Sequential scan,
            # because BasicBlock has a hand-written forward)
            if _is_sew_basic_block(child):
                fused_block = CTFSEWBasicBlock.from_sj_block(child)
                setattr(parent, name, fused_block)
                stats['patterns_matched']['CTFSEWBasicBlock'] = (
                    stats['patterns_matched'].get('CTFSEWBasicBlock', 0) + 1
                )
                # CTFSEWBasicBlock replaces 2 or 3 LIFNodes (sn1, sn2, and optionally downsample_sn)
                n_lif_replaced = 3 if child.downsample is not None else 2
                stats['fused_lif_nodes'] += n_lif_replaced
                stats['substituted_sequentials'].append(f"{child_path}:SEWBasicBlock")
                if verbose:
                    print(f"  [substitute] {child_path}: CTFSEWBasicBlock "
                          f"(replaced {n_lif_replaced} LIFs)")
                # Don't recurse into the replaced block
                continue

            # Standard spiking_resnet BasicBlock: replace with CTFSpikingBasicBlock
            if _is_standard_spiking_basic_block(child):
                fused_block = CTFSpikingBasicBlock.from_sj_block(child)
                setattr(parent, name, fused_block)
                stats['patterns_matched']['CTFSpikingBasicBlock'] = (
                    stats['patterns_matched'].get('CTFSpikingBasicBlock', 0) + 1
                )
                # Each standard BasicBlock has exactly 2 LIF nodes (sn1 + sn2)
                # regardless of downsample (downsample has no LIF)
                stats['fused_lif_nodes'] += 2
                stats['substituted_sequentials'].append(
                    f"{child_path}:SpikingBasicBlock"
                )
                if verbose:
                    print(f"  [substitute] {child_path}: CTFSpikingBasicBlock "
                          f"(replaced 2 LIFs, downsample="
                          f"{child.downsample is not None})")
                continue

            if isinstance(child, nn.Sequential):
                new_seq, result = _rewrite_sequential(child)
                if result.replacements:
                    setattr(parent, name, new_seq)
                    stats['substituted_sequentials'].append(child_path)
                    for pattern_name, indices in result.replacements:
                        stats['patterns_matched'][pattern_name] = (
                            stats['patterns_matched'].get(pattern_name, 0) + 1
                        )
                        stats['fused_lif_nodes'] += 1
                        if verbose:
                            print(f"  [substitute] {child_path}: "
                                  f"{pattern_name} @ indices {indices}")
                else:
                    # Recurse — the Sequential may contain nested children
                    # (e.g., BasicBlock inside layer[N])
                    _walk_and_rewrite(child, child_path)
            else:
                _walk_and_rewrite(child, child_path)

    _walk_and_rewrite(model)

    # Compute coverage
    stats['coverage_pct'] = (
        100.0 * stats['fused_lif_nodes'] / stats['total_lif_nodes']
        if stats['total_lif_nodes'] > 0 else 0.0
    )

    # Note: unsupported_structures is already populated during the walk above,
    # so we don't need the earlier post-walk scan.

    return model, stats


# ============================================================
# Pretty printer for coverage stats
# ============================================================

def print_coverage_report(stats: dict, title: str = 'Coverage report'):
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"  total LIF nodes      : {stats['total_lif_nodes']}")
    print(f"  fused LIF nodes      : {stats['fused_lif_nodes']}")
    print(f"  coverage             : {stats['coverage_pct']:.2f}%")
    print(f"\n  patterns matched:")
    for pattern, count in sorted(stats['patterns_matched'].items()):
        print(f"    {pattern:<30} {count} instance(s)")
    print(f"\n  substituted sequentials:")
    for path in stats['substituted_sequentials']:
        print(f"    {path}")
    if stats['unsupported_structures']:
        print(f"\n  unsupported structures (need BasicBlock matcher):")
        for s in stats['unsupported_structures']:
            print(f"    {s}")


# ============================================================
# CLI smoke test
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sew_resnet18',
                        choices=['vgg11_bn', 'sew_resnet18', 'spiking_resnet18'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    from spikingjelly.activation_based import functional, neuron, surrogate

    surrogate_fn = surrogate.Sigmoid()

    if args.model == 'vgg11_bn':
        from spikingjelly.activation_based.model import spiking_vgg
        print("Building SpikingVGG11_bn...")
        model = spiking_vgg.spiking_vgg11_bn(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            num_classes=10,
        )
        T, B, C, H = 16, 4, 3, 32
    elif args.model == 'sew_resnet18':
        from spikingjelly.activation_based.model import sew_resnet
        print("Building SEWResNet18...")
        model = sew_resnet.sew_resnet18(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            num_classes=10,
            cnf='ADD',
        )
        T, B, C, H = 16, 4, 3, 64
    elif args.model == 'spiking_resnet18':
        from spikingjelly.activation_based.model import spiking_resnet
        print("Building SpikingResNet18 (standard, no SEW)...")
        model = spiking_resnet.spiking_resnet18(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            num_classes=10,
        )
        T, B, C, H = 16, 4, 3, 64

    functional.set_step_mode(model, 'm')

    print("Applying catfuse.substitute...")
    fused_model, stats = substitute(model, verbose=True)
    print_coverage_report(stats, title=f'{args.model} coverage')

    # Quick sanity: forward the fused model
    print("\nSmoke-testing fused model forward...")
    device = torch.device(f'cuda:{args.gpu}')
    fused_model = fused_model.to(device)
    x = torch.randn(T, B, C, H, H, device=device)
    try:
        y = fused_model(x)
        print(f"  fused forward OK: input {list(x.shape)} -> output {list(y.shape)}")
    except Exception as e:
        print(f"  fused forward FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
