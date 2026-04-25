"""
catfuse.substitute — CTF substitute mechanism (Phase 3 task 3.2)

Scans a SpikingJelly nn.Module, identifies CTF-fusible patterns, and
replaces matched subgraphs with instances from catfuse_patterns.

Current support:
  - VGG-style Sequential patterns (SpikingVGG11_bn features, classifier)
  - ResNet BasicBlock patterns (SpikingResNet18/34, SEWResNet18/34)
  - ResNet Bottleneck patterns (SpikingResNet50/101/152, SEWResNet50/101/152)

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

  - **BasicBlock matcher** (ResNet18/34, SEW-ResNet18/34): match by class
    name and SEW-specific `cnf` attribute, replace with CTFSEWBasicBlock
    or CTFSpikingBasicBlock.

  - **Bottleneck matcher** (ResNet50/101/152, SEW-ResNet50/101/152): match
    by class name and SEW-specific `cnf` attribute, replace with
    CTFSEWBottleneck or CTFSpikingBottleneck.

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

from catfuse.patterns import (
    PartialFusionConvLIF,
    PartialFusionConvBNLIF,
    FusedLinearLIF,
    CTFSEWBasicBlock,
    CTFSpikingBasicBlock,
    CTFSEWBottleneck,
    CTFSpikingBottleneck,
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


def _is_sew_bottleneck(m) -> bool:
    """Check if m is a SEW-ResNet Bottleneck (has cnf attribute)."""
    cls_name = type(m).__name__
    return cls_name == 'Bottleneck' and hasattr(m, 'cnf') and m.cnf is not None


def _is_standard_spiking_bottleneck(m) -> bool:
    """Check if m is a standard spiking_resnet Bottleneck (no cnf attribute)."""
    cls_name = type(m).__name__
    return cls_name == 'Bottleneck' and not hasattr(m, 'cnf')


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

            # SEW-ResNet Bottleneck: replace with CTFSEWBottleneck
            if _is_sew_bottleneck(child):
                fused_block = CTFSEWBottleneck.from_sj_block(child)
                setattr(parent, name, fused_block)
                stats['patterns_matched']['CTFSEWBottleneck'] = (
                    stats['patterns_matched'].get('CTFSEWBottleneck', 0) + 1
                )
                # SEW Bottleneck replaces 3 or 4 LIFs (sn1, sn2, sn3, opt downsample_sn)
                n_lif_replaced = 4 if child.downsample is not None else 3
                stats['fused_lif_nodes'] += n_lif_replaced
                stats['substituted_sequentials'].append(
                    f"{child_path}:SEWBottleneck"
                )
                if verbose:
                    print(f"  [substitute] {child_path}: CTFSEWBottleneck "
                          f"(replaced {n_lif_replaced} LIFs)")
                continue

            # Standard spiking_resnet Bottleneck: replace with CTFSpikingBottleneck
            if _is_standard_spiking_bottleneck(child):
                fused_block = CTFSpikingBottleneck.from_sj_block(child)
                setattr(parent, name, fused_block)
                stats['patterns_matched']['CTFSpikingBottleneck'] = (
                    stats['patterns_matched'].get('CTFSpikingBottleneck', 0) + 1
                )
                # Standard Bottleneck has 3 LIFs (sn1, sn2, sn3), downsample has no LIF
                stats['fused_lif_nodes'] += 3
                stats['substituted_sequentials'].append(
                    f"{child_path}:SpikingBottleneck"
                )
                if verbose:
                    print(f"  [substitute] {child_path}: CTFSpikingBottleneck "
                          f"(replaced 3 LIFs, downsample="
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
# Policy-aware substitute with SparseFlow backend routing
# ============================================================

def substitute_sf(sj_model, T=16, verbose=False, force_sparse=False):
    """Policy-aware substitute: routes Conv-BN-LIF to STFusionConvBNLIF or PartialFusionConvBNLIF."""
    from spikingjelly.activation_based import neuron
    try:
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        _sf_available = True
    except ImportError:
        _sf_available = False
    from catfuse.policy import get_policy, SpatialBackend

    model = copy.deepcopy(sj_model)
    stats = {"total_lif_nodes": 0, "fused_lif_nodes": 0, "patterns_matched": {}, "routing": []}

    for _, module in model.named_modules():
        if isinstance(module, neuron.LIFNode):
            stats["total_lif_nodes"] += 1

    replacements_done = set()
    for parent_name, parent_mod in list(model.named_modules()):
        for idx in ["1", "2", "3"]:
            c = getattr(parent_mod, f"conv{idx}", None)
            b = getattr(parent_mod, f"bn{idx}", None)
            l = getattr(parent_mod, f"sn{idx}", None)
            if c is None or b is None or l is None:
                continue
            if not isinstance(c, nn.Conv2d) or not isinstance(b, nn.BatchNorm2d) or not isinstance(l, neuron.LIFNode):
                continue
            if id(l) in replacements_done:
                continue

            cin, cout = c.in_channels, c.out_channels
            ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
            stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
            H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
            row = get_policy("Conv3x3_BN_LIF", C_in=cin, C_out=cout, H=H_est, W=H_est, kernel_size=ksize, T=T)

            use_sparse = force_sparse or (_sf_available and row.spatial_backend == SpatialBackend.SPARSE_FLOW)
            full_name = f"{parent_name}.conv{idx}" if parent_name else f"conv{idx}"

            if use_sparse and _sf_available and ksize == 3 and stride_val == 1:
                b.eval()
                fused = STFusionConvBNLIF.from_sj_modules(c, b, l, K=row.K)
                pattern_name = "STFusionConvBNLIF"
            else:
                fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                pattern_name = "PartialFusionConvBNLIF"

            setattr(parent_mod, f"conv{idx}", fused)
            setattr(parent_mod, f"bn{idx}", nn.Identity())
            setattr(parent_mod, f"sn{idx}", nn.Identity())

            replacements_done.add(id(l))
            stats["fused_lif_nodes"] += 1
            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                "backend": row.spatial_backend, "K": row.K})
            if verbose:
                print(f"  [substitute_sf] {full_name}: {pattern_name} ({cin}->{cout}, k={ksize}, s={stride_val}, backend={row.spatial_backend}, K={row.K})")

    # Phase 1b: Downsample patterns (SEW-ResNet)
    # downsample = Sequential(Conv2d, BN2d) + downsample_sn = LIFNode
    for parent_name, parent_mod in list(model.named_modules()):
        ds = getattr(parent_mod, "downsample", None)
        ds_sn = getattr(parent_mod, "downsample_sn", None)
        if ds is None or ds_sn is None:
            continue
        if not isinstance(ds, nn.Sequential) or len(ds) < 2:
            continue
        if not isinstance(ds_sn, neuron.LIFNode):
            continue
        if id(ds_sn) in replacements_done:
            continue
        c, b = ds[0], ds[1]
        if not isinstance(c, nn.Conv2d) or not isinstance(b, nn.BatchNorm2d):
            continue

        cin, cout = c.in_channels, c.out_channels
        ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
        stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
        H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
        full_name = f"{parent_name}.downsample" if parent_name else "downsample"

        # Downsample is always Conv1x1 stride=2: use PartialFusionConvBNLIF (DenseKeep)
        fused = PartialFusionConvBNLIF.from_sj_modules(c, b, ds_sn)
        pattern_name = "PartialFusionConvBNLIF"

        # Replace: put fused into downsample[0], identity into [1], identity into downsample_sn
        parent_mod.downsample = nn.Sequential(fused, nn.Identity())
        parent_mod.downsample_sn = nn.Identity()

        replacements_done.add(id(ds_sn))
        stats["fused_lif_nodes"] += 1
        stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
        stats["routing"].append({"name": full_name, "pattern": pattern_name,
            "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
            "backend": "DenseKeep", "K": T})
        if verbose:
            print(f"  [substitute_sf] {full_name}: {pattern_name} ({cin}->{cout}, k={ksize}, s={stride_val})")

    # Phase 1b: Downsample patterns (SEW-ResNet)
    # downsample = Sequential(Conv2d, BN2d) + downsample_sn = LIFNode
    for parent_name, parent_mod in list(model.named_modules()):
        ds = getattr(parent_mod, "downsample", None)
        ds_sn = getattr(parent_mod, "downsample_sn", None)
        if ds is None or ds_sn is None:
            continue
        if not isinstance(ds, nn.Sequential) or len(ds) < 2:
            continue
        if not isinstance(ds_sn, neuron.LIFNode):
            continue
        if id(ds_sn) in replacements_done:
            continue
        c, b = ds[0], ds[1]
        if not isinstance(c, nn.Conv2d) or not isinstance(b, nn.BatchNorm2d):
            continue

        cin, cout = c.in_channels, c.out_channels
        ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
        stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
        H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
        full_name = f"{parent_name}.downsample" if parent_name else "downsample"

        # Downsample is always Conv1x1 stride=2: use PartialFusionConvBNLIF (DenseKeep)
        fused = PartialFusionConvBNLIF.from_sj_modules(c, b, ds_sn)
        pattern_name = "PartialFusionConvBNLIF"

        # Replace: put fused into downsample[0], identity into [1], identity into downsample_sn
        parent_mod.downsample = nn.Sequential(fused, nn.Identity())
        parent_mod.downsample_sn = nn.Identity()

        replacements_done.add(id(ds_sn))
        stats["fused_lif_nodes"] += 1
        stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
        stats["routing"].append({"name": full_name, "pattern": pattern_name,
            "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
            "backend": "DenseKeep", "K": T})
        if verbose:
            print(f"  [substitute_sf] {full_name}: {pattern_name} ({cin}->{cout}, k={ksize}, s={stride_val})")

    # Phase 1c: Generic attribute patterns (SpikFormer-style naming)
    # Scans all modules for (X, X_bn, X_lif) or (X, X_bn, X_sn) triples
    # where X is Conv2d or Linear, X_bn is BatchNorm, X_lif/X_sn is LIFNode
    for parent_name, parent_mod in list(model.named_modules()):
        # Collect child names
        child_names = {name for name, _ in parent_mod.named_children()}
        # Look for patterns: base / base_bn / base_lif (or base_sn)
        checked_bases = set()
        for cname in sorted(child_names):
            # Strip suffix to find base
            for suffix in ['_bn', '_lif', '_sn']:
                if cname.endswith(suffix):
                    base = cname[:-len(suffix)]
                    if base not in checked_bases:
                        checked_bases.add(base)
                        c = getattr(parent_mod, base, None)
                        b = getattr(parent_mod, base + '_bn', None)
                        l = getattr(parent_mod, base + '_lif', None) or getattr(parent_mod, base + '_sn', None)
                        if c is None or b is None or l is None:
                            continue
                        if not isinstance(l, neuron.LIFNode):
                            continue
                        if id(l) in replacements_done:
                            continue

                        full_name = f"{parent_name}.{base}" if parent_name else base

                        # Conv2d + BN2d + LIF
                        if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                            cin, cout = c.in_channels, c.out_channels
                            ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
                            stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
                            H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
                            row = get_policy("Conv3x3_BN_LIF", C_in=cin, C_out=cout, H=H_est, W=H_est, kernel_size=ksize, T=T)
                            use_sparse = force_sparse or (_sf_available and row.spatial_backend == SpatialBackend.SPARSE_FLOW)

                            if use_sparse and _sf_available and ksize == 3 and stride_val == 1:
                                b.eval()
                                fused = STFusionConvBNLIF.from_sj_modules(c, b, l, K=row.K)
                                pattern_name = "STFusionConvBNLIF"
                            else:
                                fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                                pattern_name = "PartialFusionConvBNLIF"

                            setattr(parent_mod, base, fused)
                            bn_attr = base + '_bn'
                            lif_attr = base + '_lif' if hasattr(parent_mod, base + '_lif') else base + '_sn'
                            setattr(parent_mod, bn_attr, nn.Identity())
                            setattr(parent_mod, lif_attr, nn.Identity())

                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                                "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                                "backend": row.spatial_backend if use_sparse else "DenseKeep", "K": row.K})

                        # Linear + BN1d + LIF
                        elif isinstance(c, nn.Linear) and isinstance(b, nn.BatchNorm1d):
                            import warnings
                            warnings.warn(
                                f"CATFuse: Linear+BN1d+LIF at {full_name}: BN1d is NOT absorbed into FusedLinearLIF. "
                                f"This path drops BN1d semantics. Skipping this pattern to preserve correctness.",
                                RuntimeWarning, stacklevel=2)
                            # Do NOT replace — skip to preserve BN1d semantics
                            continue

                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                                "shape": f"{c.in_features}->{c.out_features}",
                                "backend": "DenseKeep", "K": T})

        # Phase 1c: Generic attribute patterns (SpikFormer-style naming)
    # Scans all modules for (X, X_bn, X_lif) or (X, X_bn, X_sn) triples
    # where X is Conv2d or Linear, X_bn is BatchNorm, X_lif/X_sn is LIFNode
    for parent_name, parent_mod in list(model.named_modules()):
        # Collect child names
        child_names = {name for name, _ in parent_mod.named_children()}
        # Look for patterns: base / base_bn / base_lif (or base_sn)
        checked_bases = set()
        for cname in sorted(child_names):
            # Strip suffix to find base
            for suffix in ['_bn', '_lif', '_sn']:
                if cname.endswith(suffix):
                    base = cname[:-len(suffix)]
                    if base not in checked_bases:
                        checked_bases.add(base)
                        c = getattr(parent_mod, base, None)
                        b = getattr(parent_mod, base + '_bn', None)
                        l = getattr(parent_mod, base + '_lif', None) or getattr(parent_mod, base + '_sn', None)
                        if c is None or b is None or l is None:
                            continue
                        if not isinstance(l, neuron.LIFNode):
                            continue
                        if id(l) in replacements_done:
                            continue

                        full_name = f"{parent_name}.{base}" if parent_name else base

                        # Conv2d + BN2d + LIF
                        if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                            cin, cout = c.in_channels, c.out_channels
                            ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
                            stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
                            H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
                            row = get_policy("Conv3x3_BN_LIF", C_in=cin, C_out=cout, H=H_est, W=H_est, kernel_size=ksize, T=T)
                            use_sparse = force_sparse or (_sf_available and row.spatial_backend == SpatialBackend.SPARSE_FLOW)

                            if use_sparse and _sf_available and ksize == 3 and stride_val == 1:
                                b.eval()
                                fused = STFusionConvBNLIF.from_sj_modules(c, b, l, K=row.K)
                                pattern_name = "STFusionConvBNLIF"
                            else:
                                fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                                pattern_name = "PartialFusionConvBNLIF"

                            setattr(parent_mod, base, fused)
                            bn_attr = base + '_bn'
                            lif_attr = base + '_lif' if hasattr(parent_mod, base + '_lif') else base + '_sn'
                            setattr(parent_mod, bn_attr, nn.Identity())
                            setattr(parent_mod, lif_attr, nn.Identity())

                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                                "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                                "backend": row.spatial_backend if use_sparse else "DenseKeep", "K": row.K})

                        # Linear + BN1d + LIF
                        elif isinstance(c, nn.Linear) and isinstance(b, nn.BatchNorm1d):
                            import warnings
                            warnings.warn(
                                f"CATFuse: Linear+BN1d+LIF at {full_name}: BN1d is NOT absorbed into FusedLinearLIF. "
                                f"This path drops BN1d semantics. Skipping this pattern to preserve correctness.",
                                RuntimeWarning, stacklevel=2)
                            # Do NOT replace — skip to preserve BN1d semantics
                            continue

                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                                "shape": f"{c.in_features}->{c.out_features}",
                                "backend": "DenseKeep", "K": T})

        # Phase 1d: Numeric suffix patterns (SpikFormer SPS: conv0/bn0/lif0)
    for parent_name, parent_mod in list(model.named_modules()):
        child_names = {name for name, _ in parent_mod.named_children()}
        for i in range(10):
            # Pattern: conv{i} / bn{i} / lif{i}
            conv_name = f"conv{i}"
            bn_name = f"bn{i}"
            lif_name = f"lif{i}"
            c = getattr(parent_mod, conv_name, None)
            b = getattr(parent_mod, bn_name, None)
            l = getattr(parent_mod, lif_name, None)
            if c is None or b is None or l is None:
                continue
            if not isinstance(l, neuron.LIFNode):
                continue
            if id(l) in replacements_done:
                continue
            full_name = f"{parent_name}.{conv_name}" if parent_name else conv_name

            if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                cin, cout = c.in_channels, c.out_channels
                ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
                stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
                H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
                row = get_policy("Conv3x3_BN_LIF", C_in=cin, C_out=cout, H=H_est, W=H_est, kernel_size=ksize, T=T)
                use_sparse = force_sparse or (_sf_available and row.spatial_backend == SpatialBackend.SPARSE_FLOW)

                if use_sparse and _sf_available and ksize == 3 and stride_val == 1:
                    b.eval()
                    fused = STFusionConvBNLIF.from_sj_modules(c, b, l, K=row.K)
                    pattern_name = "STFusionConvBNLIF"
                else:
                    fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                    pattern_name = "PartialFusionConvBNLIF"

                setattr(parent_mod, conv_name, fused)
                setattr(parent_mod, bn_name, nn.Identity())
                setattr(parent_mod, lif_name, nn.Identity())

                replacements_done.add(id(l))
                stats["fused_lif_nodes"] += 1
                stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                stats["routing"].append({"name": full_name, "pattern": pattern_name,
                    "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                    "backend": row.spatial_backend if use_sparse else "DenseKeep", "K": row.K})

        # Also handle rpe_conv / rpe_bn / rpe_lif
        for prefix in ["rpe"]:
            c = getattr(parent_mod, f"{prefix}_conv", None)
            b = getattr(parent_mod, f"{prefix}_bn", None)
            l = getattr(parent_mod, f"{prefix}_lif", None)
            if c is None or b is None or l is None:
                continue
            if not isinstance(l, neuron.LIFNode) or id(l) in replacements_done:
                continue
            if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                cin, cout = c.in_channels, c.out_channels
                ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
                stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
                full_name = f"{parent_name}.{prefix}_conv" if parent_name else f"{prefix}_conv"

                fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                pattern_name = "PartialFusionConvBNLIF"

                setattr(parent_mod, f"{prefix}_conv", fused)
                setattr(parent_mod, f"{prefix}_bn", nn.Identity())
                setattr(parent_mod, f"{prefix}_lif", nn.Identity())

                replacements_done.add(id(l))
                stats["fused_lif_nodes"] += 1
                stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                stats["routing"].append({"name": full_name, "pattern": pattern_name,
                    "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                    "backend": "DenseKeep", "K": T})

        # Phase 2: Sequential patterns (VGG) — policy-aware
    for parent_name, parent_mod in list(model.named_modules()):
        for name, child in list(parent_mod.named_children()):
            if isinstance(child, nn.Sequential):
                children = list(child)
                n = len(children)
                new_children = []
                seq_replacements = []
                i = 0
                while i < n:
                    # Try 3-tuple: (Conv, BN, LIF)
                    if (i + 2 < n
                            and _is_sj_conv2d(children[i])
                            and _is_sj_bn2d(children[i + 1])
                            and _is_lif(children[i + 2])):
                        c, b, l = children[i], children[i + 1], children[i + 2]
                        if id(l) not in replacements_done:
                            cin, cout = c.in_channels, c.out_channels
                            ksize = c.kernel_size[0] if isinstance(c.kernel_size, tuple) else c.kernel_size
                            stride_val = c.stride[0] if isinstance(c.stride, tuple) else c.stride
                            H_est = 32 if cin <= 64 else (16 if cin <= 128 else 8)
                            row = get_policy("Conv3x3_BN_LIF", C_in=cin, C_out=cout, H=H_est, W=H_est, kernel_size=ksize, T=T)
                            use_sparse = force_sparse or (_sf_available and row.spatial_backend == SpatialBackend.SPARSE_FLOW)
                            full_name = f"{parent_name}.{name}.{i}" if parent_name else f"{name}.{i}"
                            if use_sparse and _sf_available and ksize == 3 and stride_val == 1:
                                b.eval()
                                fused = STFusionConvBNLIF.from_sj_modules(c, b, l, K=row.K)
                                pattern_name = "STFusionConvBNLIF"
                            else:
                                fused = PartialFusionConvBNLIF.from_sj_modules(c, b, l)
                                pattern_name = "PartialFusionConvBNLIF"
                            new_children.append(fused)
                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"][pattern_name] = stats["patterns_matched"].get(pattern_name, 0) + 1
                            stats["routing"].append({"name": full_name, "pattern": pattern_name,
                                "shape": f"{cin}->{cout}, k={ksize}, s={stride_val}",
                                "backend": row.spatial_backend, "K": row.K})
                            seq_replacements.append((pattern_name, (i, i+1, i+2)))
                        else:
                            new_children.append(children[i])
                            new_children.append(children[i+1])
                            new_children.append(children[i+2])
                        i += 3
                        continue
                    # Try 2-tuple: (Linear, LIF)
                    if (i + 1 < n
                            and _is_sj_linear(children[i])
                            and _is_lif(children[i + 1])):
                        c, l = children[i], children[i + 1]
                        if id(l) not in replacements_done:
                            fused = FusedLinearLIF.from_sj_modules(c, l)
                            new_children.append(fused)
                            replacements_done.add(id(l))
                            stats["fused_lif_nodes"] += 1
                            stats["patterns_matched"]["FusedLinearLIF"] = stats["patterns_matched"].get("FusedLinearLIF", 0) + 1
                            full_name = f"{parent_name}.{name}.{i}" if parent_name else f"{name}.{i}"
                            stats["routing"].append({"name": full_name, "pattern": "FusedLinearLIF",
                                "shape": f"{c.in_features}->{c.out_features}",
                                "backend": "DenseKeep", "K": T})
                            seq_replacements.append(("FusedLinearLIF", (i, i+1)))
                        else:
                            new_children.append(children[i])
                            new_children.append(children[i+1])
                        i += 2
                        continue
                    new_children.append(children[i])
                    i += 1
                if seq_replacements:
                    setattr(parent_mod, name, nn.Sequential(*new_children))

    stats["coverage_pct"] = 100.0 * stats["fused_lif_nodes"] / max(stats["total_lif_nodes"], 1)
    return model, stats


def print_routing_table(stats):
    routing = stats.get("routing", [])
    if not routing:
        print("  No routing decisions recorded.")
        return
    print(f"  {'Layer':<42s} {'Pattern':<24s} {'Shape':<20s} {'Backend':<12s} {'K':>3s}")
    print("  " + "-" * 103)
    for r in routing:
        print(f"  {r['name']:<42s} {r['pattern']:<24s} {r['shape']:<20s} {r['backend']:<12s} {r['K']:>3d}")
    backends = {}
    for r in routing:
        backends[r["backend"]] = backends.get(r["backend"], 0) + 1
    print(f"\n  Backend summary: {backends}")
    print(f"  Total fused: {stats['fused_lif_nodes']}/{stats['total_lif_nodes']} ({stats['coverage_pct']:.1f}%)")


# ============================================================
# CLI smoke test
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sew_resnet18',
                        choices=['vgg11_bn', 'sew_resnet18', 'spiking_resnet18',
                                 'sew_resnet50', 'spiking_resnet50'])
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
    elif args.model == 'sew_resnet50':
        from spikingjelly.activation_based.model import sew_resnet
        print("Building SEWResNet50...")
        model = sew_resnet.sew_resnet50(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            num_classes=1000,
            cnf='ADD',
        )
        T, B, C, H = 4, 2, 3, 224
    elif args.model == 'spiking_resnet50':
        from spikingjelly.activation_based.model import spiking_resnet
        print("Building SpikingResNet50...")
        model = spiking_resnet.spiking_resnet50(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            num_classes=1000,
        )
        T, B, C, H = 4, 2, 3, 224

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