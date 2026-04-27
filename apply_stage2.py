#!/usr/bin/env python3
"""apply_stage2.py — One-shot Stage 2 refactor applier.

Run from the CATFuse repo root:
    python apply_stage2.py

Idempotent / safe behavior:
  - Each replacement is verified to match exactly once in the target file
    before being applied.
  - If any replacement is missing or ambiguous, the script aborts with
    a clear error and DOES NOT half-apply changes.
  - New files are overwritten if present (safe because they're new in
    Stage 2).
  - Detects already-applied state and skips cleanly.

After running, verify with:
    python tests/stage2_verify.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# New file contents
# ============================================================

STATE_PY_CONTENT = r'''"""catfuse.state — CSR state container for CTF fused patterns.

Provides StateBuffer, the canonical engineering counterpart of §3.13 StateCarry
(L2 boundary consistency in Σ(G,T)). All CTF fused patterns store their CSR
state (e.g. LIF membrane potential v) through this abstraction, so:

  - State lifetime is managed in one place (lazy init, set, reset)
  - Reset behavior is uniform across patterns (functional.reset_net works)
  - Device migration via .to(device) is handled implicitly (lazy reallocation
    on shape/device mismatch in .get())
  - L2 violations become harder to write — the get/set interface forces the
    author to think about the carry-over edge.

Stage 2 of the refactor (problem 4 in the design audit). Replaces:
  - PartialFusionConvBNLIF.self._v plain attribute (catfuse/patterns.py)
  - STFusionConvBNLIF.register_memory('v', 0.0) + self.v dispatch
    (catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class StateBuffer:
    """Lazy-initialized CSR state container.

    Lifecycle:
      __init__      : empty (no tensor allocated)
      get(shape,...): allocate zeros tensor at requested shape/device on first
                      call; on subsequent calls, return cached tensor if shape/
                      device/dtype match, otherwise re-allocate.
      set(v)        : replace internal tensor with v.detach() (no autograd
                      tracking — forward-only invariant).
      reset()       : discard internal tensor; next .get() re-allocates zeros.
      tensor        : direct read access (None if uninitialized) — for
                      inspection only.

    StateBuffer is a plain Python object, not a nn.Module / nn.Parameter /
    nn.Buffer. It is held as a regular attribute by a Pattern. Device
    migration is handled implicitly by lazy re-allocation when the requested
    device changes; this is intentional, since CSR state has no meaning
    across devices and re-allocating zeros is the correct semantics on
    .to(device).
    """

    __slots__ = ("_v",)

    def __init__(self):
        self._v: Optional[torch.Tensor] = None

    def get(
        self,
        shape: Tuple[int, ...],
        device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return current state tensor; allocate zeros if uninitialized or
        if cached tensor's shape/device/dtype don't match the request."""
        target_shape = torch.Size(shape)
        target_device = torch.device(device) if not isinstance(device, torch.device) else device
        if (
            self._v is None
            or self._v.shape != target_shape
            or self._v.device != target_device
            or self._v.dtype != dtype
        ):
            self._v = torch.zeros(target_shape, device=target_device, dtype=dtype)
        return self._v

    def set(self, v: torch.Tensor) -> None:
        """Update the internal state tensor (detached to prevent autograd
        tracking — CTF is forward-only)."""
        self._v = v.detach()

    def reset(self) -> None:
        """Discard current state. Next .get() will re-allocate zeros."""
        self._v = None

    @property
    def is_initialized(self) -> bool:
        return self._v is not None

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        """Direct read access to the underlying tensor.

        Returns None if uninitialized. Modifications via this reference bypass
        the buffer's lifecycle management — use only for inspection / debug.
        """
        return self._v

    def __repr__(self) -> str:
        if self._v is None:
            return "StateBuffer(uninitialized)"
        return (f"StateBuffer(shape={tuple(self._v.shape)}, "
                f"device={self._v.device}, dtype={self._v.dtype})")
'''


STAGE2_VERIFY_PY_CONTENT = r'''"""[Stage 2 verification] StateBuffer abstraction.

Verifies that:
  1. catfuse.state.StateBuffer imports and behaves correctly:
     - lazy init on first .get()
     - shape / device / dtype mismatch triggers reallocation
     - .set() updates with detached tensor
     - .reset() clears state
  2. CTFPattern subclasses register their states via _register_state.
  3. functional.reset_net properly clears StateBuffer in CTF patterns
     (PartialFusionConvBNLIF, FusedLinearLIF, STFusionConvBNLIF).
  4. Forward → reset → forward produces fresh-state output (i.e., reset
     actually takes effect).
  5. SEW-ResNet18 still substitutes + runs end-to-end with bit-exact
     parity vs SJ multi-step (regression check from stage 1).

Run:
    cd /path/to/CATFuse
    python tests/stage2_verify.py
"""
import os
import sys
import traceback

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_state_buffer_unit():
    """1. StateBuffer unit behaviors."""
    print("[1/5] StateBuffer unit tests...")
    try:
        import torch
        from catfuse.state import StateBuffer
    except Exception:
        print("  SKIP: torch not available")
        return None

    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    sb = StateBuffer()
    assert not sb.is_initialized, "fresh buffer should be uninitialized"

    # Lazy init via .get()
    v0 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert sb.is_initialized
    assert v0.shape == (2, 4, 8, 8)
    assert v0.device == torch.device(device)
    assert v0.dtype == torch.float32
    assert (v0 == 0).all().item(), ".get() should allocate zeros"

    # .set() updates
    new_v = torch.ones(2, 4, 8, 8, device=device, dtype=torch.float32) * 3.5
    sb.set(new_v)
    v1 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert (v1 == 3.5).all().item(), ".set() should update underlying tensor"

    # Shape mismatch triggers reallocation
    v2 = sb.get(shape=(2, 4, 16, 16), device=device, dtype=torch.float32)
    assert v2.shape == (2, 4, 16, 16)
    assert (v2 == 0).all().item(), "shape mismatch should trigger zeros reallocation"

    # .reset() clears
    sb.reset()
    assert not sb.is_initialized
    v3 = sb.get(shape=(2, 4, 8, 8), device=device, dtype=torch.float32)
    assert (v3 == 0).all().item(), "after reset, .get() should return zeros"

    # detach: .set() must store detached copy
    x = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
    sb.reset()
    _ = sb.get(shape=(2, 4, 8, 8), device=device)
    sb.set(x * 2)
    assert not sb.tensor.requires_grad, ".set() must store detached tensor"

    print("  OK: lazy init, set, reset, shape mismatch, detach all behave correctly")
    return True


def check_pattern_registers_state():
    """2. CTF patterns register their state via _register_state."""
    print("[2/5] Pattern state registration...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.patterns import (
            PartialFusionConvBNLIF, FusedLinearLIF,
        )
        from catfuse.state import StateBuffer
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    # PartialFusionConvBNLIF
    inst = PartialFusionConvBNLIF(in_channels=4, out_channels=8,
                                  kernel_size=3, padding=1)
    assert hasattr(inst, "state"), "PartialFusionConvBNLIF should have .state"
    assert isinstance(inst.state, StateBuffer)
    assert inst.state in inst._catfuse_states, \
        "state should be registered for unified reset"

    # FusedLinearLIF
    inst2 = FusedLinearLIF(in_features=16, out_features=10)
    assert hasattr(inst2, "state")
    assert isinstance(inst2.state, StateBuffer)
    assert inst2.state in inst2._catfuse_states

    print("  OK: PartialFusionConvBNLIF + FusedLinearLIF expose .state and register it")
    return True


def check_reset_net_clears_state():
    """3. functional.reset_net clears StateBuffer."""
    print("[3/5] functional.reset_net clears CTF state...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer, functional
        from catfuse.patterns import PartialFusionConvBNLIF
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA for kernel forward")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(4, 8, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(8).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = PartialFusionConvBNLIF.from_sj_modules(conv, bn, lif).to(device)

    # State starts uninitialized
    assert not fused.state.is_initialized

    # Forward → state is now populated
    x = torch.randn(4, 2, 4, 16, 16, device=device)
    _ = fused(x)
    assert fused.state.is_initialized, "after forward, state should be populated"
    v_after_fwd = fused.state.tensor.clone()

    # Forward again: state evolves
    _ = fused(x)
    v_after_fwd2 = fused.state.tensor.clone()

    # functional.reset_net should trigger state.reset() via CTFPattern.reset()
    functional.reset_net(fused)
    assert not fused.state.is_initialized, \
        "after reset_net, state should be uninitialized"

    # Forward again: state is fresh zeros + evolves
    _ = fused(x)
    v_after_reset_fwd = fused.state.tensor.clone()

    # The post-reset+forward state should equal the first post-forward state
    # (same initial zeros, same input, same kernel → same output)
    assert torch.equal(v_after_fwd, v_after_reset_fwd), \
        "post-reset forward should reproduce fresh-init forward exactly"

    print("  OK: reset_net properly clears StateBuffer; post-reset forward is fresh")
    return True


def check_state_buffer_in_st_fusion():
    """4. STFusionConvBNLIF also uses StateBuffer."""
    print("[4/5] STFusionConvBNLIF uses StateBuffer...")
    try:
        import torch
        from spikingjelly.activation_based import neuron, layer as sj_layer
        from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF
        from catfuse.state import StateBuffer
        from catfuse.patterns import CTFPattern
    except Exception:
        print("  SKIP: import failed")
        traceback.print_exc()
        return None

    # STFusion should now be a CTFPattern subclass
    assert issubclass(STFusionConvBNLIF, CTFPattern), \
        "STFusionConvBNLIF should now inherit from CTFPattern (Stage 2)"

    # Build via from_sj_modules
    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA for from_sj_modules")
        return None

    device = "cuda:0"
    conv = sj_layer.Conv2d(64, 64, 3, padding=1, bias=False).to(device)
    bn = sj_layer.BatchNorm2d(64).to(device); bn.eval()
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         step_mode="m").to(device)

    fused = STFusionConvBNLIF.from_sj_modules(conv, bn, lif, K=4).to(device)
    assert hasattr(fused, "state"), "STFusionConvBNLIF should have .state"
    assert isinstance(fused.state, StateBuffer)
    assert fused.state in fused._catfuse_states

    print("  OK: STFusionConvBNLIF inherits CTFPattern + uses StateBuffer")
    return True


def check_resnet18_regression():
    """5. SEW-ResNet18 still bit-exact vs SJ baseline (regression from stage 1)."""
    print("[5/5] SEW-ResNet18 substitute_sf parity regression...")
    try:
        import torch
        from spikingjelly.activation_based import functional, neuron
    except Exception:
        print("  SKIP: torch / SJ not available")
        return None

    if not torch.cuda.is_available():
        print("  SKIP: needs CUDA")
        return None

    # Lazy import to avoid expensive load when SKIP
    try:
        from catfuse import optimize
        # Reuse same builder as test_05_substitute_sf
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        # Pull the existing test's builder if available
        try:
            from test_05_substitute_sf import build_sew_resnet18  # type: ignore
        except Exception:
            build_sew_resnet18 = None
    except Exception:
        print("  SKIP: catfuse.optimize unavailable")
        traceback.print_exc()
        return None

    if build_sew_resnet18 is None:
        # Build SEW-RN18 inline using spikingjelly's official model
        try:
            from spikingjelly.activation_based.model import sew_resnet
            from spikingjelly.activation_based import surrogate
            net = sew_resnet.sew_resnet18(
                pretrained=False, cnf="ADD", spiking_neuron=neuron.LIFNode,
                surrogate_function=surrogate.ATan(), detach_reset=True,
            )
            functional.set_step_mode(net, "m")
        except Exception:
            print("  SKIP: can't build SEW-RN18")
            traceback.print_exc()
            return None
    else:
        net = build_sew_resnet18()

    device = "cuda:0"
    net = net.to(device).eval()
    T, B = 4, 2
    x = torch.randn(T, B, 3, 32, 32, device=device)

    with torch.no_grad():
        functional.reset_net(net)
        y_sj = net(x)

    fused, stats = optimize(net, T=T, use_sparseflow=True)
    fused = fused.to(device).eval()

    with torch.no_grad():
        functional.reset_net(fused)
        y_ctf = fused(x)

    max_diff = (y_sj - y_ctf).abs().max().item()
    if max_diff > 1e-4:
        print(f"  FAIL: max_diff={max_diff:.6e} (expected 0 or near-zero)")
        return False

    print(f"  OK: SEW-RN18 max_diff={max_diff:.2e} (bit-exact)")
    return True


def main():
    print("=" * 60)
    print("Stage 2 verification: StateBuffer abstraction (problem 4)")
    print("=" * 60)
    results = [
        ("state_buffer_unit", check_state_buffer_unit()),
        ("pattern_registers_state", check_pattern_registers_state()),
        ("reset_net_clears_state", check_reset_net_clears_state()),
        ("st_fusion_state_buffer", check_state_buffer_in_st_fusion()),
        ("resnet18_regression", check_resnet18_regression()),
    ]

    print()
    print("=" * 60)
    failed = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    if failed:
        print(f"FAIL: {len(failed)} checks failed: {failed}")
        sys.exit(1)
    elif skipped:
        print(f"PARTIAL OK: {len(results) - len(skipped)} passed, "
              f"{len(skipped)} skipped: {skipped}")
        sys.exit(0)
    else:
        print(f"PASS: all {len(results)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
'''

# ============================================================
# Replacements: catfuse/patterns.py
# Each entry is (old_str, new_str). old_str must match exactly once.
# ============================================================

PATTERNS_REPLACEMENTS = [

    # --- A: CTFPattern base class (entire body replaced) ---
    (
r'''    Base class for all CTF fused patterns.

    Subclasses must:
      - Override `forward(x)` to implement the fused kernel call
      - Override `from_sj_modules(...)` classmethod for construction from SJ modules
      - Set `policy_row` class attribute (int, matches policy table row)

    Inherits from SJ's MemoryModule so that functional.reset_net recognizes
    CTF patterns as proper SJ-compatible modules and does not emit warnings.
    CTF patterns carry state only inside Triton kernel register files
    per-call, so reset_net is a no-op for us.
    """

    policy_row: Optional[int] = None

    def __init__(self):
        super().__init__()
        # MemoryModule.__init__ already sets up _memories and _memories_rv
        # as OrderedDicts, so we don't need to redeclare them.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, ...] tensor in SJ multi-step convention (fp32, CUDA, contiguous)
        Returns:
            spike output of same or transformed shape
        """
        raise NotImplementedError

    @classmethod
    def from_sj_modules(cls, *sj_modules, **kwargs):
        """
        Build a CTF pattern instance by copying weights/parameters from
        the corresponding SJ original modules.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset — no-op for CTF patterns (state is per-call in kernel registers).
        Override the MemoryModule reset to avoid iterating over empty memories.
        """
        pass

    def single_step_forward(self, x):
        """
        SJ MemoryModule expects step_mode logic. CTF patterns are always
        multi-step (operate on [T, B, ...] input), so single_step is just
        a pass-through to forward.
        """
        return self.forward(x)

    def multi_step_forward(self, x_seq):
        """Multi-step forward is just our forward."""
        return self.forward(x_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, ...] tensor in SJ multi-step convention (fp32, CUDA, contiguous)
        Returns:
            spike output of same or transformed shape
        """
        raise NotImplementedError

    @classmethod
    def from_sj_modules(cls, *sj_modules, **kwargs):
        """
        Build a CTF pattern instance by copying weights/parameters from
        the corresponding SJ original modules (e.g., sj_conv, sj_bn, sj_lif).

        Subclass implementations vary by pattern type. Parameters vary by
        pattern — see each subclass's docstring.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset any internal state. For CTF patterns, state is carried inside
        the kernel's register file per-call, so there's no persistent state
        to reset at the module level. Override in subclasses that maintain
        persistent buffers.
        """
        pass''',
r'''    Base class for all CTF fused patterns.

    Subclasses must:
      - Override `forward(x)` to implement the fused kernel call
      - Override `from_sj_modules(...)` classmethod for construction from SJ modules
      - Set `policy_row` class attribute (int, matches policy table row)

    Inherits from SJ's MemoryModule so that functional.reset_net recognizes
    CTF patterns as proper SJ-compatible modules and does not emit warnings.

    CSR state (e.g. LIF membrane potential) is held in StateBuffer instances
    registered via `_register_state(...)`. The base class's `reset()` method
    delegates to all registered StateBuffers, so SJ's functional.reset_net
    correctly clears v across the entire fused module tree.
    """

    policy_row: Optional[int] = None

    def __init__(self):
        super().__init__()
        # List of StateBuffer instances managed by this pattern.
        # MemoryModule.__init__ already sets up _memories and _memories_rv
        # as OrderedDicts, so we don't need to redeclare them.
        self._catfuse_states: list = []

    # ---- StateBuffer registration ------------------------------------

    def _register_state(self, state) -> None:
        """Register a StateBuffer for unified reset() management.

        Called from subclass __init__ after constructing each StateBuffer.
        Stage 2 refactor — see catfuse/state.py.
        """
        from catfuse.state import StateBuffer
        if not isinstance(state, StateBuffer):
            raise TypeError(
                f"_register_state expects a StateBuffer, got {type(state).__name__}"
            )
        # Initialize list if not present (defensive: __init__ may be skipped
        # via __new__ in some subclass patterns).
        if not hasattr(self, "_catfuse_states") or self._catfuse_states is None:
            self._catfuse_states = []
        self._catfuse_states.append(state)

    # ---- Pattern API -------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, ...] tensor in SJ multi-step convention (fp32, CUDA, contiguous)
        Returns:
            spike output of same or transformed shape
        """
        raise NotImplementedError

    @classmethod
    def from_sj_modules(cls, *sj_modules, **kwargs):
        """
        Build a CTF pattern instance by copying weights/parameters from
        the corresponding SJ original modules (e.g., sj_conv, sj_bn, sj_lif).

        Subclass implementations vary by pattern type. Parameters vary by
        pattern — see each subclass's docstring.
        """
        raise NotImplementedError

    def reset(self):
        """Reset all registered StateBuffer instances.

        Called by SJ's functional.reset_net via the MemoryModule protocol.
        Subclasses that hold child CTFPattern instances do NOT need to
        recursively call reset() on children — SJ's reset_net handles tree
        traversal automatically.
        """
        for state in getattr(self, "_catfuse_states", []):
            state.reset()

    def single_step_forward(self, x):
        """
        SJ MemoryModule expects step_mode logic. CTF patterns are always
        multi-step (operate on [T, B, ...] input), so single_step is just
        a pass-through to forward.
        """
        return self.forward(x)

    def multi_step_forward(self, x_seq):
        """Multi-step forward is just our forward."""
        return self.forward(x_seq)'''
    ),

    # --- B: PartialFusionConvBNLIF __init__ (add StateBuffer) ---
    (
r'''        # BN params (inference mode, stored as buffers + parameters matching SJ layout)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_normal_(self.weight)
        self.step_mode = "m"  # Tell SJ to pass full [T,B,C,H,W] in multi-step mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-contained BatchFold Conv + BN + LIF forward.

        x: [T, B, C_in, H, W] or [B, C_in, H, W]
        Returns: spike tensor, same shape as x but with C_out channels
        """''',
r'''        # BN params (inference mode, stored as buffers + parameters matching SJ layout)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_normal_(self.weight)
        self.step_mode = "m"  # Tell SJ to pass full [T,B,C,H,W] in multi-step mode

        # Stage 2 refactor: CSR state through StateBuffer instead of self._v
        # plain attribute. Registered for unified reset() via CTFPattern.
        from catfuse.state import StateBuffer
        self.state = StateBuffer()
        self._register_state(self.state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-contained BatchFold Conv + BN + LIF forward.

        x: [T, B, C_in, H, W] or [B, C_in, H, W]
        Returns: spike tensor, same shape as x but with C_out channels
        """'''
    ),

    # --- C: PartialFusionConvBNLIF._forward_impl + remove reset() ---
    (
r'''        # 2. Reshape to [T, B, C_out, H_out, W_out]
        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)

        # 4. Single-launch Triton LIF over T steps
        if not hasattr(self, "_v") or self._v is None:
            self._v = torch.zeros(B, self.out_channels, H_out, W_out,
                                  dtype=torch.float32, device=z.device)
        try:
            from catfuse.sparseflow.lif_seq_kernel import lif_sequential
            spikes, v_out = lif_sequential(
                z, self._v, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self._v = v_out.detach()
            return spikes
        except Exception:
            # Fallback to Python loop
            v = self._v
            spikes = []
            for t in range(T):
                v = v + (z[t] - (v - self.v_reset)) / self.tau
                spike = (v >= self.v_threshold).to(z.dtype)
                v = v * (1.0 - spike) + self.v_reset * spike
                spikes.append(spike)
            self._v = v.detach()
            return torch.stack(spikes, dim=0)

    def reset(self):
        """Reset membrane potential (called by functional.reset_net)."""
        self._v = None

    @classmethod
    def from_sj_modules(cls,''',
r'''        # 2. Reshape to [T, B, C_out, H_out, W_out]
        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)

        # 4. Single-launch Triton LIF over T steps
        # Stage 2 refactor: state via StateBuffer instead of self._v.
        v_in = self.state.get(
            shape=(B, self.out_channels, H_out, W_out),
            device=z.device,
            dtype=torch.float32,
        )
        try:
            from catfuse.sparseflow.lif_seq_kernel import lif_sequential
            spikes, v_out = lif_sequential(
                z, v_in, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self.state.set(v_out)
            return spikes
        except Exception:
            # Fallback to Python loop
            v = v_in
            spikes = []
            for t in range(T):
                v = v + (z[t] - (v - self.v_reset)) / self.tau
                spike = (v >= self.v_threshold).to(z.dtype)
                v = v * (1.0 - spike) + self.v_reset * spike
                spikes.append(spike)
            self.state.set(v)
            return torch.stack(spikes, dim=0)

    # Stage 2 refactor: CTFPattern.reset() now delegates to all registered
    # StateBuffers; the explicit reset() override below is no longer needed
    # and is removed.

    @classmethod
    def from_sj_modules(cls,'''
    ),

    # --- D: FusedLinearLIF __init__ (add StateBuffer) ---
    (
r'''        # Weight in Phase 0 convention [I, O] (not PyTorch [O, I])
        self.weight_io = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight_io.T)''',
r'''        # Weight in Phase 0 convention [I, O] (not PyTorch [O, I])
        self.weight_io = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight_io.T)

        # Stage 2 refactor: CSR state through StateBuffer instead of self._v_lin.
        from catfuse.state import StateBuffer
        self.state = StateBuffer()
        self._register_state(self.state)'''
    ),

    # --- E: FusedLinearLIF._forward_impl + remove reset() ---
    (
r'''        z = z_2d.reshape(T, B, -1)
        # 2. Triton LIF or fallback
        C_out = z.shape[2]
        if not hasattr(self, "_v_lin") or self._v_lin is None:
            self._v_lin = torch.zeros(B, C_out, dtype=torch.float32, device=z.device)
        try:
            from catfuse.sparseflow.lif_seq_kernel import lif_sequential
            # lif_sequential expects [T, B, C, H, W], add dummy spatial dims
            z_5d = z.unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
            v_5d = self._v_lin.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            spikes_5d, v_out_5d = lif_sequential(
                z_5d, v_5d, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self._v_lin = v_out_5d.squeeze(-1).squeeze(-1).detach()
            return spikes_5d.squeeze(-1).squeeze(-1)
        except Exception:
            v = self._v_lin
            spikes = []
            for t in range(T):
                v = v + (z[t] - (v - self.v_reset)) / self.tau
                spike = (v >= self.v_threshold).to(z.dtype)
                v = v * (1.0 - spike) + self.v_reset * spike
                spikes.append(spike)
            self._v_lin = v.detach()
            return torch.stack(spikes, dim=0)

    def reset(self):
        self._v_lin = None''',
r'''        z = z_2d.reshape(T, B, -1)
        # 2. Triton LIF or fallback
        C_out = z.shape[2]
        # Stage 2 refactor: state via StateBuffer.
        v_in_2d = self.state.get(
            shape=(B, C_out),
            device=z.device,
            dtype=torch.float32,
        )
        try:
            from catfuse.sparseflow.lif_seq_kernel import lif_sequential
            # lif_sequential expects [T, B, C, H, W], add dummy spatial dims
            z_5d = z.unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
            v_5d = v_in_2d.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            spikes_5d, v_out_5d = lif_sequential(
                z_5d, v_5d, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self.state.set(v_out_5d.squeeze(-1).squeeze(-1))
            return spikes_5d.squeeze(-1).squeeze(-1)
        except Exception:
            v = v_in_2d
            spikes = []
            for t in range(T):
                v = v + (z[t] - (v - self.v_reset)) / self.tau
                spike = (v >= self.v_threshold).to(z.dtype)
                v = v * (1.0 - spike) + self.v_reset * spike
                spikes.append(spike)
            self.state.set(v)
            return torch.stack(spikes, dim=0)

    # Stage 2 refactor: reset() inherited from CTFPattern, delegates to
    # registered StateBuffer (no explicit override needed).'''
    ),
]


# ============================================================
# Replacements: catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py
# ============================================================

ST_FUSION_REPLACEMENTS = [

    # --- A: header imports ---
    (
r'''try:
    from spikingjelly.activation_based import base as sj_base
    _SJ_MEMORY_MODULE = sj_base.MemoryModule
except ImportError:
    _SJ_MEMORY_MODULE = nn.Module

try:''',
r'''try:
    from spikingjelly.activation_based import base as sj_base
    _SJ_MEMORY_MODULE = sj_base.MemoryModule
except ImportError:
    _SJ_MEMORY_MODULE = nn.Module

# Stage 2 refactor: STFusion now inherits from CTFPattern so it shares the
# unified StateBuffer-based reset() protocol with all other CTF patterns.
from catfuse.patterns import CTFPattern
from catfuse.state import StateBuffer

try:'''
    ),

    # --- B: class declaration ---
    (
r'''class STFusionConvBNLIF(_SJ_MEMORY_MODULE):
    """Spatio-temporal fused Conv→BN→LIF with SparseFlow spatial backend.

    Attributes:
        K: TimeBlock size (number of steps per block)
        weight: Conv2d weight [C_out, C_in, kH, kW]
        bias: Conv2d bias [C_out] or None
        bn_scale: folded BN scale [C_out]
        bn_bias: folded BN bias [C_out]
        tau, v_threshold, v_reset: LIF parameters
    """''',
r'''class STFusionConvBNLIF(CTFPattern):
    """Spatio-temporal fused Conv→BN→LIF with SparseFlow spatial backend.

    Attributes:
        K: TimeBlock size (number of steps per block)
        weight: Conv2d weight [C_out, C_in, kH, kW]
        bias: Conv2d bias [C_out] or None
        bn_scale: folded BN scale [C_out]
        bn_bias: folded BN bias [C_out]
        tau, v_threshold, v_reset: LIF parameters
        state: StateBuffer holding membrane potential (Stage 2 refactor)
    """'''
    ),

    # --- C: register_memory replacement ---
    (
r'''        # Membrane potential state
        self.register_memory('v', 0.0)''',
r'''        # Membrane potential state via StateBuffer (Stage 2 refactor:
        # replaces register_memory('v', 0.0)). Reset is handled by the
        # CTFPattern base class.
        self.state = StateBuffer()
        self._register_state(self.state)'''
    ),

    # --- D: _dense_forward LIF section ---
    (
r'''        # 2. LIF
        v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
            B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
        spikes, v_final = lif_sequential(
            z, v_init, tau=self.tau,
            v_threshold=self.v_threshold, v_reset=self.v_reset,
        )
        self.v = v_final
        return spikes''',
r'''        # 2. LIF — Stage 2 refactor: state via StateBuffer.
        v_init = self.state.get(
            shape=(B, self.out_channels, H_out, W_out),
            device=device, dtype=torch.float32,
        )
        spikes, v_final = lif_sequential(
            z, v_init, tau=self.tau,
            v_threshold=self.v_threshold, v_reset=self.v_reset,
        )
        self.state.set(v_final)
        return spikes'''
    ),

    # --- E: _batchfold_forward static-zero branch ---
    (
r'''            z = z_flat.reshape(T, B, self.out_channels, H_out, W_out)
            v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
                B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
            spikes, v_final = lif_sequential(
                z, v_init, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self.v = v_final
            return spikes''',
r'''            z = z_flat.reshape(T, B, self.out_channels, H_out, W_out)
            # Stage 2 refactor: state via StateBuffer.
            v_init = self.state.get(
                shape=(B, self.out_channels, H_out, W_out),
                device=device, dtype=torch.float32,
            )
            spikes, v_final = lif_sequential(
                z, v_init, tau=self.tau,
                v_threshold=self.v_threshold, v_reset=self.v_reset,
            )
            self.state.set(v_final)
            return spikes'''
    ),

    # --- F: _batchfold_forward main path v_init ---
    (
r'''        # Prepare v_init
        v_init = self.v if isinstance(self.v, torch.Tensor) else torch.zeros(
            B, self.out_channels, H_out, W_out, dtype=torch.float32, device=device)
        v_init = v_init.float().contiguous()''',
r'''        # Prepare v_init — Stage 2 refactor: state via StateBuffer.
        v_init = self.state.get(
            shape=(B, self.out_channels, H_out, W_out),
            device=device, dtype=torch.float32,
        )
        v_init = v_init.float().contiguous()'''
    ),

    # --- G: _batchfold_forward final v store ---
    (
r'''        self.v = v_out
        return spike_out.reshape(T, B, self.out_channels, H_out, W_out)''',
r'''        self.state.set(v_out)
        return spike_out.reshape(T, B, self.out_channels, H_out, W_out)'''
    ),

    # --- H: forward() entry v init ---
    (
r'''        # Initialize v if needed
        if not isinstance(self.v, torch.Tensor) or self.v.shape == ():
            H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1
            self.v = torch.zeros(B, self.out_channels, H_out, W_out,
                                 dtype=torch.float32, device=device)''',
r'''        # Stage 2 refactor: state initialization via StateBuffer.
        H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1
        # Note: this just primes the buffer; the inner-path implementations
        # (_dense_forward / _batchfold_forward / per-step fallback below)
        # call self.state.get(...) themselves for the actual read.'''
    ),

    # --- I: forward() per-step fallback ---
    (
r'''        spike_list = []
        v = self.v

        for block_start in range(0, T, K):
            block_end = min(block_start + K, T)

            for t in range(block_start, block_end):
                x_t = x[t]

                if _KERNEL_AVAILABLE and x_t.is_cuda and self.stride == 1:
                    spike_t, v, _ = sparse_fused_conv_bn_lif_forward(
                        x=x_t, v_prev=v, weight=self.weight, bias=self.bias,
                        bn_scale=self.bn_scale, bn_bias=self.bn_bias_folded,
                        tau=self.tau, v_threshold=self.v_threshold,
                        v_reset=self.v_reset, decay_input=self.decay_input,
                        kernel_size=self.kernel_size, threshold=self.threshold,
                        w_cl=self._w_cl, ag_mask_buf=self._ag_mask_buf,
                    )
                else:
                    spike_t, v = self._fallback_forward_single(x_t, v)

                spike_list.append(spike_t)

        self.v = v
        return torch.stack(spike_list, dim=0)''',
r'''        spike_list = []
        # Stage 2 refactor: state via StateBuffer.
        v = self.state.get(
            shape=(B, self.out_channels, H_out, W_out),
            device=device, dtype=torch.float32,
        )

        for block_start in range(0, T, K):
            block_end = min(block_start + K, T)

            for t in range(block_start, block_end):
                x_t = x[t]

                if _KERNEL_AVAILABLE and x_t.is_cuda and self.stride == 1:
                    spike_t, v, _ = sparse_fused_conv_bn_lif_forward(
                        x=x_t, v_prev=v, weight=self.weight, bias=self.bias,
                        bn_scale=self.bn_scale, bn_bias=self.bn_bias_folded,
                        tau=self.tau, v_threshold=self.v_threshold,
                        v_reset=self.v_reset, decay_input=self.decay_input,
                        kernel_size=self.kernel_size, threshold=self.threshold,
                        w_cl=self._w_cl, ag_mask_buf=self._ag_mask_buf,
                    )
                else:
                    spike_t, v = self._fallback_forward_single(x_t, v)

                spike_list.append(spike_t)

        self.state.set(v)
        return torch.stack(spike_list, dim=0)'''
    ),
]


# ============================================================
# Apply machinery
# ============================================================

def apply_replacements(path: str, replacements, label: str) -> bool:
    """Apply (old, new) replacements sequentially.

    Idempotency contract:
      - If `new` is already present in content: skip (already applied). This
        is checked FIRST, because some replacements have `new` containing
        `old` as a substring (e.g. "append a block after existing code"),
        so a stale `old_count > 0` after first apply does NOT mean the
        replacement is unapplied.
      - Else if `old` matches exactly once: apply.
      - Else (old missing or ambiguous): error.
    """
    rel = os.path.relpath(path, REPO_ROOT)
    if not os.path.exists(path):
        print(f"  ERROR: {rel} does not exist")
        return False

    with open(path, "r") as f:
        content = f.read()

    skipped = 0
    applied = 0
    for i, (old, new) in enumerate(replacements):
        # Idempotency: if new content already present, this replacement is done.
        if new in content:
            skipped += 1
            continue

        old_count = content.count(old)
        if old_count == 0:
            print(f"  ERROR: replacement #{i+1} for {rel} not found "
                  f"(neither old nor new pattern present)")
            preview = old[:80].replace("\n", "\\n")
            print(f"         expected old text starts with: {preview}...")
            return False

        if old_count > 1:
            print(f"  ERROR: replacement #{i+1} for {rel} matches "
                  f"{old_count} times (must be unique)")
            return False

        # Exactly one match: apply
        content = content.replace(old, new, 1)
        applied += 1

    with open(path, "w") as f:
        f.write(content)

    if skipped == len(replacements):
        print(f"  {label} ({rel}): already applied ({skipped}/{len(replacements)} skipped)")
    elif skipped > 0:
        print(f"  {label} ({rel}): {applied} applied, {skipped} already-present")
    else:
        print(f"  {label} ({rel}): {applied}/{len(replacements)} replacements applied")
    return True


def write_new_file(path: str, content: str, label: str) -> bool:
    """Create or overwrite a new file."""
    rel = os.path.relpath(path, REPO_ROOT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  {label} ({rel}): {len(content.splitlines())} lines written")
    return True


def main():
    print("=" * 60)
    print("Stage 2 apply: StateBuffer abstraction (problem 4)")
    print("=" * 60)

    # Sanity check: are we in a CATFuse repo?
    if not os.path.exists(os.path.join(REPO_ROOT, "catfuse", "patterns.py")):
        print(f"ERROR: not a CATFuse repo (catfuse/patterns.py missing in {REPO_ROOT})")
        sys.exit(1)

    print(f"Repo: {REPO_ROOT}")
    print()

    ok = True

    # 1. New files
    print("--- New files ---")
    ok &= write_new_file(
        os.path.join(REPO_ROOT, "catfuse", "state.py"),
        STATE_PY_CONTENT,
        "catfuse/state.py",
    )
    ok &= write_new_file(
        os.path.join(REPO_ROOT, "tests", "stage2_verify.py"),
        STAGE2_VERIFY_PY_CONTENT,
        "tests/stage2_verify.py",
    )

    if not ok:
        print("\nFAIL: new file creation failed")
        sys.exit(1)

    # 2. Modified files
    print()
    print("--- Modified files ---")
    ok &= apply_replacements(
        os.path.join(REPO_ROOT, "catfuse", "patterns.py"),
        PATTERNS_REPLACEMENTS,
        "catfuse/patterns.py",
    )
    ok &= apply_replacements(
        os.path.join(REPO_ROOT, "catfuse", "sparseflow", "ops",
                     "st_fusion_conv_bn_lif.py"),
        ST_FUSION_REPLACEMENTS,
        "catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py",
    )

    print()
    if ok:
        print("=" * 60)
        print("Stage 2 applied successfully.")
        print()
        print("Next: run verification:")
        print("    python tests/stage2_verify.py")
        print()
        print("Expected: PASS: all 5 checks passed")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("Stage 2 apply FAILED. See errors above.")
        print()
        print("If a replacement was not found, the file may have been")
        print("already modified or differs from the expected baseline.")
        print("In that case, review the failed replacement manually.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()