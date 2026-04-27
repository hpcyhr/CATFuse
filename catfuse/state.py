"""catfuse.state — CSR state container for CTF fused patterns.

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
