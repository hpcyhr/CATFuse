"""catfuse.implementations.base — abstractions for Definition 3.16.

Per Definition 3.16.1 (Operator Implementation), an Implementation is a
bit-exact realization of σ ∈ Σ(G,T) that takes (input, spec, params, state)
and produces output spikes while updating state.

Per Definition 3.16.2 (Implementation Set Impl(σ)), multiple Implementations
may exist for the same σ, all bit-exact equivalent to each other on every
input.

Per Corollary 3.17 (Implementation Selection preserves equivalence), a
runtime selector π : Ctx → Impl(σ) may freely choose among them at each
call without changing the global semantics.

The selector itself (Definition 3.16.3) is implemented at the Pattern level
(see STFusionConvBNLIF) and is OUT OF SCOPE for this module — base.py defines
only the contract that all implementations must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Spec & Params
# ============================================================

@dataclass(frozen=True)
class ConvLIFSpec:
    """Static topology of a Conv→(BN)→LIF op σ.

    Frozen at pattern construction; does not change during forward.
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    has_conv_bias: bool   # whether the conv carries a bias term
    has_bn: bool          # whether BN was present (folded into bn_scale/bn_bias)
    tau: float
    v_threshold: float
    v_reset: Optional[float]   # None = soft reset; float = hard reset
    decay_input: bool = True
    bn_eps: float = 1e-5

    def output_hw(self, h_in: int, w_in: int):
        """Compute (H_out, W_out) for given input spatial size."""
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out


class ConvLIFParams:
    """Runtime tensor params of σ (references, not copies).

    The pattern owns these tensors as nn.Parameter / nn.Buffer; ConvLIFParams
    just bundles them for passing to an Implementation. Implementations must
    not modify the tensors here (they are the pattern's source of truth).

    bn_scale and bn_bias are PRE-FOLDED affine BN params:
      bn_scale = bn_weight / sqrt(running_var + eps)
      bn_bias  = bn_bias_raw - running_mean * bn_scale
    For the Conv→BN→LIF pattern, applying these to z = Conv(x) yields:
      z' = z * bn_scale[None, :, None, None] + bn_bias[None, :, None, None]
    which equals BN(Conv(x)) in eval mode.
    """
    __slots__ = ("weight", "bias", "bn_scale", "bn_bias")

    def __init__(
        self,
        weight: torch.Tensor,                    # [C_out, C_in, k, k]
        bias: Optional[torch.Tensor],            # [C_out] or None
        bn_scale: Optional[torch.Tensor],        # [C_out] or None (no BN)
        bn_bias: Optional[torch.Tensor],         # [C_out] or None (no BN)
    ):
        self.weight = weight
        self.bias = bias
        self.bn_scale = bn_scale
        self.bn_bias = bn_bias


@dataclass(frozen=True)
class IOCost:
    """Analytic HBM access count, broken down by tensor.

    Each field is in BYTES. The breakdown matches the §3.9 cost model:

      x_load:      input activation read from HBM
      w_load:      conv weight read from HBM (per kernel launch)
      z_io:        intermediate conv output traffic (write + read)
                   — eliminated by StreamFuse, this is the §3.9 savings target
      v_io:        membrane state carry (read + write per TimeBlock boundary)
                   — for baseline lif_sequential: just initial+final (2·HWC)
                   — for CTF with TimeBlock(K): 2·ceil(T/K)·HWC
      spike_write: output spike tensor write to HBM

    intermediate_io = z_io + v_io is the "moves between time steps" term
    that §3.9 reduces from O(T·HWC) to O(T/K·HWC).

    total = sum of all fields above.
    """
    x_load: int
    w_load: int
    z_io: int
    v_io: int
    spike_write: int
    schedule: str       # human-readable schedule descriptor
    num_blocks: int     # number of TimeBlock instances

    @property
    def intermediate_io(self) -> int:
        """The 'inter-step state movement' term (§3.9 target)."""
        return self.z_io + self.v_io

    @property
    def total(self) -> int:
        return self.x_load + self.w_load + self.z_io + self.v_io + self.spike_write

    def as_dict(self) -> dict:
        return {
            "x_load": self.x_load,
            "w_load": self.w_load,
            "z_io": self.z_io,
            "v_io": self.v_io,
            "spike_write": self.spike_write,
            "intermediate_io": self.intermediate_io,
            "total": self.total,
            "schedule": self.schedule,
            "num_blocks": self.num_blocks,
        }


# ============================================================
# Implementation ABC
# ============================================================

class Implementation(nn.Module, ABC):
    """Abstract base for one element of Impl(σ).

    Subclasses must:
      - Set class attribute `name` (str): e.g. "DenseKeep", "SparseFlow".
      - Override forward(x, spec, params, state) producing bit-exact output.
      - Override analytic_io_cost(spec, T, B, H_in, W_in, K) returning the
        §3.9 HBM traffic prediction for that scheduling choice.

    Bit-exactness contract (Corollary 3.17):
        ∀ x, ∀ a, b ∈ Impl(σ):
            a.forward(x, spec, params, state_a) ≡ b.forward(x, spec, params, state_b)
        pointwise on every (T, B, C_out, H_out, W_out) entry, given identical
        x, spec, params, and identical state initial condition.

    Note that bit-exactness DOES NOT imply equal I/O cost — that's exactly
    why analytic_io_cost is a separate method. Two impls in Impl(σ) may
    produce identical outputs while moving very different numbers of bytes
    through HBM. That delta is the §3.9 optimization target.

    Implementations may carry IMPL-LOCAL caches (e.g. a pre-folded weight,
    a channel-last layout, prescan buffers). Such caches must be invalidated
    via `reset_caches()` if the pattern's params change after construction.
    """

    name: str = "<unset>"

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,                # [T, B, C_in, H, W], fp32, CUDA, contiguous
        spec: ConvLIFSpec,
        params: ConvLIFParams,
        state: "StateBuffer",           # type: ignore[name-defined]  (catfuse.state.StateBuffer)
    ) -> torch.Tensor:                  # [T, B, C_out, H_out, W_out], fp32
        """Run the bit-exact realization of σ.

        Reads state via state.get(...), writes via state.set(v_final).
        """
        ...

    @abstractmethod
    def analytic_io_cost(
        self,
        spec: ConvLIFSpec,
        T: int, B: int, H_in: int, W_in: int,
        K: int = None,
        *,
        dtype_bytes: int = 4,
    ) -> IOCost:
        """Predict HBM traffic in bytes for one forward pass.

        This is the §3.9 cost model evaluated symbolically — it does NOT
        run the kernel and does NOT depend on input data. It captures the
        worst-case (dense input) HBM access count of the schedule the impl
        realizes.

        K is the TimeBlock(K) parameter (Definition 3.13). For impls that
        do not parametrize on K (e.g. DenseKeep is pinned at K=T via
        BatchFold), K is ignored and the result reflects the impl's fixed
        schedule. For impls that DO use K (e.g. SparseFlow), K=None means
        "use my default" (typically K=T, lean batchfold).

        dtype_bytes is the tensor element size. Default 4 (fp32). Pass 2
        for fp16 weight/activation paths.

        Static-zero short-circuits and sparsity-driven w_load reductions
        are NOT modeled here — they are data-dependent. Empirical
        measurements may come in lower than this prediction.
        """
        ...

    @abstractmethod
    def schedule_decomposition(
        self,
        spec: ConvLIFSpec,
        T: int,
        K: int = None,
    ) -> "ScheduleDecomposition":
        """Return the §3.13 primitive-transform decomposition of this impl's schedule.

        Each Implementation realizes one specific σ ∈ Σ(G,T) — a composition
        of the four §3.13 primitives (TimeBlock, BatchFold, StreamFuse,
        StateCarry). This method makes that composition explicit, both for
        documentation (which form does this impl correspond to?) and for
        verification (does the decomposition obey TSI/CSR typing rules?).

        Returns a ScheduleDecomposition whose .verify() must pass for the
        impl to be considered a legal element of CTF(G,T) per Definition 3.13.

        K parameterizes TimeBlock for impls that support it; ignored otherwise.
        """
        ...

    def reset_caches(self) -> None:
        """Invalidate impl-local caches. Default: no-op.

        Override in subclasses that hold lazy caches dependent on params
        (e.g. DenseKeep's pre-folded weight, SparseFlow's _w_cl).
        """
        pass


# ============================================================
# Shared utilities
# ============================================================

def static_zero_forward(
    spec: ConvLIFSpec,
    params: ConvLIFParams,
    T: int,
    B: int,
    H_out: int,
    W_out: int,
    device,
    state,
    lif_sequential_fn,
) -> torch.Tensor:
    """Bit-exact short-circuit when input x is identically zero.

    When x ≡ 0, Conv(x) ≡ 0, and z = BN(0) + bias = bn_bias (or conv_bias if
    no BN), broadcast to (T*B, C_out, H_out, W_out). This avoids invoking
    the conv kernel at all. The LIF step still runs (state evolves under
    constant input).

    Math identity (verified): for any conv with weight W and any bias b,
        Conv(0; W, b) = b broadcast to spatial dims
    Combined with BN(z; γ_folded, β_folded) = z * γ + β:
        BN(Conv(0)) = b * γ + β    if conv has bias
                    = β            otherwise
    This is what we materialize as z_4d below.

    All implementations should call this when x.any() == False, since the
    result is independent of the implementation choice.
    """
    if spec.has_bn:
        if spec.has_conv_bias:
            # z = conv_bias * bn_scale + bn_bias
            z_per_channel = params.bias * params.bn_scale + params.bn_bias
        else:
            # z = bn_bias
            z_per_channel = params.bn_bias
        z_4d = z_per_channel.view(1, -1, 1, 1).expand(
            T * B, spec.out_channels, H_out, W_out
        ).contiguous()
    elif spec.has_conv_bias:
        z_4d = params.bias.view(1, -1, 1, 1).expand(
            T * B, spec.out_channels, H_out, W_out
        ).contiguous()
    else:
        z_4d = torch.zeros(
            T * B, spec.out_channels, H_out, W_out,
            dtype=torch.float32, device=device,
        )
    z = z_4d.reshape(T, B, spec.out_channels, H_out, W_out)

    v_in = state.get(
        shape=(B, spec.out_channels, H_out, W_out),
        device=device, dtype=torch.float32,
    )
    spikes, v_out = lif_sequential_fn(
        z, v_in, tau=spec.tau,
        v_threshold=spec.v_threshold, v_reset=spec.v_reset,
    )
    state.set(v_out)
    return spikes


# ============================================================
# Schedule decomposition — §3.13 primitive transforms
# ============================================================

# Operator classification per §3.2.
# TSI = Time-Shape Invariant: each timestep computes independently of others
# CSR = Causal State Recurrent: state at t depends on previous timesteps
# TGO = Time-Global Operator: requires all timesteps simultaneously (out of CTF scope)
TSI_OPS = frozenset({"Conv", "BN", "Linear", "Add", "Pool"})
CSR_OPS = frozenset({"LIF", "PSN"})
TGO_OPS = frozenset({"Attention", "GroupNorm", "LayerNorm"})


@dataclass(frozen=True)
class ScheduleTransform:
    """One application of a §3.13 primitive transform.

    Four primitives, with their well-formedness constraints:
      TimeBlock(K)      : K >= 1; partitions T steps into ceil(T/K) blocks
      BatchFold(op)     : op ∈ TSI_OPS; folds time into batch dim for op
      StreamFuse(u, v)  : u ∈ TSI_OPS, v ∈ CSR_OPS; on-chip dataflow u → v
      StateCarry(op)    : op ∈ CSR_OPS; precise state transfer at block boundaries
    """
    primitive: str
    args: tuple   # (K,) | (op,) | (u, v) — fixed positional per primitive

    # ---- Factories ----
    @classmethod
    def TimeBlock(cls, K: int) -> "ScheduleTransform":
        return cls("TimeBlock", (int(K),))

    @classmethod
    def BatchFold(cls, op: str) -> "ScheduleTransform":
        return cls("BatchFold", (str(op),))

    @classmethod
    def StreamFuse(cls, u: str, v: str) -> "ScheduleTransform":
        return cls("StreamFuse", (str(u), str(v)))

    @classmethod
    def StateCarry(cls, op: str) -> "ScheduleTransform":
        return cls("StateCarry", (str(op),))

    # ---- Display ----
    def __str__(self) -> str:
        if self.primitive == "TimeBlock":
            return f"TimeBlock(K={self.args[0]})"
        if self.primitive == "BatchFold":
            return f"BatchFold({self.args[0]})"
        if self.primitive == "StreamFuse":
            return f"StreamFuse({self.args[0]}, {self.args[1]})"
        if self.primitive == "StateCarry":
            return f"StateCarry({self.args[0]})"
        return f"{self.primitive}{self.args}"

    # ---- Per-primitive type rules (Lemma 3.14 well-formedness) ----
    def check_typing(self) -> "list[str]":
        errors = []
        if self.primitive == "TimeBlock":
            if len(self.args) != 1:
                errors.append(f"TimeBlock takes 1 arg, got {len(self.args)}")
            elif self.args[0] < 1:
                errors.append(f"TimeBlock(K) requires K >= 1, got {self.args[0]}")
        elif self.primitive == "BatchFold":
            if len(self.args) != 1:
                errors.append(f"BatchFold takes 1 arg, got {len(self.args)}")
            else:
                op = self.args[0]
                if op not in TSI_OPS:
                    errors.append(
                        f"BatchFold({op}) ILLEGAL — op must be TSI; "
                        f"{op} is {'CSR' if op in CSR_OPS else 'unknown/TGO'}"
                    )
        elif self.primitive == "StreamFuse":
            if len(self.args) != 2:
                errors.append(f"StreamFuse takes 2 args, got {len(self.args)}")
            else:
                u, v = self.args
                if u not in TSI_OPS:
                    errors.append(f"StreamFuse({u}, {v}) ILLEGAL — u={u} must be TSI")
                if v not in CSR_OPS:
                    errors.append(f"StreamFuse({u}, {v}) ILLEGAL — v={v} must be CSR")
        elif self.primitive == "StateCarry":
            if len(self.args) != 1:
                errors.append(f"StateCarry takes 1 arg, got {len(self.args)}")
            else:
                op = self.args[0]
                if op not in CSR_OPS:
                    errors.append(
                        f"StateCarry({op}) ILLEGAL — op must be CSR; "
                        f"{op} is {'TSI' if op in TSI_OPS else 'unknown/TGO'}"
                    )
        else:
            errors.append(f"Unknown primitive '{self.primitive}'")
        return errors


@dataclass(frozen=True)
class ScheduleDecomposition:
    """A schedule σ ∈ Σ(G,T) expressed as a composition of §3.13 primitives.

    Convention: `transforms` is in INNERMOST-FIRST order. The full schedule
    applied to a baseline reference σ_ref is:

        σ = transforms[-1] ∘ transforms[-2] ∘ ... ∘ transforms[0] ∘ σ_ref

    i.e. transforms[0] is applied first; transforms[-1] last (outermost).
    The display string reads outermost-first, matching the paper convention:

        "TimeBlock(K=4) ∘ StreamFuse(Conv, LIF) ∘ StateCarry(LIF)"

    Each Implementation produces a ScheduleDecomposition via its
    schedule_decomposition() method. The .verify() method walks the
    decomposition and checks well-formedness (§3.13 typing rules).

    `form` labels which §3.8 form this decomposition realizes:
      "form_1" : TimeBlock + BatchFold (cuDNN-style baseline)
      "form_2" : TimeBlock + StreamFuse + StateCarry (§3.9 CTF)
      "form_3" : reserved for future composite forms
      "custom" : a decomposition not matching the canonical forms
    """
    transforms: tuple   # tuple[ScheduleTransform, ...]
    form: str           # "form_1" | "form_2" | "form_3" | "custom"
    description: str = ""

    def __str__(self) -> str:
        # Outermost first (paper convention)
        return " ∘ ".join(str(t) for t in reversed(self.transforms))

    def verify(self) -> "tuple[bool, list[str]]":
        """Check well-formedness against §3.13.

        Returns (ok, errors). Errors include:
          - per-transform typing violations (BatchFold on CSR, etc.)
          - missing TimeBlock (Definition 3.13 requires chunking)
          - StreamFuse without corresponding StateCarry on its CSR target
            (orphan stream fuse — the carried state must persist somewhere)
        """
        errors = []
        # Per-transform typing
        for i, t in enumerate(self.transforms):
            for e in t.check_typing():
                errors.append(f"transform[{i}] {t}: {e}")

        # Whole-decomposition rules
        prims = [t.primitive for t in self.transforms]
        if "TimeBlock" not in prims:
            errors.append(
                "Missing TimeBlock — Definition 3.13 requires a chunked schedule "
                "(use TimeBlock(K=T) for the trivial single-block case)"
            )

        # If StreamFuse(u, v) appears with v ∈ CSR, then StateCarry(v) must
        # also appear — the CSR state has to be carried at block boundaries
        # somewhere, since StreamFuse only handles intra-block dataflow.
        sf_csr_targets = set()
        for t in self.transforms:
            if t.primitive == "StreamFuse" and len(t.args) == 2:
                sf_csr_targets.add(t.args[1])
        sc_targets = {t.args[0] for t in self.transforms
                      if t.primitive == "StateCarry" and len(t.args) == 1}
        for v in sf_csr_targets:
            if v not in sc_targets:
                errors.append(
                    f"StreamFuse(_, {v}) without StateCarry({v}) — "
                    f"the CSR state of {v} must be carried at block boundaries"
                )

        # form labels are advisory; check they're a known value
        if self.form not in ("form_1", "form_2", "form_3", "custom"):
            errors.append(f"unknown form '{self.form}'")

        return (len(errors) == 0, errors)
