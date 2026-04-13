"""
CTF §3.5 Corollary 3.7 reverse check:
Does "BatchFold over T" on a CSR operator (LIF) actually break semantics?

The paper claims this transformation is illegal because it violates dependency
edge (v, t-1) → (v, t). We verify by:
  1. Implementing LIF the correct way: sequential T steps with state carry
  2. Implementing LIF the "illegal" way: treat T as batch dim, all steps parallel
  3. Comparing outputs

Expected: illegal version produces DIFFERENT spikes. If it produces the SAME
spikes, the paper's prohibition is too strong and needs qualification.
"""
import torch
import torch.nn as nn
import statistics

device = 'cuda:0'
torch.manual_seed(0)

T, B, C, H, W = 8, 4, 16, 8, 8
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0


def lif_sequential(x_seq):
    """Correct LIF: sequential, state carries across time."""
    T_ = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T_):
        v = v + (x_seq[t] - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0), v


def lif_batchfold_illegal(x_seq):
    """
    Illegal: pretend all T steps are independent batch elements.
    Each 'step' sees v = 0 (no state carry). This matches what would happen
    if you mechanically applied BatchFold to LIF.
    """
    T_, B_, C_, H_, W_ = x_seq.shape
    x_flat = x_seq.reshape(T_ * B_, C_, H_, W_)
    v_init = torch.zeros_like(x_flat)  # no state carry
    v = v_init + (x_flat - (v_init - V_RESET)) / TAU
    s = (v >= V_TH).float()
    # (reset is irrelevant here because we never see v again)
    return s.reshape(T_, B_, C_, H_, W_)


def lif_batchfold_with_carry_attempt(x_seq):
    """
    Another "illegal" variant: try to be clever by pre-computing the running
    state as if the recurrence were linear, then apply nonlinearity. This is
    WRONG because the fire-reset nonlinearity breaks linearity, but some
    optimizers might try it.
    """
    T_ = x_seq.shape[0]
    # Pretend it's a linear RNN: v_t = sum_{s<=t} (1-1/tau)^(t-s) * x_s / tau
    # Compute all v_t in parallel, then apply threshold independently
    decay = 1 - 1.0 / TAU
    v_accum = torch.zeros_like(x_seq[0])
    vs = []
    for t in range(T_):
        v_accum = decay * v_accum + x_seq[t] / TAU
        vs.append(v_accum)
    v_stack = torch.stack(vs, dim=0)  # [T, ...]
    # Spike independently at each step (no reset)
    s = (v_stack >= V_TH).float()
    return s


def main():
    print("=" * 78)
    print("§3.5 Corollary 3.7 reverse check: BatchFold on CSR (LIF)")
    print(f"Shape: T={T}, B={B}, C={C}, H=W={H}")
    print("=" * 78)
    print()

    # Scale inputs so we get meaningful spike rates
    x = torch.randn(T, B, C, H, W, device=device) * 1.5

    s_correct, _ = lif_sequential(x)
    s_illegal = lif_batchfold_illegal(x)
    s_linear_attempt = lif_batchfold_with_carry_attempt(x)

    print("Spike rates:")
    print(f"  sequential (correct):          {s_correct.mean().item():.4f}")
    print(f"  batchfold (no state carry):    {s_illegal.mean().item():.4f}")
    print(f"  linear precompute (no reset):  {s_linear_attempt.mean().item():.4f}")
    print()

    print("Equality checks vs correct:")
    diff_illegal = (s_correct - s_illegal).abs().max().item()
    diff_linear = (s_correct - s_linear_attempt).abs().max().item()
    match_illegal = torch.equal(s_correct, s_illegal)
    match_linear = torch.equal(s_correct, s_linear_attempt)
    print(f"  vs batchfold:       equal={match_illegal}, max_diff={diff_illegal}")
    print(f"  vs linear precomp:  equal={match_linear}, max_diff={diff_linear}")
    print()

    # Count how many spike positions differ
    n_total = s_correct.numel()
    n_diff_illegal = (s_correct != s_illegal).sum().item()
    n_diff_linear = (s_correct != s_linear_attempt).sum().item()
    print(f"Differing spike positions (out of {n_total:,}):")
    print(f"  batchfold:      {n_diff_illegal:,}  ({n_diff_illegal/n_total*100:.2f}%)")
    print(f"  linear precomp: {n_diff_linear:,}  ({n_diff_linear/n_total*100:.2f}%)")
    print()

    print("Interpretation:")
    if not match_illegal and not match_linear:
        print("  ✓ Both illegal transforms produce different outputs.")
        print("  ✓ Corollary 3.7's prohibition is confirmed: BatchFold on CSR")
        print("    genuinely breaks semantics.")
    else:
        print("  ✗ At least one 'illegal' transform produced the SAME output.")
        print("  ✗ Corollary 3.7 is too strong. The transformation is not")
        print("    always illegal — we need to characterize when it's safe.")


if __name__ == '__main__':
    main()