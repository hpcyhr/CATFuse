"""
CTF §3.5 Corollary 3.8 reverse check:
Does introducing future-time dependencies break semantics?

The paper forbids edges (v, t') → (u, t) where t' > t. We verify by
constructing an "illegal" LIF that reads x_{t+1} at step t, then compare
to the correct sequential LIF. Expected: different outputs.
"""
import torch
import torch.nn as nn

device = 'cuda:0'
torch.manual_seed(0)

T, B, C, H, W = 8, 4, 16, 8, 8
TAU = 2.0
V_TH = 1.0
V_RESET = 0.0


def lif_correct(x_seq):
    """σ_ref: step t uses x_t."""
    T_ = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T_):
        v = v + (x_seq[t] - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


def lif_future_peek(x_seq):
    """
    Illegal: step t reads x_{t+1} (look one step ahead).
    Last step falls back to x_T (no future available).
    This violates Corollary 3.8 — it's an edge from time t+1 into time t.
    """
    T_ = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T_):
        x_use = x_seq[t + 1] if t + 1 < T_ else x_seq[t]
        v = v + (x_use - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


def lif_future_average(x_seq):
    """
    Illegal variant: step t uses mean(x_t, x_{t+1}) — a common smoothing
    trick that would be tempting in a 'full sequence known upfront' setting.
    Still violates Corollary 3.8.
    """
    T_ = x_seq.shape[0]
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T_):
        if t + 1 < T_:
            x_use = 0.5 * (x_seq[t] + x_seq[t + 1])
        else:
            x_use = x_seq[t]
        v = v + (x_use - (v - V_RESET)) / TAU
        s = (v >= V_TH).float()
        v = v * (1 - s) + V_RESET * s
        spikes.append(s)
    return torch.stack(spikes, dim=0)


def main():
    print("=" * 78)
    print("§3.5 Corollary 3.8 reverse check: future dependency")
    print(f"Shape: T={T}, B={B}, C={C}, H=W={H}")
    print("=" * 78)
    print()

    x = torch.randn(T, B, C, H, W, device=device) * 1.5

    s_ref = lif_correct(x)
    s_peek = lif_future_peek(x)
    s_avg = lif_future_average(x)

    print("Spike rates:")
    print(f"  correct:        {s_ref.mean().item():.4f}")
    print(f"  future-peek:    {s_peek.mean().item():.4f}")
    print(f"  future-average: {s_avg.mean().item():.4f}")
    print()

    n_total = s_ref.numel()
    for name, s_var in [("future-peek", s_peek), ("future-average", s_avg)]:
        exact = torch.equal(s_ref, s_var)
        n_diff = (s_ref != s_var).sum().item()
        max_d = (s_ref - s_var).abs().max().item()
        print(f"{name:<18} equal={exact}, max_diff={max_d}, "
              f"n_diff={n_diff:,} ({n_diff/n_total*100:.2f}%)")

    print()
    print("Expected: both illegal variants produce different spikes.")
    print("If either matches, Corollary 3.8's prohibition is incomplete.")


if __name__ == '__main__':
    main()