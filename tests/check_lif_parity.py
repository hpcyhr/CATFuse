"""[§3.9] LIF reference parity check vs SpikingJelly LIFNode.

Verifies that our lif_step_ref() formula produces bit-exact output
identical to SJ's LIFNode in both single-step and multi-step modes.
This is a prerequisite for the TVM/CTF benchmark — if parity fails here,
everything downstream is suspect.
"""
import torch
from spikingjelly.activation_based import neuron

TAU = 2.0
V_TH = 1.0
V_RESET = 0.0


def lif_step_ref(z, v_prev):
    """Matches SJ LIFNode(tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
                           decay_input=True, hard reset) forward pass bit-exactly."""
    H = v_prev + (z - v_prev) / TAU
    S = (H >= V_TH).float()
    V = H * (1 - S) + V_RESET * S
    return V, S


def main():
    torch.manual_seed(42)
    T, N, C, H, W = 8, 1, 64, 32, 32
    # Use input that will produce a mix of spiking/non-spiking steps
    z_seq = torch.randn(T, N, C, H, W, device='cuda') * 0.8

    # ---- Our reference ----
    v = torch.zeros(N, C, H, W, device='cuda')
    s_ours = []
    for t in range(T):
        v, s = lif_step_ref(z_seq[t], v)
        s_ours.append(s)
    s_ours = torch.stack(s_ours)
    v_ours_final = v.clone()

    # ---- SJ single-step ----
    lif_s = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        decay_input=True, detach_reset=False,
    ).cuda()
    lif_s.reset()
    s_sj_s = []
    for t in range(T):
        s_sj_s.append(lif_s(z_seq[t]))
    s_sj_s = torch.stack(s_sj_s)
    v_sj_s_final = lif_s.v.clone() if isinstance(lif_s.v, torch.Tensor) else None

    diff_s = (s_ours - s_sj_s).abs().max().item()
    v_diff_s = (v_ours_final - v_sj_s_final).abs().max().item() if v_sj_s_final is not None else -1
    print(f"[single-step] max |spike diff|   = {diff_s:.2e}")
    print(f"[single-step] max |v_final diff| = {v_diff_s:.2e}")

    # ---- SJ multi-step ----
    lif_m = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        decay_input=True, detach_reset=False, step_mode='m',
    ).cuda()
    lif_m.reset()
    s_sj_m = lif_m(z_seq)
    v_sj_m_final = lif_m.v.clone() if isinstance(lif_m.v, torch.Tensor) else None

    diff_m = (s_ours - s_sj_m).abs().max().item()
    v_diff_m = (v_ours_final - v_sj_m_final).abs().max().item() if v_sj_m_final is not None else -1
    print(f"[multi-step]  max |spike diff|   = {diff_m:.2e}")
    print(f"[multi-step]  max |v_final diff| = {v_diff_m:.2e}")

    # ---- Verdict ----
    all_exact = (diff_s == 0.0 and diff_m == 0.0
                 and (v_diff_s == 0.0 or v_diff_s < 0)
                 and (v_diff_m == 0.0 or v_diff_m < 0))
    if all_exact:
        print("[PASS] lif_step_ref is bit-exact with SpikingJelly LIFNode")
    else:
        # Non-zero diff — diagnose
        print("[FAIL] Parity breaks. Diagnostics:")
        if diff_s > 0:
            wrong = (s_ours != s_sj_s).nonzero()
            if len(wrong) > 0:
                idx = tuple(wrong[0].tolist())
                print(f"  First disagreeing spike at index {idx}:")
                print(f"    ours = {s_ours[idx].item()}, sj = {s_sj_s[idx].item()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
