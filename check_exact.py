"""
check_bitexact.py — Are SJ torch and SJ cupy backward bit-exact to each other?

If YES: we have a clear target. Triton kernel should match both.
If NO:  SJ's own two backends disagree, so "bit-exact to SJ" is ambiguous —
        we'll pick one as the reference (torch backend, since it's the
        canonical PyTorch autograd path).
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate

device = 'cuda:0'
torch.manual_seed(0)

T, B, C, H, W = 8, 4, 16, 8, 8  # small for fast check
TAU, V_TH, V_RESET = 2.0, 1.0, 0.0


def run(backend, x):
    """Run LIF forward+backward, return output and input grad."""
    x = x.detach().clone().requires_grad_(True)
    lif = neuron.LIFNode(
        tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
        surrogate_function=surrogate.ATan(alpha=2.0),
        step_mode='m', backend=backend,
    ).to(device)
    functional.reset_net(lif)
    s = lif(x)
    # Use a fixed grad_output for reproducibility
    grad_out = torch.ones_like(s)
    s.backward(grad_out)
    return s.detach(), x.grad.detach()


x_init = torch.randn(T, B, C, H, W, device=device) * 2.0

s_torch, gx_torch = run('torch', x_init)
s_cupy, gx_cupy = run('cupy', x_init)

print(f"Forward spike match:       {torch.equal(s_torch, s_cupy)}")
print(f"Forward max diff:          {(s_torch - s_cupy).abs().max().item()}")
print()
print(f"Backward grad_x bit-exact: {torch.equal(gx_torch, gx_cupy)}")
print(f"Backward max abs diff:     {(gx_torch - gx_cupy).abs().max().item()}")
print(f"Backward max rel diff:     {((gx_torch - gx_cupy).abs() / (gx_torch.abs() + 1e-8)).max().item()}")
print()
print(f"grad_x_torch stats: mean={gx_torch.mean().item():.6f}, std={gx_torch.std().item():.6f}")
print(f"grad_x_cupy  stats: mean={gx_cupy.mean().item():.6f}, std={gx_cupy.std().item():.6f}")