"""
debug_spike_rate.py — verify that our input + warmup strategy gives a healthy
spike rate profile across all LIF layers in SJ spiking_resnet18.

Run:  python benchmarks/debug_spike_rate.py
Expected: after warmup, most LIF nodes have spike_rate in [0.01, 0.3]
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional as sj_func
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg


device = 'cuda:0'
T = 4
B = 8


def build_net(ctor_name, ctor_module):
    ctor = getattr(ctor_module, ctor_name)
    torch.manual_seed(0)
    net = ctor(
        spiking_neuron=neuron.LIFNode,
        tau=2.0, v_threshold=1.0, v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, num_classes=10,
    ).to(device)
    sj_func.set_step_mode(net, 'm')
    return net


def make_direct_encoded_input(seed, T, batch):
    """Direct encoding: same image broadcast T times."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    img = torch.randn(batch, 3, 32, 32, generator=gen) * 0.5 + 0.5
    x = img.unsqueeze(0).expand(T, -1, -1, -1, -1).contiguous()
    return x.to(device)


def collect_spike_rates(net):
    rates = {}
    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            def make_hook(n):
                def fn(mod, inp, out):
                    rates[n] = out.float().mean().item()
                return fn
            hooks.append(m.register_forward_hook(make_hook(name)))
    return rates, hooks


def test_config(net_name, ctor_module, warmup_iters=5, bn_warmup=True,
                weight_scale=1.0):
    print(f"\n{'='*70}")
    print(f"{net_name}  warmup_iters={warmup_iters}  bn_warmup={bn_warmup}  "
          f"weight_scale={weight_scale}")
    print('='*70)

    net = build_net(net_name, ctor_module)

    if weight_scale != 1.0:
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.mul_(weight_scale)

    x = make_direct_encoded_input(seed=0, T=T, batch=B)

    # Optional BN warmup in train mode
    if bn_warmup:
        net.train()
        with torch.no_grad():
            for _ in range(warmup_iters):
                sj_func.reset_net(net)
                _ = net(x)

    net.eval()
    rates, hooks = collect_spike_rates(net)

    sj_func.reset_net(net)
    with torch.no_grad():
        y = net(x)

    for h in hooks:
        h.remove()

    print(f"  output: mean={y.mean().item():.4f}  std={y.std().item():.4f}")
    nonzero = [(k, v) for k, v in rates.items() if v > 0]
    zero = [k for k, v in rates.items() if v == 0]
    print(f"  LIF layers with spike_rate > 0: {len(nonzero)}/{len(rates)}")
    if nonzero:
        vals = [v for _, v in nonzero]
        print(f"    min={min(vals):.4f}  max={max(vals):.4f}  "
              f"mean={sum(vals)/len(vals):.4f}")
    if zero and len(zero) < 6:
        print(f"  zero-spike LIF layers: {zero}")


def main():
    # Baseline: no fix
    test_config('spiking_resnet18', spiking_resnet,
                warmup_iters=0, bn_warmup=False, weight_scale=1.0)

    # Fix A: BN warmup only
    test_config('spiking_resnet18', spiking_resnet,
                warmup_iters=5, bn_warmup=True, weight_scale=1.0)

    # Fix B: BN warmup + light weight scale
    test_config('spiking_resnet18', spiking_resnet,
                warmup_iters=5, bn_warmup=True, weight_scale=2.0)

    # Fix C: BN warmup + heavy weight scale
    test_config('spiking_resnet18', spiking_resnet,
                warmup_iters=5, bn_warmup=True, weight_scale=4.0)

    # Test on VGG11_bn too
    test_config('spiking_vgg11_bn', spiking_vgg,
                warmup_iters=5, bn_warmup=True, weight_scale=2.0)

    # Test on ResNet-50 (bottleneck structure, check it's not different)
    test_config('spiking_resnet50', spiking_resnet,
                warmup_iters=5, bn_warmup=True, weight_scale=2.0)


if __name__ == '__main__':
    main()