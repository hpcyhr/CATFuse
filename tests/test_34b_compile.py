"""Test 34b: torch.compile on CATFuse — fresh model per config."""
import sys, time, statistics
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP, N_ITER, N_REPEAT = 30, 100, 3

def bench(fn):
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER): fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

print("=" * 80)

for name, build_fn, T, B, C, H in [
    ("SEW-RN18 CIFAR", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("VGG11 CIFAR", lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
        tau=2.0, num_classes=10), 4, 2, 3, 32),
    ("SEW-RN18 ImgNet", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
]:
    print(f"\n--- {name} (T={T}, B={B}, {H}×{H}) ---")
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ baseline
    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    sj_ms = bench(lambda: (functional.reset_net(net), net(x)))

    # CATFuse-Dense with Triton LIF (current)
    from catfuse.substitute import substitute_sf
    net2 = build_fn().to(device).eval()
    functional.set_step_mode(net2, 'm')
    net_d, _ = substitute_sf(net2, T=T)
    net_d = net_d.to(device).eval()
    functional.set_step_mode(net_d, 'm')
    def dense_fn():
        functional.reset_net(net_d)
        for m in net_d.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): return net_d(x)
    dense_ms = bench(dense_fn)

    # CATFuse with PyTorch LIF (compilable) — patch forward
    from catfuse.patterns import PartialFusionConvBNLIF
    _orig = PartialFusionConvBNLIF._forward_impl

    def _pytorch_lif_impl(self, x_in):
        T2, B2 = x_in.shape[0], x_in.shape[1]
        self._ensure_bn_folded()
        x_4d = x_in.reshape(T2 * B2, x_in.shape[2], x_in.shape[3], x_in.shape[4])
        z_4d = F.conv2d(x_4d, self._w_fused, bias=self._b_fused,
                        stride=self.stride, padding=self.padding)
        H_out, W_out = z_4d.shape[2], z_4d.shape[3]
        z = z_4d.reshape(T2, B2, self.out_channels, H_out, W_out)
        if not hasattr(self, "_v") or self._v is None:
            self._v = torch.zeros(B2, self.out_channels, H_out, W_out,
                                  dtype=torch.float32, device=z.device)
        v = self._v
        decay = 1.0 - 1.0 / self.tau
        recip = 1.0 / self.tau
        spikes = []
        for t in range(T2):
            v = v * decay + z[t] * recip
            sp = (v >= self.v_threshold).float()
            v = v * (1.0 - sp) + self.v_reset * sp
            spikes.append(sp)
        self._v = v
        return torch.stack(spikes, dim=0)

    PartialFusionConvBNLIF._forward_impl = _pytorch_lif_impl

    # Build fresh model with PyTorch LIF
    net3 = build_fn().to(device).eval()
    functional.set_step_mode(net3, 'm')
    net_p, _ = substitute_sf(net3, T=T)
    net_p = net_p.to(device).eval()
    functional.set_step_mode(net_p, 'm')

    def pytorch_fn():
        functional.reset_net(net_p)
        for m in net_p.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad(): return net_p(x)
    pytorch_ms = bench(pytorch_fn)

    # Now try torch.compile on the PyTorch-LIF model
    try:
        net4 = build_fn().to(device).eval()
        functional.set_step_mode(net4, 'm')
        net_comp, _ = substitute_sf(net4, T=T)
        net_comp = net_comp.to(device).eval()
        functional.set_step_mode(net_comp, 'm')
        net_compiled = torch.compile(net_comp, mode='reduce-overhead', fullgraph=False)

        # Warmup compilation
        for _ in range(10):
            functional.reset_net(net_compiled)
            for m in net_compiled.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): _ = net_compiled(x)

        def compiled_fn():
            functional.reset_net(net_compiled)
            for m in net_compiled.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad(): return net_compiled(x)
        compiled_ms = bench(compiled_fn)
    except Exception as e:
        compiled_ms = float('inf')
        print(f"  compile failed: {e}")

    PartialFusionConvBNLIF._forward_impl = _orig

    print(f"  SJ:             {sj_ms:.2f} ms")
    print(f"  Dense(Triton):  {dense_ms:.2f} ms ({sj_ms/dense_ms:.2f}×)")
    print(f"  Dense(PyTorch): {pytorch_ms:.2f} ms ({sj_ms/pytorch_ms:.2f}×)")
    if compiled_ms != float('inf'):
        print(f"  Dense+compile:  {compiled_ms:.2f} ms ({sj_ms/compiled_ms:.2f}×)")

print("=" * 80)
