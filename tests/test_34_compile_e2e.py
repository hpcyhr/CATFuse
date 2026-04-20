"""
Test 34: torch.compile on CATFuse-Dense fused model.

torch.compile(reduce-overhead) internally uses CUDA Graphs
and can fuse cuDNN conv + element-wise LIF ops.
"""
import sys, time, statistics
sys.path.insert(0, '.')
import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet, spiking_vgg
from catfuse.substitute import substitute_sf

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 30
N_ITER = 100
N_REPEAT = 3

def bench(fn):
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    return statistics.median(times)

print("=" * 80)
print("torch.compile acceleration on CATFuse-Dense")
print("=" * 80)

for name, build_fn, T, B, C, H in [
    ("SEW-RN18 ImgNet", lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=1000,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 1, 3, 224),
    ("SEW-RN18 CIFAR",  lambda: sew_resnet.sew_resnet18(pretrained=False, num_classes=10,
        cnf='ADD', spiking_neuron=neuron.LIFNode, tau=2.0), 4, 2, 3, 32),
    ("VGG11 CIFAR",     lambda: spiking_vgg.spiking_vgg11_bn(spiking_neuron=neuron.LIFNode,
        tau=2.0, num_classes=10), 4, 2, 3, 32),
]:
    print(f"\n--- {name} (T={T}, B={B}, {H}×{H}) ---")

    net = build_fn().to(device).eval()
    functional.set_step_mode(net, 'm')
    x = torch.rand(T, B, C, H, H, device=device)

    # SJ baseline
    sj_ms = bench(lambda: (functional.reset_net(net), net(x)))

    # CATFuse-Dense (current best)
    net_d, _ = substitute_sf(net, T=T)
    net_d = net_d.to(device).eval()
    functional.set_step_mode(net_d, 'm')

    def dense_fn():
        functional.reset_net(net_d)
        for m in net_d.modules():
            if hasattr(m, 'reset'): m.reset()
        with torch.no_grad():
            return net_d(x)

    dense_ms = bench(dense_fn)

    # CATFuse-Dense + torch.compile
    # Need a compilable wrapper that doesn't use Triton kernels
    # Replace lif_sequential with PyTorch ops for compile compatibility
    # Monkey-patch PartialFusionConvBNLIF to use PyTorch LIF instead of Triton
    from catfuse.patterns import PartialFusionConvBNLIF, FusedLinearLIF

    _orig_pfcbl_fwd = PartialFusionConvBNLIF._forward_impl

    def _pytorch_forward_impl(self, x):
        T, B = x.shape[0], x.shape[1]
        self._ensure_bn_folded()
        x_4d = x.reshape(T * B, x.shape[2], x.shape[3], x.shape[4])
        if not x_4d.any():
            H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1
            z_4d = self._b_fused.view(1, -1, 1, 1).expand(T * B, -1, H_out, W_out).clone()
        else:
            z_4d = torch.nn.functional.conv2d(x_4d, self._w_fused, bias=self._b_fused,
                            stride=self.stride, padding=self.padding)
        H_out, W_out = z_4d.shape[2], z_4d.shape[3]
        z = z_4d.reshape(T, B, self.out_channels, H_out, W_out)

        if not hasattr(self, "_v") or self._v is None:
            self._v = torch.zeros(B, self.out_channels, H_out, W_out,
                                  dtype=torch.float32, device=z.device)
        v = self._v
        decay = 1.0 - 1.0 / self.tau
        recip = 1.0 / self.tau
        spikes = []
        for t in range(T):
            v = v * decay + z[t] * recip
            sp = (v >= self.v_threshold).float()
            v = v * (1.0 - sp) + self.v_reset * sp
            spikes.append(sp)
        self._v = v
        return torch.stack(spikes, dim=0)

    # Patch and compile
    PartialFusionConvBNLIF._forward_impl = _pytorch_forward_impl

    net_c, _ = substitute_sf(net, T=T)
    net_c = net_c.to(device).eval()
    functional.set_step_mode(net_c, 'm')

    try:
        # Compile the full model
        net_compiled = torch.compile(net_c, mode='reduce-overhead', fullgraph=False)

        # Warmup compile (first few calls trigger compilation)
        for _ in range(10):
            functional.reset_net(net_compiled)
            for m in net_compiled.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad():
                _ = net_compiled(x)

        def compiled_fn():
            functional.reset_net(net_compiled)
            for m in net_compiled.modules():
                if hasattr(m, 'reset'): m.reset()
            with torch.no_grad():
                return net_compiled(x)

        compiled_ms = bench(compiled_fn)
    except Exception as e:
        compiled_ms = float('inf')
        print(f"  torch.compile failed: {e}")

    # Restore
    PartialFusionConvBNLIF._forward_impl = _orig_pfcbl_fwd

    print(f"  SJ:              {sj_ms:.2f} ms")
    print(f"  Dense:           {dense_ms:.2f} ms ({sj_ms/dense_ms:.2f}×)")
    if compiled_ms != float('inf'):
        print(f"  Dense+compile:   {compiled_ms:.2f} ms ({sj_ms/compiled_ms:.2f}×)")
        print(f"  compile vs Dense: {dense_ms/compiled_ms:.2f}×")

print("\n" + "=" * 80)
