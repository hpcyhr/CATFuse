"""
Test 28: Multi-architecture end-to-end test.

Tests CATFuse-SF on:
  1. VGG11-BN (Sequential Conv→BN→LIF + MaxPool + Linear→LIF)
  2. SEW-ResNet18 (BasicBlock Conv→BN→LIF + Add→LIF + AvgPool)
  3. SpikFormer (if available: Conv + Attention + FFN)

For each: substitute_sf → coverage → parity → wall-clock
"""
import sys, os, time, statistics
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

device = 'cuda:0'
torch.backends.cudnn.benchmark = False
N_WARMUP = 20
N_ITER = 50
N_REPEAT = 3


def bench(model, x_in, label):
    model.eval()
    times = []
    for _ in range(N_REPEAT):
        for _ in range(N_WARMUP):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); torch.cuda.synchronize()
        for _ in range(N_ITER):
            functional.reset_net(model)
            with torch.no_grad():
                _ = model(x_in)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / N_ITER * 1000)
    med = statistics.median(times)
    print(f"    [{label}] {med:.2f} ms/iter")
    return med


def test_architecture(name, build_fn, T, B, C, H):
    print(f"\n{'=' * 70}")
    print(f"Architecture: {name}")
    print(f"Input: T={T}, B={B}, C={C}, H={H}")
    print(f"{'=' * 70}")

    # 1. Build model
    try:
        net = build_fn()
        net = net.to(device).eval()
        functional.set_step_mode(net, 'm')
    except Exception as e:
        print(f"  BUILD FAILED: {e}")
        return

    x = torch.rand(T, B, C, H, H, device=device)

    # 2. Count LIF nodes
    lif_count = sum(1 for _, m in net.named_modules() if isinstance(m, neuron.LIFNode))
    print(f"\n  LIF nodes: {lif_count}")

    # 3. SJ baseline
    print("\n  --- SJ baseline ---")
    functional.reset_net(net)
    with torch.no_grad():
        ref_out = net(x)
    print(f"    Output: {ref_out.shape}, mean={ref_out.mean().item():.4f}")
    sj_ms = bench(net, x, "SJ")

    # 4. CATFuse-Dense (original substitute)
    print("\n  --- CATFuse-Dense ---")
    try:
        from catfuse.substitute import substitute
        net_dense, stats_dense = substitute(net, verbose=False)
        net_dense = net_dense.to(device).eval()
        functional.set_step_mode(net_dense, 'm')
        print(f"    Coverage: {stats_dense['fused_lif_nodes']}/{stats_dense['total_lif_nodes']} "
              f"({stats_dense['coverage_pct']:.0f}%)")
        print(f"    Patterns: {stats_dense['patterns_matched']}")

        functional.reset_net(net_dense)
        with torch.no_grad():
            dense_out = net_dense(x)
        diff = (ref_out - dense_out).abs().max().item()
        print(f"    Parity: max_diff={diff:.6f}")
        dense_ms = bench(net_dense, x, "Dense")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        dense_ms = float('inf')

    # 5. CATFuse-SF (policy-aware)
    print("\n  --- CATFuse-SF ---")
    try:
        from catfuse.substitute import substitute_sf, print_routing_table
        net_sf, stats_sf = substitute_sf(net, T=T, verbose=False)
        print_routing_table(stats_sf)
        net_sf = net_sf.to(device).eval()
        functional.set_step_mode(net_sf, 'm')

        functional.reset_net(net_sf)
        with torch.no_grad():
            sf_out = net_sf(x)
        diff = (ref_out - sf_out).abs().max().item()
        print(f"    Parity: max_diff={diff:.6f}")
        sf_ms = bench(net_sf, x, "SF")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        sf_ms = float('inf')

    # 6. Summary
    print(f"\n  --- Summary ---")
    print(f"    SJ:    {sj_ms:.2f} ms")
    if dense_ms != float('inf'):
        print(f"    Dense: {dense_ms:.2f} ms ({sj_ms/dense_ms:.2f}x)")
    if sf_ms != float('inf'):
        print(f"    SF:    {sf_ms:.2f} ms ({sj_ms/sf_ms:.2f}x)")


# ================================================================
# Architecture 1: VGG11-BN
# ================================================================
def build_vgg11():
    from spikingjelly.activation_based.model import spiking_vgg
    return spiking_vgg.spiking_vgg11_bn(
        spiking_neuron=neuron.LIFNode, tau=2.0,
        num_classes=10,
    )

test_architecture("VGG11-BN (CIFAR-10)", build_vgg11, T=4, B=2, C=3, H=32)

# ================================================================
# Architecture 2: SEW-ResNet18
# ================================================================
def build_sew_resnet18():
    from spikingjelly.activation_based.model import sew_resnet
    net = sew_resnet.sew_resnet18(
        pretrained=False, num_classes=10, cnf='ADD',
        spiking_neuron=neuron.LIFNode, tau=2.0,
    )
    ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu')
        net.load_state_dict(sd.get('model', sd), strict=False)
    return net

test_architecture("SEW-ResNet18 (CIFAR-10)", build_sew_resnet18, T=4, B=2, C=3, H=32)

# ================================================================
# Architecture 3: SEW-ResNet18 ImageNet scale
# ================================================================
def build_sew_resnet18_imagenet():
    from spikingjelly.activation_based.model import sew_resnet
    return sew_resnet.sew_resnet18(
        pretrained=False, num_classes=1000, cnf='ADD',
        spiking_neuron=neuron.LIFNode, tau=2.0,
    )

test_architecture("SEW-ResNet18 (ImageNet 224)", build_sew_resnet18_imagenet, T=4, B=1, C=3, H=224)

# ================================================================
# Architecture 4: SpikingResNet18 (standard, no SEW)
# ================================================================
def build_spiking_resnet18():
    from spikingjelly.activation_based.model import spiking_resnet
    return spiking_resnet.spiking_resnet18(
        spiking_neuron=neuron.LIFNode, tau=2.0,
        num_classes=10,
    )

test_architecture("SpikingResNet18 (CIFAR-10)", build_spiking_resnet18, T=4, B=2, C=3, H=32)

# ================================================================
# Architecture 5: VGG11-BN ImageNet scale
# ================================================================
test_architecture("VGG11-BN (ImageNet 224)", build_vgg11, T=4, B=1, C=3, H=224)

print("\n" + "=" * 70)
print("MULTI-ARCHITECTURE AUDIT COMPLETE")
print("=" * 70)
