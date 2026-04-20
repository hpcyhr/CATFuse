"""
Diagnostic: verify SparseFlow kernel is actually invoked,
measure per-layer spike rates with real data, and do
single-layer latency comparison (sparse vs cuDNN).
"""
import sys, os, time
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import sew_resnet

device = 'cuda:0'
T, B = 4, 2
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("Diagnostic: SparseFlow integration health check")
print("=" * 70)

# ============================================================
# 1. Load model + get real spike patterns
# ============================================================
print("\n--- 1. Collecting real spike patterns ---")

net = sew_resnet.sew_resnet18(pretrained=False, num_classes=10, cnf='ADD',
                               spiking_neuron=neuron.LIFNode, tau=2.0)
ckpt = 'checkpoints/sew_resnet18_cifar10_lif_best.pth'
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd.get('model', sd), strict=False)
net = net.to(device).eval()
functional.set_step_mode(net, 'm')

# Use real CIFAR-10 or synthetic
try:
    import torchvision, torchvision.transforms as transforms
    tf = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=False)
    imgs, _ = next(iter(loader))
    x_input = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1).to(device)
    print(f"  Using real CIFAR-10 data: {x_input.shape}")
except:
    x_input = torch.rand(T, B, 3, 32, 32, device=device) * 0.3
    print(f"  Using synthetic data: {x_input.shape}")

# Capture intermediate spike tensors with hooks
layer_data = {}
def make_hook(name, conv_mod):
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        layer_data[name] = {
            'input': inp.detach().clone(),
            'output': output.detach().clone(),
            'conv': conv_mod,
        }
    return hook_fn

hooks = []
for name, mod in net.named_modules():
    if isinstance(mod, neuron.LIFNode):
        # Find parent block to get the conv
        parent_name = '.'.join(name.split('.')[:-1])
        parent = dict(net.named_modules()).get(parent_name)
        if parent is None:
            continue
        sn_idx = name.split('.')[-1]  # sn1 or sn2
        idx = sn_idx.replace('sn', '')
        conv = getattr(parent, f'conv{idx}', None)
        bn = getattr(parent, f'bn{idx}', None)
        if conv is not None:
            hooks.append(mod.register_forward_hook(make_hook(name, conv)))

functional.reset_net(net)
with torch.no_grad():
    _ = net(x_input)

for h in hooks:
    h.remove()

# ============================================================
# 2. Per-layer spike rate analysis
# ============================================================
print("\n--- 2. Per-layer spike rates (real data) ---")
print(f"  {'Layer':<35s} {'Shape':<25s} {'Spike rate':>10s} {'Sparsity':>10s} {'Zero tiles%':>12s}")
print("  " + "-" * 95)

for name, info in layer_data.items():
    out = info['output']
    # out is [T, B, C, H, W] in multi-step mode or [B, C, H, W] single-step
    rate = out.mean().item()
    sparsity = (1.0 - rate) * 100

    # Estimate tile-level zero rate (8x8 blocks)
    if out.dim() == 5:
        flat = out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])  # [T*B, C, H, W]
    else:
        flat = out
    n, c, h, w = flat.shape
    bh, bw = min(8, h), min(8, w)
    if h >= bh and w >= bw:
        gh, gw = h // bh, w // bw
        # reshape to tiles
        tiled = flat[:, :, :gh*bh, :gw*bw].reshape(n, c, gh, bh, gw, bw)
        tile_sums = tiled.abs().sum(dim=(1, 3, 5))  # [n, gh, gw]
        zero_tiles = (tile_sums == 0).float().mean().item() * 100
    else:
        zero_tiles = 0.0

    shape_str = f"{'x'.join(str(s) for s in out.shape)}"
    print(f"  {name:<35s} {shape_str:<25s} {rate:>9.4f}% {sparsity:>9.1f}% {zero_tiles:>11.1f}%")

# ============================================================
# 3. Verify Triton kernel invocation
# ============================================================
print("\n--- 3. Triton kernel invocation check ---")

from catfuse.sparseflow.ops.st_fusion_conv_bn_lif import STFusionConvBNLIF, _KERNEL_AVAILABLE
print(f"  _KERNEL_AVAILABLE = {_KERNEL_AVAILABLE}")

# Create a test module and check if sparse kernel runs
test_conv = nn.Conv2d(128, 128, 3, padding=1, bias=True).to(device)
test_bn = nn.BatchNorm2d(128).to(device).eval()

class MockLIF:
    tau = 2.0; v_threshold = 1.0; v_reset = 0.0; decay_input = True

sf_mod = STFusionConvBNLIF.from_sj_modules(test_conv, test_bn, MockLIF(), K=4).to(device).eval()

# Feed sparse spike input
x_sparse = (torch.rand(2, B, 128, 14, 14, device=device) > 0.95).float()
print(f"  Test input: {x_sparse.shape}, spike_rate={x_sparse.mean().item():.3f}")

# Patch forward to print which path is taken
import catfuse.sparseflow.fused_conv_bn_lif_kernel as fk
_orig_fn = fk.sparse_fused_conv_bn_lif_forward

call_log = []
def _traced_fn(*args, **kwargs):
    call_log.append('triton_kernel_called')
    return _orig_fn(*args, **kwargs)

fk.sparse_fused_conv_bn_lif_forward = _traced_fn

# Also need to patch the reference in the ops module
import catfuse.sparseflow.ops.st_fusion_conv_bn_lif as ops_mod
ops_mod.sparse_fused_conv_bn_lif_forward = _traced_fn

sf_mod.v = 0.0
with torch.no_grad():
    out = sf_mod(x_sparse)

fk.sparse_fused_conv_bn_lif_forward = _orig_fn
ops_mod.sparse_fused_conv_bn_lif_forward = _orig_fn

if call_log:
    print(f"  [OK] Triton kernel was called {len(call_log)} times (expected {x_sparse.shape[0]})")
else:
    print(f"  [WARN] Triton kernel was NOT called — falling back to PyTorch")

# ============================================================
# 4. Single-layer micro-benchmark: sparse vs cuDNN
# ============================================================
print("\n--- 4. Single-layer micro-benchmark ---")

configs = [
    ("C=128, H=14", 128, 128, 14),
    ("C=256, H=7",  256, 256, 7),
    ("C=512, H=4",  512, 512, 4),
]

for label, cin, cout, spatial in configs:
    conv = nn.Conv2d(cin, cout, 3, padding=1, bias=True).to(device)
    bn = nn.BatchNorm2d(cout).to(device).eval()
    sf_layer = STFusionConvBNLIF.from_sj_modules(conv, bn, MockLIF(), K=2).to(device).eval()

    for sparsity_pct in [50, 90, 95, 99]:
        thresh = 1.0 - sparsity_pct / 100.0
        x_test = (torch.rand(2, B, cin, spatial, spatial, device=device) > (1 - thresh)).float()
        actual_rate = x_test.mean().item()

        # cuDNN path
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                for t in range(2):
                    z = conv(x_test[t])
                    z = bn(z)
        torch.cuda.synchronize()
        cudnn_ms = (time.perf_counter() - t0) / 100 * 1000

        # STFusionConvBNLIF path
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            sf_layer.v = 0.0
            with torch.no_grad():
                _ = sf_layer(x_test)
        torch.cuda.synchronize()
        sf_ms = (time.perf_counter() - t0) / 100 * 1000

        ratio = cudnn_ms / sf_ms if sf_ms > 0 else 0
        faster = "SF wins" if ratio > 1 else "cuDNN wins"
        print(f"  {label} sparsity={sparsity_pct:2d}% (actual={actual_rate:.3f}): "
              f"cuDNN={cudnn_ms:.2f}ms  SF={sf_ms:.2f}ms  ratio={ratio:.2f}x ({faster})")

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
