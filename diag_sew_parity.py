"""
Phase 4 task 4.2 diagnostic — verify SEW-ResNet18 is not degenerate.

Questions:
  1. What is the mean spike rate at each CTFSEWBasicBlock output?
  2. What fraction of the final fc input is zero?
  3. Is the sj_torch stdev (10% CV) due to SJ's Python per-step overhead, or
     something else?

If spike rates are ~0 and fc input is mostly zero, then the "3.13x speedup" is
actually measuring kernel launch overhead on an essentially-dead network, not
real compute. The result is still valid (CTF saves launches) but the framing
must change.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import copy
import torch
import torch.nn as nn

from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.activation_based import functional, neuron, surrogate
from catfuse_substitute import substitute

device = torch.device('cuda:0')
torch.manual_seed(0)

# Build SEWResNet18
model = sew_resnet.sew_resnet18(
    spiking_neuron=neuron.LIFNode,
    surrogate_function=surrogate.Sigmoid(),
    detach_reset=True,
    num_classes=10,
    cnf='ADD',
)
functional.set_step_mode(model, 'm')

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.running_mean.data.normal_(0, 0.1)
        m.running_var.data.uniform_(0.5, 1.5)

sj_torch = model.to(device).eval()
ctf_model, cov = substitute(sj_torch)
ctf_model = ctf_model.to(device).eval()

# ----- Diagnosis 1: Per-block spike rate trace on CTF model -----
print("=" * 70)
print("Diagnosis 1: spike rate at each CTFSEWBasicBlock output (CTF model)")
print("=" * 70)

# Hook each CTFSEWBasicBlock forward to capture its output
block_outputs = {}
hooks = []

def make_hook(name):
    def hook(module, input, output):
        block_outputs[name] = {
            'spike_rate': output.mean().item(),
            'spike_max': output.max().item(),
            'spike_min': output.min().item(),
            'shape': list(output.shape),
        }
    return hook

for name, module in ctf_model.named_modules():
    cls = type(module).__name__
    if cls == 'CTFSEWBasicBlock':
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

x = (torch.randn(16, 32, 3, 64, 64, device=device, dtype=torch.float32) * 1.0).contiguous()

with torch.no_grad():
    functional.reset_net(ctf_model)
    out_ctf = ctf_model(x)

for h in hooks:
    h.remove()

for name, info in block_outputs.items():
    print(f"  {name:<20} spike_rate={info['spike_rate']:.6f}  "
          f"max={info['spike_max']:.4f}  min={info['spike_min']:.4f}  "
          f"shape={info['shape']}")

print(f"\n  Final output (fc): max_abs={out_ctf.abs().max().item():.6e}  "
      f"mean_abs={out_ctf.abs().mean().item():.6e}")
print(f"  Fraction of output that is exactly zero: "
      f"{(out_ctf == 0).float().mean().item():.4f}")

# ----- Diagnosis 2: fc layer input -----
print("\n" + "=" * 70)
print("Diagnosis 2: what does fc layer see as input?")
print("=" * 70)

# Hook the fc input by hooking the avgpool output (which feeds fc after flatten)
fc_input = {}
def fc_hook(module, input, output):
    fc_input['out'] = output.detach()

# avgpool is at ctf_model.avgpool
for name, m in ctf_model.named_modules():
    if name == 'avgpool':
        h = m.register_forward_hook(fc_hook)
        break

with torch.no_grad():
    functional.reset_net(ctf_model)
    _ = ctf_model(x)

h.remove()

avgpool_out = fc_input['out']
print(f"  avgpool output shape: {list(avgpool_out.shape)}")
print(f"  max_abs: {avgpool_out.abs().max().item():.6e}")
print(f"  mean_abs: {avgpool_out.abs().mean().item():.6e}")
print(f"  fraction == 0: {(avgpool_out == 0).float().mean().item():.4f}")
print(f"  fraction < 1e-6: {(avgpool_out.abs() < 1e-6).float().mean().item():.4f}")

# ----- Diagnosis 3: sj_torch wall-clock distribution -----
print("\n" + "=" * 70)
print("Diagnosis 3: sj_torch wall-clock distribution (30 single-iter timings)")
print("=" * 70)

# Warmup
for _ in range(20):
    functional.reset_net(sj_torch)
    with torch.no_grad():
        _ = sj_torch(x)
torch.cuda.synchronize()

samples = []
for i in range(30):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    functional.reset_net(sj_torch)
    with torch.no_grad():
        _ = sj_torch(x)
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end))

import statistics
print(f"  sj_torch per-iter samples (ms):")
for i, s in enumerate(samples):
    print(f"    iter {i}: {s:.4f}")
print(f"\n  stats: min={min(samples):.4f} max={max(samples):.4f} "
      f"mean={statistics.mean(samples):.4f} stdev={statistics.stdev(samples):.4f}")

# Same for ctf
print("\n" + "=" * 70)
print("Diagnosis 3b: ctf wall-clock distribution (30 single-iter timings)")
print("=" * 70)
for _ in range(20):
    functional.reset_net(ctf_model)
    with torch.no_grad():
        _ = ctf_model(x)
torch.cuda.synchronize()

samples_ctf = []
for i in range(30):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    functional.reset_net(ctf_model)
    with torch.no_grad():
        _ = ctf_model(x)
    end.record()
    torch.cuda.synchronize()
    samples_ctf.append(start.elapsed_time(end))

print(f"  ctf per-iter samples (ms):")
for i, s in enumerate(samples_ctf):
    print(f"    iter {i}: {s:.4f}")
print(f"\n  stats: min={min(samples_ctf):.4f} max={max(samples_ctf):.4f} "
      f"mean={statistics.mean(samples_ctf):.4f} stdev={statistics.stdev(samples_ctf):.4f}")