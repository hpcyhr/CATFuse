"""
bench_e2e.py — CTF end-to-end benchmark on Spiking-ResNet/VGG family

Supports two dataset modes:
  - cifar10:  random [3,32,32] input, num_classes=10  (for smoke/regression)
  - imagenet: real ImageNet val images resized to [3,224,224], num_classes=1000
              networks loaded with pretrained=True (ANN ImageNet weights)

Protocol:
  - Forward only (torch.no_grad)
  - Multi-step LIF (step_mode='m')
  - Backends: sj_torch, sj_cupy, CTF (when available)
  - Wall-clock: median over 11 repeats of 100 iters after 20 warmup
  - Structural check only (numerical parity is validated at kernel level v9-v15)
  - Spike-rate diagnostic printed for first config in each run

Usage (CIFAR-10 smoke, random init):
  python bench_e2e.py --net resnet18 --T 4 --batch 32 --dataset cifar10 --skip-ctf

Usage (ImageNet, pretrained, recommended):
  python bench_e2e.py --net resnet18 --T 16 --batch 16 --dataset imagenet \
                       --pretrained --skip-ctf

Usage (full e2e scan):
  python bench_e2e.py --net all --T 4,8,16,32 --batch 16 --dataset imagenet \
                       --pretrained --skip-ctf
"""

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional as sj_func
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg


# ============================================================
# Constants
# ============================================================

NETS_RESNET = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
NETS_VGG = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
ALL_NETS = NETS_RESNET + NETS_VGG

LIF_TAU = 2.0
LIF_V_TH = 1.0
LIF_V_RESET = 0.0

# Dataset-specific constants
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'input_shape': (3, 32, 32),
    },
    'imagenet': {
        'num_classes': 1000,
        'input_shape': (3, 224, 224),
    },
}

DEFAULT_IMAGENET_PATH = '/data/yhr/CATFuse/data/imagenet-val'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_WARMUP = 20
N_ITER = 100
N_REPEAT = 11


# ============================================================
# SpikingJelly network factory
# ============================================================

def make_sj_net(net_name: str, num_classes: int, backend: str,
                pretrained: bool = False):
    """
    Create a SpikingJelly network with LIF neurons on the specified backend.
    
    If pretrained=True, loads torchvision ANN ImageNet weights into the
    Conv/BN layers (SpikingJelly handles this internally via its own
    pretrained flag).
    
    Note: pretrained implies num_classes=1000 (ImageNet). Passing
    num_classes<1000 with pretrained=True will cause SJ to fail the
    state_dict load for the final FC layer.
    """
    common_kwargs = dict(
        spiking_neuron=neuron.LIFNode,
        pretrained=pretrained,
        tau=LIF_TAU,
        v_threshold=LIF_V_TH,
        v_reset=LIF_V_RESET,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    if not pretrained:
        # Only override num_classes when NOT loading pretrained weights
        common_kwargs['num_classes'] = num_classes
    
    if net_name in NETS_RESNET:
        ctor = getattr(spiking_resnet, f'spiking_{net_name}')
        net = ctor(**common_kwargs)
    elif net_name in NETS_VGG:
        ctor = getattr(spiking_vgg, f'spiking_{net_name}')
        net = ctor(**common_kwargs)
    else:
        raise ValueError(f"Unknown net: {net_name}")
    
    sj_func.set_step_mode(net, 'm')
    if backend == 'cupy':
        sj_func.set_backend(net, 'cupy', instance=neuron.LIFNode)
    elif backend == 'torch':
        sj_func.set_backend(net, 'torch', instance=neuron.LIFNode)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return net


# ============================================================
# CTF factory (placeholder)
# ============================================================

def make_ctf_net(net_name: str, num_classes: int, sj_state_dict: dict,
                 pretrained: bool = False):
    raise NotImplementedError(
        "CTF net factory not yet implemented. "
        "Pending: ctf_ops/ shape-generic wrappers."
    )


# ============================================================
# Input generation
# ============================================================

def make_cifar10_input(T: int, batch: int, device: torch.device, seed: int = 0):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(T, batch, 3, 32, 32, generator=gen) * 0.5
    return x.to(device)


def load_imagenet_batch(imagenet_path: str, T: int, batch: int,
                        device: torch.device, seed: int = 0):
    """
    Load `batch` images from a flat ImageNet val directory, resize + center-crop
    to 224x224, normalize with ImageNet mean/std, then broadcast across T.
    
    Returns: [T, B, 3, 224, 224] float32 tensor on device.
    """
    from PIL import Image
    from torchvision import transforms
    
    # Deterministic image selection
    all_files = sorted(os.listdir(imagenet_path))
    jpegs = [f for f in all_files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    if len(jpegs) < batch:
        raise RuntimeError(
            f"Only {len(jpegs)} images in {imagenet_path}, need {batch}"
        )
    
    rng = random.Random(seed)
    picked = rng.sample(jpegs, batch)
    
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    imgs = []
    for fname in picked:
        p = os.path.join(imagenet_path, fname)
        img = Image.open(p).convert('RGB')
        imgs.append(tfm(img))
    batch_tensor = torch.stack(imgs, dim=0)         # [B, 3, 224, 224]
    x_seq = batch_tensor.unsqueeze(0).expand(T, -1, -1, -1, -1).contiguous()
    return x_seq.to(device)


def make_input_seq(dataset: str, imagenet_path: str, T: int, batch: int,
                   device: torch.device, seed: int = 0):
    if dataset == 'cifar10':
        return make_cifar10_input(T, batch, device, seed)
    elif dataset == 'imagenet':
        return load_imagenet_batch(imagenet_path, T, batch, device, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================
# Forward helpers
# ============================================================

def sj_forward(net, x_seq):
    sj_func.reset_net(net)
    return net(x_seq)


def ctf_forward(net, x_seq):
    return net(x_seq)


# ============================================================
# Spike rate diagnostic (printed for first config per run)
# ============================================================

def collect_spike_rates(net, x_seq):
    """
    Attach forward hooks to every LIFNode, run one forward, return
    a dict of {layer_name: mean_spike_rate}.
    """
    rates = {}
    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.LIFNode):
            def make_hook(n):
                def fn(mod, inp, out):
                    rates[n] = out.float().mean().item()
                return fn
            hooks.append(m.register_forward_hook(make_hook(name)))
    
    sj_func.reset_net(net)
    with torch.no_grad():
        _ = net(x_seq)
    
    for h in hooks:
        h.remove()
    return rates


def summarize_spike_rates(rates: dict):
    if not rates:
        return {'n_lif': 0, 'summary': 'no LIF nodes found'}
    vals = list(rates.values())
    nonzero = [v for v in vals if v > 1e-6]
    return {
        'n_lif': len(vals),
        'n_nonzero': len(nonzero),
        'min': min(vals),
        'max': max(vals),
        'mean': sum(vals) / len(vals),
        'mean_nonzero': (sum(nonzero) / len(nonzero)) if nonzero else 0.0,
    }


# ============================================================
# Structural check
# ============================================================

def structural_check(reference, candidate, degenerate_abs_max_threshold=1e-4):
    """
    Structural/sanity check. NOT a numerical parity proof.
    Numerical parity is validated at kernel level (kernels v9-v15).
    """
    assert reference.shape == candidate.shape, \
        f"shape mismatch: {reference.shape} vs {candidate.shape}"
    diff = (reference - candidate).abs()
    max_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    ref_max_abs = reference.abs().max().item()
    cand_max_abs = candidate.abs().max().item()
    scale = max(ref_max_abs, cand_max_abs)
    degenerate = scale < degenerate_abs_max_threshold
    rel_diff = None if degenerate else max_diff / scale
    
    ref_argmax_step = reference.argmax(dim=-1)
    cand_argmax_step = candidate.argmax(dim=-1)
    argmax_match_step = (ref_argmax_step == cand_argmax_step).float().mean().item() * 100
    
    ref_argmax_meaned = reference.mean(dim=0).argmax(dim=-1)
    cand_argmax_meaned = candidate.mean(dim=0).argmax(dim=-1)
    argmax_match_meaned = (ref_argmax_meaned == cand_argmax_meaned).float().mean().item() * 100
    
    return {
        'shape': list(reference.shape),
        'max_diff': max_diff,
        'mean_abs_diff': mean_abs_diff,
        'ref_max_abs': ref_max_abs,
        'cand_max_abs': cand_max_abs,
        'rel_diff': rel_diff,
        'degenerate': degenerate,
        'argmax_match_stepwise_pct': argmax_match_step,
        'argmax_match_time_mean_pct': argmax_match_meaned,
    }


# ============================================================
# Timing
# ============================================================

def cuda_time_one_shot(fn, n_iter: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def cuda_time_stats(fn, n_iter: int = N_ITER, n_repeat: int = N_REPEAT) -> dict:
    samples = [cuda_time_one_shot(fn, n_iter) for _ in range(n_repeat)]
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'stdev_ms': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'n_iter': n_iter,
        'n_repeat': n_repeat,
    }


def measure_peak_memory(fn, warmup_iters: int = 2) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    return {
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
    }


# ============================================================
# Per-config run
# ============================================================

def run_one_config(net_name: str, T: int, batch: int, device: torch.device,
                   dataset: str, imagenet_path: str, pretrained: bool,
                   skip_ctf: bool = False, print_spike_rates: bool = False) -> dict:
    print(f"\n{'='*78}")
    print(f"Config: net={net_name}, T={T}, batch={batch}, device={device}")
    print(f"        dataset={dataset}, pretrained={pretrained}")
    print(f"{'='*78}")
    
    cfg = DATASET_CONFIGS[dataset]
    num_classes = cfg['num_classes']
    
    result = {
        'net': net_name,
        'T': T,
        'batch': batch,
        'device': str(torch.cuda.get_device_name(0)),
        'dataset': dataset,
        'num_classes': num_classes,
        'input_shape': list(cfg['input_shape']),
        'pretrained': pretrained,
        'seed': 0,
    }
    
    # --- Step 1: build SJ networks ---
    torch.manual_seed(0)
    print("\nStep 1: Building SJ torch network...")
    try:
        sj_torch_net = make_sj_net(net_name, num_classes, 'torch',
                                   pretrained=pretrained).to(device).eval()
    except Exception as e:
        print(f"  FAILED to build sj_torch: {type(e).__name__}: {e}")
        result['error'] = f"sj_torch build: {e}"
        return result
    
    print("Building SJ cupy network...")
    sj_cupy_net = None
    try:
        torch.manual_seed(0)
        sj_cupy_net = make_sj_net(net_name, num_classes, 'cupy',
                                  pretrained=pretrained).to(device).eval()
        result['cupy_available'] = True
    except Exception as e:
        print(f"  FAILED to build sj_cupy: {type(e).__name__}: {e}")
        result['cupy_available'] = False
    
    # --- Step 2: CTF (placeholder) ---
    ctf_net = None
    if not skip_ctf:
        print("Building CTF network...")
        try:
            sj_state = sj_torch_net.state_dict()
            ctf_net = make_ctf_net(net_name, num_classes, sj_state,
                                   pretrained=pretrained).to(device).eval()
        except NotImplementedError as e:
            print(f"  CTF net not yet implemented, skipping")
            result['ctf_status'] = 'not_implemented'
        except Exception as e:
            print(f"  FAILED to build CTF: {type(e).__name__}: {e}")
            result['ctf_status'] = f'error: {e}'
    else:
        result['ctf_status'] = 'skipped_by_flag'
    
    # --- Step 3: input ---
    x_seq = make_input_seq(dataset, imagenet_path, T, batch, device, seed=0)
    print(f"\nInput shape: {list(x_seq.shape)}")
    
    # --- Step 4: spike rate diagnostic (first config only) ---
    if print_spike_rates:
        print("\nStep 4a: Spike-rate diagnostic (sj_torch network)")
        rates = collect_spike_rates(sj_torch_net, x_seq)
        summary = summarize_spike_rates(rates)
        print(f"  LIF nodes: {summary['n_lif']}, "
              f"nonzero: {summary['n_nonzero']}")
        if summary['n_nonzero'] > 0:
            print(f"  spike rates: min={summary['min']:.4f}, "
                  f"max={summary['max']:.4f}, "
                  f"mean(all)={summary['mean']:.4f}, "
                  f"mean(nonzero only)={summary['mean_nonzero']:.4f}")
        result['spike_rate_summary'] = summary
    
    # --- Step 4: structural check ---
    print("\nStep 4b: Structural check")
    with torch.no_grad():
        out_sj_torch = sj_forward(sj_torch_net, x_seq)
        print(f"  sj_torch output: shape={list(out_sj_torch.shape)}, "
              f"mean={out_sj_torch.mean().item():.4f}, "
              f"std={out_sj_torch.std().item():.4f}")
        
        if sj_cupy_net is not None:
            out_sj_cupy = sj_forward(sj_cupy_net, x_seq)
            check_sj = structural_check(out_sj_torch, out_sj_cupy)
            result['structural_sj_torch_vs_cupy'] = check_sj
            degen_tag = ' [DEGENERATE]' if check_sj['degenerate'] else ''
            print(f"  sj_torch vs sj_cupy: max_diff={check_sj['max_diff']:.2e}, "
                  f"argmax_step={check_sj['argmax_match_stepwise_pct']:.1f}%, "
                  f"argmax_mean={check_sj['argmax_match_time_mean_pct']:.1f}%"
                  f"{degen_tag}")
        
        if ctf_net is not None:
            out_ctf = ctf_forward(ctf_net, x_seq)
            check_ctf = structural_check(out_sj_torch, out_ctf)
            result['structural_sj_torch_vs_ctf'] = check_ctf
            degen_tag = ' [DEGENERATE]' if check_ctf['degenerate'] else ''
            print(f"  sj_torch vs CTF:     max_diff={check_ctf['max_diff']:.2e}, "
                  f"argmax_step={check_ctf['argmax_match_stepwise_pct']:.1f}%, "
                  f"argmax_mean={check_ctf['argmax_match_time_mean_pct']:.1f}%"
                  f"{degen_tag}")
    
    # --- Step 5: wall-clock ---
    print("\nStep 5: Wall-clock")
    
    def bench_sj_torch():
        with torch.no_grad():
            _ = sj_forward(sj_torch_net, x_seq)
    
    for _ in range(N_WARMUP):
        bench_sj_torch()
    torch.cuda.synchronize()
    result['wall_sj_torch'] = cuda_time_stats(bench_sj_torch)
    print(f"  sj_torch:  {result['wall_sj_torch']['median_ms']:.3f} ms "
          f"± {result['wall_sj_torch']['stdev_ms']:.3f}")
    
    if sj_cupy_net is not None:
        def bench_sj_cupy():
            with torch.no_grad():
                _ = sj_forward(sj_cupy_net, x_seq)
        for _ in range(N_WARMUP):
            bench_sj_cupy()
        torch.cuda.synchronize()
        result['wall_sj_cupy'] = cuda_time_stats(bench_sj_cupy)
        print(f"  sj_cupy:   {result['wall_sj_cupy']['median_ms']:.3f} ms "
              f"± {result['wall_sj_cupy']['stdev_ms']:.3f}")
    
    if ctf_net is not None:
        def bench_ctf():
            with torch.no_grad():
                _ = ctf_forward(ctf_net, x_seq)
        for _ in range(N_WARMUP):
            bench_ctf()
        torch.cuda.synchronize()
        result['wall_ctf'] = cuda_time_stats(bench_ctf)
        print(f"  CTF:       {result['wall_ctf']['median_ms']:.3f} ms "
              f"± {result['wall_ctf']['stdev_ms']:.3f}")
    
    # --- Step 6: peak memory ---
    print("\nStep 6: Peak GPU memory")
    result['mem_sj_torch'] = measure_peak_memory(bench_sj_torch)
    print(f"  sj_torch:  {result['mem_sj_torch']['peak_allocated_mb']:.1f} MB")
    if sj_cupy_net is not None:
        result['mem_sj_cupy'] = measure_peak_memory(bench_sj_cupy)
        print(f"  sj_cupy:   {result['mem_sj_cupy']['peak_allocated_mb']:.1f} MB")
    if ctf_net is not None:
        result['mem_ctf'] = measure_peak_memory(bench_ctf)
        print(f"  CTF:       {result['mem_ctf']['peak_allocated_mb']:.1f} MB")
    
    # --- Speedups ---
    if 'wall_sj_cupy' in result:
        t_cupy = result['wall_sj_cupy']['median_ms']
        t_torch = result['wall_sj_torch']['median_ms']
        result['speedup_cupy_over_torch'] = t_torch / t_cupy
        print(f"\nSpeedups:")
        print(f"  sj_cupy/sj_torch: {result['speedup_cupy_over_torch']:.3f}x")
        if 'wall_ctf' in result:
            t_ctf = result['wall_ctf']['median_ms']
            result['speedup_ctf_over_torch'] = t_torch / t_ctf
            result['speedup_ctf_over_cupy'] = t_cupy / t_ctf
            print(f"  CTF/sj_torch:     {result['speedup_ctf_over_torch']:.3f}x")
            print(f"  CTF/sj_cupy:      {result['speedup_ctf_over_cupy']:.3f}x")
    
    del sj_torch_net
    if sj_cupy_net is not None:
        del sj_cupy_net
    if ctf_net is not None:
        del ctf_net
    torch.cuda.empty_cache()
    
    return result


# ============================================================
# OOM-retry wrapper
# ============================================================

def run_with_oom_retry(net_name: str, T: int, batch: int, device,
                       dataset: str, imagenet_path: str, pretrained: bool,
                       skip_ctf: bool, print_spike_rates: bool):
    """
    Try (net, T, batch). On OOM, retry with batch=max(1, batch//2).
    Max 2 retries.
    """
    tried_batches = []
    current_batch = batch
    for attempt in range(3):
        try:
            tried_batches.append(current_batch)
            result = run_one_config(
                net_name, T, current_batch, device,
                dataset, imagenet_path, pretrained,
                skip_ctf=skip_ctf,
                print_spike_rates=(print_spike_rates and attempt == 0),
            )
            if current_batch != batch:
                result['batch_fallback_from'] = batch
                result['batch_attempts'] = tried_batches
            return result
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM at batch={current_batch}] net={net_name}, T={T}")
            torch.cuda.empty_cache()
            if current_batch <= 1:
                return {
                    'net': net_name, 'T': T, 'batch': current_batch,
                    'error': 'OOM at batch=1', 'batch_attempts': tried_batches,
                }
            current_batch = max(1, current_batch // 2)
            print(f"  retrying with batch={current_batch}")
    return {
        'net': net_name, 'T': T, 'batch': current_batch,
        'error': f'OOM after {len(tried_batches)} attempts',
        'batch_attempts': tried_batches,
    }


# ============================================================
# Main
# ============================================================

def parse_list(s: str, allowed: list = None):
    if s.lower() == 'all':
        return allowed if allowed is not None else None
    items = [x.strip() for x in s.split(',')]
    if allowed is not None:
        for item in items:
            if item not in allowed:
                raise ValueError(f"Unknown item '{item}', allowed: {allowed}")
    return items


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser(description="CTF end-to-end benchmark")
    p.add_argument('--net', type=str, default='resnet18',
                   help=f"Network(s) or 'all'. Options: {ALL_NETS}")
    p.add_argument('--T', type=str, default='16', help="Time steps, comma-separated")
    p.add_argument('--batch', type=int, default=16, help="Batch size")
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--dataset', type=str, default='imagenet',
                   choices=['cifar10', 'imagenet'])
    p.add_argument('--imagenet-path', type=str, default=DEFAULT_IMAGENET_PATH,
                   help="Flat directory containing ImageNet val JPEGs")
    p.add_argument('--pretrained', action='store_true',
                   help="Load ANN ImageNet pretrained weights into Conv/BN")
    p.add_argument('--output-dir', type=str, default='./results/e2e_v100')
    p.add_argument('--skip-ctf', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-oom-retry', action='store_true',
                   help="Disable automatic batch halving on OOM")
    args = p.parse_args()
    
    # Validation
    if args.dataset == 'imagenet' and not args.pretrained:
        print("WARNING: --dataset imagenet without --pretrained")
        print("         Results will be meaningless (see CIFAR-10 run_log.txt)")
    if args.pretrained and args.dataset != 'imagenet':
        print("WARNING: --pretrained forces num_classes=1000 (ImageNet)")
        print("         but --dataset is not imagenet. This will likely fail.")
    
    nets = parse_list(args.net, allowed=ALL_NETS)
    Ts = parse_int_list(args.T)
    batch = args.batch
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    
    print(f"Device:        {torch.cuda.get_device_name(device)}")
    print(f"Dataset:       {args.dataset}")
    print(f"Pretrained:    {args.pretrained}")
    if args.dataset == 'imagenet':
        print(f"ImageNet path: {args.imagenet_path}")
    print(f"Nets:          {nets}")
    print(f"T list:        {Ts}")
    print(f"Batch:         {batch} (with OOM retry: {not args.no_oom_retry})")
    print(f"Skip CTF:      {args.skip_ctf}")
    print(f"Output:        {args.output_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print(f"\n[DRY RUN] Total configs: {len(nets) * len(Ts)}")
        for net in nets:
            for T in Ts:
                print(f"  {net}, T={T}, batch={batch}")
        return
    
    all_results = []
    failed = []
    t_total_start = time.time()
    is_first_config = True
    
    for net in nets:
        for T in Ts:
            try:
                if args.no_oom_retry:
                    result = run_one_config(
                        net, T, batch, device,
                        args.dataset, args.imagenet_path, args.pretrained,
                        skip_ctf=args.skip_ctf,
                        print_spike_rates=is_first_config,
                    )
                else:
                    result = run_with_oom_retry(
                        net, T, batch, device,
                        args.dataset, args.imagenet_path, args.pretrained,
                        skip_ctf=args.skip_ctf,
                        print_spike_rates=is_first_config,
                    )
                is_first_config = False
                
                if 'error' in result:
                    failed.append(result)
                else:
                    all_results.append(result)
                    b_tag = result.get('batch', batch)
                    fname = f"e2e_{net}_T{T}_B{b_tag}.json"
                    with open(output_dir / fname, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n  → saved to {output_dir / fname}")
            except Exception as e:
                print(f"\n[ERROR] net={net}, T={T}: {type(e).__name__}: {e}")
                failed.append({
                    'net': net, 'T': T, 'batch': batch,
                    'reason': f'{type(e).__name__}: {e}',
                })
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
    
    t_total = time.time() - t_total_start
    
    print(f"\n{'='*78}")
    print(f"Summary: {len(all_results)} succeeded, {len(failed)} failed, "
          f"total time {t_total/60:.1f} min")
    print(f"{'='*78}")
    
    summary_path = output_dir / 'e2e_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'pretrained': args.pretrained,
                'batch': batch,
                'Ts': Ts,
                'nets': nets,
            },
            'results': all_results,
            'failed': failed,
            'total_time_sec': t_total,
        }, f, indent=2)
    print(f"Full summary saved to {summary_path}")
    
    print(f"\n{'net':<14} {'T':<4} {'B':<4} {'sj_torch':<14} {'sj_cupy':<14} "
          f"{'c/t':<8}")
    print('-' * 72)
    for r in all_results:
        net = r['net']
        T = r['T']
        b = r.get('batch', batch)
        t_torch = r.get('wall_sj_torch', {}).get('median_ms', float('nan'))
        t_cupy = r.get('wall_sj_cupy', {}).get('median_ms', float('nan'))
        sp = r.get('speedup_cupy_over_torch', float('nan'))
        print(f"{net:<14} {T:<4} {b:<4} {t_torch:<14.3f} {t_cupy:<14.3f} "
              f"{sp:<8.3f}")


if __name__ == '__main__':
    main()