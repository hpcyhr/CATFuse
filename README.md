# CATFuse-SF

**Certified Spatio-Temporal Fusion for Efficient SNN Inference**

CATFuse-SF combines two orthogonal optimization dimensions for multi-step spiking neural network inference:

- **CATFuse (temporal)**: CTF schedule transforms (TimeBlock, BatchFold, StreamFuse, StateCarry) eliminate redundant HBM data movement across time steps and operator boundaries
- **SparseFlow (spatial)**: Block-level prescan + grouped sparse convolution exploits the natural high sparsity of spike activations to skip inactive computation

## Quick start

```python
import catfuse

# One-line drop-in replacement for SpikingJelly models
fused_model, stats = catfuse.optimize(sj_model)
```

## Architecture

```
SpikingJelly model
       │
       ▼
┌─────────────────────────────────────┐
│  CATFuse-SF Framework               │
│                                     │
│  1. Pattern matching + CTF cert     │
│  2. Per-layer spatio-temporal policy │
│                                     │
│  Spatial backend (per-layer):       │
│  ┌──────────┬──────────┬──────────┐ │
│  │StaticZero│SparseFlow│DenseKeep │ │
│  │ (全零)   │ (高稀疏)  │ (低稀疏) │ │
│  └────┬─────┴────┬─────┴────┬─────┘ │
│       └──────────┼──────────┘       │
│                  ▼                  │
│  Temporal fused tail (shared):      │
│  BN → LIF → StateCarry → spike     │
└─────────────────────────────────────┘
       │
       ▼
  Spike output [T, B, C, H, W]
```

## Three execution paths

| Path | When | Conv backend | z location | I/O ratio | Compute |
|------|------|-------------|-----------|-----------|---------|
| **DenseKeep** | Low sparsity / compute-bound | cuDNN | HBM | (3+2/K)/7 | 100% dense |
| **SparseFlow** | High sparsity (>80%) | Triton sparse | On-chip | (1+2/K)/7 | r × dense |
| **StaticZero** | All-zero input (100%) | Skipped | Immediate | ~0 | ~0 |

## Project structure

```
CATFuse/
├── catfuse/                     # Core framework package
│   ├── patterns.py              #   Fused pattern library (nn.Module wrappers)
│   ├── substitute.py            #   Module substitution mechanism
│   ├── policy.py                #   Spatio-temporal policy engine
│   ├── kernels/                 #   CATFuse temporal Triton kernels
│   │   ├── lif_fwd.py           #     Multi-step LIF (TimeBlock + StateCarry)
│   │   ├── conv_bn_lif.py       #     Fused Conv→BN→LIF (dense)
│   │   ├── add_lif.py           #     Fused Add→LIF
│   │   ├── linear_lif.py        #     Fused Linear→LIF
│   │   └── avgpool_lif.py       #     Fused AvgPool→LIF
│   └── sparseflow/              #   SparseFlow spatial backend
│       ├── config.py            #     Dispatch thresholds and constants
│       ├── prescan.py           #     Prescan kernels (bitmask construction)
│       ├── sparse_conv2d_kernel.py   # Sparse Conv2d Triton kernel
│       ├── fused_conv_lif_kernel.py  # Fused sparse Conv+LIF kernel
│       ├── dispatch.py          #     EGD: StaticZero/Sparse/DenseKeep
│       ├── registry.py          #     Spike op detection
│       ├── analyzer.py          #     torch.fx graph analysis
│       └── ops/                 #     nn.Module wrappers
│           ├── sparse_conv2d.py
│           ├── sparse_fused_conv_lif.py
│           └── static_zero_conv2d.py
├── benchmarks/                  # All benchmark scripts
├── training/                    # Checkpoint training scripts
├── checkpoints/                 # Trained model weights
└── models/                      # Model definitions
```

## Differentiation

| System | Temporal fusion | Cross-type fusion | Spatial sparsity | Correctness cert |
|--------|----------------|-------------------|-----------------|-----------------|
| SpikingJelly | ✗ | ✗ | ✗ | N/A (reference) |
| Chronos | tTILE batching | ✗ (Conv/LIF always separate) | ✗ | Informal |
| Helios | ✗ (per-layer only) | Thread-anchored | Binary tile skip | ✗ |
| **CATFuse-SF** | TimeBlock+StateCarry | StreamFuse (z on-chip) | Prescan+EGD 3-path | CTF Σ(G,T) |

## Hardware

- Primary: NVIDIA V100-SXM2-32GB (sm_70)
- Secondary: NVIDIA A100-SXM4-40GB (sm_80)
- Software: PyTorch 2.1+, Triton 2.1+, SpikingJelly 0.0.0.0.14

## License

See LICENSE file.
