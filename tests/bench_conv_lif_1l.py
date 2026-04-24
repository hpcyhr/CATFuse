"""[§3.9] Conv -> LIF single-layer benchmark.

Configs measured:
  eager : SJ torch multi-step (batch-folded conv + chained LIF)
  tvm_a : single-step Conv compiled by TVM + Python T-loop + Python LIF
  tvm_b : T-unrolled Conv+LIF static graph compiled by TVM
  tvm_c : T-folded-into-batch Conv compiled by TVM + Python LIF loop

Primary metric: analytical HBM bytes (§3.9 ground truth).
Secondary metrics: wall-clock ms, bit-exact spike parity.

HBM formula for 1L (per timestep M=N*C*H*W fp32 = 4M bytes):
  Conv ops: R x + W z = 2M
  LIF ops:  R z + R v_prev + W v + W s = 4M
  Total per step = 6M
  Over T: 6*T*M elements = 24*T*M bytes
  Best-case TVM-B per-step fusion kills z_t -> 4*T*M elements
"""
import os, sys, time, statistics, tempfile, warnings
import numpy as np
import torch
import torch.nn as nn
import onnx
import tvm
from tvm import relax
from tvm import dlight as dl
from tvm.relax.frontend import onnx as relax_onnx
from spikingjelly.activation_based import neuron, functional

warnings.filterwarnings("ignore")

# ---- Env check ----
print(f"TVM {tvm.__version__}  |  PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
from tvm.contrib import nvcc as _nvcc
print(f"TVM CUDA path: {_nvcc.find_cuda_path()}")
assert "12" in _nvcc.find_cuda_path(), f"CUDA version mismatch; did you forget to source env.sh?"

# ---- Constants ----
TAU, V_TH, V_RESET = 2.0, 1.0, 0.0
DEVICE, DTYPE = 'cuda', torch.float32
N, C, Hs, Ws = 1, 64, 32, 32
T_VALUES = [4, 8, 16]
N_WARMUP, N_ITERS, N_REPEATS = 20, 50, 3
TMP_DIR = tempfile.mkdtemp(prefix='ctf_bench_')
print(f"Temp ONNX dir: {TMP_DIR}\n")

# ---- LIF reference (bit-exact w/ SJ) ----
def lif_step_ref(z, v_prev):
    h = v_prev + (z - v_prev) / TAU
    s = (h >= V_TH).float()
    v = h * (1 - s) + V_RESET * s
    return v, s

# ---- Analytical HBM ----
def hbm_bytes(tag, T, N, C, H, W, bpe=4):
    M = N*C*H*W
    if tag in ('eager', 'tvm_a', 'tvm_c'):
        return 6*T*M*bpe
    if tag == 'tvm_b_best':
        return 4*T*M*bpe
    raise ValueError(tag)

# ---- DLPack interop (zero-copy torch <-> TVM) ----
def torch_to_tvm(x):
    return tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
def tvm_to_torch(x):
    return torch.utils.dlpack.from_dlpack(x.to_dlpack())

# ---- TVM build helper ----
def build_relax_from_onnx(onnx_path):
    mod = relax_onnx.from_onnx(onnx.load(onnx_path))
    target = tvm.target.Target("cuda")
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    ex = relax.build(mod, target=target)
    return relax.VirtualMachine(ex, tvm.cuda(0))

# =============================================================================
# Config 1: eager (SJ torch multi-step)
# =============================================================================
class EagerSJ(nn.Module):
    def __init__(self, C, conv_weight):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(conv_weight)
        self.lif = neuron.LIFNode(
            tau=TAU, v_threshold=V_TH, v_reset=V_RESET,
            decay_input=True, backend='torch', step_mode='m',
        )
    def forward(self, x_seq):
        functional.reset_net(self)
        T_, N_ = x_seq.shape[:2]
        z = self.conv(x_seq.reshape(T_*N_, *x_seq.shape[2:]))
        z = z.reshape(T_, N_, *z.shape[1:])
        return self.lif(z)

# =============================================================================
# Config 2: TVM-A (single-step Conv + Python loop)
# =============================================================================
class TVMA:
    def __init__(self, N, C, H, W, conv_weight):
        conv = nn.Conv2d(C, C, 3, padding=1, bias=False).eval()
        with torch.no_grad():
            conv.weight.copy_(conv_weight)
        path = os.path.join(TMP_DIR, f'tvm_a_N{N}.onnx')
        torch.onnx.export(conv, torch.randn(N, C, H, W), path,
                          input_names=['x'], output_names=['z'], opset_version=17)
        self.vm = build_relax_from_onnx(path)
        self.v_shape = (N, C, H, W)
    def run(self, x_seq):
        T_ = x_seq.shape[0]
        v = torch.zeros(self.v_shape, device=DEVICE, dtype=DTYPE)
        s_list = []
        for t in range(T_):
            x_t = x_seq[t].contiguous()
            z_t = tvm_to_torch(self.vm["main"](torch_to_tvm(x_t)))
            v, s = lif_step_ref(z_t, v)
            s_list.append(s)
        return torch.stack(s_list)

# =============================================================================
# Config 3: TVM-B (T-unrolled Conv + LIF static graph)
# =============================================================================
class UnrolledConvLIF(nn.Module):
    def __init__(self, T, C, conv_weight):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(C, C, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(conv_weight)
    def forward(self, x_seq, v_init):
        v = v_init
        s_list = []
        for t in range(self.T):
            z = self.conv(x_seq[t])
            h = v + (z - v) / TAU
            s = (h >= V_TH).float()
            v = h * (1 - s) + V_RESET * s
            s_list.append(s)
        return torch.stack(s_list), v

class TVMB:
    def __init__(self, T, N, C, H, W, conv_weight):
        m = UnrolledConvLIF(T, C, conv_weight).eval()
        path = os.path.join(TMP_DIR, f'tvm_b_T{T}.onnx')
        torch.onnx.export(
            m, (torch.randn(T, N, C, H, W), torch.zeros(N, C, H, W)), path,
            input_names=['x_seq', 'v_init'], output_names=['s_seq', 'v_final'],
            opset_version=17,
        )
        self.vm = build_relax_from_onnx(path)
        self.v_shape = (N, C, H, W)
    def run(self, x_seq):
        v_init = torch.zeros(self.v_shape, device=DEVICE, dtype=DTYPE)
        out = self.vm["main"](torch_to_tvm(x_seq), torch_to_tvm(v_init))
        # Relax VM returns tuple of NDArray
        s_tvm = out[0] if isinstance(out, (list, tuple)) else out
        return tvm_to_torch(s_tvm)

# =============================================================================
# Config 4: TVM-C (T-folded-into-batch Conv + Python LIF)
# =============================================================================
class TVMC:
    def __init__(self, T, N, C, H, W, conv_weight):
        conv = nn.Conv2d(C, C, 3, padding=1, bias=False).eval()
        with torch.no_grad():
            conv.weight.copy_(conv_weight)
        path = os.path.join(TMP_DIR, f'tvm_c_T{T}_N{N}.onnx')
        torch.onnx.export(conv, torch.randn(T*N, C, H, W), path,
                          input_names=['x'], output_names=['z'], opset_version=17)
        self.vm = build_relax_from_onnx(path)
        self.v_shape = (N, C, H, W)
        self.N = N
    def run(self, x_seq):
        T_, N_ = x_seq.shape[:2]
        x_flat = x_seq.reshape(T_*N_, *x_seq.shape[2:]).contiguous()
        z_flat = tvm_to_torch(self.vm["main"](torch_to_tvm(x_flat)))
        z = z_flat.reshape(T_, N_, *z_flat.shape[1:])
        v = torch.zeros(self.v_shape, device=DEVICE, dtype=DTYPE)
        s_list = []
        for t in range(T_):
            v, s = lif_step_ref(z[t], v)
            s_list.append(s)
        return torch.stack(s_list)

# =============================================================================
# Benchmark + parity
# =============================================================================
def bench(fn, x):
    times = []
    for _ in range(N_REPEATS):
        for _ in range(N_WARMUP):
            _ = fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            out = fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)/N_ITERS*1000)
    return statistics.median(times), out

def parity_report(s_ref, s_test):
    if s_ref.shape != s_test.shape:
        return f"SHAPE_MISMATCH {s_test.shape}"
    flip = (s_ref != s_test).float().mean().item()
    return "bit-exact" if flip == 0.0 else f"{flip*100:.4f}% flipped"

# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(0)
    conv_weight = torch.randn(C, C, 3, 3) * (1.0 / 3)  # scaled for reasonable spike rate

    all_rows = []
    for T in T_VALUES:
        print(f"========== T = {T} ==========")
        torch.manual_seed(0)
        # clip to non-negative to simulate spike-like inputs
        x_seq = (torch.randn(T, N, C, Hs, Ws, device=DEVICE) * 0.5 + 0.3).clamp_(0)

        # eager reference
        print("  build eager... ", end='', flush=True)
        eager = EagerSJ(C, conv_weight).to(DEVICE).eval()
        print("ok")
        with torch.no_grad():
            fn_eager = lambda x: eager(x)
            ms_e, s_ref = bench(fn_eager, x_seq)
        spike_rate = s_ref.mean().item()
        hbm_e = hbm_bytes('eager', T, N, C, Hs, Ws)
        print(f"  eager  {ms_e:>8.3f} ms  HBM {hbm_e/2**20:>7.2f} MB  spike_rate={spike_rate:.3f}")

        # tvm_a
        print("  build tvm_a... ", end='', flush=True)
        tvm_a = TVMA(N, C, Hs, Ws, conv_weight)
        print("ok")
        with torch.no_grad():
            ms_a, s_a = bench(tvm_a.run, x_seq)
        par_a = parity_report(s_ref, s_a)
        hbm_a = hbm_bytes('tvm_a', T, N, C, Hs, Ws)
        print(f"  tvm_a  {ms_a:>8.3f} ms  HBM {hbm_a/2**20:>7.2f} MB  parity={par_a}")

        # tvm_b
        print(f"  build tvm_b (T={T} unrolled)... ", end='', flush=True)
        tvm_b = TVMB(T, N, C, Hs, Ws, conv_weight)
        print("ok")
        with torch.no_grad():
            ms_b, s_b = bench(tvm_b.run, x_seq)
        par_b = parity_report(s_ref, s_b)
        hbm_b_best = hbm_bytes('tvm_b_best', T, N, C, Hs, Ws)
        print(f"  tvm_b  {ms_b:>8.3f} ms  HBM {hbm_b_best/2**20:>7.2f} MB (best fusion)  parity={par_b}")

        # tvm_c
        print("  build tvm_c... ", end='', flush=True)
        tvm_c = TVMC(T, N, C, Hs, Ws, conv_weight)
        print("ok")
        with torch.no_grad():
            ms_c, s_c = bench(tvm_c.run, x_seq)
        par_c = parity_report(s_ref, s_c)
        hbm_c = hbm_bytes('tvm_c', T, N, C, Hs, Ws)
        print(f"  tvm_c  {ms_c:>8.3f} ms  HBM {hbm_c/2**20:>7.2f} MB  parity={par_c}")

        all_rows.append((T, ms_e, hbm_e, ms_a, hbm_a, par_a, ms_b, hbm_b_best, par_b, ms_c, hbm_c, par_c))
        print()

    # Summary table
    print("=" * 92)
    print(f"SUMMARY  N={N}  C={C}  H={Hs}  W={Ws}  fp32  (A100-80GB-PCIe)")
    print("=" * 92)
    print(f"{'T':>3} {'cfg':<7} {'ms':>9} {'HBM MB':>9}  {'parity':<25}")
    print("-" * 92)
    for row in all_rows:
        T, ms_e, hbm_e, ms_a, hbm_a, par_a, ms_b, hbm_b, par_b, ms_c, hbm_c, par_c = row
        print(f"{T:>3} {'eager':<7} {ms_e:>9.3f} {hbm_e/2**20:>9.2f}  {'(ref)':<25}")
        print(f"{T:>3} {'tvm_a':<7} {ms_a:>9.3f} {hbm_a/2**20:>9.2f}  {par_a:<25}")
        print(f"{T:>3} {'tvm_b':<7} {ms_b:>9.3f} {hbm_b/2**20:>9.2f}  {par_b:<25}")
        print(f"{T:>3} {'tvm_c':<7} {ms_c:>9.3f} {hbm_c/2**20:>9.2f}  {par_c:<25}")
        print("-" * 92)

if __name__ == "__main__":
    main()
