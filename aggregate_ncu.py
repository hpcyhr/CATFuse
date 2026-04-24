"""Aggregate ncu CSV: skip the ncu log banner (first ~6 lines)
and find the real CSV header starting with \"ID\"."""
import csv
from io import StringIO
from pathlib import Path

LD = 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum'
ST = 'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum'


def total_traffic(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()

    # Find the real CSV header line (starts with "ID")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break

    if header_idx is None:
        return 0, 0, 0

    csv_text = ''.join(lines[header_idx:])
    reader = csv.DictReader(StringIO(csv_text))

    total_ld = 0
    total_st = 0
    n_rows_ld = 0
    for row in reader:
        metric = row.get('Metric Name', '')
        val_str = row.get('Metric Value', '0').replace(',', '')
        try:
            val = int(val_str)
        except ValueError:
            continue
        if metric == LD:
            total_ld += val
            n_rows_ld += 1
        elif metric == ST:
            total_st += val

    return total_ld, total_st, n_rows_ld


print(f"{'File':<36}{'load MB':>12}{'store MB':>12}{'total MB':>12}{'#kernels':>10}")
print("-" * 82)
results = {}
for f in sorted(Path("/data/dagongcheng/yhrtest/CATFuse").glob("ncu_*_*.csv")):
    ld, st, n = total_traffic(f)
    results[f.name] = (ld, st, n)
    print(f"{f.name:<36}{ld/1e6:>12.2f}{st/1e6:>12.2f}{(ld+st)/1e6:>12.2f}{n:>10}")

print()
print("Per-GPU comparison (baseline vs CUDA Graph):")
print("-" * 82)
for gpu in ['A100', 'V100']:
    bf = f"ncu_{gpu}_baseline.csv"
    cf = f"ncu_{gpu}_cudagraph.csv"
    if bf in results and cf in results:
        b_total = sum(results[bf][:2])
        c_total = sum(results[cf][:2])
        if b_total > 0:
            diff_pct = (c_total - b_total) / b_total * 100
            print(f"  {gpu}: baseline={b_total/1e6:.2f} MB, "
                  f"cudagraph={c_total/1e6:.2f} MB, "
                  f"delta={diff_pct:+.2f}%")
        else:
            print(f"  {gpu}: baseline total = 0 (unexpected)")

print()
print("Expected: |delta| < 2% because CUDA Graph replay executes the same")
print("CUDA kernels with unchanged kernel bodies. Any residual difference")
print("reflects minor kernel-schedule reordering or ncu replay variance.")
