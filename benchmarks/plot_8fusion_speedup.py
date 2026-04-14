"""
Plot wall-clock speedup vs TSI reduction dim (K_total) for the 8 fusions.

This figure visualizes the three-way scope of CTF wall-clock realization:
  - memory-bound TSI (K_total = 1):    speedup 1.26x - 2.17x
  - compute-light TSI (K_total = 128): speedup ~0.95x
  - compute-bound TSI (K_total >= 512): speedup 0.55x - 0.69x

Run:
    python plot_8fusion_speedup.py

Output:
    fig_8fusion_speedup.pdf
    fig_8fusion_speedup.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 8 fusion data points
# Each entry: (name, K_total, speedup, category, label_offset)
# K_total semantics:
#   - For elementwise / pool TSI: K_total = 1 (no reduction)
#   - For matmul / conv TSI:      K_total = (effective fan-in dimension)
#     - Linear(I=512, O=512):     K_total = 512
#     - Conv 1x1, C_in=128:       K_total = 128
#     - Conv 3x3, C_in=128:       K_total = 128*3*3 = 1152
# label_offset is (dx, dy) in axis units for text placement
# ============================================================

fusions = [
    # (name,              K_total, speedup, category,        offset)
    ("Add → LIF",              1,  1.558, "memory-bound",  ( 1.3,  0.04)),
    ("AvgPool → LIF",          1,  1.259, "memory-bound",  ( 1.3, -0.06)),
    ("Add → BN → LIF",         1,  2.165, "memory-bound",  ( 1.3,  0.00)),
    ("Conv 1×1 → LIF",       128,  0.947, "compute-light", ( 1.3,  0.04)),
    ("Linear → LIF",         512,  0.689, "compute-bound", ( 1.3,  0.04)),
    ("Conv 3×3 s=1 → LIF",  1152,  0.616, "compute-bound", (-1.5,  0.06)),
    ("Conv 3×3 s=2 → LIF",  1152,  0.546, "compute-bound", (-1.5, -0.07)),
    ("Conv → BN → LIF",     1152,  0.608, "compute-bound", ( 1.3, -0.01)),
]

color_map = {
    "memory-bound":  "#2E7D32",   # dark green
    "compute-light": "#F9A825",   # amber
    "compute-bound": "#C62828",   # dark red
}
marker_map = {
    "memory-bound":  "o",
    "compute-light": "s",
    "compute-bound": "D",
}

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)

# Reference horizontal lines
ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
ax.axhline(y=2.0, color="grey", linestyle=":",  linewidth=1, alpha=0.4, zorder=1)
ax.text(1750, 1.02, "break-even (speedup = 1)", color="grey", fontsize=8,
        ha="right", va="bottom")
ax.text(1750, 2.02, "Add+BN headline (2.17×)", color="grey", fontsize=8,
        ha="right", va="bottom")

# Plot each fusion
for name, K_total, speedup, cat, (dx, dy) in fusions:
    # Use K_total + small jitter for visibility on log scale (avoid x=1 collapse)
    x = K_total
    ax.scatter(x, speedup,
               color=color_map[cat],
               marker=marker_map[cat],
               s=110,
               edgecolor="black",
               linewidth=0.7,
               zorder=3)
    # Adjust label placement: for x=1 cluster, fan out vertically
    if K_total == 1:
        # Three points cluster at x=1; spread labels
        ax.annotate(name, (x, speedup),
                    xytext=(x + dx, speedup + dy),
                    fontsize=9, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="black",
                                    lw=0.5, alpha=0.5))
    else:
        ax.annotate(name, (x, speedup),
                    xytext=(x * (1.15 if dx > 0 else 0.6),
                            speedup + dy),
                    fontsize=9,
                    ha="left" if dx > 0 else "right",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color="black",
                                    lw=0.5, alpha=0.5))

# Axes
ax.set_xscale("log")
ax.set_xlim(0.5, 2500)
ax.set_ylim(0.4, 2.5)
ax.set_xlabel(r"TSI reduction dimension $K_\mathrm{total}$",
              fontsize=11)
ax.set_ylabel(r"Wall-clock speedup ($t_\mathrm{nof}\,/\,t_\mathrm{fus}$)",
              fontsize=11)
ax.set_title("CTF wall-clock realization vs TSI compute weight\n"
             "(8 fusions, V100-SXM2 + Triton 2.1.0, T=16, K=T)",
             fontsize=11)

# Set log-ticks at meaningful K_total values
ax.set_xticks([1, 10, 100, 128, 512, 1152, 1000])
ax.set_xticklabels(["1", "10", "100", "128", "512", "1152", ""])

# Grid
ax.grid(True, which="major", axis="y", alpha=0.3)
ax.grid(True, which="major", axis="x", alpha=0.15)

# Legend by category
legend_handles = [
    mpatches.Patch(color=color_map["memory-bound"],
                   label="memory-bound TSI (Add, Pool)"),
    mpatches.Patch(color=color_map["compute-light"],
                   label="compute-light TSI (Conv 1×1)"),
    mpatches.Patch(color=color_map["compute-bound"],
                   label="compute-bound TSI (Linear, Conv 3×3)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
          framealpha=0.92)

# Optional: shaded region for "fusion realized" (speedup > 1)
ax.axhspan(1.0, 2.5, alpha=0.04, color="green", zorder=0)
ax.axhspan(0.4, 1.0, alpha=0.04, color="red",   zorder=0)

plt.tight_layout()

# Save
plt.savefig("fig_8fusion_speedup.pdf", bbox_inches="tight")
plt.savefig("fig_8fusion_speedup.png", bbox_inches="tight", dpi=150)

print("Saved fig_8fusion_speedup.pdf and .png")
print()
print("Data summary:")
print(f"{'Fusion':<25} {'K_total':>8} {'speedup':>9} {'category':>16}")
for name, K, s, cat, _ in fusions:
    print(f"{name:<25} {K:>8} {s:>9.3f} {cat:>16}")