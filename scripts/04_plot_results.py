"""Generate all result plots for Phase 2 and Phase 3.

Run:
    python scripts/04_plot_results.py

Saves all plots to results/plots/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")          # no display needed — save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_BO   = "#2563EB"    # blue  — DKL-UCB
C_RAND = "#DC2626"    # red   — random
C_GOOD = "#16A34A"    # green — "good zone"
C_WARN = "#F59E0B"    # amber — "warning zone"

def save(fig, name):
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")

# ── load data ─────────────────────────────────────────────────────────────────
bo   = pd.read_csv("results/bo_ucb_beta0.2_results.csv")
rand = pd.read_csv("results/bo_random_results.csv")
p2   = pd.read_csv("results/surrogate_metrics_exactgpsurrogate.csv")
loss = pd.read_csv("results/training_loss_exactgpsurrogate.csv")

print("Generating plots…")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training loss curve (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss["epoch"] + 1, loss["loss"], color=C_BO, lw=2, label="MLL loss")
ax.set_xlabel("Training epoch", fontsize=12)
ax.set_ylabel("Negative MLL loss", fontsize=12)
ax.set_title("Phase 2 — DKL Joint Training Loss", fontsize=14, fontweight="bold")
ax.annotate(f"Start: {loss['loss'].iloc[0]:.2f}",
            xy=(1, loss['loss'].iloc[0]), xytext=(10, loss['loss'].iloc[0] - 0.08),
            fontsize=10, color="gray")
ax.annotate(f"End: {loss['loss'].iloc[-1]:.2f}",
            xy=(100, loss['loss'].iloc[-1]), xytext=(70, loss['loss'].iloc[-1] + 0.08),
            fontsize=10, color=C_BO)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
save(fig, "01_training_loss.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Phase 2 accuracy metrics bar chart
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("Phase 2 — Model Accuracy Metrics", fontsize=14, fontweight="bold")

metrics_labels = ["MAE (eV)", "RMSE (eV)", "R²"]
val_vals  = [p2[p2.split=="val"].iloc[0].mae,
             p2[p2.split=="val"].iloc[0].rmse,
             p2[p2.split=="val"].iloc[0].r2]
test_vals = [p2[p2.split=="test"].iloc[0].mae,
             p2[p2.split=="test"].iloc[0].rmse,
             p2[p2.split=="test"].iloc[0].r2]
targets   = [0.45, 0.65, 0.70]   # approximate literature targets
target_labels = ["Target ≤0.45", "Target ≤0.65", "Target ≥0.70"]

colors_val  = ["#2563EB", "#2563EB", "#2563EB"]
colors_test = ["#93C5FD", "#93C5FD", "#93C5FD"]

for i, (ax, label, vv, tv, tgt, tgt_label) in enumerate(
        zip(axes, metrics_labels, val_vals, test_vals, targets, target_labels)):
    bars = ax.bar(["Val", "Test"], [vv, tv],
                  color=[C_BO, "#93C5FD"], width=0.5, edgecolor="white", linewidth=1.5)
    # target line
    if label == "R²":
        ax.axhline(tgt, color=C_GOOD, lw=1.5, ls="--", label=tgt_label)
        ax.set_ylim(0, 1.0)
    else:
        ax.axhline(tgt, color=C_GOOD, lw=1.5, ls="--", label=tgt_label)
        ax.set_ylim(0, max(vv, tv) * 1.4)
    for bar, val in zip(bars, [vv, tv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F8FAFC")

save(fig, "02_phase2_accuracy.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Phase 2 calibration metrics
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Phase 2 — Calibration Metrics (val split)", fontsize=14, fontweight="bold")

cal_metrics = ["NLL", "ENCE", "Miscal. Area", "Spearman ρ", "Coverage@95"]
val_row = p2[p2.split == "val"].iloc[0]
cal_vals   = [val_row.nll, val_row.ence, val_row.miscal_area,
              val_row.spearman_rho, val_row.coverage_95]
cal_targets_lo = [None, 0.06, 0.04, 0.3,  0.92]
cal_targets_hi = [None, 0.10, 0.07, None, 0.98]
cal_status     = ["lower=better", "target 0.06–0.10", "target 0.04–0.07",
                  "higher=better", "target ≈0.95"]

for ax, name, val, lo, hi, status in zip(
        axes, cal_metrics, cal_vals, cal_targets_lo, cal_targets_hi, cal_status):
    # colour bar by whether it's in target range
    if lo is not None and hi is not None:
        c = C_GOOD if lo <= val <= hi else C_WARN
    elif lo is not None:
        c = C_GOOD if val >= lo else C_WARN
    else:
        c = C_BO   # NLL — just show value

    ax.bar(["Val"], [abs(val)], color=c, width=0.4, edgecolor="white")
    ax.text(0, abs(val) + abs(val)*0.05, f"{val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

    if lo is not None:
        ax.axhline(lo, color="gray", lw=1.2, ls="--")
    if hi is not None:
        ax.axhline(hi, color="gray", lw=1.2, ls="--")

    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel(status, fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F8FAFC")

save(fig, "03_phase2_calibration.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Best gap found over cycles
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(bo["cycle"],   bo["best_so_far"],   color=C_BO,   lw=2.5, label="DKL-UCB (β=0.2)")
ax.plot(rand["cycle"], rand["best_so_far"], color=C_RAND, lw=2.5, label="Random search",
        linestyle="--")

# Mark every top-50 discovery
top50_bo = bo[bo.is_top50]
ax.scatter(top50_bo["cycle"], top50_bo["best_so_far"],
           color=C_BO, s=100, zorder=5, marker="*", label="Top-50 material found")

# Horizontal reference lines
ax.axhline(10.79, color="black", lw=1, ls=":", alpha=0.5, label="Dataset max (10.79 eV)")
ax.axhline(7.02,  color=C_GOOD,  lw=1, ls=":", alpha=0.6, label="Top-50 threshold (7.02 eV)")
ax.axhline(5.05,  color=C_WARN,  lw=1, ls=":", alpha=0.6, label="Top-10% threshold (5.05 eV)")

# Final values annotation
ax.annotate(f"9.58 eV", xy=(100, 9.58), xytext=(85, 9.75),
            fontsize=11, color=C_BO, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_BO, lw=1.5))
ax.annotate(f"6.40 eV", xy=(100, 6.40), xytext=(83, 5.9),
            fontsize=11, color=C_RAND, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_RAND, lw=1.5))

ax.set_xlabel("BO Cycle (number of DFT experiments)", fontsize=12)
ax.set_ylabel("Best band gap found so far (eV)", fontsize=12)
ax.set_title("Phase 3 — Best Material Discovered vs Number of Experiments",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
ax.set_ylim(4, 11.5)
ax.set_facecolor("#F8FAFC")

save(fig, "04_best_gap_over_cycles.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Cumulative top-10% hits
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(bo["cycle"],   bo["cumul_top10pct"],   color=C_BO,   lw=2.5, label="DKL-UCB (β=0.2)")
ax.plot(rand["cycle"], rand["cumul_top10pct"], color=C_RAND, lw=2.5, label="Random search",
        linestyle="--")

# Expected random line (10% hit rate)
cycles = np.arange(1, 101)
expected_random = cycles * 0.10
ax.plot(cycles, expected_random, color="gray", lw=1.5, ls=":", alpha=0.7,
        label="Expected random (10% rate)")

# Efficiency ratio annotation at cycle 100
ax.annotate("4.7× more efficient\nthan random",
            xy=(100, 47), xytext=(65, 40),
            fontsize=11, color=C_BO, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_BO, lw=1.5))

ax.set_xlabel("BO Cycle (number of DFT experiments)", fontsize=12)
ax.set_ylabel("Cumulative top-10% materials found", fontsize=12)
ax.set_title("Phase 3 — Efficiency: How Many High-Gap Materials Found?",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
ax.set_facecolor("#F8FAFC")

save(fig, "05_cumulative_top10pct.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6 — Cumulative top-50 hits
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.step(bo["cycle"],   bo["cumul_top50"],   color=C_BO,   lw=2.5,
        where="post", label="DKL-UCB (β=0.2)")
ax.step(rand["cycle"], rand["cumul_top50"], color=C_RAND, lw=2.5,
        where="post", linestyle="--", label="Random search")

# Expected random (50/3351 ≈ 1.49% hit rate)
expected_r50 = cycles * (50 / 3351)
ax.plot(cycles, expected_r50, color="gray", lw=1.5, ls=":", alpha=0.7,
        label="Expected random (1.5% rate)")

ax.set_xlabel("BO Cycle (number of DFT experiments)", fontsize=12)
ax.set_ylabel("Cumulative top-50 materials found", fontsize=12)
ax.set_title("Phase 3 — Rare Material Discovery (Top-50 by Band Gap ≥7.02 eV)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
ax.set_facecolor("#F8FAFC")

save(fig, "06_cumulative_top50.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7 — Gap acquired per cycle (scatter)
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Phase 3 — Band Gap of Each Acquisition (per cycle)",
             fontsize=13, fontweight="bold")

for ax, df, color, label in [(ax1, bo, C_BO, "DKL-UCB"),
                              (ax2, rand, C_RAND, "Random")]:
    sc = ax.scatter(df["cycle"], df["gap_acquired"],
                    c=df["gap_acquired"], cmap="YlOrRd",
                    s=40, alpha=0.8, vmin=0, vmax=10.8)
    ax.axhline(7.02, color="green", lw=1, ls="--", alpha=0.7, label="Top-50 (7.02 eV)")
    ax.axhline(5.05, color="orange", lw=1, ls="--", alpha=0.7, label="Top-10% (5.05 eV)")
    ax.set_ylabel("Band gap acquired (eV)", fontsize=11)
    ax.set_title(label, fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 11)

plt.colorbar(sc, ax=[ax1, ax2], label="Band gap (eV)", shrink=0.8)
ax2.set_xlabel("BO Cycle", fontsize=12)

save(fig, "07_acquisitions_per_cycle.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8 — Summary comparison dashboard
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
fig.suptitle("DKL-BO Summary Dashboard", fontsize=16, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

# Panel A: best gap
ax_a = fig.add_subplot(gs[0, :2])
ax_a.plot(bo["cycle"],   bo["best_so_far"],   color=C_BO,   lw=2, label="DKL-UCB")
ax_a.plot(rand["cycle"], rand["best_so_far"], color=C_RAND, lw=2, linestyle="--", label="Random")
ax_a.axhline(7.02, color=C_GOOD, lw=1, ls=":", alpha=0.6)
ax_a.axhline(10.79, color="black", lw=1, ls=":", alpha=0.4)
ax_a.set_xlabel("Cycle", fontsize=10)
ax_a.set_ylabel("Best gap (eV)", fontsize=10)
ax_a.set_title("A — Best Discovery Over Time", fontsize=11, fontweight="bold")
ax_a.legend(fontsize=9)
ax_a.grid(alpha=0.3)
ax_a.set_facecolor("#F8FAFC")

# Panel B: KPI summary
ax_b = fig.add_subplot(gs[0, 2])
ax_b.axis("off")
kpis = [
    ("Best gap",      "9.58 eV", "6.40 eV"),
    ("Top-50 hits",   "9",       "0"),
    ("Top-10% hits",  "47",      "10"),
    ("Efficiency",    "4.7×",    "1.0×"),
]
ax_b.text(0.5, 0.97, "Key Results (100 cycles)",
          ha="center", va="top", fontsize=11, fontweight="bold",
          transform=ax_b.transAxes)
ax_b.text(0.55, 0.87, "DKL", ha="center", fontsize=10, color=C_BO, fontweight="bold",
          transform=ax_b.transAxes)
ax_b.text(0.85, 0.87, "Random", ha="center", fontsize=10, color=C_RAND, fontweight="bold",
          transform=ax_b.transAxes)
for i, (name, bo_v, rand_v) in enumerate(kpis):
    y = 0.75 - i * 0.16
    ax_b.text(0.05, y, name,    ha="left",   fontsize=10, transform=ax_b.transAxes)
    ax_b.text(0.55, y, bo_v,   ha="center", fontsize=11, fontweight="bold",
              color=C_BO,  transform=ax_b.transAxes)
    ax_b.text(0.85, y, rand_v, ha="center", fontsize=11, fontweight="bold",
              color=C_RAND, transform=ax_b.transAxes)

# Panel C: cumulative top-10%
ax_c = fig.add_subplot(gs[1, :2])
ax_c.plot(bo["cycle"],   bo["cumul_top10pct"],   color=C_BO,   lw=2, label="DKL-UCB")
ax_c.plot(rand["cycle"], rand["cumul_top10pct"], color=C_RAND, lw=2, linestyle="--", label="Random")
ax_c.plot(cycles, cycles * 0.10, color="gray", lw=1.5, ls=":", alpha=0.6, label="Expected random")
ax_c.set_xlabel("Cycle", fontsize=10)
ax_c.set_ylabel("Top-10% hits", fontsize=10)
ax_c.set_title("C — Cumulative High-Gap Discoveries", fontsize=11, fontweight="bold")
ax_c.legend(fontsize=9)
ax_c.grid(alpha=0.3)
ax_c.set_facecolor("#F8FAFC")

# Panel D: Phase 2 accuracy radar-style bars
ax_d = fig.add_subplot(gs[1, 2])
metrics = ["MAE\n(eV)", "RMSE\n(eV)", "R²"]
v_vals  = [0.4463, 0.6082, 0.6990]
targets_d = [0.45,  0.65,   0.70]
colors_d  = [C_GOOD if (v <= t if i < 2 else v >= t) else C_WARN
             for i, (v, t) in enumerate(zip(v_vals, targets_d))]
bars = ax_d.bar(metrics, v_vals, color=colors_d, width=0.5, edgecolor="white")
for bar, tgt, higher in zip(bars, targets_d, [False, False, True]):
    ax_d.axhline(tgt, color="gray", lw=1, ls="--", alpha=0.6)
    ax_d.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.02,
              f"{bar.get_height():.3f}",
              ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_d.set_title("D — Phase 2 Accuracy (Val)", fontsize=11, fontweight="bold")
ax_d.set_ylim(0, 1.1)
ax_d.grid(axis="y", alpha=0.3)
ax_d.set_facecolor("#F8FAFC")

save(fig, "08_summary_dashboard.png")

print()
print("All 8 plots saved to results/plots/")

# ═══════════════════════════════════════════════════════════════════════════
# BETA SWEEP COMPARISON PLOTS (09–12)
# These only run when multiple beta result files exist in results/
# Run: python scripts/03_run_bo.py data.db_path=... bo.beta=0.0
#      python scripts/03_run_bo.py data.db_path=... bo.beta=0.5  etc.
# ═══════════════════════════════════════════════════════════════════════════

BETA_COLORS = {
    0.0: "#7C3AED",   # purple  — pure exploitation
    0.2: "#2563EB",   # blue    — our default
    0.5: "#0891B2",   # cyan
    1.0: "#16A34A",   # green
    2.0: "#EA580C",   # orange  — heavy exploration
}

# Discover which beta CSVs are present
beta_files = {}
for beta_val, color in BETA_COLORS.items():
    path = Path(f"results/bo_ucb_beta{beta_val}_results.csv")
    if path.exists():
        beta_files[beta_val] = (pd.read_csv(path), color)

if len(beta_files) >= 2:
    print("\nBeta sweep files found — generating comparison plots 09–12…")

    # ── Plot 09: Best gap over cycles for all betas ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    for beta_val, (df, color) in sorted(beta_files.items()):
        lw   = 3.0 if beta_val == 0.2 else 1.8
        lbl  = f"β={beta_val}" + (" (default)" if beta_val == 0.2 else "")
        ax.plot(df["cycle"], df["best_so_far"], color=color, lw=lw, label=lbl)

    ax.plot(rand["cycle"], rand["best_so_far"], color=C_RAND, lw=1.8,
            linestyle="--", label="Random baseline")

    ax.axhline(10.79, color="black", lw=1, ls=":", alpha=0.4, label="Dataset max (10.79 eV)")
    ax.axhline(7.02,  color="gray",  lw=1, ls=":", alpha=0.5, label="Top-50 threshold (7.02 eV)")

    ax.set_xlabel("BO Cycle (number of experiments)", fontsize=12)
    ax.set_ylabel("Best band gap found so far (eV)", fontsize=12)
    ax.set_title("Beta Sweep — Best Material Discovered vs Experiments",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(4, 11.5)
    ax.set_facecolor("#F8FAFC")
    save(fig, "09_beta_best_gap_comparison.png")

    # ── Plot 10: Cumulative top-10% hits for all betas ───────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    for beta_val, (df, color) in sorted(beta_files.items()):
        lw  = 3.0 if beta_val == 0.2 else 1.8
        lbl = f"β={beta_val}" + (" (default)" if beta_val == 0.2 else "")
        ax.plot(df["cycle"], df["cumul_top10pct"], color=color, lw=lw, label=lbl)

    ax.plot(rand["cycle"], rand["cumul_top10pct"], color=C_RAND, lw=1.8,
            linestyle="--", label="Random baseline")
    ax.plot(cycles, cycles * 0.10, color="gray", lw=1.2, ls=":", alpha=0.6,
            label="Expected random (10%)")

    ax.set_xlabel("BO Cycle (number of experiments)", fontsize=12)
    ax.set_ylabel("Cumulative top-10% materials found", fontsize=12)
    ax.set_title("Beta Sweep — Search Efficiency Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_facecolor("#F8FAFC")
    save(fig, "10_beta_top10pct_comparison.png")

    # ── Plot 11: Final results bar chart per beta ────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Beta Sweep — Final Results After 100 Cycles",
                 fontsize=14, fontweight="bold")

    sorted_betas  = sorted(beta_files.keys())
    bar_colors    = [beta_files[b][1] for b in sorted_betas]
    best_gaps     = [beta_files[b][0]["best_so_far"].iloc[-1]  for b in sorted_betas]
    top10_hits    = [beta_files[b][0]["cumul_top10pct"].iloc[-1] for b in sorted_betas]
    top50_hits    = [beta_files[b][0]["cumul_top50"].iloc[-1]  for b in sorted_betas]
    x_labels      = [f"β={b}" for b in sorted_betas]

    # Left: best gap
    bars = ax_l.bar(x_labels, best_gaps, color=bar_colors, width=0.55,
                    edgecolor="white", linewidth=1.5)
    ax_l.axhline(rand["best_so_far"].iloc[-1], color=C_RAND, lw=1.5,
                 ls="--", label=f"Random ({rand['best_so_far'].iloc[-1]:.2f} eV)")
    ax_l.axhline(10.79, color="black", lw=1, ls=":", alpha=0.4, label="Dataset max")
    for bar, val in zip(bars, best_gaps):
        ax_l.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{val:.2f}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold")
    ax_l.set_ylabel("Best band gap found (eV)", fontsize=11)
    ax_l.set_title("Best Material Found", fontsize=12, fontweight="bold")
    ax_l.legend(fontsize=9)
    ax_l.set_ylim(0, 12)
    ax_l.grid(axis="y", alpha=0.3)
    ax_l.set_facecolor("#F8FAFC")

    # Right: top-10% hits
    bars2 = ax_r.bar(x_labels, top10_hits, color=bar_colors, width=0.55,
                     edgecolor="white", linewidth=1.5)
    ax_r.axhline(rand["cumul_top10pct"].iloc[-1], color=C_RAND, lw=1.5,
                 ls="--", label=f"Random ({int(rand['cumul_top10pct'].iloc[-1])} hits)")
    for bar, val in zip(bars2, top10_hits):
        ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  str(int(val)), ha="center", va="bottom",
                  fontsize=10, fontweight="bold")
    ax_r.set_ylabel("Cumulative top-10% materials found", fontsize=11)
    ax_r.set_title("Search Efficiency (Top-10% Hits)", fontsize=12, fontweight="bold")
    ax_r.legend(fontsize=9)
    ax_r.grid(axis="y", alpha=0.3)
    ax_r.set_facecolor("#F8FAFC")

    save(fig, "11_beta_final_results_bars.png")

    # ── Plot 12: Beta sweep summary table ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    fig.suptitle("Beta Sweep — Complete Results Summary",
                 fontsize=14, fontweight="bold", y=1.02)

    rand_best   = rand["best_so_far"].iloc[-1]
    rand_top10  = int(rand["cumul_top10pct"].iloc[-1])
    rand_top50  = int(rand["cumul_top50"].iloc[-1])

    col_labels = ["β value", "Best gap (eV)", "vs Random", "Top-10% hits",
                  "Efficiency vs random", "Top-50 hits"]
    table_data = []
    for b in sorted_betas:
        df_b    = beta_files[b][0]
        bg      = df_b["best_so_far"].iloc[-1]
        t10     = int(df_b["cumul_top10pct"].iloc[-1])
        t50     = int(df_b["cumul_top50"].iloc[-1])
        eff     = t10 / rand_top10 if rand_top10 > 0 else float("inf")
        vs_rand = f"+{bg - rand_best:.2f} eV" if bg >= rand_best else f"{bg - rand_best:.2f} eV"
        table_data.append([f"β={b}", f"{bg:.2f}", vs_rand,
                            str(t10), f"{eff:.1f}×", str(t50)])

    # Add random row at the end
    table_data.append(["Random (baseline)", f"{rand_best:.2f}", "—",
                        str(rand_top10), "1.0×", str(rand_top50)])

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # Colour the header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1E3A5F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour each beta row to match its line colour, last row is random
    for i, b in enumerate(sorted_betas):
        hex_col = beta_files[b][1]
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(hex_col + "30")  # 30 = ~19% opacity

    # Random row
    for j in range(len(col_labels)):
        tbl[len(sorted_betas) + 1, j].set_facecolor("#DC262620")

    save(fig, "12_beta_sweep_summary_table.png")

    print("Beta comparison plots 09–12 saved.")

else:
    if len(beta_files) == 1:
        print(f"\nOnly 1 beta file found ({list(beta_files.keys())[0]}).")
    else:
        print("\nNo beta sweep files found.")
    print("Run experiments with different beta values to generate plots 09–12:")
    print("  python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.beta=0.0")
    print("  python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.beta=0.5")
    print("  python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.beta=1.0")
    print("  python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.beta=2.0")

print()
print("Files:")
for p in sorted(PLOTS_DIR.iterdir()):
    print(f"  {p.name}")
