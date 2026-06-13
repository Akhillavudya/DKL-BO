"""Phase 4 — plots in the same style as Plan 1's Phase-3 figures (04_plot_results.py).

Clean line + bar charts comparing the four methods on band-gap maximization. No
statistical-test graphs. Seed-averaged lines with a light ±std band.

Outputs (results/phase4/plots/)
  best_gap_over_cycles.png   cumulative_top10pct.png
  cumulative_top50.png       final_results_bars.png
Plus results/phase4/summary.csv (per-method final numbers).

Usage
-----
    python scripts/13_plot_phase4.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PH4 = REPO / "results" / "phase4"
RUNS = PH4 / "runs"
PLOTS = PH4 / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Method display order, labels, colours (matches 04_plot_results.py palette).
METHODS = ["dkl_pretrained", "std_gp", "dkl_cold", "random"]
LABELS = {"dkl_pretrained": "DKL (pre-trained)", "std_gp": "Standard GP",
          "dkl_cold": "DKL (cold / untrained)", "random": "Random"}
COLORS = {"dkl_pretrained": "#2563EB", "std_gp": "#16A34A",
          "dkl_cold": "#F59E0B", "random": "#9CA3AF"}


def save(fig, name):
    p = PLOTS / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")


def load():
    frames = [pd.read_csv(c) for c in sorted(RUNS.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No runs in {RUNS}. Run 12_run_phase4_bo.py first.")
    return pd.concat(frames, ignore_index=True)


def stack(df, method, col):
    """[n_seeds, n_cycles] for one method, plus the cycle axis."""
    sub = df[df.method == method]
    piv = sub.pivot_table(index="seed", columns="cycle", values=col)
    return piv.to_numpy(dtype=float), piv.columns.to_numpy()


def line_plot(df, col, ylabel, title, fname, optimum=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in METHODS:
        if m not in df.method.unique():
            continue
        Y, cyc = stack(df, m, col)
        mean, std = Y.mean(0), Y.std(0)
        ax.plot(cyc, mean, color=COLORS[m], lw=2.2, label=LABELS[m])
        ax.fill_between(cyc, mean - std, mean + std, color=COLORS[m], alpha=0.12)
    if optimum is not None:
        ax.axhline(optimum, ls="--", color="k", alpha=0.55,
                   label=f"pool optimum ({optimum:.2f} eV)")
    ax.set_xlabel("BO cycle", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, fname)


def bar_plot(df):
    finals = (df.sort_values("cycle").groupby(["method", "seed"]).tail(1))
    agg = finals.groupby("method").agg(
        best=("best_so_far", "mean"),
        top50=("cumul_top50", "mean"),
        top10=("cumul_top10pct", "mean"),
    ).reindex([m for m in METHODS if m in finals.method.unique()])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("best", "Best band gap found (eV)", "Best material found"),
              ("top50", "count", "Top-50 rare materials found"),
              ("top10", "count", "Top-10% materials found")]
    for ax, (col, ylab, title) in zip(axes, panels):
        colors = [COLORS[m] for m in agg.index]
        ax.bar([LABELS[m] for m in agg.index], agg[col], color=colors)
        ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(agg[col]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Phase 4 — final results (mean over 10 seeds)", fontsize=15, fontweight="bold")
    save(fig, "final_results_bars.png")
    return agg


def main():
    df = load()
    pool = pd.read_parquet(REPO / "data" / "cache" / "metadata_phase4.parquet")
    optimum = float(pool[pool.split4 == "pool"].target.max())

    print("Generating Phase-4 plots…")
    line_plot(df, "best_so_far", "Best band gap found (eV)",
              "Phase 4 — best band gap found vs BO cycle",
              "best_gap_over_cycles.png", optimum=optimum)
    line_plot(df, "cumul_top10pct", "Cumulative top-10% hits",
              "Phase 4 — top-10% materials discovered",
              "cumulative_top10pct.png")
    line_plot(df, "cumul_top50", "Cumulative top-50 hits",
              "Phase 4 — top-50 rare materials discovered",
              "cumulative_top50.png")
    agg = bar_plot(df)

    agg.to_csv(PH4 / "summary.csv")
    print(f"\nSummary (mean over seeds):\n{agg.round(2)}")
    print(f"\nDone. Figures in {PLOTS}")


if __name__ == "__main__":
    main()
