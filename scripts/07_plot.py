"""Phase 5 (clean rebuild) — figures for all 4 tasks x 5 methods.

For each task produces:
  {task}_curves.png   best-so-far vs cycle  +  cumulative top-10% vs cycle
                      (seed-mean lines with light 95% CI bands; pool optimum dashed)
  {task}_bars.png     final best / top-50 / top-10% bars (mean over seeds)

Reads the same three run folders as scripts/06_stats.py. Outputs → results/rebuild/plots/.

Usage
    python scripts/07_plot.py
    python scripts/07_plot.py --outdir results/rebuild/plots_30seed
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE_DIR = REPO / "data" / "cache"
OUT = REPO / "results" / "rebuild"
RUN_DIRS = [OUT / "runs", OUT / "runs_finetune", OUT / "runs_coldlive"]
PLOTS = OUT / "plots"

TASKS = {
    "gap_max":   ("gap",   "none",  "max"),
    "gap_min":   ("gap",   "none",  "min"),
    "emass_min": ("emass", "log10", "min"),
    "emass_max": ("emass", "log10", "max"),
}
METHODS = ["dkl_frozen", "dkl_finetune", "dkl_cold_live", "std_gp", "random"]
LABELS = {"dkl_frozen": "DKL (pre-trained, frozen)",
          "dkl_finetune": "DKL (fine-tuned, live)",
          "dkl_cold_live": "DKL (cold, live)",
          "std_gp": "Standard GP", "random": "Random"}
COLORS = {"dkl_frozen": "#2563EB", "dkl_finetune": "#1E3A8A",
          "dkl_cold_live": "#D97706", "std_gp": "#16A34A", "random": "#9CA3AF"}
UNITS = {"gap": "gap (eV)", "emass": "log10(emass)"}


def load():
    frames = []
    for d in RUN_DIRS:
        frames += [pd.read_csv(c) for c in sorted(d.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError("No run CSVs. Run scripts 04 & 05 first.")
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["method"].replace({"dkl": "dkl_frozen"})
    return df


def task_optimum(master, col, transform, direction):
    s = master[master.split == "pool"][col].astype("float64")
    s = np.log10(s) if transform == "log10" else s
    return float(s.max() if direction == "max" else s.min())


def stack(df, method, col):
    piv = df[df.method == method].pivot_table(index="seed", columns="cycle", values=col)
    return piv.to_numpy(float), piv.columns.to_numpy()


def curves(df, task, optimum, ylab):
    sub = df[df.task == task]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for col, ax, title, show_opt in [
        ("best_so_far", axes[0], f"{task}: best found vs cycle", True),
        ("cumul_top10pct", axes[1], f"{task}: cumulative top-10% found", False),
    ]:
        for m in METHODS:
            if m not in sub.method.unique():
                continue
            Y, cyc = stack(sub, m, col)
            mean = Y.mean(0)
            n = max(Y.shape[0], 1)
            ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ax.plot(cyc, mean, color=COLORS[m], lw=2.2, label=LABELS[m])
            ax.fill_between(cyc, mean - ci, mean + ci, color=COLORS[m], alpha=0.13)
        if show_opt:
            ax.axhline(optimum, ls="--", color="k", alpha=0.5,
                       label=f"pool optimum ({optimum:.2f})")
            ax.set_ylabel(f"best {ylab}")
        else:
            ax.set_ylabel("cumulative top-10% hits")
        ax.set_xlabel("BO cycle")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{task}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def bars(df, task):
    sub = df[df.task == task]
    finals = sub.sort_values("cycle").groupby(["method", "seed"]).tail(1)
    agg = finals.groupby("method").agg(best=("best_so_far", "mean"),
                                       top50=("cumul_top50", "mean"),
                                       top10=("cumul_top10pct", "mean"))
    agg = agg.reindex([m for m in METHODS if m in agg.index])
    n = int(finals.groupby("method").size().max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, title) in zip(axes, [("best", "Best found"),
                                       ("top50", "Top-50 found"),
                                       ("top10", "Top-10% found")]):
        ax.bar([LABELS[m] for m in agg.index], agg[col],
               color=[COLORS[m] for m in agg.index])
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(agg[col]):
            ax.text(i, v, f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    fig.suptitle(f"{task} — final results (mean over {n} seeds)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS / f"{task}_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    global PLOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=PLOTS,
                    help="where to write figures (default results/rebuild/plots)")
    args = ap.parse_args()
    PLOTS = args.outdir
    PLOTS.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(CACHE_DIR / "master.parquet")
    df = load()
    for task, (col, transform, direction) in TASKS.items():
        opt = task_optimum(master, col, transform, direction)
        curves(df, task, opt, UNITS[col])
        bars(df, task)
        print(f"  {task}: saved {task}_curves.png + {task}_bars.png")
    print(f"\nAll figures → {PLOTS}")


if __name__ == "__main__":
    main()
