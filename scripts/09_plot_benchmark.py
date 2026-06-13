"""Publication figures for the GP-BO vs DKL-BO benchmark.

Reads results/benchmark/runs/*.csv (+ per_run_metrics.csv, offline_surrogate.csv)
and writes figures to results/benchmark/plots/:

  <task>_best_found.png    — best target vs iteration (mean ± 95% CI over seeds)
  <task>_regret.png        — simple-regret curve (log-y)
  <task>_topk.png          — cumulative top-50 / top-10% vs iteration
  summary_regret_auc.png   — cross-task regret-AUC bars (mean ± std, all methods)
  offline_surrogate.png    — descriptor-GP (and DKL) accuracy/calibration, if present

Usage
-----
    python scripts/09_plot_benchmark.py
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

BENCH = REPO / "results" / "benchmark"
RUNS = BENCH / "runs"
PLOTS = BENCH / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

COLORS = {"dkl": "#2563EB", "std_gp": "#16A34A", "random": "#9CA3AF"}
LABELS = {"dkl": "DKL-BO", "std_gp": "Std GP-BO", "random": "Random"}
META = {
    "metadata.parquet": REPO / "data/cache/metadata.parquet",
    "metadata_emass.parquet": REPO / "data/cache/metadata_emass.parquet",
}
TASK_META = {
    "gap_max": "metadata.parquet", "gap_min": "metadata.parquet",
    "emass_min": "metadata_emass.parquet", "emass_max": "metadata_emass.parquet",
}
TASK_UNIT = {"gap_max": "eV", "gap_min": "eV", "emass_min": "m*", "emass_max": "m*"}


def save(fig, name):
    p = PLOTS / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")


def _mean_ci(arr):
    """Mean and 95% CI half-width (1.96·SEM) over axis 0."""
    m = arr.mean(0)
    n = arr.shape[0]
    sem = arr.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
    return m, 1.96 * sem


def _stack(df, col):
    """[n_seeds, n_cycles] matrix for one (task, method), aligned by cycle."""
    piv = df.pivot_table(index="seed", columns="cycle", values=col)
    return piv.to_numpy(dtype=float), piv.columns.to_numpy()


def load_runs():
    frames = [pd.read_csv(c) for c in sorted(RUNS.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No runs in {RUNS}. Run 07_run_benchmark.py first.")
    return pd.concat(frames, ignore_index=True)


def plot_task(all_df, task, optimum, sign):
    sub = all_df[all_df.task == task]
    methods = [m for m in ("dkl", "std_gp", "random") if m in sub.method.unique()]
    unit = TASK_UNIT[task]

    # ── best-found vs iteration ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        Y, cyc = _stack(sub[sub.method == m], "best_so_far")
        mean, ci = _mean_ci(Y)
        ax.plot(cyc, mean, color=COLORS[m], lw=2, label=LABELS[m])
        ax.fill_between(cyc, mean - ci, mean + ci, color=COLORS[m], alpha=0.18)
    ax.axhline(optimum, ls="--", color="k", alpha=0.5, label=f"dataset optimum ({optimum:.2f})")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel(f"Best found ({unit})")
    ax.set_title(f"{task} — best-found vs iteration (mean ± 95% CI, {sub.seed.nunique()} seeds)",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, f"{task}_best_found.png")

    # ── simple regret (log-y) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        Y, cyc = _stack(sub[sub.method == m], "best_so_far")
        regret = optimum * sign - (sign * Y)         # internal regret ≥ 0
        regret = np.clip(regret, 1e-6, None)
        mean, ci = _mean_ci(regret)
        ax.plot(cyc, mean, color=COLORS[m], lw=2, label=LABELS[m])
        ax.fill_between(cyc, np.clip(mean - ci, 1e-6, None), mean + ci,
                        color=COLORS[m], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("BO iteration"); ax.set_ylabel("Simple regret (log)")
    ax.set_title(f"{task} — simple regret", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    save(fig, f"{task}_regret.png")

    # ── cumulative top-k ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, ax, title in [("cumul_top50", axes[0], "Cumulative top-50"),
                           ("cumul_top10pct", axes[1], "Cumulative top-10%")]:
        for m in methods:
            Y, cyc = _stack(sub[sub.method == m], col)
            mean, ci = _mean_ci(Y)
            ax.plot(cyc, mean, color=COLORS[m], lw=2, label=LABELS[m])
            ax.fill_between(cyc, mean - ci, mean + ci, color=COLORS[m], alpha=0.18)
        ax.set_xlabel("BO iteration"); ax.set_ylabel("count")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"{task} — rare-material discovery", fontweight="bold")
    save(fig, f"{task}_topk.png")


def plot_summary(per_run):
    tasks = sorted(per_run.task.unique())
    methods = [m for m in ("dkl", "std_gp", "random") if m in per_run.method.unique()]
    x = np.arange(len(tasks)); w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(methods):
        means, errs = [], []
        for t in tasks:
            g = per_run[(per_run.task == t) & (per_run.method == m)].regret_auc
            means.append(g.mean())
            errs.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        ax.bar(x + i * w, means, w, yerr=errs, capsize=3,
               color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Regret-AUC (lower = better)")
    ax.set_title("Sample efficiency across tasks (mean ± std over seeds)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    save(fig, "summary_regret_auc.png")


def plot_offline():
    p = BENCH / "offline_surrogate.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df["key"] = df["method"] + "/" + df["target"]
    for ax, metric, title in [("r2", "R²", "Accuracy (R², higher=better)"),
                              ("mae", "MAE", "MAE (lower=better)"),
                              ("coverage_95", "Coverage@95", "Calibration (→0.95)")]:
        a = axes[["r2", "mae", "coverage_95"].index(ax)]
        a.bar(df["key"], df[ax], color="#2563EB")
        a.set_title(title); a.set_ylabel(metric)
        a.tick_params(axis="x", rotation=30)
        if ax == "coverage_95":
            a.axhline(0.95, ls="--", color="k", alpha=0.5)
    fig.suptitle("Offline surrogate quality (held-out test split)", fontweight="bold")
    save(fig, "offline_surrogate.png")


def main():
    all_df = load_runs()
    metas = {f: pd.read_parquet(p) for f, p in META.items() if p.exists()}
    print("Generating benchmark plots…")
    for task in sorted(all_df.task.unique()):
        meta = metas.get(TASK_META[task])
        if meta is None:
            continue
        sign = 1.0 if all_df[all_df.task == task].direction.iloc[0] == "max" else -1.0
        optimum = float((sign * meta["target"]).max() * sign)  # original-unit optimum
        plot_task(all_df, task, optimum, sign)

    per_run_path = BENCH / "per_run_metrics.csv"
    if per_run_path.exists():
        plot_summary(pd.read_csv(per_run_path))
    plot_offline()
    print(f"Done. Figures in {PLOTS}")


if __name__ == "__main__":
    main()
