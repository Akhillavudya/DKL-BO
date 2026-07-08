"""Phase 5 (clean rebuild) — statistics: which BO differences are REAL?

Turns the means from Phases 3-4 into journal-grade numbers: per-method mean ± 95%
bootstrap CI, and paired Wilcoxon signed-rank tests (vs Std-GP) per task. The pairing is
valid because every method shares the same per-seed init set (same seed → same
random.sample), so the methods are matched samples across seeds.

Reads all runs from the three folders (Phase-3 frozen + Phase-4 finetune + cold-live):
  results/rebuild/runs/            std_gp, dkl(→dkl_frozen), random
  results/rebuild/runs_finetune/   dkl_finetune
  results/rebuild/runs_coldlive/   dkl_cold_live

Metrics per run (per task; emass uses log10 target):
  regret_auc      mean simple-regret over cycles   (lower = better)
  final_best      best value found
  final_top50     cumulative top-50 hits           (higher = better)
  final_top10pct  cumulative top-10% hits          (higher = better)

Outputs (results/rebuild/)
  per_run_metrics.csv   one row per (task, method, seed)
  summary_stats.csv     per (task, method) mean ± 95% CI
  stats_pairs.csv       paired Wilcoxon vs Std-GP, per task & metric

Usage
    python scripts/06_stats.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CACHE_DIR = REPO / "data" / "cache"
OUT = REPO / "results" / "rebuild"
RUN_DIRS = [OUT / "runs", OUT / "runs_finetune", OUT / "runs_coldlive"]

# task -> (target column, transform, direction)
TASKS = {
    "gap_max":   ("gap",   "none",  "max"),
    "gap_min":   ("gap",   "none",  "min"),
    "emass_min": ("emass", "log10", "min"),
    "emass_max": ("emass", "log10", "max"),
}
METHODS = ["std_gp", "dkl_frozen", "dkl_finetune", "dkl_cold_live", "random"]
METRICS = ["regret_auc", "final_best", "final_top50", "final_top10pct"]


def task_optimum(df_master, col, transform, direction):
    pool = df_master[df_master.split == "pool"]
    s = pool[col].astype("float64")
    s = np.log10(s) if transform == "log10" else s
    return float(s.max() if direction == "max" else s.min())


def bootstrap_ci(vals, n_boot=10000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_runs():
    frames = []
    for d in RUN_DIRS:
        for csv in sorted(d.glob("*.csv")):
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError("No run CSVs found. Run scripts 04 & 05 first.")
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["method"].replace({"dkl": "dkl_frozen"})
    return df


def per_run_metrics(df, optima):
    rows = []
    for (task, method, seed), g in df.groupby(["task", "method", "seed"]):
        g = g.sort_values("cycle")
        direction = TASKS[task][2]
        opt = optima[task]
        best = g["best_so_far"].to_numpy()
        regret = (opt - best) if direction == "max" else (best - opt)   # >= 0
        rows.append({
            "task": task, "method": method, "seed": int(seed),
            "regret_auc": float(np.mean(regret)),
            "final_best": float(best[-1]),
            "final_top50": int(g["cumul_top50"].iloc[-1]),
            "final_top10pct": int(g["cumul_top10pct"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def summary(per_run):
    rows = []
    for (task, method), g in per_run.groupby(["task", "method"]):
        rec = {"task": task, "method": method, "n_seeds": len(g)}
        for m in METRICS:
            lo, hi = bootstrap_ci(g[m].to_numpy())
            rec[f"{m}_mean"] = float(g[m].mean())
            rec[f"{m}_ci_lo"], rec[f"{m}_ci_hi"] = lo, hi
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_vs_stdgp(per_run):
    rows = []
    for task in TASKS:
        base = per_run[(per_run.task == task) & (per_run.method == "std_gp")].set_index("seed")
        for method in [m for m in METHODS if m != "std_gp"]:
            cur = per_run[(per_run.task == task) & (per_run.method == method)].set_index("seed")
            seeds = sorted(set(base.index) & set(cur.index))
            if len(seeds) < 2:
                continue
            for metric in METRICS:
                x = cur.loc[seeds, metric].to_numpy(float)    # method
                y = base.loc[seeds, metric].to_numpy(float)   # std_gp
                diff = x - y
                if np.allclose(diff, 0):
                    pval = float("nan")
                else:
                    try:
                        pval = float(wilcoxon(x, y).pvalue)
                    except ValueError:
                        pval = float("nan")
                better_lower = metric == "regret_auc"
                win = float(np.mean(x < y) if better_lower else np.mean(x > y))
                rows.append({
                    "task": task, "method": method, "metric": metric,
                    "n_seeds": len(seeds), "method_mean": float(x.mean()),
                    "std_gp_mean": float(y.mean()), "mean_diff": float(diff.mean()),
                    "wilcoxon_p": pval, "method_win_rate": win,
                })
    return pd.DataFrame(rows)


def main():
    master = pd.read_parquet(CACHE_DIR / "master.parquet")
    optima = {t: task_optimum(master, *TASKS[t]) for t in TASKS}

    df = load_runs()
    per_run = per_run_metrics(df, optima)
    per_run.to_csv(OUT / "per_run_metrics.csv", index=False)
    summ = summary(per_run)
    summ.to_csv(OUT / "summary_stats.csv", index=False)
    pairs = paired_vs_stdgp(per_run)
    pairs.to_csv(OUT / "stats_pairs.csv", index=False)

    print("=" * 72)
    print("PHASE 5 STATS — paired Wilcoxon vs Std-GP (★ = p<0.05).")
    print("regret-AUC: lower better | top-10%: higher better. Pool optima:")
    print("  " + "  ".join(f"{t}={optima[t]:.2f}" for t in TASKS))
    print("=" * 72)
    for task in TASKS:
        print(f"\n  {task}")
        print(f"    {'method':14s} | {'regretAUC':>9s} {'top10%':>7s} {'top50':>6s} | vs Std-GP(top10%)")
        print("    " + "-" * 64)
        s = summ[summ.task == task].set_index("method")
        for m in [x for x in METHODS if x in s.index]:
            verdict = ""
            if m != "std_gp":
                r = pairs[(pairs.task == task) & (pairs.method == m) &
                          (pairs.metric == "final_top10pct")]
                if not r.empty:
                    p = r.wilcoxon_p.iloc[0]
                    md = r.mean_diff.iloc[0]
                    star = "★" if (p < 0.05) else " "
                    side = "wins" if md > 0 else "loses"
                    verdict = f"{star} {side} p={p:.3f}"
            print(f"    {m:14s} | {s.loc[m,'regret_auc_mean']:9.3f} "
                  f"{s.loc[m,'final_top10pct_mean']:7.1f} "
                  f"{s.loc[m,'final_top50_mean']:6.1f} | {verdict}")
    print(f"\nSaved → per_run_metrics.csv, summary_stats.csv, stats_pairs.csv  (in {OUT})")


if __name__ == "__main__":
    main()
