"""Phase 5 (SNUMAT side-test) — statistics: which BO differences are REAL?

Turns the Phase 3-4 means into journal-grade numbers: per-method mean ± 95% bootstrap CI, and
paired Wilcoxon signed-rank tests (vs Std-GP) per task. The pairing is valid because every method
shares the same per-seed init set (same seed → same random.sample).

Reads all runs from the three folders:
  results/runs/            std_gp, dkl(→dkl_frozen), random
  results/runs_finetune/   dkl_finetune
  results/runs_coldlive/   dkl_cold_live

Metrics per run (per task): regret_auc (lower better), final_best, final_top50, final_top10pct.

Outputs (results/): per_run_metrics.csv, summary_stats.csv, stats_pairs.csv
"""

import sys
from pathlib import Path

SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lib import config

RUN_DIRS = [config.RESULTS_DIR / "runs", config.RESULTS_DIR / "runs_finetune",
            config.RESULTS_DIR / "runs_coldlive"]

# task -> direction (band gap only, no transform).
TASKS = {"gap_max": "max", "gap_min": "min"}
METHODS = ["std_gp", "dkl_frozen", "dkl_finetune", "dkl_cold_live", "random"]
METRICS = ["regret_auc", "final_best", "final_top50", "final_top10pct"]


def task_optimum(master, direction):
    s = master[master.split == "pool"]["gap"].astype("float64")
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
        direction = TASKS[task]
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
    master = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    optima = {t: task_optimum(master, TASKS[t]) for t in TASKS}

    df = load_runs()
    per_run = per_run_metrics(df, optima)
    per_run.to_csv(config.RESULTS_DIR / "per_run_metrics.csv", index=False)
    summ = summary(per_run)
    summ.to_csv(config.RESULTS_DIR / "summary_stats.csv", index=False)
    pairs = paired_vs_stdgp(per_run)
    pairs.to_csv(config.RESULTS_DIR / "stats_pairs.csv", index=False)

    print("=" * 72)
    print("SNUMAT PHASE 5 STATS — paired Wilcoxon vs Std-GP (★ = p<0.05).")
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
    print(f"\nSaved → per_run_metrics.csv, summary_stats.csv, stats_pairs.csv  (in {config.RESULTS_DIR})")


if __name__ == "__main__":
    main()
