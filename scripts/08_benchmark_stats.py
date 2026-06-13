"""Aggregate benchmark runs and run the paired statistical comparison.

Reads every results/benchmark/runs/{task}_{method}_seed*.csv produced by
07_run_benchmark.py and produces:

  results/benchmark/per_run_metrics.csv   — one row per (task, method, seed)
  results/benchmark/summary_<task>.csv     — per-method aggregates (mean ± 95% CI)
  results/benchmark/stats_<task>.csv       — paired DKL-vs-StdGP tests
  results/benchmark/tables/benchmark_summary.csv / .tex  — publication tables
  results/benchmark/offline_surrogate.csv  — descriptor-GP accuracy + calibration
                                             (+ DKL gap metrics from script 02 if present)

Primary sample-efficiency metric: AUC of the simple-regret curve (lower = better).
Because both methods share the same per-(task,seed) init set, DKL and Std-GP are a
PAIRED sample → we use the Wilcoxon signed-rank test.

Usage
-----
    python scripts/08_benchmark_stats.py --db_path data/raw/c2db.db
"""

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.eval.metrics_calibration import compute_calibration_metrics
from dklbo.models.surrogate import ExactGPSurrogate

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PAIR = ("dkl", "std_gp")   # the comparison of interest


def _bootstrap_ci(vals, n_boot=10000, seed=0):
    """Percentile 95% CI of the mean via bootstrap."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _per_run_metrics(runs_dir: Path, meta_by_task: dict) -> pd.DataFrame:
    """One row per (task, method, seed) with the scalar BO metrics."""
    rows = []
    for csv in sorted(runs_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        task = df["task"].iloc[0]
        method = df["method"].iloc[0]
        seed = int(df["seed"].iloc[0])
        sign = 1.0 if df["direction"].iloc[0] == "max" else -1.0

        targets = meta_by_task[task]["target"].to_numpy()
        opt_internal = float(np.max(sign * targets))

        best_internal = sign * df["best_so_far"].to_numpy()
        regret = opt_internal - best_internal            # >= 0, monotic non-increasing
        rows.append({
            "task": task, "method": method, "seed": seed,
            "direction": df["direction"].iloc[0],
            "final_best": float(df["best_so_far"].iloc[-1]),
            "regret_auc": float(np.mean(regret)),        # area under regret curve (mean)
            "simple_regret_final": float(regret[-1]),
            "final_top50": int(df["cumul_top50"].iloc[-1]),
            "final_top10pct": int(df["cumul_top10pct"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def _summaries(per_run: pd.DataFrame, out_dir: Path):
    """Per-(task, method) aggregates with 95% bootstrap CIs."""
    metrics = ["final_best", "regret_auc", "simple_regret_final",
               "final_top50", "final_top10pct"]
    all_rows = []
    for task, sub in per_run.groupby("task"):
        rows = []
        for method, g in sub.groupby("method"):
            rec = {"task": task, "method": method, "n_seeds": len(g)}
            for m in metrics:
                lo, hi = _bootstrap_ci(g[m].to_numpy())
                rec[f"{m}_mean"] = float(g[m].mean())
                rec[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
                rec[f"{m}_ci_lo"] = lo
                rec[f"{m}_ci_hi"] = hi
            rows.append(rec)
        summ = pd.DataFrame(rows)
        summ.to_csv(out_dir / f"summary_{task}.csv", index=False)
        all_rows.append(summ)
    return pd.concat(all_rows, ignore_index=True)


def _paired_stats(per_run: pd.DataFrame, out_dir: Path):
    """Wilcoxon signed-rank DKL vs Std-GP per task, on regret_auc and final_best."""
    a, b = PAIR
    stats_rows = []
    for task, sub in per_run.groupby("task"):
        pa = sub[sub.method == a].set_index("seed")
        pb = sub[sub.method == b].set_index("seed")
        seeds = sorted(set(pa.index) & set(pb.index))
        if len(seeds) < 2:
            continue
        for metric, lower_is_better in [("regret_auc", True), ("final_best", None)]:
            x = pa.loc[seeds, metric].to_numpy()   # dkl
            y = pb.loc[seeds, metric].to_numpy()   # std_gp
            diff = x - y
            # Wilcoxon (skip if all differences are zero)
            if np.allclose(diff, 0):
                pval, rbc = float("nan"), 0.0
            else:
                try:
                    stat, pval = wilcoxon(x, y)
                    pval = float(pval)
                except ValueError:
                    pval = float("nan")
                # matched-pairs rank-biserial effect size
                nz = diff[diff != 0]
                ranks = pd.Series(np.abs(nz)).rank().to_numpy()
                w_pos = ranks[nz > 0].sum()
                w_neg = ranks[nz < 0].sum()
                rbc = float((w_pos - w_neg) / (w_pos + w_neg)) if (w_pos + w_neg) else 0.0

            if lower_is_better is True:
                dkl_better = float(np.mean(x < y))     # lower regret wins
            else:                                       # final_best: direction-aware
                sign = 1.0 if sub["direction"].iloc[0] == "max" else -1.0
                dkl_better = float(np.mean(sign * x > sign * y))

            stats_rows.append({
                "task": task, "metric": metric, "n_seeds": len(seeds),
                "dkl_mean": float(x.mean()), "std_gp_mean": float(y.mean()),
                "mean_diff_dkl_minus_stdgp": float(diff.mean()),
                "wilcoxon_p": pval, "rank_biserial": rbc,
                "dkl_win_rate": dkl_better,
            })
    stats = pd.DataFrame(stats_rows)
    if stats.empty:
        logger.warning("Paired stats need ≥2 seeds with both methods — skipping "
                       "(run more seeds via 07_run_benchmark.py).")
        return stats
    for task, sub in stats.groupby("task"):
        sub.to_csv(out_dir / f"stats_{task}.csv", index=False)
    stats.to_csv(out_dir / "stats_all.csv", index=False)
    return stats


def _offline_surrogate(bench_dir: Path, cache_dir: Path, db_path: str) -> pd.DataFrame:
    """Descriptor-GP accuracy + calibration on the held-out test split (both targets)."""
    rows = []
    for tag, meta_file in [("gap", "metadata.parquet"), ("emass", "metadata_emass.parquet")]:
        desc_path = bench_dir / f"descriptors_{tag}.parquet"
        meta_path = cache_dir / meta_file
        if not desc_path.exists() or not meta_path.exists():
            continue
        meta = pd.read_parquet(meta_path)
        desc = pd.read_parquet(desc_path).set_index("uid").loc[meta["uid"]]
        X = desc.to_numpy(dtype="float64")
        y = meta["target"].to_numpy(dtype="float64")
        split = meta["split"].to_numpy()

        tr, te = split == "train", split == "test"
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-9] = 1.0
        Xtr = torch.tensor((X[tr] - mu) / sd); Xte = torch.tensor((X[te] - mu) / sd)
        # Standardize the target on the train split (un-standardize before metrics).
        ym, ys = float(y[tr].mean()), float(y[tr].std()) or 1.0
        ytr = torch.tensor((y[tr] - ym) / ys)
        yte = torch.tensor(y[te])

        gp = ExactGPSurrogate(n_epochs=150, ard=True)
        gp.fit(Xtr, ytr)
        gp.eval_mode()
        mean_s, std_s = gp.predict(Xte)
        mean = mean_s * ys + ym          # back to original units
        std = std_s * ys

        acc = compute_accuracy_metrics(yte.float(), mean.float())
        cal = compute_calibration_metrics(yte.float(), mean.float(), std.float())
        rows.append({
            "target": tag, "method": "std_gp", "n_test": int(te.sum()),
            "mae": acc.mae, "rmse": acc.rmse, "r2": acc.r2,
            "nll": cal.nll, "ence": cal.ence, "coverage_95": cal.coverage_95,
            "spearman_rho": cal.spearman_rho,
        })
        logger.info(f"[offline] std_gp/{tag}: MAE={acc.mae:.3f} R2={acc.r2:.3f} "
                    f"cov95={cal.coverage_95:.3f}")

    # Pull DKL gap metrics from script 02 output if present (band gap only).
    dkl_csv = REPO / "results" / "surrogate_metrics_exactgpsurrogate.csv"
    if dkl_csv.exists():
        m = pd.read_csv(dkl_csv)
        test = m[m.get("split", "") == "test"]
        if not test.empty:
            r = test.iloc[0]
            rows.append({
                "target": "gap", "method": "dkl", "n_test": int(r.get("n", -1)),
                "mae": float(r.get("mae", float("nan"))),
                "rmse": float(r.get("rmse", float("nan"))),
                "r2": float(r.get("r2", float("nan"))),
                "nll": float(r.get("nll", float("nan"))),
                "ence": float(r.get("ence", float("nan"))),
                "coverage_95": float(r.get("coverage_95", float("nan"))),
                "spearman_rho": float(r.get("spearman_rho", float("nan"))),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", default="data/raw/c2db.db")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    bench_dir = REPO / args.results_dir / "benchmark"
    runs_dir = bench_dir / "runs"
    cache_dir = REPO / "data" / "cache"
    tables_dir = bench_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not any(runs_dir.glob("*.csv")):
        raise FileNotFoundError(f"No runs in {runs_dir}. Run 07_run_benchmark.py first.")

    # Metadata per task (for the optimum used in regret).
    meta_files = {"metadata.parquet": None, "metadata_emass.parquet": None}
    for f in meta_files:
        p = cache_dir / f
        if p.exists():
            meta_files[f] = pd.read_parquet(p)
    task_meta_map = {
        "gap_max": meta_files["metadata.parquet"],
        "gap_min": meta_files["metadata.parquet"],
        "emass_min": meta_files["metadata_emass.parquet"],
        "emass_max": meta_files["metadata_emass.parquet"],
    }

    per_run = _per_run_metrics(runs_dir, task_meta_map)
    per_run.to_csv(bench_dir / "per_run_metrics.csv", index=False)
    logger.info(f"Per-run metrics: {len(per_run)} rows → per_run_metrics.csv")

    summary = _summaries(per_run, bench_dir)
    stats = _paired_stats(per_run, bench_dir)

    # Publication summary table (compact: mean ± std per task/method).
    pub = []
    for (task, method), g in per_run.groupby(["task", "method"]):
        pub.append({
            "task": task, "method": method,
            "final_best": f"{g.final_best.mean():.3f} ± {g.final_best.std(ddof=1):.3f}",
            "regret_auc": f"{g.regret_auc.mean():.3f} ± {g.regret_auc.std(ddof=1):.3f}",
            "top50": f"{g.final_top50.mean():.1f}",
            "top10pct": f"{g.final_top10pct.mean():.1f}",
        })
    pub_df = pd.DataFrame(pub).sort_values(["task", "method"])
    pub_df.to_csv(tables_dir / "benchmark_summary.csv", index=False)
    (tables_dir / "benchmark_summary.tex").write_text(pub_df.to_latex(index=False))

    offline = _offline_surrogate(bench_dir, cache_dir, args.db_path)
    if not offline.empty:
        offline.to_csv(bench_dir / "offline_surrogate.csv", index=False)

    # Console digest
    logger.info("\n==== PAIRED DKL vs Std-GP (regret_auc, lower=better) ====")
    digest = stats[stats.metric == "regret_auc"] if not stats.empty else pd.DataFrame()
    for _, r in digest.iterrows():
        verdict = "DKL better" if r.mean_diff_dkl_minus_stdgp < 0 else "Std-GP better"
        logger.info(
            f"  {r.task:10s}  dkl={r.dkl_mean:.3f}  stdgp={r.std_gp_mean:.3f}  "
            f"p={r.wilcoxon_p:.4f}  win_rate={r.dkl_win_rate:.2f}  → {verdict}"
        )
    logger.info(f"\nTables → {tables_dir}/benchmark_summary.csv")


if __name__ == "__main__":
    main()
