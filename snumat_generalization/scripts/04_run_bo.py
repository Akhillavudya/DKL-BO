"""Phase 3 (SNUMAT side-test) — the BO contest: Std-GP vs DKL vs Random.

Two search tasks (band gap only — SNUMAT has no effective mass), three methods, many seeds.
Every method hunts the SAME held-out pool through the SAME EI acquisition and the SAME ExactGP,
with identical per-seed init sets — so the only difference is how each material is described:

  std_gp : handcrafted descriptors        (data/cache/descriptors.parquet)
  dkl    : 32-d pre-trained embeddings     (results/embeddings_gap.parquet)
  random : no features (the floor everyone must beat)

Tasks:
  gap_max   maximize HSE band gap
  gap_min   minimize HSE band gap

Per-run CSVs go to results/runs/ (idempotent — rerun resumes). A mean-over-seeds summary is
printed and saved to results/bo_summary.csv.

Usage
    python snumat_generalization/scripts/04_run_bo.py                  # 10 seeds, 100 cycles
    python snumat_generalization/scripts/04_run_bo.py --seeds 1 --cycles 5   # quick smoke test
"""

import argparse
import logging
import sys
from pathlib import Path

SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))

import pandas as pd
from omegaconf import OmegaConf

from lib import config
from dklbo.baselines.feature_bo_loop import FeatureBOLoop
from dklbo.bo.baselines import RandomBaseline
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUNS = config.RESULTS_DIR / "runs"

# task -> direction. DKL uses the single gap encoder (embeddings_gap.parquet).
TASKS = {"gap_max": "max", "gap_min": "min"}


def features(path: Path, pool: pd.DataFrame):
    d = pd.read_parquet(path).set_index("uid").loc[pool.uid]
    return d.index.tolist(), d.to_numpy("float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--cycles", type=int, default=config.N_CYCLES)
    ap.add_argument("--n_init", type=int, default=config.N_INIT)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    cfg = OmegaConf.create({"acquisition": "ei", "beta": 0.0, "xi": 0.01,
                            "n_init": args.n_init, "n_cycles": args.cycles})

    df = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    pool_base = df[df.split == "pool"].reset_index(drop=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pool={len(pool_base)} | seeds={args.seeds} cycles={args.cycles} "
                f"init={args.n_init} | acquisition=EI")

    feat_paths = {
        "std_gp": config.CACHE_DIR / "descriptors.parquet",
        "dkl": config.RESULTS_DIR / "embeddings_gap.parquet",
    }

    n_done = n_skip = 0
    for task, direction in TASKS.items():
        pool = pool_base.copy()
        pool["target"] = pool["gap"].astype("float64").to_numpy()
        loaded = {m: features(p, pool) for m, p in feat_paths.items()}

        for method in ["std_gp", "dkl", "random"]:
            for seed in seeds:
                out = RUNS / f"{task}__{method}__seed{seed}.csv"
                if out.exists():
                    n_skip += 1
                    continue
                seed_everything(seed)
                if method == "random":
                    res = RandomBaseline(meta_df=pool, cfg=cfg, seed=seed,
                                         direction=direction).run()
                else:
                    uids, X = loaded[method]
                    res = FeatureBOLoop(X=X, uids=uids, meta_df=pool, cfg=cfg, seed=seed,
                                        direction=direction, ard=True, gp_epochs=100).run()
                res.insert(0, "task", task)
                res.insert(1, "method", method)
                res.insert(2, "seed", seed)
                res.to_csv(out, index=False)
                n_done += 1
        logger.info(f"  [{task}] done")

    logger.info(f"Runs written={n_done} skipped={n_skip}. Summarizing…")
    summarize()


def summarize() -> None:
    """Mean over seeds of final best_so_far, top50, top10pct, per task & method."""
    files = sorted(RUNS.glob("*.csv"))
    finals = []
    for f in files:
        d = pd.read_csv(f)
        last = d.iloc[-1]
        finals.append({"task": last.task, "method": last.method, "seed": last.seed,
                       "best": last.best_so_far, "top50": last.cumul_top50,
                       "top10pct": last.cumul_top10pct})
    fin = pd.DataFrame(finals)
    summary = (fin.groupby(["task", "method"])[["best", "top50", "top10pct"]]
               .mean().round(3).reset_index())
    summary.to_csv(config.RESULTS_DIR / "bo_summary.csv", index=False)

    order = {"std_gp": 0, "dkl": 1, "random": 2}
    print("\n" + "=" * 64)
    print("SNUMAT PHASE 3 — BO contest (mean over seeds). best = best value found")
    print("=" * 64)
    for task in TASKS:
        sub = summary[summary.task == task].copy()
        sub["o"] = sub.method.map(order)
        sub = sub.sort_values("o")
        print(f"\n  {task}")
        print(f"    {'method':8s} | {'best':>8s} {'top50':>7s} {'top10%':>7s}")
        print("    " + "-" * 36)
        for _, r in sub.iterrows():
            print(f"    {r.method:8s} | {r.best:8.3f} {r.top50:7.1f} {r.top10pct:7.1f}")
    print(f"\nSaved → {config.RESULTS_DIR/'bo_summary.csv'}")


if __name__ == "__main__":
    main()
