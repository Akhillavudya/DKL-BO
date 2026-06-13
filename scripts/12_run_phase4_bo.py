"""Phase 4 — hunt the held-out POOL with four methods (band-gap maximization).

All four methods search the SAME pool (materials none of them saw labels for), through
the SAME GP engine and acquisition, with identical per-seed init sets — so the only
difference is how each material is described:

  dkl_pretrained : 32-d embeddings from the encoder pre-trained on the train split
  dkl_cold       : 32-d embeddings from a random-init encoder (control)
  std_gp         : 43 handcrafted descriptors (reuse descriptors_gap.parquet)
  random         : no features (sanity floor)

Writes results/phase4/runs/{method}_seed{S}.csv (idempotent — resumable).

Usage
-----
    python scripts/12_run_phase4_bo.py bo=ei data.db_path=data/raw/c2db.db
    # smoke: python scripts/12_run_phase4_bo.py bo=ei bo.n_cycles=10 +phase4.seeds='[0]'
"""

import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from dklbo.baselines.feature_bo_loop import FeatureBOLoop
from dklbo.bo.baselines import RandomBaseline
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)

SEEDS = list(range(10))
# feature method -> embeddings/descriptors parquet (relative to results/)
FEATURE_FILES = {
    "dkl_pretrained": "phase4/embeddings_pretrained.parquet",
    "dkl_cold":       "phase4/embeddings_cold.parquet",
    "std_gp":         "benchmark/descriptors_gap.parquet",
}


def _features(path: Path, pool: pd.DataFrame):
    """Load a feature parquet and align rows to the pool uid order."""
    d = pd.read_parquet(path).set_index("uid").loc[pool.uid]
    return d.index.tolist(), d.to_numpy(dtype="float64")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seeds = list(cfg.get("phase4", {}).get("seeds", SEEDS))
    df = pd.read_parquet(REPO / "data" / "cache" / "metadata_phase4.parquet")
    pool = df[df.split4 == "pool"].reset_index(drop=True)
    logger.info(f"Pool: {len(pool)} materials  |  best gap {pool.target.max():.2f} eV  "
                f"|  acquisition={cfg.bo.acquisition}  cycles={cfg.bo.n_cycles}")

    runs_dir = REPO / cfg.results_dir / "phase4" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = 0
    for method in ["dkl_pretrained", "dkl_cold", "std_gp", "random"]:
        if method in FEATURE_FILES:
            uids, X = _features(REPO / cfg.results_dir / FEATURE_FILES[method], pool)
        for seed in seeds:
            out = runs_dir / f"{method}_seed{seed}.csv"
            if out.exists():
                n_skip += 1
                continue
            seed_everything(seed)
            if method == "random":
                res = RandomBaseline(meta_df=pool, cfg=cfg.bo, seed=seed,
                                     direction="max").run()
            else:
                res = FeatureBOLoop(X=X, uids=uids, meta_df=pool, cfg=cfg.bo, seed=seed,
                                    direction="max", ard=True, gp_epochs=100).run()
            res.insert(0, "method", method)
            res.insert(1, "seed", seed)
            res.to_csv(out, index=False)
            n_done += 1
            logger.info(f"  {method}/seed{seed}  → best={res.best_so_far.iloc[-1]:.3f}  "
                        f"top50={res.cumul_top50.iloc[-1]}  top10%={res.cumul_top10pct.iloc[-1]}")

    logger.info(f"Phase-4 BO complete: {n_done} runs written, {n_skip} skipped.")


if __name__ == "__main__":
    main()
