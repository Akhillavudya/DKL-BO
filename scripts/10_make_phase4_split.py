"""Phase 4 — build a gap-stratified train/pool split for the warm-start experiment.

Why a new split?
----------------
The original split puts ALL rare high-gap materials in `train` (train best=10.79 eV,
val best=6.55, test best=6.46). If we pre-train DKL on train and then "hunt" the same
pool, it is memorising answers, not discovering — and unfair to Standard GP. So we make
a fresh split where rare high-gap materials are SPREAD into a held-out `pool` that no
method ever sees the labels of.

Leakage safety
--------------
We still split by structural `prototype` (never by individual rows), so near-duplicate
structures cannot straddle the train/pool boundary. We stratify the *prototype groups*
by their maximum band gap so the rare high-gap groups are distributed across both sides.

Output
------
`data/cache/metadata_phase4.parquet` — same columns as metadata.parquet plus
`split4 ∈ {train, pool}`. Graphs are reused from the existing LMDB cache.

Usage
-----
    python scripts/10_make_phase4_split.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from dklbo.data.c2db_loader import verify_no_split_leakage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
POOL_FRAC = 0.30      # ~30% of prototype groups become the held-out hunting pool
N_STRATA = 4          # quartile bins by prototype max-gap
SEED = 42
TOP50_GAP = 7.02      # dataset top-50 threshold (gap_hse, from project notes)


def main() -> None:
    meta_path = REPO / "data" / "cache" / "metadata.parquet"
    df = pd.read_parquet(meta_path)
    logger.info(f"Loaded {len(df)} materials from {meta_path.name}")

    # One row per prototype group with its max gap (drives stratification).
    grp = df.groupby("prototype")["target"].max().reset_index()
    grp.columns = ["prototype", "max_gap"]

    # Quartile strata by max gap; high-gap prototypes get spread across train/pool.
    grp["stratum"] = pd.qcut(grp["max_gap"], q=N_STRATA, labels=False, duplicates="drop")

    rng = np.random.default_rng(SEED)
    pool_protos: set = set()
    for stratum, sub in grp.groupby("stratum"):
        protos = sub["prototype"].to_numpy()
        rng.shuffle(protos)
        n_pool = max(1, int(round(len(protos) * POOL_FRAC)))
        pool_protos.update(protos[:n_pool])

    df = df.copy()
    df["split4"] = np.where(df["prototype"].isin(pool_protos), "pool", "train")

    # Leakage check reuses the existing verifier (expects a 'split' column).
    verify_no_split_leakage(df[["prototype", "split4"]].rename(columns={"split4": "split"}))

    pool = df[df["split4"] == "pool"]
    train = df[df["split4"] == "train"]
    n_top50_pool = int((pool["target"] >= TOP50_GAP).sum())

    logger.info("=" * 56)
    logger.info(f"Phase-4 split  |  train={len(train)}  pool={len(pool)}")
    logger.info(f"  train best gap = {train['target'].max():.2f} eV")
    logger.info(f"  pool  best gap = {pool['target'].max():.2f} eV")
    logger.info(f"  top-50-class materials (gap≥{TOP50_GAP}) in pool = {n_top50_pool}")
    logger.info("=" * 56)

    if n_top50_pool == 0:
        raise AssertionError(
            "Pool contains no rare high-gap materials — the hunt would be pointless. "
            "Increase POOL_FRAC or check the stratification."
        )

    out = REPO / "data" / "cache" / "metadata_phase4.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
