"""Phase 1 (clean rebuild) — build ONE master dataset shared by every study.

Why this script exists
----------------------
The rebuilt project compares three methods (Random, Std-GP, DKL-BO) across three
studies (band gap, effective mass, and both combined). For the comparison to be
fair, *every* study and *every* method must search the exact same list of
materials. So we build a single "master" dataset here, once.

What "master" means
-------------------
Only materials that have BOTH a valid band gap (gap_hse) AND a valid electron
effective mass (emass_cbm). Effective mass is only meaningful for a semiconductor,
so emass materials are a subset of the gap materials — the intersection is just the
emass set. We keep both numbers as explicit columns (`gap`, `emass`).

What it produces
----------------
1. data/cache/master.parquet      — the one dataset (uid, gap, emass, prototype, split)
2. data/cache/descriptors.parquet — handcrafted features for Std-GP, same rows/order
3. (reuses) data/cache/graphs_<hash>.lmdb — crystal graphs for DKL (NOT rebuilt:
   graphs are structure-only and identical to the existing gap cache)

One shared split
----------------
A prototype-aware train/pool split (no near-duplicate structure leaks across the
boundary), gap-stratified so rare high-gap materials are spread into the held-out
pool instead of all sitting in train. `train` = what a model may learn from /
pre-train on; `pool` = the held-out hunting ground that BO searches and that
prediction accuracy is measured on.

Run
---
    python scripts/01_build_dataset.py            # uses data/raw/c2db.db
    python scripts/01_build_dataset.py --db_path /path/to/c2db.db
"""

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from ase.db import connect

from dklbo.baselines.descriptors import build_descriptors, FEATURE_NOTES
from dklbo.data.cache import GraphCache
from dklbo.data.c2db_loader import verify_no_split_leakage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
CACHE_DIR = REPO / "data" / "cache"

# --- fixed choices (documented so the dataset is reproducible) ---------------
GAP_MIN = 0.01        # semiconductor filter: drop metals (zero-gap spike)
POOL_FRAC = 0.30      # fraction of prototype GROUPS held out as the hunting pool
N_STRATA = 4          # gap quartile bins used to spread rare materials into pool
SEED = 42

# Graph preprocessing config — MUST match the existing structure-only cache so we
# can reuse it instead of rebuilding identical graphs.
GRAPH_PREPROC = {
    "radius": 8.0,
    "vacuum_cutoff": 4.0,
    "max_neighbors": 12,
    "target": "gap_hse",
    "gap_min": GAP_MIN,
}


def build_master_table(db_path: str) -> pd.DataFrame:
    """Scan C2DB once and keep materials with BOTH a valid gap and a valid emass."""
    db = connect(db_path)
    records = []
    n_no_gap = n_metal = n_no_emass = n_bad_emass = 0

    for row in db.select():
        kvp = row.key_value_pairs

        gap = kvp.get("gap_hse")
        if gap is None:
            n_no_gap += 1
            continue
        gap = float(gap)
        if gap <= GAP_MIN:                       # metal / zero-gap → drop
            n_metal += 1
            continue

        em = kvp.get("emass_cbm")
        if em is None:
            n_no_emass += 1
            continue
        em = float(em)
        if not math.isfinite(em) or em <= 0:     # inf / zero effective mass → drop
            n_bad_emass += 1
            continue

        lg = kvp.get("layergroup") or kvp.get("lgnum") or kvp.get("international") or ""
        sg = str(kvp.get("number", ""))
        prototype = f"{lg}_{sg}" if lg else (sg or row.formula)

        records.append({
            "id": row.id,
            "uid": kvp.get("uid") or str(row.id),
            "formula": row.formula,
            "prototype": str(prototype),
            "gap": gap,
            "emass": em,
            "n_atoms": row.natoms,
        })

    if not records:
        raise ValueError("No materials with both a valid gap_hse and emass_cbm found.")

    df = pd.DataFrame(records).reset_index(drop=True)
    logger.info("=" * 64)
    logger.info("Step 1 — master intersection dataset (gap AND emass)")
    logger.info("=" * 64)
    logger.info(f"  kept           : {len(df)} materials")
    logger.info(f"  dropped        : {n_no_gap} no-gap, {n_metal} metals(gap<= {GAP_MIN}), "
                f"{n_no_emass} no-emass, {n_bad_emass} inf/zero emass")
    logger.info(f"  gap   range eV : {df['gap'].min():.3f} – {df['gap'].max():.3f}")
    logger.info(f"  emass range m* : {df['emass'].min():.3f} – {df['emass'].max():.3f}")
    return df


def add_train_pool_split(df: pd.DataFrame) -> pd.DataFrame:
    """Prototype-aware, gap-stratified train/pool split (no structure leakage)."""
    grp = df.groupby("prototype")["gap"].max().reset_index()
    grp.columns = ["prototype", "max_gap"]
    grp["stratum"] = pd.qcut(grp["max_gap"], q=N_STRATA, labels=False, duplicates="drop")

    rng = np.random.default_rng(SEED)
    pool_protos: set = set()
    for _, sub in grp.groupby("stratum"):
        protos = sub["prototype"].to_numpy()
        rng.shuffle(protos)
        n_pool = max(1, int(round(len(protos) * POOL_FRAC)))
        pool_protos.update(protos[:n_pool])

    df = df.copy()
    df["split"] = np.where(df["prototype"].isin(pool_protos), "pool", "train")
    verify_no_split_leakage(df)  # asserts zero prototype overlap train<->pool

    train, pool = df[df.split == "train"], df[df.split == "pool"]
    logger.info("=" * 64)
    logger.info("Step 2 — shared train/pool split (gap-stratified, by prototype)")
    logger.info("=" * 64)
    logger.info(f"  train = {len(train)}   pool = {len(pool)}")
    logger.info(f"  pool gap   range : {pool['gap'].min():.3f} – {pool['gap'].max():.3f} eV")
    logger.info(f"  pool emass range : {pool['emass'].min():.3f} – {pool['emass'].max():.3f}")
    # Coverage of the rare tails the four search tasks aim at, inside the pool:
    for col, frac, tail in [("gap", 0.985, "high"), ("gap", 0.015, "low"),
                            ("emass", 0.985, "high"), ("emass", 0.015, "low")]:
        thr = df[col].quantile(frac)
        n = int((pool[col] >= thr).sum()) if tail == "high" else int((pool[col] <= thr).sum())
        logger.info(f"  pool has {n:3d} of the {tail}-{col} tail (1.5% extreme)")
    return df


def verify_alignment(df: pd.DataFrame, desc: pd.DataFrame, db_path: str) -> None:
    """Confirm graphs + descriptors cover the SAME materials in the SAME order."""
    logger.info("=" * 64)
    logger.info("Step 4 — fairness check: graphs + descriptors aligned to master")
    logger.info("=" * 64)

    # Descriptors: same uids, same order as master.
    if list(desc["uid"]) != list(df["uid"]):
        raise AssertionError("Descriptor rows are not aligned to master.parquet order.")
    logger.info(f"  descriptors: {len(desc)} rows aligned  ✓  "
                f"({desc.shape[1]-1} features)")

    # Graphs: every master uid present in the existing structure-only cache.
    cache = GraphCache(str(CACHE_DIR), GRAPH_PREPROC, readonly=True)
    if not cache.exists():
        raise FileNotFoundError(
            f"Graph cache {cache.db_path} not found. The rebuild reuses the existing "
            f"structure-only cache; build it before running this script."
        )
    missing = [u for u in df["uid"] if u not in cache]
    n_cache = len(cache)
    cache.close()
    if missing:
        raise AssertionError(f"{len(missing)} master uids missing from graph cache "
                             f"(e.g. {missing[:5]}).")
    logger.info(f"  graphs: all {len(df)} master uids present in "
                f"{cache.db_path.name} (cache holds {n_cache})  ✓")
    logger.info("  → Std-GP and DKL-BO will search the identical material set.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", default=str(REPO / "data" / "raw" / "c2db.db"))
    args = ap.parse_args()
    if not Path(args.db_path).exists():
        raise FileNotFoundError(f"C2DB database not found: {args.db_path}")

    df = build_master_table(args.db_path)
    df = add_train_pool_split(df)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    master_path = CACHE_DIR / "master.parquet"
    df.to_parquet(master_path, index=False)
    logger.info(f"  saved → {master_path}  ({len(df)} rows)")

    # Step 3 — handcrafted descriptors for Std-GP, in master row order.
    logger.info("=" * 64)
    logger.info("Step 3 — handcrafted descriptors for Std-GP")
    logger.info("=" * 64)
    logger.info(f"  {FEATURE_NOTES}")
    uids, X, names = build_descriptors(df, args.db_path)
    desc = pd.DataFrame(X, columns=names)
    desc.insert(0, "uid", uids)
    desc_path = CACHE_DIR / "descriptors.parquet"
    desc.to_parquet(desc_path, index=False)
    logger.info(f"  saved → {desc_path}  ({X.shape[0]} × {X.shape[1]})")

    verify_alignment(df, desc, args.db_path)

    logger.info("=" * 64)
    logger.info("Phase 1 complete. One dataset, graphs + descriptors aligned.")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
