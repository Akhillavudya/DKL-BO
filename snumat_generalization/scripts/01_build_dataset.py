"""Phase 1 (SNUMAT generalization side-test) — build the band-gap dataset.

Mirror of the C2DB rebuild's `scripts/01_build_dataset.py`, but for the SNUMAT
database of ~10k 3D bulk crystals (band gap only, no effective mass). It produces
ONE dataset shared by every method (Random, Std-GP, DKL-BO):

  data/cache/master.parquet      — uid, formula, prototype, gap (HSE), metadata, split
  data/cache/descriptors.parquet — handcrafted features for Std-GP, same row order
  data/cache/graphs_<hash>.lmdb  — 3D crystal graphs for DKL (built fresh here)

A prototype-aware, gap-stratified train/pool split keeps near-duplicate structures
on one side of the boundary; `pool` is the held-out hunting ground for BO and the
test set for prediction accuracy.

Run
---
    python snumat_generalization/scripts/01_build_dataset.py
    python snumat_generalization/scripts/01_build_dataset.py --data_dir /path/to/Dataset
"""

import argparse
import logging
import sys
from pathlib import Path

# Make both the local lib and the reused dklbo package importable.
SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))                 # -> import lib.*
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))  # -> import dklbo.*

import pandas as pd
import torch

from lib import config
from lib.snumat_loader import load_snumat, add_train_pool_split
from lib.descriptors_snumat import build_descriptors_snumat, FEATURE_NOTES
from lib.graph_builder_3d import atoms_to_graph_3d
from dklbo.data.cache import GraphCache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_graph_cache(df: pd.DataFrame, atoms_map: dict) -> GraphCache:
    """Build the 3D graph LMDB cache (fresh) and return a read handle."""
    logger.info("=" * 64)
    logger.info("Step 3 - build 3D crystal graph cache (no vacuum filter)")
    logger.info("=" * 64)
    cache = GraphCache(str(config.CACHE_DIR), config.GRAPH_PREPROC, readonly=False)

    def _items():
        for _, r in df.iterrows():
            g = atoms_to_graph_3d(
                atoms_map[r["uid"]],
                radius=config.GRAPH_PREPROC["radius"],
                max_neighbors=config.GRAPH_PREPROC["max_neighbors"],
                atom_feat_dim=config.GRAPH_PREPROC["atom_feat_dim"],
            )
            g.y = torch.tensor([r["gap"]], dtype=torch.float32)  # target carried for convenience
            yield r["uid"], g

    cache.put_batch(_items())
    cache.close()
    logger.info(f"  built {len(df)} graphs -> {config.CACHE_DIR}/graphs_{cache.hash}.lmdb")
    return GraphCache(str(config.CACHE_DIR), config.GRAPH_PREPROC, readonly=True)


def verify_alignment(df: pd.DataFrame, desc: pd.DataFrame) -> None:
    logger.info("=" * 64)
    logger.info("Step 5 - fairness check: graphs + descriptors aligned to master")
    logger.info("=" * 64)
    if list(desc["uid"]) != list(df["uid"]):
        raise AssertionError("Descriptor rows are not aligned to master.parquet order.")
    logger.info(f"  descriptors: {len(desc)} rows aligned  ({desc.shape[1]-1} features)")

    cache = GraphCache(str(config.CACHE_DIR), config.GRAPH_PREPROC, readonly=True)
    missing = [u for u in df["uid"] if u not in cache]
    n_cache = len(cache)
    cache.close()
    if missing:
        raise AssertionError(f"{len(missing)} master uids missing from graph cache "
                             f"(e.g. {missing[:5]}).")
    logger.info(f"  graphs: all {len(df)} master uids present (cache holds {n_cache})")
    logger.info("  -> Std-GP and DKL-BO will search the identical material set.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.DEFAULT_DATA_DIR)
    args = ap.parse_args()

    df, atoms_map = load_snumat(args.data_dir)
    df = add_train_pool_split(df, pool_frac=config.POOL_FRAC,
                              n_strata=config.N_STRATA, seed=config.SEED)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    master_path = config.CACHE_DIR / "master.parquet"
    df.to_parquet(master_path, index=False)
    logger.info(f"  saved -> {master_path}  ({len(df)} rows)")

    # Step 3 — graphs.
    build_graph_cache(df, atoms_map)

    # Step 4 — descriptors.
    logger.info("=" * 64)
    logger.info("Step 4 - handcrafted descriptors for Std-GP")
    logger.info("=" * 64)
    logger.info(f"  {FEATURE_NOTES}")
    uids, X, names = build_descriptors_snumat(df, atoms_map)
    desc = pd.DataFrame(X, columns=names)
    desc.insert(0, "uid", uids)
    desc_path = config.CACHE_DIR / "descriptors.parquet"
    desc.to_parquet(desc_path, index=False)
    logger.info(f"  saved -> {desc_path}  ({X.shape[0]} x {X.shape[1]})")

    verify_alignment(df, desc)

    logger.info("=" * 64)
    logger.info("Phase 1 complete. One SNUMAT dataset, graphs + descriptors aligned.")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
