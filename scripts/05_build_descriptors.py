"""Build handcrafted descriptor matrices for the Standard GP-BO baseline.

Produces one parquet per target universe (band-gap and effective-mass) under
`results/benchmark/`. Each parquet has a `uid` column plus one column per
descriptor feature. The benchmark runner (07) loads these and standardizes them
inside the BO loop, so this script stores the RAW (un-standardized) features for
transparency and reuse.

Usage
-----
    # band-gap universe (uses metadata.parquet)
    python scripts/05_build_descriptors.py data.db_path=data/raw/c2db.db

    # effective-mass universe (uses metadata_emass.parquet)
    python scripts/05_build_descriptors.py data=c2db_emass data.db_path=data/raw/c2db.db
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import pandas as pd
from omegaconf import DictConfig

from dklbo.baselines.descriptors import build_descriptors, FEATURE_NOTES

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cache_dir = Path(cfg.data.cache_dir)
    meta_file = cfg.data.get("meta_file", "metadata.parquet")
    meta_path = cache_dir / meta_file
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run 01_build_cache.py (gap) or "
            f"06_build_emass_dataset.py (emass) first."
        )
    df = pd.read_parquet(meta_path)
    logger.info(f"Building descriptors for {len(df)} materials from {meta_file}")
    logger.info(FEATURE_NOTES)

    uids, X, names = build_descriptors(df, cfg.data.db_path)

    out_df = pd.DataFrame(X, columns=names)
    out_df.insert(0, "uid", uids)

    out_dir = Path(cfg.results_dir) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tag the file by metadata source so gap/emass universes never collide.
    tag = "emass" if "emass" in meta_file else "gap"
    out_path = out_dir / f"descriptors_{tag}.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info(f"Descriptors saved → {out_path}  ({X.shape[0]} × {X.shape[1]})")


if __name__ == "__main__":
    main()
