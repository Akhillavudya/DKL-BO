"""Build the effective-mass benchmark dataset (metadata + prototype-aware split).

The DKL-BO/GP-BO benchmark adds two effective-mass tasks (min / max emass_cbm).
Effective mass is only physically meaningful for a semiconductor, so we keep the
same gap_hse > gap_min semiconductor filter and additionally require a finite,
positive `emass_cbm` (some entries are `inf` or 0 and are dropped).

Graphs are structure-only and already cached for the band-gap task, so this
script does NOT rebuild graphs — it only writes `metadata_emass.parquet` (same
columns as `metadata.parquet`, but `target` = emass_cbm) and asserts every uid
is present in the existing LMDB graph cache.

Usage
-----
    python scripts/06_build_emass_dataset.py data=c2db_emass data.db_path=data/raw/c2db.db
"""

import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import pandas as pd
from ase.db import connect
from omegaconf import DictConfig

from dklbo.data.c2db_loader import _prototype_aware_split, verify_no_split_leakage
from dklbo.data.cache import GraphCache
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    target = cfg.data.target            # emass_cbm
    gap_min = float(cfg.data.gap_min)
    db = connect(cfg.data.db_path)

    records = []
    n_no_gap = n_no_emass = n_nonfinite = 0
    for row in db.select():
        kvp = row.key_value_pairs

        gap = kvp.get("gap_hse")
        if gap is None or float(gap) <= gap_min:   # semiconductor filter
            n_no_gap += 1
            continue

        em = kvp.get(target)
        if em is None:
            n_no_emass += 1
            continue
        em = float(em)
        if not math.isfinite(em) or em <= 0:       # drop inf / zero effective mass
            n_nonfinite += 1
            continue

        lg = kvp.get("layergroup") or kvp.get("lgnum") or kvp.get("international") or ""
        sg = str(kvp.get("number", ""))
        prototype = f"{lg}_{sg}" if lg else (sg or row.formula)
        uid = kvp.get("uid") or str(row.id)

        records.append({
            "id": row.id,
            "uid": uid,
            "formula": row.formula,
            "prototype": str(prototype),
            "target": em,
            "n_atoms": row.natoms,
        })

    if not records:
        raise ValueError(f"No materials with finite positive '{target}' and gap_hse>{gap_min}.")

    df = pd.DataFrame(records)
    logger.info(
        f"emass dataset: {len(df)} materials  |  "
        f"range {df['target'].min():.3f}–{df['target'].max():.3f}  "
        f"(dropped: {n_no_gap} non-semiconductor, {n_no_emass} no-emass, "
        f"{n_nonfinite} inf/zero)"
    )

    df = _prototype_aware_split(df, cfg.data.val_frac, cfg.data.test_frac, cfg.seed)
    verify_no_split_leakage(df)
    counts = df["split"].value_counts()
    logger.info(f"Split: train={counts.get('train',0)} val={counts.get('val',0)} test={counts.get('test',0)}")

    # Assert all uids are present in the existing (structure-only) graph cache.
    graph_preproc = {
        "radius": cfg.data.radius,
        "vacuum_cutoff": cfg.data.vacuum_cutoff,
        "max_neighbors": cfg.data.max_neighbors,
        "target": cfg.data.get("graph_cache_target", "gap_hse"),
        "gap_min": cfg.data.gap_min,
    }
    cache = GraphCache(cfg.data.cache_dir, graph_preproc, readonly=True)
    if not cache.exists():
        raise FileNotFoundError(
            f"Graph cache {cache.db_path} not found. Run 01_build_cache.py first."
        )
    missing = [u for u in df["uid"] if u not in cache]
    cache.close()
    if missing:
        raise AssertionError(
            f"{len(missing)} emass uids are missing from the graph cache "
            f"(e.g. {missing[:5]}). The benchmark reuses the band-gap graph cache; "
            f"rebuild it if the structure set changed."
        )
    logger.info(f"All {len(df)} emass uids present in graph cache {cache.db_path.name}.")

    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / cfg.data.get("meta_file", "metadata_emass.parquet")
    df.to_parquet(out_path, index=False)
    logger.info(f"Effective-mass metadata saved → {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
