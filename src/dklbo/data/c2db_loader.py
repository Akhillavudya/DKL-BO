"""Load C2DB ASE database into a metadata DataFrame with prototype-aware splits.

Key design decisions (see READme.md):
- gap_min=0.01 eV removes the zero-gap metal spike from the distribution
- Splits by structural prototype (not random rows) to prevent data leakage from
  near-duplicate structures that are common in 2D materials databases
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ase.db import connect

logger = logging.getLogger(__name__)


def load_c2db(
    db_path: str,
    target: str = "gap_hse",
    gap_min: float = 0.01,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load C2DB, filter, and return a metadata DataFrame with train/val/test split.

    Each row corresponds to one material that has a valid target label.

    Returns
    -------
    DataFrame with columns:
        id        : ASE database row id (int)
        uid       : unique material identifier (str)
        formula   : chemical formula
        prototype : structural prototype used for leakage-free splitting
        target    : band-gap value in eV (float)
        n_atoms   : number of atoms in the unit cell
        split     : "train" | "val" | "test"
    """
    db_path = str(db_path)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"C2DB database not found: {db_path}")

    db = connect(db_path)
    records = []
    total_rows = 0
    n_with_target = 0
    n_metals = 0

    for row in db.select():
        total_rows += 1
        kvp = row.key_value_pairs

        # Skip materials without the target label
        if target not in kvp:
            continue
        n_with_target += 1

        gap_val = float(kvp[target])

        # Semiconductor filter: skip metals (zero-gap spike distorts GP calibration)
        if gap_val <= gap_min:
            n_metals += 1
            continue

        # Prototype key used for split grouping.
        # C2DB uses "layergroup" (e.g. "p6m2") as the 2D structural prototype.
        # We combine layergroup + spacegroup number so that different decorations
        # of the same prototype (MoS2 vs MoSe2 both have layergroup "p3m1")
        # still land in the same split — preventing near-duplicate leakage.
        lg = kvp.get("layergroup") or kvp.get("lgnum") or kvp.get("international") or ""
        sg = str(kvp.get("number", ""))
        prototype = f"{lg}_{sg}" if lg else (sg or row.formula)

        uid = kvp.get("uid") or str(row.id)

        records.append(
            {
                "id": row.id,
                "uid": uid,
                "formula": row.formula,
                "prototype": str(prototype),
                "target": gap_val,
                "n_atoms": row.natoms,
            }
        )

    if not records:
        raise ValueError(
            f"No records found with target='{target}' and gap_min={gap_min}. "
            f"Check the db_path and target field name."
        )

    df = pd.DataFrame(records)

    # Stats were collected in one pass above — report them now
    logger.info(f"C2DB database: {total_rows} total materials")
    logger.info(f"  {n_with_target} have target='{target}'  ({100*n_with_target/total_rows:.1f}%)")
    logger.info(f"  {n_metals} are metals (gap <= {gap_min} eV) — filtered out")
    logger.info(f"  {len(df)} semiconductors remain for modelling")
    logger.info(f"  Target range: {df['target'].min():.3f} – {df['target'].max():.3f} eV")

    df = _prototype_aware_split(df, val_frac, test_frac, random_seed)

    counts = df["split"].value_counts()
    logger.info(f"  Split: train={counts.get('train',0)}  val={counts.get('val',0)}  test={counts.get('test',0)}")

    return df.reset_index(drop=True)


def _prototype_aware_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Assign train/val/test by shuffling prototype groups.

    All materials sharing a prototype land in the same split, so near-duplicate
    structures cannot leak information across the split boundary.
    """
    rng = np.random.default_rng(seed)
    prototypes = df["prototype"].unique()
    rng.shuffle(prototypes)

    n = len(prototypes)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    val_set = set(prototypes[:n_val])
    test_set = set(prototypes[n_val : n_val + n_test])

    def _assign(proto: str) -> str:
        if proto in val_set:
            return "val"
        if proto in test_set:
            return "test"
        return "train"

    df = df.copy()
    df["split"] = df["prototype"].map(_assign)
    return df


def verify_no_split_leakage(df: pd.DataFrame) -> None:
    """Assert zero prototype overlap across splits (call this after load_c2db)."""
    groups = {split: set(sub["prototype"]) for split, sub in df.groupby("split")}
    splits = list(groups.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = groups[s1] & groups[s2]
            if overlap:
                raise AssertionError(
                    f"Data leakage: {len(overlap)} prototypes shared between "
                    f"'{s1}' and '{s2}': {list(overlap)[:5]}"
                )
    logger.info("Split leakage check passed: zero prototype overlap across splits.")
