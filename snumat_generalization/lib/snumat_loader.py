"""Load the SNUMAT 3D bulk-crystal dataset into a metadata DataFrame + ASE Atoms.

The SNUMAT dataset is one JSON file per material (in `Dataset/`), each holding:
    Band_gap_HSE, Band_gap_GGA, Band_gap_*_optical : band gaps (eV)
    Direct_or_indirect[_HSE]                        : band-structure label
    Structure_rlx                                   : a VASP5 POSCAR string
    Space_group_rlx                                 : space-group number
    SOC                                             : spin-orbit-coupling flag (bool)
    Magnetic_ordering                               : NM / FM / AFM / ...
    SNUMAT_id                                        : unique id (e.g. SM-19222)
    ICSD_number

This replaces `dklbo.data.c2db_loader` (which reads an ASE .db of 2D materials).
The target is the HSE band gap. Effective mass does not exist in this dataset, so
there is only the band-gap study.
"""

import io
import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read as ase_read
from pymatgen.core import Composition

logger = logging.getLogger(__name__)

GAP_MIN = 0.01      # drop near-zero / metallic gaps (mirrors the C2DB rebuild)
TARGET_FIELD = "Band_gap_HSE"


def _prototype_key(formula: str, space_group) -> str:
    """3D analogue of the C2DB layergroup+spacegroup prototype key.

    Anonymised stoichiometry (e.g. "ABC3") + space-group number groups different
    chemical decorations of the same structural prototype so near-duplicate
    structures cannot leak across the train/pool split boundary.
    """
    try:
        anon = Composition(formula).anonymized_formula
    except Exception:
        anon = formula
    sg = str(space_group) if space_group is not None else ""
    return f"{anon}_{sg}"


def load_snumat(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, Atoms]]:
    """Scan `data_dir`/*.json and return (metadata DataFrame, uid -> ASE Atoms).

    DataFrame columns:
        id, uid, formula, prototype, gap, n_atoms, soc, magnetic_ordering, space_group
    Rows with a missing/non-finite/<=GAP_MIN HSE gap or an unparseable structure are
    skipped; the counts are logged.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON materials found in {data_path}")

    records = []
    atoms_map: Dict[str, Atoms] = {}
    n_no_gap = n_metal = n_bad_struct = 0

    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            n_bad_struct += 1
            continue

        gap = d.get(TARGET_FIELD)
        if gap is None:
            n_no_gap += 1
            continue
        gap = float(gap)
        if not math.isfinite(gap) or gap <= GAP_MIN:     # metal / zero-gap -> drop
            n_metal += 1
            continue

        struct_str = d.get("Structure_rlx")
        if not struct_str:
            n_bad_struct += 1
            continue
        try:
            atoms = ase_read(io.StringIO(struct_str), format="vasp")
        except Exception:
            n_bad_struct += 1
            continue

        uid = str(d.get("SNUMAT_id") or f.stem)
        formula = atoms.get_chemical_formula()
        sg = d.get("Space_group_rlx")

        records.append({
            "id": len(records),
            "uid": uid,
            "formula": formula,
            "prototype": _prototype_key(formula, sg),
            "gap": gap,
            "n_atoms": len(atoms),
            "soc": bool(d.get("SOC", False)),
            "magnetic_ordering": str(d.get("Magnetic_ordering", "NM")),
            "space_group": int(sg) if sg is not None else -1,
        })
        atoms_map[uid] = atoms

    if not records:
        raise ValueError(f"No materials with a valid {TARGET_FIELD} found in {data_path}.")

    df = pd.DataFrame(records).reset_index(drop=True)
    # Drop any duplicate SNUMAT ids (keep first) so uid is a clean key.
    before = len(df)
    df = df.drop_duplicates(subset="uid", keep="first").reset_index(drop=True)
    n_dup = before - len(df)

    logger.info("=" * 64)
    logger.info("Step 1 - SNUMAT master dataset (HSE band gap, 3D bulk crystals)")
    logger.info("=" * 64)
    logger.info(f"  scanned        : {len(files)} JSON files")
    logger.info(f"  kept           : {len(df)} materials")
    logger.info(f"  dropped        : {n_no_gap} no-gap, {n_metal} metals(gap<={GAP_MIN}), "
                f"{n_bad_struct} unparseable/struct-missing, {n_dup} duplicate uids")
    logger.info(f"  gap range eV   : {df['gap'].min():.3f} - {df['gap'].max():.3f}")
    logger.info(f"  n_atoms range  : {int(df['n_atoms'].min())} - {int(df['n_atoms'].max())}")
    return df, atoms_map


def verify_no_split_leakage(df: pd.DataFrame) -> None:
    """Assert zero prototype overlap across splits."""
    groups = {split: set(sub["prototype"]) for split, sub in df.groupby("split")}
    splits = list(groups.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = groups[s1] & groups[s2]
            if overlap:
                raise AssertionError(
                    f"Data leakage: {len(overlap)} prototypes shared between "
                    f"'{s1}' and '{s2}': {list(overlap)[:5]}")
    logger.info("  split leakage check passed: zero prototype overlap.")


def add_train_pool_split(df: pd.DataFrame, pool_frac: float = 0.30,
                         n_strata: int = 4, seed: int = 42) -> pd.DataFrame:
    """Prototype-aware, gap-stratified train/pool split (mirrors scripts/01)."""
    grp = df.groupby("prototype")["gap"].max().reset_index()
    grp.columns = ["prototype", "max_gap"]
    grp["stratum"] = pd.qcut(grp["max_gap"], q=n_strata, labels=False, duplicates="drop")

    rng = np.random.default_rng(seed)
    pool_protos: set = set()
    for _, sub in grp.groupby("stratum"):
        protos = sub["prototype"].to_numpy()
        rng.shuffle(protos)
        n_pool = max(1, int(round(len(protos) * pool_frac)))
        pool_protos.update(protos[:n_pool])

    df = df.copy()
    df["split"] = np.where(df["prototype"].isin(pool_protos), "pool", "train")
    verify_no_split_leakage(df)

    train, pool = df[df.split == "train"], df[df.split == "pool"]
    logger.info("=" * 64)
    logger.info("Step 2 - train/pool split (gap-stratified, by prototype)")
    logger.info("=" * 64)
    logger.info(f"  train = {len(train)}   pool = {len(pool)}")
    logger.info(f"  pool gap range : {pool['gap'].min():.3f} - {pool['gap'].max():.3f} eV")
    for frac, tail in [(0.985, "high"), (0.015, "low")]:
        thr = df["gap"].quantile(frac)
        n = int((pool["gap"] >= thr).sum()) if tail == "high" else int((pool["gap"] <= thr).sum())
        logger.info(f"  pool has {n:3d} of the {tail}-gap tail (1.5% extreme)")
    return df
