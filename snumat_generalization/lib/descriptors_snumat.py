"""Handcrafted descriptors for the Std-GP baseline on the SNUMAT 3D dataset.

This is the SNUMAT analogue of `dklbo.baselines.descriptors`. To keep the Std-GP
vs DKL-BO comparison FAIR, descriptors encode only **composition** and
**geometry/structure** — the same kind of information the CGCNN reads from the
crystal — and deliberately EXCLUDE any electronic-structure DFT output (the band
gaps themselves and the direct/indirect label), which would let the descriptor GP
cheat.

Two feature families
--------------------
1. Composition (pymatgen) — Magpie-style stoichiometry-weighted statistics.
   REUSED unchanged from the C2DB module (`_composition_features`), which depends
   only on the formula string.
2. Geometry / structure — computed from the ASE Atoms object (cell + species) plus
   a few cheap categorical fields from the SNUMAT JSON:
       n_atoms, volume_per_atom, density, packing_fraction,
       space_group (numeric), soc (0/1), magnetic_ordering (ordinal code).

EXCLUDED as leakage: Band_gap_HSE/GGA[_optical], Direct_or_indirect[_HSE].
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import covalent_radii

# Reuse the composition-feature helper from the main package (read-only import;
# the C2DB descriptor module is NOT modified).
from dklbo.baselines.descriptors import _composition_features

logger = logging.getLogger(__name__)

FEATURE_NOTES = (
    "Composition (pymatgen) + 3D geometry/structure (cell, species, space group, "
    "SOC, magnetic ordering). Band gaps and direct/indirect labels are excluded to "
    "keep the GP-BO vs DKL-BO comparison fair."
)


def _geometry_features(atoms: Atoms) -> Tuple[List[float], List[str]]:
    """Cheap geometry descriptors from the cell + species (no DFT outputs)."""
    n = len(atoms)
    vol = float(atoms.get_volume())
    masses = float(atoms.get_masses().sum())
    # Sum of atomic (covalent-sphere) volumes / cell volume — a packing proxy.
    radii = np.array([covalent_radii[int(z)] for z in atoms.get_atomic_numbers()])
    sphere_vol = float((4.0 / 3.0) * math.pi * np.sum(radii ** 3))

    feats = [
        float(n),
        vol / n if n else math.nan,                 # volume per atom
        masses / vol if vol > 0 else math.nan,       # density (amu / A^3)
        sphere_vol / vol if vol > 0 else math.nan,   # packing fraction
    ]
    names = ["geom_n_atoms", "geom_vol_per_atom", "geom_density", "geom_packing"]
    return feats, names


def build_descriptors_snumat(
    meta_df: pd.DataFrame,
    atoms_map: Dict[str, Atoms],
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Build the [N, D] descriptor matrix aligned to `meta_df` row order.

    Parameters
    ----------
    meta_df : DataFrame with columns [uid, formula, space_group, soc, magnetic_ordering].
    atoms_map : uid -> ASE Atoms (for geometry features).

    Returns
    -------
    uids, X (float64, NaNs mean-imputed column-wise), feature_names.
    Features are NOT standardized here — the caller standardizes on train statistics.
    """
    # Deterministic ordinal vocab for magnetic ordering.
    mag_vocab = {v: i for i, v in enumerate(sorted(meta_df["magnetic_ordering"].astype(str).unique()))}

    rows: List[List[float]] = []
    feature_names: List[str] | None = None
    uids: List[str] = []

    for _, r in meta_df.iterrows():
        uid = r["uid"]
        uids.append(uid)

        comp_feats, comp_names = _composition_features(r["formula"])
        geo_feats, geo_names = _geometry_features(atoms_map[uid])

        cat_feats = [
            float(r.get("space_group", -1)),
            float(bool(r.get("soc", False))),
            float(mag_vocab.get(str(r.get("magnetic_ordering", "NM")), -1)),
        ]
        cat_names = ["struct_space_group", "struct_soc", "struct_magnetic_code"]

        row_feats = comp_feats + geo_feats + cat_feats
        if feature_names is None:
            feature_names = comp_names + geo_names + cat_names
        rows.append(row_feats)

    X = np.asarray(rows, dtype=np.float64)

    # Mean-impute any NaNs column-wise.
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    logger.info(
        f"  built descriptors: {X.shape[0]} materials x {X.shape[1]} features "
        f"({len(comp_names)} composition + {len(geo_names)} geometry + {len(cat_names)} "
        f"categorical). Imputed {int(nan_mask.sum())} missing cells."
    )
    return uids, X, feature_names
