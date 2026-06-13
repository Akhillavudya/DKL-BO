"""Handcrafted material descriptors for the Standard GP-BO baseline.

The descriptor-based GP is the traditional comparator to DKL-BO. To make the
comparison *fair*, the descriptors deliberately encode only **composition** and
**geometry/stability** — the same kind of information the CGCNN sees from the
crystal structure. We explicitly exclude electronic-structure DFT outputs
(PBE gap, CBM/VBM, polarizability, Fermi level) because those are themselves
expensive DFT quantities correlated with the target; feeding them would let the
descriptor GP "cheat" and make the benchmark meaningless.

Two feature families
--------------------
1. Composition (pymatgen) — Magpie-style stoichiometry-weighted statistics
   (mean, std, min, max, range) over per-element properties.
2. Structure / stability (C2DB key_value_pairs) — n_atoms, thickness, formation
   energy, energy-above-hull, inversion symmetry, magnetism, plus ordinal-encoded
   Bravais type and layergroup.

`build_descriptors` returns an [N, D] matrix aligned to the uid order of the
metadata DataFrame, ready to drop into `ExactGPSurrogate.fit`. Standardization
(z-score, fit on the train split) is the caller's responsibility — see
`scripts/05_build_descriptors.py`.
"""

import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from ase.db import connect
from pymatgen.core import Composition, Element

logger = logging.getLogger(__name__)

# Per-element properties used for composition statistics. Chosen because they are
# defined for (nearly) every element; the few missing values are mean-imputed.
_ELEMENT_PROPS = ["Z", "atomic_mass", "X", "group", "row", "mendeleev_no", "atomic_radius"]

# Stoichiometry-weighted statistics computed per property (Magpie-style).
_STATS = ["mean", "std", "min", "max", "range"]

# Structural / stability fields pulled from C2DB key_value_pairs. Booleans are
# mapped to {0,1}; categoricals (bravais_type, layergroup) are ordinal-encoded.
_NUMERIC_STRUCT = ["thickness", "hform", "ehull"]
_BOOL_STRUCT = ["has_inversion_symmetry", "is_magnetic"]
_CATEGORICAL_STRUCT = ["bravais_type", "layergroup"]

# Electronic-structure DFT outputs that MUST NOT enter the descriptors (leakage).
EXCLUDED_LEAKAGE_FIELDS = [
    "gap", "gap_dir", "gap_hse", "gap_dir_hse", "gap_dir_nosoc",
    "cbm", "cbm_hse", "vbm", "vbm_hse", "emass_cbm", "emass_vbm",
    "efermi", "efermi_hse", "evac", "evacdiff", "dipz",
    "alphax", "alphay", "alphaz",
    "alphax_el", "alphay_el", "alphaz_el",
    "alphax_lat", "alphay_lat", "alphaz_lat",
]

FEATURE_NOTES = (
    "Composition (pymatgen) + geometry/stability (C2DB). Electronic-structure DFT "
    "outputs are excluded to keep the GP-BO vs DKL-BO comparison fair."
)


def _element_property(el: Element, prop: str) -> float:
    """Return a float element property, or NaN if undefined (e.g. EN of a noble gas)."""
    val = getattr(el, prop, None)
    if val is None:
        return math.nan
    try:
        f = float(val)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _composition_features(formula: str) -> Tuple[List[float], List[str]]:
    """Stoichiometry-weighted statistics over element properties for one formula."""
    comp = Composition(formula)
    fracs = comp.fractional_composition.as_dict()  # {symbol: weight}, sums to 1
    elements = [Element(sym) for sym in fracs]
    weights = np.array([fracs[sym] for sym in fracs], dtype=float)

    feats: List[float] = []
    names: List[str] = []
    for prop in _ELEMENT_PROPS:
        vals = np.array([_element_property(el, prop) for el in elements], dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            stats = {s: math.nan for s in _STATS}
        else:
            v = vals[finite]
            w = weights[finite]
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            mean = float(np.sum(w * v))
            std = float(math.sqrt(max(np.sum(w * (v - mean) ** 2), 0.0)))
            stats = {
                "mean": mean,
                "std": std,
                "min": float(v.min()),
                "max": float(v.max()),
                "range": float(v.max() - v.min()),
            }
        for s in _STATS:
            feats.append(stats[s])
            names.append(f"comp_{prop}_{s}")
    return feats, names


def _structural_lookup(db_path: str) -> dict:
    """Map uid -> {field: value} for the structural fields we need, in one DB pass."""
    db = connect(str(db_path))
    needed = set(_NUMERIC_STRUCT + _BOOL_STRUCT + _CATEGORICAL_STRUCT)
    out: dict = {}
    for row in db.select():
        kvp = row.key_value_pairs
        uid = kvp.get("uid") or str(row.id)
        rec = {f: kvp.get(f) for f in needed}
        rec["n_atoms"] = row.natoms
        out[uid] = rec
    return out


def _encode_categoricals(values: List, vocab: dict) -> List[int]:
    """Stable ordinal encoding; unseen/None -> -1."""
    return [vocab.get(v, -1) for v in values]


def build_descriptors(
    meta_df: pd.DataFrame,
    db_path: str,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Build the descriptor matrix for the materials in `meta_df`.

    Parameters
    ----------
    meta_df : DataFrame with at least columns [uid, formula]. Row order defines
              the output row order.
    db_path : path to the C2DB ASE database (for structural fields).

    Returns
    -------
    uids          : list[str]      — row order (matches meta_df)
    X             : np.ndarray [N, D] float64, NaNs mean-imputed (column means)
    feature_names : list[str]      — length D, one name per column

    Notes
    -----
    Features are NOT standardized here — the caller standardizes using train-split
    statistics so the GP sees zero-mean/unit-variance inputs without test leakage.
    """
    struct = _structural_lookup(db_path)

    # Build categorical vocabularies (deterministic: sorted by first appearance value)
    cat_vocabs = {}
    for field in _CATEGORICAL_STRUCT:
        seen = []
        for uid in meta_df["uid"]:
            v = struct.get(uid, {}).get(field)
            if v is not None and v not in seen:
                seen.append(v)
        cat_vocabs[field] = {v: i for i, v in enumerate(sorted(map(str, seen)))}

    rows: List[List[float]] = []
    feature_names: List[str] | None = None
    uids: List[str] = []

    for _, r in meta_df.iterrows():
        uid = r["uid"]
        uids.append(uid)
        comp_feats, comp_names = _composition_features(r["formula"])

        s = struct.get(uid, {})
        struct_feats: List[float] = []
        struct_names: List[str] = []

        struct_feats.append(float(s.get("n_atoms", r.get("n_atoms", math.nan))))
        struct_names.append("struct_n_atoms")

        for f in _NUMERIC_STRUCT:
            v = s.get(f)
            struct_feats.append(float(v) if v is not None else math.nan)
            struct_names.append(f"struct_{f}")

        for f in _BOOL_STRUCT:
            v = s.get(f)
            struct_feats.append(float(bool(v)) if v is not None else math.nan)
            struct_names.append(f"struct_{f}")

        for f in _CATEGORICAL_STRUCT:
            v = s.get(f)
            code = cat_vocabs[f].get(str(v), -1) if v is not None else -1
            struct_feats.append(float(code))
            struct_names.append(f"struct_{f}_code")

        row_feats = comp_feats + struct_feats
        if feature_names is None:
            feature_names = comp_names + struct_names
        rows.append(row_feats)

    X = np.asarray(rows, dtype=np.float64)

    # Mean-impute any NaNs (missing element props or structural fields) column-wise.
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    logger.info(
        f"Built descriptors: {X.shape[0]} materials × {X.shape[1]} features "
        f"({len(comp_names)} composition + {len(struct_names)} structural). "
        f"Imputed {int(nan_mask.sum())} missing cells."
    )
    return uids, X, feature_names
