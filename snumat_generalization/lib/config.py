"""Shared paths and constants for the SNUMAT generalization side-test.

Centralised so every script (01..07) agrees on the cache hash, split, and BO budget.
"""

from pathlib import Path

# snumat_generalization/  (this file is in snumat_generalization/lib/)
SNUMAT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SNUMAT_ROOT.parent
SRC_DIR = REPO_ROOT / "src"                      # for importing the reused `dklbo` package

CACHE_DIR = SNUMAT_ROOT / "data" / "cache"
RESULTS_DIR = SNUMAT_ROOT / "results"

# Default location of the SNUMAT JSON materials (overridable via --data_dir).
DEFAULT_DATA_DIR = "/home/roy/Desktop/Bandgap_prediction/Dataset"

# --- fixed dataset choices (mirror the C2DB rebuild) -------------------------
GAP_MIN = 0.01
POOL_FRAC = 0.30
N_STRATA = 4
SEED = 42

# --- graph preprocessing -----------------------------------------------------
# This dict defines the LMDB cache hash; build (01) and read (02..05) must match.
# Note: 3D bulk crystals -> NO vacuum_cutoff (see lib/graph_builder_3d.py).
GRAPH_PREPROC = {
    "dataset": "snumat",
    "radius": 8.0,
    "max_neighbors": 12,
    "atom_feat_dim": 100,
    "target": "Band_gap_HSE",
    "gap_min": GAP_MIN,
    "dim": "3d",
}

# --- BO contest budget (mirror the C2DB rebuild) -----------------------------
N_INIT = 10
N_CYCLES = 100
N_SEEDS = 10
GAP_TASKS = ["gap_max", "gap_min"]
