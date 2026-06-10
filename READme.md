# DKL-BO Project — Context Summary

## What This Project Is
Pool-based active learning to discover high-band-gap 2D materials from C2DB database.
Method: Deep Kernel Learning Bayesian Optimisation (DKL-BO) = CGCNN encoder + Gaussian Process + UCB acquisition.

## Your Data
- C2DB database: ~17,000 materials total
- ~8,000 have band-gap labels (gap_hse)
- Primary target: `gap_hse` (HSE06 band gap in eV)

## Key References
- **[P1] Kiyohara & Kumagai (2025)** — the method we implement (CGCNN + GP, attention pooling, Matérn-5/2, UCB)
- **[P2] Mamun, Yang & Yue (2026)** — scaling fix (SVGP) + calibration metrics
- **[P3] Lyu et al. (2023)** — transfer learning strategy (pre-train on Eform → fine-tune on sparse target)

## Pipeline (4 steps from prof's reference PDF)
| Step | Script | Does |
|------|--------|------|
| 1 | `step1_c2db_loader.py` | Load C2DB → filter → CIF files + metadata CSV |
| 2 | `step2_cgcnn_graph.py` | CIF files → crystal graph tensors (.pt) |
| 3 | `step3_dkl_bo.py` | DKL-BO active learning loop |
| 4 | `step4_analysis.py` | Convergence plots + top materials |

## Folder Structure Created
```
dkl-bo-c2db/
├── configs/
│   ├── bo/
│   ├── data/
│   ├── experiment/
│   └── model/
├── results/
├── scripts/
├── src/
│   └── dklbo/
│       ├── bo/
│       ├── data/
│       ├── eval/
│       ├── models/
│       └── utils/
└── tests/
```

## Key Design Decisions
- **min_gap = 0.01 eV** filter removes metals (zero-gap spike in C2DB)
- **vacuum_cutoff = 4.0 Å** blocks false bonds across c-direction vacuum gap (critical 2D fix)
- **Attention pooling** not average — variable unit cell sizes in C2DB
- **Exact GP** default (N < 1000); swap to **SVGP** if OOM errors appear
- **N_INIT = 10** — small initial set is better ([P1] finding)
- **UCB beta = 0.2** — prof's recommendation
- **Prototype-aware split** — prevents data leakage (near-duplicate structures in C2DB)
- **pretrain = false** for gap_hse (8000 points, enough data)
- **pretrain = true** for gap_gw (only ~200 points, needs Eform transfer)

## Implementation Plan Phases
| Phase | Goal | Status |
|-------|------|--------|
| 0 | Env setup + data audit | ✅ Folders created |
| 1 | Graph cache (build once) | ✅ Complete |
| 2 | Offline surrogate eval (no loop) | 🔲 Not started |
| 3 | BO loop + baselines | 🔲 Not started |
| 4 | Scaling + transfer learning | 🔲 Not started |
| 5 | Analysis + write-up | 🔲 Not started |

## What Is Done (Phase 1)
- `pyproject.toml` — pinned dependencies (torch, gpytorch, botorch, ase, PyG, lmdb, hydra)
- `configs/` — all Hydra config groups (data, model, bo, experiment)
- `src/dklbo/data/c2db_loader.py` — ASE db → metadata DataFrame + prototype-aware split
- `src/dklbo/data/graph_builder.py` — vacuum-aware 2D graph construction (ATOM_FEAT_DIM=90, BOND_FEAT_DIM=10)
- `src/dklbo/data/cache.py` — LMDB graph store with config-hash invalidation
- `src/dklbo/utils/seed.py` — deterministic seeding
- `src/dklbo/utils/profiling.py` — per-cycle memory + wall-clock profiling
- `scripts/01_build_cache.py` — Hydra CLI to build the cache from C2DB
- `tests/test_vacuum_cutoff.py` — 7 correctness tests for the vacuum filter + MoS2 coordination

## How to Run Phase 1
```bash
pip install -e .
python scripts/01_build_cache.py data.db_path=/path/to/c2db.db
pytest tests/test_vacuum_cutoff.py -v
```

## Next Step
Phase 2: Offline surrogate validation (`02_eval_surrogate.py`)
- CGCNN encoder (`src/dklbo/models/cgcnn_encoder.py`)
- Exact GP + SVGP surrogates (`src/dklbo/models/surrogate.py`, `dkl.py`)
- Calibration metrics suite (`src/dklbo/eval/`)