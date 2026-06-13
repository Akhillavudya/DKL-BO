# Benchmark Design — Standard GP-BO vs DKL-BO

## The question

Do **learned graph representations** (CGCNN → GP, "DKL-BO") give better sample
efficiency for 2D-materials discovery than a **traditional handcrafted-descriptor
GP** ("Standard GP-BO")? We answer it on the C2DB database across four optimisation
tasks with a paired, statistically-tested protocol.

## What makes the comparison fair

The two methods are made identical in everything except the feature source:

| Component | DKL-BO | Standard GP-BO |
|-----------|--------|----------------|
| Features | CGCNN 32-d learned embedding | 43 handcrafted composition + structure descriptors |
| GP backend | `ExactGPSurrogate` (Matérn-5/2) | **the same** `ExactGPSurrogate` (Matérn-5/2) |
| Acquisition | Expected Improvement | Expected Improvement |
| Init set | `random.sample(uids, n_init)` @ seed | **identical** (same seed, same universe) |
| Direction handling | sign trick (`y_internal = sign·y`) | identical sign trick |
| Oracle / pool / budget | same `metadata.parquet`, same `n_cycles` | same |

Because the per-(task, seed) initial set is identical, DKL and Std-GP form a
**paired sample** → we use the **Wilcoxon signed-rank** test.

### No-leakage descriptor rule

Descriptors use **composition** (pymatgen Magpie-style stoichiometry-weighted
statistics over Z, mass, electronegativity, group, row, mendeleev number, atomic
radius) and **geometry/stability** (n_atoms, thickness, formation energy, energy
above hull, inversion symmetry, magnetism, Bravais/layergroup codes). They
**exclude** every electronic-structure DFT output (PBE gap, CBM/VBM, polarizability,
Fermi level) — see `EXCLUDED_LEAKAGE_FIELDS` in `baselines/descriptors.py`. Those
quantities are themselves expensive DFT outputs correlated with the target;
including them would let the descriptor GP trivially "cheat". Both methods therefore
see only structure + composition — exactly what the CGCNN sees.

## Tasks

| Task | Target | Direction | Universe | Meaning |
|------|--------|-----------|----------|---------|
| `gap_max`   | `gap_hse`   | max | 3,351 | widest-gap insulators |
| `gap_min`   | `gap_hse`   | min | 3,351 | smallest-gap (>0.01 eV) semiconductors |
| `emass_min` | `emass_cbm` | min | 2,667 | **high-mobility** (low effective mass) |
| `emass_max` | `emass_cbm` | max | 2,667 | flat-band / heavy-carrier materials |

> Note: the original brief named `emass_cb_1`; that column does not exist in this
> C2DB dump. The real column is `emass_cbm` (electron effective mass at the CBM).
> Entries with `inf`/0 effective mass are dropped. Graphs are structure-only, so the
> effective-mass tasks **reuse the band-gap graph cache** — only the metadata/split
> differs (`metadata_emass.parquet`).

## Minimization: the sign trick

The GP and acquisition always **maximize**. A minimization task is handled by training
the GP on `y_internal = -y_true` and selecting `argmax`. Results are reported back in
original units (`best_so_far = sign · best_internal`). Top-K is defined on the internal
scale, so a min task's "top-50" are the 50 *lowest*-valued materials. This one mechanism
lives in `BOLoop`, `FeatureBOLoop`, and `RandomBaseline` so all three agree.

## Pipeline

```
01_build_cache.py            # (existing) band-gap graph cache — reused by all tasks
06_build_emass_dataset.py    # emass metadata + prototype split (no graph rebuild)
05_build_descriptors.py      # descriptors_gap.parquet, descriptors_emass.parquet
07_run_benchmark.py          # 4 tasks × {dkl, std_gp, random} × 10 seeds → runs/*.csv
08_benchmark_stats.py        # regret-AUC, Wilcoxon, bootstrap CIs, offline accuracy
09_plot_benchmark.py         # best-found / regret / top-k / summary figures
```

Run order:
```bash
python scripts/06_build_emass_dataset.py data=c2db_emass data.db_path=data/raw/c2db.db
python scripts/05_build_descriptors.py data.db_path=data/raw/c2db.db
python scripts/05_build_descriptors.py data=c2db_emass data.db_path=data/raw/c2db.db
python scripts/07_run_benchmark.py bo=ei data.db_path=data/raw/c2db.db
python scripts/08_benchmark_stats.py --db_path data/raw/c2db.db
python scripts/09_plot_benchmark.py
```
`07` is idempotent (skips existing `runs/*.csv`) so the sweep is resumable.

## Metrics

- **Optimization**: best-found vs iteration, simple regret + **regret-AUC**
  (primary sample-efficiency metric), cumulative top-50 / top-10%.
- **Statistics**: Wilcoxon signed-rank (DKL vs Std-GP), 95% bootstrap CIs,
  matched-pairs rank-biserial effect size, per-seed win rate.
- **Surrogate quality** (held-out test split): MAE, RMSE, R² + calibration
  (NLL, ENCE, Coverage@95, Spearman ρ). Descriptor-GP is computed in `08`; the DKL
  band-gap numbers are pulled from `02_eval_surrogate.py` output when present.

## Interpretation guidelines

- **DKL wins** if its regret-AUC is significantly lower (Wilcoxon p < 0.05 with a
  non-trivial rank-biserial effect and win-rate > 0.5), especially on the gap tasks
  where 3D bonding topology is informative → the learned representation buys sample
  efficiency that justifies the CGCNN's complexity.
- **Std-GP competitive or better** on a task → handcrafted composition/geometry
  descriptors already capture the signal; the encoder isn't worth it *there*.
- Both outcomes are publishable and honest. The framework is built to **report
  whichever is true**, not to favour DKL.

## Files added

```
src/dklbo/baselines/{descriptors.py, feature_bo_loop.py}
src/dklbo/bo/acquisition.py   (+ ei)        bo/loop.py (+ direction, train_y)
src/dklbo/models/dkl.py       (+ train_y override)
configs/bo/ei.yaml  configs/data/c2db_emass.yaml  configs/benchmark/bench.yaml
scripts/05..09_*.py           tests/test_benchmark.py
```
