# SNUMAT Generalization Side-Test

**Goal.** Show that the main project's headline result — *pre-trained DKL-BO beats
Std-GP on band-gap discovery* — **generalizes** from the C2DB 2D-materials database to a
second, structurally different dataset: **SNUMAT**, ~10k **3D bulk crystals** (ICSD-derived),
band gap only (HSE).

This is a **self-contained side experiment** for the paper. It does **not** modify the main
`src/dklbo` package or the existing `scripts/` — it only adds a new data layer here and reuses
the dataset-agnostic model/BO code from `dklbo` by import.

## Layout
```
lib/    snumat_loader.py, graph_builder_3d.py, descriptors_snumat.py, config.py
scripts/ 01..07  (mirror the C2DB rebuild, band-gap only)
data/cache/      master.parquet, descriptors.parquet, graphs_<hash>.lmdb
results/         runs/, runs_finetune/, runs_coldlive/, *.csv, plots/
docs/            per-phase explainers
```

## What's different vs the C2DB rebuild
- **3D bulk crystals** → graphs use no 2D vacuum/z-filter (`lib/graph_builder_3d.py`).
- **Band gap only (no emass)** → one study, two search tasks: `gap_max`, `gap_min`.
- **Data source** = SNUMAT JSON files (`Band_gap_HSE` + VASP POSCAR `Structure_rlx`).
- Target = **HSE** gap; the GGA gap and direct/indirect labels are excluded as leakage features.

## Run (phase by phase)
```
python snumat_generalization/scripts/01_build_dataset.py        # data + graphs + descriptors
python snumat_generalization/scripts/02_pretrain_encoder.py     # CGCNN encoder + embeddings
python snumat_generalization/scripts/03_eval_accuracy.py        # Std-GP vs DKL accuracy
python snumat_generalization/scripts/04_run_bo.py               # BO contest (3 methods)
python snumat_generalization/scripts/05_run_bo_finetune.py      # live finetune (+ --cold)
python snumat_generalization/scripts/06_stats.py                # CIs + paired Wilcoxon
python snumat_generalization/scripts/07_plot.py                 # curves + bars
```
Data source defaults to `/home/roy/Desktop/Bandgap_prediction/Dataset` (override `--data_dir`).
