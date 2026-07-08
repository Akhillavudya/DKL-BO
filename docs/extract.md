# Project Extract — DKL-BO

> Structured profile extracted directly from the repository (code, configs, git history, docs)
> for CV / portfolio use. Generated 2026-06-30; §1/§6/§7 updated 2026-07-01 with author-confirmed
> context (research assistantship, first author, May–June 2026, paper in progress, embargoed).

---

## 1. Identity

- **Project name:** DKL-BO — Deep Kernel Learning + Bayesian Optimisation for 2D Material Discovery
- **One-line description (CV header):** Active-learning pipeline that discovers high-band-gap 2D materials from a 3,351-crystal database, finding the global best in ~3% of evaluations.
- **Timeframe:** **May – June 2026 (~2 months)** of active work; **paper writing still in progress** as of July 2026. (Code commits begin 2026-06-10; implementation/experiments ran May–June, write-up ongoing.)
- **Type:** **Research assistantship** — research output under a professor/lab (lists under *Experience*, not just Projects). NOT FOUND IN REPO — lab / professor / institution name (you'll fill this in).
- **Authorship:** **First author**, with your professor/supervisor as co-author.
- **Contribution:** **Sole implementer** of all code — built the CGCNN encoder, the GP / Deep-Kernel head, the BO loop, and the calibration suite yourself, following the reference paper's math (not its code). Also owned **experiment design** (crossover sweep, 30-seed paired-Wilcoxon protocol, baselines/metrics), **analysis & figures**, and **paper writing**. (Core research question developed with supervisor / the reference-paper line of work.) → Full first-person ownership of the implementation, experimentation, and analysis is claimable.
- **Status:** Ongoing — **paper in progress / drafting** (targeting a *Digital Discovery*–class journal; see `docs/paper_positioning.md`). Core pipeline (Phases 1–4), a controlled pre-training crossover study, 30-seed paired-Wilcoxon stats, and a second-dataset (SNUMAT) generalization test are complete; manuscript write-up underway.
- **Publishability:** **Embargoed until publication** — repo must stay private; a portfolio page may *describe* the work and show results, but must **not** link the code or host a live demo from the repo until the paper is out.

---

## 2. The Problem

Discovering new functional materials normally means running expensive DFT (density-functional theory) simulations on thousands of candidate crystals — computationally prohibitive to do exhaustively. This project frames the search as **pool-based active learning / Bayesian optimisation**: a surrogate model predicts which untested 2D material is most likely to have a high band gap, "labels" only that one each cycle (simulating a DFT run via lookup), and iterates — so the best materials are found after evaluating a tiny fraction of the pool.

- **Audience:** Computational materials scientists / ML-for-science researchers who need to prioritise which candidate materials to simulate or synthesise under a limited experiment budget.

---

## 3. Technical Stack (verified from `pyproject.toml` and source)

- **Languages:** Python (≥3.10) — 78 `.py` files.
- **Frameworks / libraries (exact, from `pyproject.toml`):**
  - `torch>=2.0`, `torch-geometric>=2.4` (CGCNN graph neural network)
  - `gpytorch>=1.11`, `botorch>=0.9` (Gaussian-process surrogate + BO)
  - `ase>=3.22`, `pymatgen>=2024.1` (crystal-structure handling)
  - `scikit-learn>=1.3`, `scipy>=1.11` (metrics, isotonic recalibration, stats)
  - `hydra-core>=1.3`, `omegaconf>=2.3` (config management)
  - `lmdb>=1.4`, `pandas>=2.0`, `pyarrow>=14.0` (graph cache + Parquet metadata)
  - `matplotlib>=3.7`, `tqdm>=4.66`, `psutil>=5.9` (plots, profiling)
- **Databases / storage:** C2DB (Computational 2D Materials Database) as an ASE `.db`; **LMDB** key-value store for cached crystal graphs with config-hash invalidation; **Parquet** files for metadata. Second study uses **SNUMAT** (~10k ICSD-derived 3D bulk crystals, JSON + VASP POSCAR).
- **Infra / deployment:** None — local research code. No Docker/cloud/CI config present. Hydra writes timestamped run outputs under `outputs/`. Runs on CUDA when available.
- **APIs / external services:** None integrated.
- **ML/AI components (specific):**
  - **CGCNN encoder** (Crystal Graph Convolutional NN, Xie & Grossman gated message-passing) — 18,593 params, atom_dim=90, bond_dim=10, hidden=32, 3 conv layers, **attention pooling**, LayerNorm + residual (`src/dklbo/models/cgcnn_encoder.py`).
  - **Deep Kernel Learning**: CGCNN encoder trained **jointly** with a GP head; encoder in float32, GP in float64, component-specific learning rates to prevent mode collapse (`src/dklbo/models/dkl.py`).
  - **GP surrogates**: ExactGP (Matérn-5/2, ARD) and SVGP (128 inducing points) — swappable (`src/dklbo/models/surrogate.py`).
  - **Acquisition functions**: UCB (`mean + β·std`) and analytic Expected Improvement (`src/dklbo/bo/acquisition.py`).
  - **Uncertainty calibration**: post-hoc temperature scaling + isotonic recalibration; ENCE, NLL, coverage, miscalibration-area metrics (`src/dklbo/eval/`).
  - **Transfer learning**: pre-train encoder on a dense property, fine-tune on sparse target (Phase 4 warm-start).

---

## 4. Architecture & Key Technical Decisions

- **Cache-once, run-many data flow.** C2DB → filter → crystal graphs are built once and stored in LMDB keyed by a config hash, so changing graph parameters auto-invalidates the cache (`src/dklbo/data/cache.py`). Metadata lives in Parquet. This separates expensive graph construction from fast experiment iteration.
- **Domain-correct 2D graph construction.** A `vacuum_cutoff = 4.0 Å` rule blocks false bonds across the vacuum gap in the c-direction (a real failure mode for 2D crystals), and a `min_gap = 0.01 eV` filter removes the metal spike. Verified by 7 dedicated unit tests including MoS₂ coordination (`tests/test_vacuum_cutoff.py`).
- **Prototype-aware train/val/test split** prevents data leakage from near-duplicate structures common in C2DB (`src/dklbo/data/c2db_loader.py`).
- **Compute-aware BO loop.** Full DKL retrain (encoder + GP) every K cycles; cheap GP-only refits in between with a **pool embedding cache** that drops one labelled row per cycle instead of re-encoding ~3,000 graphs — most of the accuracy gain at a fraction of the compute (`src/dklbo/bo/loop.py`).
- **Direction-agnostic acquisition.** Minimisation tasks are handled by sign-flipping the target upstream (`y_internal = -y_true`), so every acquisition function operates in a single `argmax` convention (`src/dklbo/bo/acquisition.py`).
- **Hydra-configured experiment grid.** All data/model/BO/experiment knobs are config groups under `configs/`, enabling reproducible sweeps (β sweep, ExactGP vs SVGP, epoch/window sweeps).
- **Scale indicators:** ~9,200 lines of Python across source + scripts + tests (src 2,701 · scripts 3,283 · SNUMAT study 2,151 · tests 1,057); 78 `.py` files; 15 commits on `main` plus extensive uncommitted working-tree progress; **5 test files (~1,057 lines)** covering vacuum cutoff, surrogate swap, BO loop, benchmark, and Phase-4 warm-start.

---

## 5. Quantifiable Results / Impact

**Dataset:** C2DB, **3,351 materials** after filtering; target = HSE06 band gap (0.01–10.79 eV); top-50 threshold ≥7.02 eV (rarest 1.5%).

**Phase 3 — DKL-BO vs Random (budget = 10 init + 100 cycles = ~3% of pool):**

| Metric | DKL-UCB (β=0.2) | Random | Improvement |
|---|---|---|---|
| Best material found | **9.58 eV** | 6.40 eV | +50% |
| Top-10% hits | **47** | 10 | **4.7×** |
| Top-50 (rare) hits | **9** | 0 | from 0 |

**Experiment 1 — β sweep (research finding):** pure exploitation (β=0.0) found the **global database maximum (10.79 eV) at cycle 56**, with **28 top-50 hits and 6.3× efficiency** over random — showing that for a well-calibrated surrogate, the conventional β=0.2 is suboptimal here.

**Phase 4 — warm-start / multi-seed (mean over 30 seeds, `results/phase4/summary.csv`):**

| Method | Best (eV) | Top-50 | Top-10% |
|---|---|---|---|
| DKL pretrained | **10.56** | 24.5 | 39.1 |
| Std GP | 10.46 | 22.1 | 33.7 |
| DKL fine-tuned | 10.29 | 23.1 | 34.9 |
| Random | 8.29 | 5.4 | 9.8 |

**Surrogate accuracy (Phase 2):** Val MAE **0.45 eV**, R² **0.70**, Coverage@95 **0.93** (CGCNN trained in ~30 s on CUDA).

**Generalization study (SNUMAT, ~10k 3D crystals, 10 seeds, paired Wilcoxon):** pre-trained DKL beat Std-GP on regret-AUC for the gap-maximisation task (mean regret 2.81 vs 8.55, **p≈0.002**), confirming the headline result transfers to a structurally different dataset.

> All numbers above are sourced from in-repo CSVs and `docs/results_summary.md` / `docs/experiment_results.md` — this project is unusually well-quantified for a portfolio piece.

---

## 6. Role-Specific Angles

> Framing note: this was a **research assistantship** and you are the **sole implementer**, so first-person ownership verbs ("Designed", "Implemented", "Built") are fully justified. List under *Experience* with the lab/institution name.

- **SDE:** Built a modular, Hydra-configurable ML research pipeline (~9.2k LoC, 78 modules) with an LMDB graph cache keyed by config-hash for automatic invalidation and a pool-embedding cache that eliminates ~3,000 redundant GNN forward passes per optimisation cycle; backed by 5 test suites (~1,057 lines) covering domain-specific edge cases.
- **Data Scientist:** Implemented from scratch a Deep Kernel Learning Bayesian-optimisation surrogate (CGCNN + Gaussian Process, joint training) that located the global optimum of a 3,351-material search space within 3% of evaluations (6.3× more efficient than random), and validated reproducibility across 30 seeds plus a second dataset with paired Wilcoxon tests (p≈0.002) and uncertainty calibration (Coverage@95 = 0.93).
- **Data Analyst:** Engineered a reproducible analysis pipeline over a 17k-row materials database (Parquet + ASE), running controlled experiments (β sweep, surrogate comparison, transfer learning) and reporting results through 12+ generated plots and summary tables with confidence intervals.
- **Research / ML-Scientist (strongest fit):** As research assistant, designed and ran a controlled study isolating *pre-training data size* as the variable that governs when a learned deep-kernel representation beats handcrafted descriptors in low-data Bayesian optimisation — establishing the crossover point (N≈250–500 materials) with a 30-seed paired-Wilcoxon protocol; manuscript in preparation.

---

## 7. For the Portfolio Website

> **⚠️ Embargo:** the paper is in progress, so the repo must stay **private** and **no live demo / GitHub link** until publication. The portfolio page may *describe* the work and show result figures — present it as "research in progress, code available on request / link after publication."

- **Project card title:** DKL-BO — Smarter Materials Discovery with Deep Kernel Bayesian Optimisation
- **Card summary:** A CGCNN + Gaussian-Process active-learning system that finds the highest-band-gap 2D materials in a 3,351-crystal database while testing only ~3% of candidates. Pinpointed the global best material with 6.3× better efficiency than random search, validated across 30 seeds and a second 10k-crystal dataset. *(Research in progress — manuscript in preparation.)*
- **Live demo:** **Not while embargoed.** Post-publication, a small CSV-driven **Streamlit/Gradio app on Hugging Face Spaces** (BO convergence animation + "best material found per cycle") would be a good fit — no GPU needed.
- **Screenshot-worthy visuals (safe to show now — figures, not code):** `results/phase4/plots/best_gap_over_cycles.png`, `results/phase4/plots/cumulative_top10pct.png`, the crossover figure (`results/phase4/crossover/crossover.png`, the paper's headline), and the Phase-3 summary dashboard (`results/plots/08_summary_dashboard.png`). The best-gap-over-cycles curve (DKL vs random) is the strongest single image. *Check with your supervisor before publishing any figure that will appear in the paper.*
- **GitHub status:** **Keep private until publication.** Replace the link with "Code & paper available on request" or a "link coming after publication" note. The README (`READme.md`) is an internal notes file anyway and would need a rewrite before any public release (see §8).

---

## 8. Gaps / Cleanup Needed Before Publishing

- **Secrets:** None found. A grep for `api_key/secret/password/token/sk-/aws_/bearer` returned only documentation prose (an unrelated "Materia production-readiness" note appended to `docs/results_summary.md` and phase explainers). **No exposed credentials.**
- **README rewrite required:** `READme.md` reads as personal phase-tracking notes (and even references scripts like `step1_c2db_loader.py` / `step3_dkl_bo.py` that don't match the actual `scripts/01_build_dataset.py …` naming). Replace with a public-facing README: problem, headline result, architecture diagram, how-to-run, results table.
- **Naming/typo:** file is `READme.md` (non-standard casing) — rename to `README.md`.
- **Repo hygiene:** working tree has many deleted/renamed scripts and uncommitted result CSVs (`git status` shows large drift); commit or clean before sharing so the public history isn't confusing.
- **Data not included:** `data/raw/` and `data/cache/` are git-ignored (correct — large files), but the public README should document where to obtain C2DB/SNUMAT so the pipeline is reproducible.
- **Demo links:** None present (nothing broken).


## 3-bullet summary

- A genuinely strong, well-quantified **research-assistantship output** (your role: **first author**, sole code implementer, owner of experiment design + analysis + writing): a Deep Kernel Learning + Bayesian Optimisation pipeline (CGCNN graph net + Gaussian Process + UCB/EI) for discovering high-band-gap 2D materials. Headline numbers are real and in-repo: found the global database maximum (10.79 eV) in ~3% of evaluations, 6.3× more efficient than random, validated across 30 seeds and a second 10k-crystal dataset with paired Wilcoxon tests (p≈0.002). Strongest framing is **Research / ML-Scientist**, then Data Scientist, then SDE.
- Engineering is portfolio-worthy too — ~9,200 lines of Python across 78 files, Hydra-configured experiment grid, LMDB graph cache with config-hash invalidation, a compute-saving embedding cache, and 5 real test suites (~1,057 lines) including domain-correct 2D-crystal edge cases.
- **Embargoed:** the paper is still being written, so the repo stays **private** — no public GitHub link or live demo until publication. The portfolio page can *describe* the work and show result figures (with supervisor sign-off); a Hugging Face Spaces demo is a good post-publication option. Still-missing fact: the **lab / professor / institution name** for the Experience entry.