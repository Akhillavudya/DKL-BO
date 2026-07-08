# DKL-BO — Paper Source Document (Rebuild Phase)

> **Scope.** This document is built **only** from the clean *rebuild* of the project
> (scripts `01_`–`07_` + `exp_*` extensions, `results/rebuild/`, `src/dklbo/`,
> `data/cache/`). The `_archive_old/` folder (old pipeline, different numbers) is
> deliberately excluded. The separate `snumat_generalization/` folder is a
> second-dataset replication study and is only pointed to, not merged in here.
>
> **Evidence labels used throughout:**
> **MEASURED** = value read from a results/log/code artifact (path given).
> **CLAIMED** = asserted in docs/comments but not backed by a saved artifact.
> **MISSING** = a standard paper element that does not exist in the repo.
>
> **Critical provenance caveat (read first).** The rebuild is **not committed to git**.
> `git log` contains only the *old* pipeline (`35521ab … d11b40f`, 2026‑06‑10 → 06‑13),
> and `READme.md` documents that old pipeline (scripts `step1…step4`, 3351 materials),
> **not** this rebuild. Every rebuild claim below is traceable to a working-tree file,
> not to a commit. Timeline for the rebuild therefore comes from file mtimes / run logs,
> not commit history.

---

## 0. Paper Framing

**Working titles (options):**
1. *Learned vs. Handcrafted Representations for Bayesian Optimisation of 2D Materials: A Controlled 30-Seed Benchmark on C2DB.*
2. *When Does Deep Kernel Learning Beat a Descriptor GP? Task-Dependent Gains in Active Discovery of Band Gap and Effective Mass.*
3. *Search ≠ Accuracy: An Honest Head-to-Head of DKL-BO and Standard GP-BO for 2D Semiconductor Discovery.*

**Research domain / subfield:** Machine learning for materials discovery; Bayesian
optimisation / active learning; graph neural networks for crystals (surrogate modelling).

**Target venue/format:** TBD (fits a materials-informatics or ML-for-science workshop/journal;
e.g. *npj Computational Materials*, *Digital Discovery*, or an ML4Materials workshop).

**Draft elevator abstract (~150 words, only what exists):**
> Deep Kernel Learning Bayesian Optimisation (DKL-BO) couples a crystal graph neural
> network (CGCNN) encoder to a Gaussian-Process surrogate to guide the discovery of 2D
> materials. Whether its *learned* representation actually beats classical *handcrafted*
> descriptors is usually asserted rather than tested under matched conditions. We build a
> single intersection dataset of 2,667 C2DB monolayers with both band gap and electron
> effective mass, hold out a prototype-disjoint pool of 766, and run a controlled contest:
> Random vs. a 43-descriptor Standard GP vs. DKL (pre-trained/frozen, fine-tuned, and
> cold) — all sharing the identical pool, Expected-Improvement acquisition, ExactGP
> backend, and per-seed initial sets, over 100 cycles × 30 seeds with paired Wilcoxon
> tests. Pre-trained DKL significantly widens discovery of high-band-gap materials
> (top-10% hits 42.7 vs 29.8, p≈2×10⁻⁶) and of constrained-window targets, but not the
> single best material; effective-mass tasks are statistically tied. Pre-training is
> necessary — cold DKL collapses.

**Candidate research questions / hypotheses:**
1. Under strictly matched search conditions, does a learned CGCNN representation
   outperform handcrafted composition/geometry descriptors for BO-driven discovery — and
   for *which* objectives (max/min gap, max/min effective mass, target windows)?
2. Is any DKL advantage attributable to *pre-training* (representation transfer) rather
   than to the acquisition/optimiser? (Tested via a cold, from-scratch control.)
3. Does predictive accuracy (R²/MAE) predict *search* success, or can a method with lower
   accuracy still discover more top materials (search ≠ accuracy)?

---

## 1. Contributions

1. **A strictly-controlled representation benchmark for materials BO** — same held-out pool,
   same ExactGP backend, same EI acquisition, same per-seed init sets; the *only* variable
   is how a material is described (43 handcrafted descriptors vs 32-d learned embedding vs
   none). *Contribution type: experimental methodology / engineering.* Backed by
   `scripts/04_run_bo.py`, `src/dklbo/baselines/feature_bo_loop.py`.
2. **Honest, statistically-tested, task-resolved findings (30 seeds, paired Wilcoxon +
   bootstrap CIs).** The DKL advantage is *not* universal: it is significant for **band-gap
   maximisation discovery breadth** and **constrained-window breadth**, absent/reversed for
   single-champion and effective-mass tasks. *Type: scientific finding (incl. negative
   results).* Backed by `results/rebuild/summary_stats.csv`, `stats_pairs.csv`,
   `winmax_stats.csv`, `winmin_stats.csv`.
3. **A pre-training necessity result via a cold control.** A random-init encoder trained
   live (`--cold`) collapses on band gap (final_best p≈6×10⁻⁴ worse than Std-GP), isolating
   *representation transfer* as the causal ingredient. Backed by
   `scripts/05_run_bo_finetune.py`, `stats_pairs.csv` (`dkl_cold_live` rows).
4. **A "search ≠ accuracy" demonstration.** On effective mass *both* methods have negative
   R² (pointwise prediction fails), yet BO still harvests top-k materials; on band gap
   DKL is slightly *less* accurate (R² 0.55 vs 0.60) yet discovers more rare materials.
   Backed by `results/rebuild/accuracy.csv` vs `summary_stats.csv`.
5. **Two objective-shaping extensions** rarely benchmarked together: (a) heavy-tailed target
   handling via `log10(effective mass)`; (b) *constrained-window* acquisitions
   (`window`, `window_max`, `window_min`) for target-band search, with closed-form partial
   expectations. Backed by `src/dklbo/bo/acquisition.py`, `scripts/exp_window*_bo.py`.

**Novelty honesty.** The *method* (CGCNN+GP DKL-BO, Matérn-5/2, attention pooling, warm-start
fine-tuning) is adopted from prior work ([P1] Kiyohara & Kumagai; [P2] Mamun/Yang/Yue), not
invented here. The defensible novelty is the **controlled comparison + honest negative/mixed
results + the constrained-window and log-emass extensions on C2DB**. To strengthen novelty:
add a second dataset (the `snumat_generalization/` study), multi-property/multi-objective
search, and/or an analysis of *why* the learned embedding helps only on gap-max.

---

## 2. Problem & Motivation

- **Problem.** Discovering 2D semiconductors with target electronic properties (high band
  gap, low/high electron effective mass, or a gap inside a device-relevant window) requires
  expensive DFT. Active learning / BO chooses which few materials to evaluate. The surrogate's
  *representation* of a crystal is the key design choice.
- **What's hard.** (i) Properties are heterogeneous — band gap is smooth-ish; effective mass
  is heavy-tailed over ~5 orders of magnitude. (ii) Near-duplicate crystal prototypes cause
  train/test leakage. (iii) The community often reports DKL wins without matched baselines,
  so it is unclear whether gains come from the learned representation or from incidental
  differences in acquisition/tuning.
- **Gap addressed.** A like-for-like, multi-seed, significance-tested comparison of learned
  vs handcrafted representations under identical BO machinery, with a cold control to
  attribute any gain to pre-training.
- **Significance/impact.** Guidance on *when* DKL is worth its complexity vs a cheap
  descriptor GP; an honest map of task-dependent behaviour; reusable constrained-window
  acquisitions for device-targeted search.

---

## 3. Background & Related Work Signals

**Built on / competes with (inferred from code, imports, comments):**
- **CGCNN** — Crystal Graph Convolutional NN; gated message passing exactly per Xie &
  Grossman (2018) (`src/dklbo/models/cgcnn_encoder.py` docstring cites it). [CITATION NEEDED: Xie & Grossman, PRL 2018]
- **Deep Kernel Learning** — jointly-trained NN feature extractor + GP kernel
  (`src/dklbo/models/dkl.py`). [CITATION NEEDED: Wilson et al., AISTATS 2016]
- **Gaussian Processes / Matérn-5/2 kernel, ARD** (`src/dklbo/models/surrogate.py`, GPyTorch).
  [CITATION NEEDED: Rasmussen & Williams 2006]
- **Expected Improvement** acquisition (`src/dklbo/bo/acquisition.py`).
  [CITATION NEEDED: Jones, Schonlau & Welch 1998; Mockus]
- **SVGP / inducing-point GP** (implemented but not used in reported rebuild results; scaling
  fix from [P2]). [CITATION NEEDED: Hensman et al. 2013/2015]
- **Magpie-style composition descriptors** via pymatgen (`src/dklbo/baselines/descriptors.py`).
  [CITATION NEEDED: Ward et al. 2016 (Magpie); Ong et al. 2013 (pymatgen)]
- **C2DB dataset** (`data/raw/c2db.db`, ASE db; fields `gap_hse`, `emass_cbm`, `hform`,
  `ehull`, `layergroup`, …). [CITATION NEEDED: Haastrup et al. 2018; Gjerding et al. 2021 (C2DB)]
- **Frameworks:** PyTorch, PyTorch Geometric, GPyTorch, BoTorch, ASE, scikit-learn, SciPy.
  [CITATION NEEDED for each: Paszke 2019; Fey & Lenssen 2019; Gardner 2018; Balandat 2020;
  Larsen 2017 (ASE)]

**Project's own reference tags (from README/comments), to be resolved to full citations:**
- **[P1] Kiyohara & Kumagai (2025)** — the DKL-BO template (CGCNN+GP, attention pooling,
  Matérn-5/2, UCB; Std-GP vs DKL; no validation set in BO). [CITATION NEEDED — verify authors/year/venue]
- **[P2] Mamun, Yang & Yue (2026)** — SVGP scaling + calibration metrics + LayerNorm-for-DKL.
  [CITATION NEEDED — verify]
- **[P3] Lyu et al. (2023)** — transfer learning (pretrain → fine-tune on sparse target).
  [CITATION NEEDED — verify]

**Key concepts a reader needs:** crystal graph representation; gated graph convolution;
attention pooling; GP surrogate with Matérn-5/2 + ARD; marginal likelihood training;
Expected Improvement; simple regret; prototype-aware (leakage-free) splitting; heavy-tailed
target transformation; constrained/level-set acquisition (partial expectation over a band).

---

## 4. Methodology / Approach

### 4.1 Pipeline architecture (data/control flow; file per component)

```
data/raw/c2db.db  (ASE database, ~C2DB monolayers)
      │  scripts/01_build_dataset.py  +  src/dklbo/baselines/descriptors.py
      ▼
data/cache/master.parquet         2,667 materials w/ gap AND emass, prototype-aware split
data/cache/descriptors.parquet    2,667 × 43 handcrafted features (Std-GP)
data/cache/graphs_e98e27ea.lmdb   crystal graphs (REUSED, structure-only; radius 8, vac 4.0)
      │
      ├── scripts/02_pretrain_encoder.py  (CGCNN encoder + ExactGP, per target gap/emass_log)
      │        → results/rebuild/encoder_{gap,emass_log}.pt, embeddings_{...}.parquet (2667×32)
      │        components: src/dklbo/models/{cgcnn_encoder,dkl,surrogate}.py, data/{cache,dataset}.py
      │
      ├── scripts/03_eval_accuracy.py     (ExactGP on train→pool; MAE/RMSE/R²/cov95)
      │        → results/rebuild/accuracy.csv, plots/accuracy.png
      │        (+ scripts/exp_epoch_sweep.py → epoch_sweep.csv: overfitting ablation)
      │
      ├── scripts/04_run_bo.py            (Std-GP, frozen-DKL, Random; EI; 30 seeds×100 cyc)
      │        engine: src/dklbo/baselines/feature_bo_loop.py, bo/baselines.py, bo/acquisition.py
      │        → results/rebuild/runs/*.csv (360 files), bo_summary.csv
      │
      ├── scripts/05_run_bo_finetune.py   (warm-start fine-tune DKL + cold control)
      │        engine: src/dklbo/bo/loop.py (real BOLoop, retrain_every_k=5)
      │        → runs_finetune/*.csv (120), runs_coldlive/*.csv (120), bo_finetune_summary.csv
      │
      ├── scripts/06_stats.py             (bootstrap 95% CI + paired Wilcoxon vs Std-GP)
      │        → per_run_metrics.csv, summary_stats.csv, stats_pairs.csv
      │
      ├── scripts/07_plot.py              (curves + bars per task) → plots/, plots_30seed/
      │
      └── scripts/exp_window*_bo.py       (constrained-window / target-band search extensions)
               → winmax_*.csv, winmin_*.csv, window_*.csv, plots/win*.png
```

### 4.2 Core algorithms (pseudocode, with source)

**(a) Prototype-aware, gap-stratified split** — `scripts/01_build_dataset.py:add_train_pool_split`
```
group materials by structural prototype  (layergroup_spacegroup)
stratum(prototype) = gap-quartile of its max gap        # N_STRATA = 4
for each stratum:
    shuffle prototypes with rng(SEED=42)
    move ceil(POOL_FRAC=0.30 * n_protos) prototypes → POOL
assert no prototype appears in both train and pool       # verify_no_split_leakage
```

**(b) Standard GP-BO over static features** — `src/dklbo/baselines/feature_bo_loop.py:run`
```
standardize X (z-score over all candidates)              # features are cheap/known, not leakage
y_internal = sign * y_true                                # sign=+1 max, -1 min
labelled ← random.sample(all_uids, n_init=10)  using seed
for cycle in 1..100:
    fit ExactGP(ARD Matérn-5/2) on (X[labelled], y_internal[labelled])   # 100 epochs, early stop
    mean,std ← GP.predict(X[pool])
    scores ← EI(mean, std, best_f=incumbent, xi=0.01)
    pick argmax(scores); reveal oracle value; update incumbent, top50/top10% counters
    move picked uid: pool → labelled
```

**(c) DKL-BO with live fine-tuning / cold** — `src/dklbo/bo/loop.py:run` + `models/dkl.py:fit`
```
encoder ← warm-start from encoder_{target}.pt   (cold: random init)
labelled ← random.sample(all_uids, n_init=10)  using seed
for cycle in 0..99:
    if cycle % retrain_every_k(=5) == 0:
        DKL.fit(labelled_graphs, y_internal):               # joint encoder+GP
            Phase A: GP warmup 20 epochs on frozen embeddings
            Phase B: 50 epochs joint; Adam groups {encoder_lr=1e-3, gp_lr=1e-2}
            final: standardize embeddings, refit GP
        invalidate embedding cache
    else:
        refit GP only on cached labelled embeddings (50 epochs)   # encoder frozen
    embeddings ← encode(pool)  (cached between full retrains)
    mean,std ← GP.predict(standardize(embeddings)); std_cal = std * τ(=1.0)
    scores ← EI(mean, std_cal, best_f=incumbent, xi=0.01)
    pick argmax; reveal oracle; update; move pool→labelled
```
*Frozen DKL* (Phase 3, method `dkl`) skips the encoder entirely: it feeds the **pre-computed**
`embeddings_{target}.parquet` into `FeatureBOLoop` (algorithm b), i.e. a static learned
feature matrix. So `dkl_frozen` and `std_gp` differ *only* in the feature matrix.

**(d) Gated CGCNN convolution** — `src/dklbo/models/cgcnn_encoder.py:CGCNNConv`
```
m_ij = sigmoid(W_f · [h_i, h_j, e_ij]) ⊙ softplus(W_s · [h_i, h_j, e_ij])
h_i' = LayerNorm(h_i + Σ_{j∈N(i)} m_ij)
```
Encoder = embed(90→32) → 3× conv → attention pool (softmax within graph) → 1× FC(softplus) → 32-d.

### 4.3 Mathematical formulation (LaTeX-ready)

**GP surrogate.** ExactGP with constant mean $m$, scaled Matérn-5/2 kernel, Gaussian noise:
$$k(x,x') = \sigma_f^2\left(1+\sqrt{5}r+\tfrac{5}{3}r^2\right)\exp(-\sqrt5\,r),\quad
r=\sqrt{\textstyle\sum_{d=1}^{D}\frac{(x_d-x'_d)^2}{\ell_d^2}}$$
ARD gives one lengthscale $\ell_d$ per input dim (`ard_num_dims=D`). Hyperparameters
$\{m,\sigma_f,\{\ell_d\},\sigma_n^2\}$ fit by maximising the exact marginal log-likelihood
(minimising $-\mathrm{MLL}$) with Adam (`surrogate.py:fit`).

**Deep kernel.** $k_{\text{DKL}}(g,g') = k\big(\phi_\theta(g),\phi_\theta(g')\big)$, with
$\phi_\theta$ the CGCNN encoder; $\theta$ and GP hyperparameters trained jointly on
$-\mathrm{MLL}$ (`dkl.py:fit`).

**Expected Improvement** (maximise convention; `acquisition.py:ei`):
$$\mathrm{EI}(x) = (\mu-f^*-\xi)\,\Phi(z) + \sigma\,\phi(z),\qquad z=\frac{\mu-f^*-\xi}{\sigma},\quad \xi=0.01,$$
$\mathrm{EI}=0$ where $\sigma=0$. Minimisation uses $y_{\text{int}}=-y$.

**Constrained-window "partial expectation"** (`acquisition.py:window_max`), with
$\alpha=(w_{lo}-\mu)/\sigma$, $\beta=(w_{hi}-\mu)/\sigma$:
$$a_{\max}(x)=\mathbb{E}\!\left[y\,\mathbf{1}\{w_{lo}\le y\le w_{hi}\}\right]
=\mu\big(\Phi(\beta)-\Phi(\alpha)\big)-\sigma\big(\phi(\beta)-\phi(\alpha)\big)\;(+\,\beta_{\text{ucb}}\sigma P_{\text{in}}).$$
Window-min rewards headroom $w_{hi}-y$; window-probability uses $P_{\text{in}}=\Phi(\beta)-\Phi(\alpha)$.

**Metrics** (`06_stats.py`): simple regret $r_t=\lvert y^*_{\text{pool}}-\text{best}_t\rvert$;
$\text{regret\_auc}=\frac1T\sum_t r_t$; `final_top50`/`final_top10pct` = cumulative count of
acquired materials whose internal value clears the pool's top-50 / top-10% threshold.
Accuracy: MAE, RMSE, $R^2$, and Coverage@95 $=\frac1N\sum \mathbf{1}\{|y-\mu|\le 1.96\sigma\}$.

### 4.4 Key design decisions & justification (from code/comments)
- **Single intersection dataset (gap ∧ emass).** Guarantees every study/method searches the
  identical material set (`01_build_dataset.py` docstring).
- **Prototype-aware + gap-stratified split.** Prevents near-duplicate leakage while ensuring
  rare high-gap materials exist in the pool to be discovered.
- **Fairness by construction.** Std-GP and frozen-DKL share GP backend, acquisition, init
  sets, and standardisation; only the feature matrix differs.
- **Descriptors deliberately exclude DFT electronic outputs** (`descriptors.py:EXCLUDED_LEAKAGE_FIELDS`)
  so the descriptor GP cannot "cheat" with quantities correlated to the target.
- **`log10(emass)`** for the heavy-tailed effective mass (range 0.0012–136); min-emass ≡
  min-log-emass (same materials).
- **ExactGP only** in reported results (pool N≤776 ≪ SVGP threshold); SVGP implemented but unused.
- **Component-specific LRs + LayerNorm + embedding standardisation** to prevent DKL mode collapse.
- **Attention pooling** (not mean) for variable C2DB unit-cell sizes.

---

## 5. Implementation Details (reproducibility)

**Language / runtime:** Python 3.13.9 (`venv`); package `dklbo` v0.1.0, `requires-python>=3.10`
(`pyproject.toml`).

**Exact library versions (MEASURED, `venv/bin/pip freeze`):**
| Library | Version | Library | Version |
|---|---|---|---|
| torch | 2.12.0 | gpytorch | 1.15.2 |
| torch-geometric | 2.8.0 | botorch | 0.18.1 |
| numpy | 2.4.6 | scipy | 1.17.1 |
| pandas | 3.0.3 | scikit-learn | 1.9.0 |
| ase | 3.28.0 | pymatgen | 2026.5.4 (core 2026.5.18) |
| lmdb | 2.2.1 | matplotlib | 3.10.9 |
| omegaconf | 2.3.0 | hydra-core | 1.3.2 |

> Note: installed versions **exceed** the `pyproject.toml` floors (`torch>=2.0`, etc.). The
> paper should pin the *installed* versions above for reproducibility.

**Model / architecture (MEASURED, `configs/model/cgcnn.yaml`, `02_pretrain_encoder.py`):**
CGCNN encoder — atom_dim 90, bond_dim 10, hidden_dim 32, n_conv 3, n_fc 1, pooling=attention,
LayerNorm+residual, softplus activations, `spectral_norm=false`. Output embedding dim 32.
GP head — ExactGP, ScaleKernel(Matérn ν=2.5), ConstantMean, GaussianLikelihood.

**Graph construction (MEASURED, `GRAPH_PREPROC` in scripts):** radius 8.0 Å, vacuum_cutoff 4.0 Å,
max_neighbors 12, target gap_hse, gap_min 0.01. Cache `graphs_e98e27ea.lmdb` reused (structure-only).

**Full hyperparameter list (MEASURED):**
| Group | Params |
|---|---|
| Split | SEED=42, POOL_FRAC=0.30, N_STRATA=4, GAP_MIN=0.01 |
| Encoder pretrain (`02`) | 100 epochs joint, gp_pretrain 50, gp_final 300, encoder_lr 1e-3, gp_lr 1e-2, GP head ard=False, standardize=True; emass uses `--log` (log10) |
| Accuracy GP (`03`) | ExactGP ARD=True, 200 epochs, train-standardized features & target |
| BO (`ei.yaml` / `04`,`05`) | acquisition=EI, xi=0.01, n_init=10, n_cycles=100, **30 seeds** (seeds 0–29), FeatureBOLoop GP: ard=True, 100 epochs/cycle |
| DKL live loop (`05`) | retrain_every_k=5, n_joint_epochs=50, n_pretrain_epochs=20, gp_refit_epochs=50, GP head ard=False |
| GP fit | Adam lr=0.01, early-stop patience=20, warm_start=True, float64 inference |
| Stats (`06`) | bootstrap n_boot=10000 (seed 0), paired Wilcoxon signed-rank |

**Hardware (MEASURED):** NVIDIA RTX A4000, 16 GB (`nvidia-smi`); run logs show `device=cuda`.
CPU/RAM not recorded.

**Seeds / determinism:** `src/dklbo/utils/seed.py:seed_everything` seeds Python/NumPy/PyTorch;
called per BO run with the run seed. Split uses fixed SEED=42. Encoder pretrain uses SEED=42.
*Note:* only the BO **init set** varies across the 30 seeds; the **data split and pre-trained
encoders are single fixed realizations** (see Gaps §13).

**Setup to reproduce:**
```
pip install -e .                                   # or use venv/
python scripts/01_build_dataset.py --db_path data/raw/c2db.db
python scripts/02_pretrain_encoder.py --target gap
python scripts/02_pretrain_encoder.py --target emass --log
python scripts/03_eval_accuracy.py
python scripts/04_run_bo.py --seeds 30 --cycles 100
python scripts/05_run_bo_finetune.py --seeds 30 --cycles 100
python scripts/05_run_bo_finetune.py --seeds 30 --cycles 100 --cold
python scripts/06_stats.py
python scripts/07_plot.py --outdir results/rebuild/plots_30seed
# extensions:
python scripts/exp_epoch_sweep.py ; python scripts/exp_window_max_bo.py ; python scripts/exp_window_min_bo.py
```
> Reproducibility risk: `04`/`05` default to `--seeds 10`; the reported results are 30 seeds
> (confirmed by `summary_stats.csv:n_seeds=30` and 360/120/120 run files). Pass `--seeds 30`.

---

## 6. Experimental Setup

**Dataset (MEASURED, `data/cache/master.parquet`):**
- Source: C2DB (`data/raw/c2db.db`, ASE). Kept only materials with valid `gap_hse` > 0.01 eV
  **and** valid finite `emass_cbm` > 0 (intersection).
- Size: **2,667 materials**, across **60 structural prototypes**.
- Split: **train 1,901 / pool 766** (prototype-disjoint, gap-stratified).
- Ranges: gap 0.010–10.792 eV (median 2.166; **pool** max 8.681 — the 10.79 champion sits in
  train); emass 0.0012–136.19 m₀ (median 0.810; pool log10 range −2.80…+2.13).
- Preprocessing: metals dropped (gap≤0.01); crystal graphs via vacuum-aware builder; 43
  handcrafted descriptors (35 composition + 8 structural), NaNs mean-imputed, z-scored on train.
- License: C2DB terms [CITATION NEEDED / verify license].

**Baselines / comparison methods:**
- **Random** (no model) — floor. `src/dklbo/bo/baselines.py`.
- **Standard GP** — 43 descriptors + ExactGP ARD. `feature_bo_loop.py` + `descriptors.py`.
- **DKL-frozen** — pre-trained 32-d embeddings (static). **DKL-finetune** — warm-start,
  live joint retrain. **DKL-cold** — random-init, live (control). `05_run_bo_finetune.py`.
- *No comparison to other GNN surrogates or literature systems* (see Gaps).

**Evaluation metrics:** (search) regret-AUC, final_best, final_top50, final_top10pct;
(accuracy) MAE, RMSE, R², Coverage@95; (constrained) best in-window gap, cumulative in-window top-k.

**Protocol:** 4 search tasks {gap_max, gap_min, emass_min, emass_max} × 5 methods × 100
cycles × 30 seeds (n_init=10). EI throughout. Paired across seeds (shared init) → paired
Wilcoxon vs Std-GP; 95% bootstrap CIs (10,000 resamples). Accuracy = single train→pool fit.

---

## 7. Results (MEASURED unless noted; source paths given)

### 7.1 Prediction accuracy — held-out pool (766) — `results/rebuild/accuracy.csv`
| Target | Method | MAE | RMSE | R² | Cov@95 |
|---|---|---|---|---|---|
| gap | Std-GP | 0.745 | 0.995 | **0.604** | 0.948 |
| gap | DKL | 0.759 | 1.065 | 0.546 | 0.898 |
| emass (raw) | Std-GP | 1.861 | 6.167 | **−0.099** | 0.967 |
| emass (raw) | DKL | 1.962 | 6.029 | −0.050 | 0.943 |

`emass_scale_compare.csv`: in **log10** space R²_log = 0.125 (Std-GP) vs **0.136 (DKL)** —
both weak, DKL marginally better. **Takeaway:** Std-GP is *more* accurate on gap and better
calibrated; effective mass is essentially unpredictable pointwise for both (R²≤0).

### 7.2 BO search — 30 seeds — `summary_stats.csv` (mean) + `stats_pairs.csv` (paired Wilcoxon vs Std-GP)
final_top10pct (higher better), final_best, regret-AUC (lower better). ★ = p<0.05.

| Task | Method | top10% | top50 | best | regret-AUC | vs Std-GP (key p) |
|---|---|---|---|---|---|---|
| **gap_max** | Std-GP | 29.8 | 22.0 | 8.678 | 0.714 | — |
| | **DKL-frozen** | **42.7** | **31.5** | 8.569 | 0.853 | top10% p=1.7e‑6 ★win; top50 p=2.3e‑6 ★win; best p=1.8e‑3 ★**lose** |
| | DKL-finetune | 35.4 | 25.3 | 8.400 | 1.095 | top10% p=7.6e‑4 ★win; best p=0.017 ★lose |
| | DKL-cold | 30.2 | 20.3 | 7.854 | 1.588 | best p=5.9e‑4 ★lose; top-k tie → pretrain needed |
| | Random | 10.0 | 6.8 | 7.500 | 2.012 | all ★lose |
| **gap_min** | Std-GP | 27.8 | 19.1 | 0.0242 | 0.0778 | — |
| | DKL-frozen | 28.4 | 20.6 | 0.0248 | 0.0628 | all tie (top10% p=0.69) |
| | DKL-finetune | 26.1 | 17.0 | 0.0278 | 0.0784 | top50 p=0.028 ★lose |
| | DKL-cold | 24.8 | 15.5 | 0.0320 | 0.0998 | best p=5.0e‑3 ★lose; top50 p=2.7e‑3 ★lose |
| **emass_min** (log10) | Std-GP | 21.2 | 15.9 | −1.971 | 1.213 | — |
| | DKL-frozen | 21.0 | 15.7 | −1.744 | 1.344 | all tie |
| | DKL-finetune | 21.5 | 16.7 | **−2.103** | 1.100 | all tie (best p=0.53) |
| | DKL-cold | 19.3 | 14.7 | −1.987 | 1.211 | all tie |
| **emass_max** (log10) | Std-GP | **19.5** | 14.6 | **1.656** | 0.694 | — |
| | DKL-frozen | 18.0 | 13.3 | 1.500 | 0.823 | regret p=0.036 ★lose; top-k tie |
| | DKL-finetune | 18.7 | 13.9 | 1.544 | 0.784 | all tie |
| | DKL-cold | 15.9 | 11.4 | 1.494 | 0.821 | best p=0.023 ★lose; top10% p=0.036 ★lose |

**Headline (statistically clean, 30 seeds):** DKL's representation gives a **significant edge
only for band-gap-maximisation discovery breadth** (top-50 & top-10%), and that edge
**requires pre-training** (cold collapses). For the **single best** gap-max material, **Std-GP
is significantly better**. gap-min and both effective-mass tasks are **statistically tied**
with descriptors; Std-GP is nominally best on emass_max.

### 7.3 Constrained-window search — 30 seeds — `winmax_stats.csv`, `winmin_stats.csv`
Window 0.7–3.0 eV. cumulative in-window top-50 (higher better):
| Objective | Method | in-win top50 | vs Std-GP |
|---|---|---|---|
| window-**max** | Std-GP 8.33 / **DKL-finetune 11.63** / DKL-frozen 7.43 / Random 6.97 | | finetune p=5.5e‑4 ★win |
| window-**min** | Std-GP 9.93 / **DKL-finetune 16.13** / DKL-frozen 12.4 / Random 6.37 | | finetune p=2.1e‑5 ★win; frozen p=0.016 ★win |

**New finding:** on *constrained/target-window* discovery breadth, **live fine-tuned DKL wins
significantly** (both directions) — a DKL advantage beyond plain gap-max. Best-in-window gap
values are all near the constrained optimum (~2.99 / ~0.70) and differ only marginally.
The **window-probability** variant (`window_stats.csv`) was run at **only 2 seeds** →
underpowered, not significant (report as preliminary or MISSING).

### 7.4 Ablation — encoder training budget — `results/rebuild/epoch_sweep.csv` (`exp_epoch_sweep.py`)
DKL train-R² rises to ~0.995 by 300–400 epochs while **pool-R² peaks early (~0.63 at ~50
epochs) then plateaus/declines (~0.58–0.62)** and never clearly beats Std-GP pool-R² 0.604.
**Takeaway:** more encoder epochs → overfitting, not better generalisation (justifies the
modest 100-epoch pretrain).

### 7.5 Variance / error bars
**MEASURED:** 95% bootstrap CIs for every (task,method) in `summary_stats.csv`; paired Wilcoxon
p-values + win-rates in `stats_pairs.csv` (n=30). Per-run values in `per_run_metrics.csv`.

### 7.6 Qualitative
Discovery curves + bars exist as figures (§8). Champion-material identities are **not saved in a
table** (acquired uids are in per-run `runs/*.csv` `uid` column but not aggregated) → MISSING as
a clean artifact.

---

## 8. Figures & Tables Plan

**Already exist (`results/rebuild/plots/` and `plots_30seed/`):**
- `accuracy.png` — Std-GP vs DKL MAE/RMSE/R², gap & emass. *(from accuracy.csv)*
- `{gap_max,gap_min,emass_min,emass_max}_curves.png` — best-so-far & cumulative top-10% vs
  cycle, seed-mean ± 95% CI (the 30-seed set is in `plots_30seed/`).
- `{task}_bars.png` — final best/top50/top10% bars.
- `epoch_sweep_gap.png` — overfitting ablation.
- `winmax_*.png`, `winmin_*.png`, `window_*` — constrained-window results + crossover plots.

**Need to generate:**
- **Architecture / pipeline diagram** (schematic; none exists) — from §4.1.
- **Main results table** (Table: task × method × {top10%, top50, best, regret, p}) — from
  `summary_stats.csv` + `stats_pairs.csv`.
- **Prototype-split / dataset schematic** (2,667 → train 1,901 / pool 766).
- **Champion-materials table** (top discovered formulas per task) — must be aggregated from
  `runs/*.csv` (script to write).
- Optional: calibration / reliability figure (coverage exists in `accuracy.csv`; full
  calibration curve not saved).

---

## 9. Discussion & Limitations

**What the results support:** (1) Under matched conditions, a *pre-trained* learned
representation significantly improves **discovery breadth for band-gap maximisation** and for
**constrained-window** targets. (2) The gain is **causally tied to pre-training** (cold
control fails). (3) **Search ≠ accuracy**: DKL is slightly less accurate on gap yet discovers
more; emass is unpredictable pointwise yet still searchable. (4) For **single-champion** gap
and for **effective-mass** objectives, a cheap descriptor GP is as good or better.

**What they do NOT support:** a blanket "DKL beats Std-GP" claim; any emass advantage
(all tied); any single-best-material advantage (Std-GP wins gap-max best).

**Honest limitations / threats to validity:**
- **Single dataset split.** One prototype-disjoint realization (SEED=42) over only **60
  prototype groups** (~18 pool groups). Seeds vary only the BO *init set*, not the split or
  the pre-trained encoder → results are conditional on one split; no split-level CI.
- **Single accuracy fit.** `accuracy.csv` is one train→pool fit (no seed variation / CI).
- **Scale.** Pool = 766, budget = 100 cycles; small by industrial standards; ExactGP-only.
- **Effective mass unpredictable** (R²≤0) — search claims there rest on ranking, and are tied.
- **Window-probability extension underpowered** (2 seeds).
- **Descriptor design choices** (which fields, ordinal encodings, mean-imputation) affect the
  Std-GP baseline strength; not sensitivity-tested.
- **Provenance:** rebuild uncommitted; README describes a different pipeline (confusion risk).
- **Compute fairness not quantified** (per-cycle timings exist per-run but no aggregate
  wall-clock comparison of DKL vs Std-GP).

---

## 10. Future Work
- Repeat across **multiple prototype splits** (and report split-level CIs) to remove the
  single-split threat — the highest-value next experiment.
- **Second dataset** generalisation (the `snumat_generalization/` study: 3D ICSD crystals,
  gap-only) to test whether the gap-max breadth win transfers.
- **Multi-objective / combined** gap+emass search (dataset already supports it).
- Aggregate **compute-vs-quality** trade-off (DKL fine-tune cost vs Std-GP).
- Investigate *why* the learned embedding helps only on gap-max (embedding-space analysis of
  the high-gap cluster).
- Add stronger baselines (other descriptor sets/GNN surrogates); recalibration (τ) sweep.

---

## 11. Reproducibility Checklist
| Item | Status | Note |
|---|---|---|
| Code runnable | **Partial** | Scripts run, but **uncommitted**; README is for the old pipeline |
| Seeds set | **Yes** | `seed_everything`; SEED=42 split; BO seeds 0–29 |
| Single command for main results | **No** | Multi-step; and defaults are `--seeds 10` not 30 |
| Data available | **Partial** | `c2db.db` present locally; external availability/license TBD |
| Trained weights available | **Yes (local)** | `encoder_gap.pt`, `encoder_emass_log.pt` in `results/rebuild/` |
| Results artifacts saved | **Yes** | run CSVs (600 files), summary/stats CSVs, plots |
| Determinism across split/encoder | **Partial** | Only BO init varies over seeds; split & encoder are single fixed runs |
| Environment pinned | **Partial** | `pyproject.toml` has floors; exact versions only via `pip freeze` (listed §5) |

---

## 12. Citation-Worthy Dependencies
| Item | What it is | Where used |
|---|---|---|
| C2DB | 2D materials DFT database (gap_hse, emass_cbm, …) | `data/raw/c2db.db`, all data |
| CGCNN (Xie & Grossman 2018) | Crystal graph conv net; gated message passing | `models/cgcnn_encoder.py` |
| Deep Kernel Learning (Wilson 2016) | joint NN+GP kernel | `models/dkl.py` |
| Matérn-5/2 GP + ARD (Rasmussen & Williams) | surrogate kernel | `models/surrogate.py` |
| Expected Improvement (Jones 1998) | acquisition | `bo/acquisition.py` |
| SVGP / inducing points (Hensman 2015) | scalable GP (implemented, unused) | `models/surrogate.py` |
| Magpie descriptors (Ward 2016) | composition statistics | `baselines/descriptors.py` |
| pymatgen (Ong 2013) | Composition/Element props | `baselines/descriptors.py` |
| ASE (Larsen 2017) | read C2DB db, structures | `data/*`, `descriptors.py` |
| PyTorch (Paszke 2019) | tensors/autograd | everywhere |
| PyTorch Geometric (Fey 2019) | graph batching / message passing | encoder, loaders |
| GPyTorch (Gardner 2018) | GP implementation | `surrogate.py` |
| BoTorch (Balandat 2020) | (dependency; BO utilities) | env (usage to confirm) |
| scikit-learn / SciPy | wilcoxon, utils | `06_stats.py` |
| LMDB | graph cache store | `data/cache.py` |
| Project refs [P1] Kiyohara & Kumagai; [P2] Mamun/Yang/Yue; [P3] Lyu et al. | method template, scaling/calibration, transfer | README/comments — **verify full citations** |

---

## 13. GAPS FOR PAPER-READINESS (ranked)

1. **[ESSENTIAL] Single train/pool split (one SEED=42, 60 prototypes).** Seeds vary only BO
   init, so all results are conditional on one split. **Run ≥5–10 independent prototype
   splits** and report split-level variability. *Without this, the significance is
   overstated.*
2. **[ESSENTIAL] Accuracy has no error bars.** `accuracy.csv` is a single fit. Repeat over
   seeds/splits for MAE/RMSE/R² CIs.
3. **[HIGH] Confirm the 30-seed provenance & re-pin defaults.** Reported numbers are 30 seeds
   (`summary_stats.csv`), but scripts default to 10; ensure the paper's stated protocol
   matches the artifacts and set defaults to 30.
4. **[HIGH] Commit the rebuild & fix the README.** Currently uncommitted; README documents a
   *different* pipeline — a reviewer running the repo would be misled.
5. **[MEDIUM] Window-probability extension underpowered (2 seeds).** Either scale to 30 seeds
   or drop from the paper; constrained window-max/min are already at 30 seeds (keep).
6. **[MEDIUM] No compute/wall-clock comparison.** Aggregate the per-cycle `train_time_s` to
   quantify DKL-finetune cost vs Std-GP.
7. **[MEDIUM] Baseline breadth.** Only Random + one descriptor GP. Consider a second
   descriptor set or GNN surrogate to strengthen "learned > handcrafted" claims.
8. **[MEDIUM] Champion-material tables/qualitative results not aggregated** (only raw uids in
   run CSVs).
9. **[LOW] Descriptor sensitivity** (encoding/imputation choices) untested.
10. **[LOW] Generalisation to a second dataset** exists separately (`snumat_generalization/`);
    fold in to broaden claims.

---

## 14. Evidence Appendix (claim → source)

| Claim | Source (path / function / artifact) | Label |
|---|---|---|
| 2,667 materials, train 1901 / pool 766, 60 prototypes | `data/cache/master.parquet` (verified via pandas) | MEASURED |
| gap 0.01–10.79 eV; emass 0.0012–136.19; pool gap max 8.681 | `master.parquet` | MEASURED |
| 43 handcrafted descriptors (35 comp + 8 struct) | `data/cache/descriptors.parquet` (44 cols −uid); `baselines/descriptors.py` | MEASURED |
| Split algorithm (proto-aware, gap-stratified, POOL_FRAC 0.30, SEED 42) | `scripts/01_build_dataset.py:add_train_pool_split` | MEASURED (code) |
| CGCNN arch (32-d, 3 conv, attn pool, LayerNorm) | `configs/model/cgcnn.yaml`, `models/cgcnn_encoder.py` | MEASURED (code) |
| ExactGP Matérn-5/2 + ARD, MLL, Adam, patience 20 | `models/surrogate.py` | MEASURED (code) |
| EI formula + xi=0.01; window partial-expectation acquisitions | `bo/acquisition.py` | MEASURED (code) |
| BO protocol: EI, n_init 10, 100 cycles, retrain_every_k 5 | `configs/bo/ei.yaml`, `scripts/04`,`05` | MEASURED (code) |
| **30 seeds** used (not 10) | `summary_stats.csv:n_seeds=30`; run-file counts 360/120/120 | MEASURED |
| Accuracy table (gap R² 0.604 vs 0.546; emass R²≤0; R²_log 0.125/0.136) | `results/rebuild/accuracy.csv`, `emass_scale_compare.csv` | MEASURED |
| gap_max: DKL-frozen top10% 42.7 vs 29.8, p=1.7e‑6; best p=1.8e‑3 (lose) | `summary_stats.csv`, `stats_pairs.csv` | MEASURED |
| Cold DKL collapses (gap_max best p=5.9e‑4 worse) | `stats_pairs.csv` (dkl_cold_live) | MEASURED |
| emass & gap_min tasks tied vs Std-GP | `stats_pairs.csv` | MEASURED |
| Constrained window: finetune wins top50 (max p=5.5e‑4, min p=2.1e‑5) | `winmax_stats.csv`, `winmin_stats.csv` | MEASURED |
| Epoch-sweep overfitting (pool-R² plateaus ~0.6) | `epoch_sweep.csv` | MEASURED |
| Hardware NVIDIA RTX A4000 16 GB; device=cuda | `nvidia-smi`; `*_run.log` | MEASURED |
| Library versions (torch 2.12.0, gpytorch 1.15.2, …) | `venv` pip freeze | MEASURED |
| "DKL wins because ranking not accuracy" narrative | comments/docstrings | CLAIMED (supported by §7.1 vs §7.2 contrast) |
| [P1]/[P2]/[P3] specific author/year/venue | `READme.md`, comments | CLAIMED — needs verification |
| C2DB total ~17,000 / ~8,000 gap-labelled | `READme.md` (old-pipeline doc) | CLAIMED — not re-verified for rebuild |
| Multiple-split robustness; accuracy CIs; wall-clock comparison | — | MISSING |
| Champion-material aggregate table; full calibration curves | — | MISSING (raw data in run CSVs) |
