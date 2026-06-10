# DKL‑BO Pipeline for 2D Band‑Gap Discovery — Benchmarking & Phased Implementation Plan

**Prepared as a Research‑Software‑Engineering review of the reference brief**
**Primary goal:** predict / discover high‑band‑gap 2D materials in C2DB (~17,000 materials, ~8,000 with a band‑gap label).
**Reference method:** Deep Kernel Learning + Bayesian Optimisation (DKL‑BO): a CGCNN encoder feeding a Gaussian‑Process surrogate, driving an active‑learning loop.

This document grounds every recommendation in the three assigned papers:

- **[P1] Kiyohara & Kumagai (2025)**, *Bayesian Optimization with Gaussian Processes Assisted by Deep Learning for Material Designs*, J. Phys. Chem. Lett. 16, 5244. — the direct architectural template (CGCNN + GP, attention pooling, Matérn‑5/2, UCB).
- **[P2] Mamun, Yang & Yue (2026)**, *Deep graph kernel learning for material & atomic‑level uncertainty quantification in adsorption‑energy prediction*, Digital Discovery 5, 1568. — the scaling fix (Sparse Variational GP) and the calibration‑metric toolkit.
- **[P3] Lyu, Hu, Chuai & Chen (2023)**, *Efficient Bayesian Optimization with Deep Kernel Learning and Transformer Pre‑trained on Multiple Heterogeneous Datasets*, ICLR 2023. — the transfer / pre‑training strategy across heterogeneous properties.

---

## 0. Reframing the problem (read this first)

The brief is written as "discover materials". In practice, because all ~8,000 band‑gap values already exist in C2DB, this is **pool‑based active learning over a closed candidate set**, not open‑ended generation. That distinction matters for everything that follows:

- The "search space" is the C2DB pool itself. Each BO cycle the acquisition function ranks the *currently unlabelled* materials and "reveals" the best one (simulating a DFT calculation you already have the answer to).
- Because the true maximum band gap in the pool is **known**, the *best‑found‑value vs. cycle* curve and *cycles‑to‑target* are exactly measurable — this is what makes rigorous benchmarking possible (this is precisely how [P1] reports its results).
- "Predict band gap" and "discover high‑band‑gap material" are two different success criteria. **Benchmark them separately**: predictive accuracy/calibration of the surrogate (offline, no loop) *and* search efficiency of the loop (online). Conflating them is the most common evaluation mistake.

A second data‑specific point: the C2DB band‑gap distribution is **bimodal with a large spike at 0 eV** (metals/semimetals). A single GP regressing raw band gap across metals + semiconductors will be badly calibrated. Decide your stance early (Section 3, pitfall B).

---

## 1. Architecture Validation & Optimisation

### 1.1 Is the 4‑step design sound?

The sequential `step1 → step2 → step3 → step4` decomposition is conceptually correct and matches the data‑flow of [P1]. The problems are **not** in the logic but in three engineering seams that will bite you as soon as you run more than a handful of experiments.

### 1.2 Bottleneck 1 — Disk serialisation between steps

**Symptom.** The flat design writes CIFs to disk in step 2, re‑reads them, emits per‑graph `.pt` files, then step 3 re‑loads thousands of tiny files every single run. For 17k materials this is thousands of small‑file I/O operations and silent re‑computation on every experiment.

**Fix.**
- Build the graph cache **once** into a single sharded store rather than 17k loose `.pt` files. Use either a small number of `torch.save` shards keyed by material ID, or an **LMDB** store (this is exactly how OC20 in [P2] handles millions of graphs). Loose‑file overhead dominates at this scale.
- Persist metadata (id, formula, target value, split, n_atoms, n_edges, filter flags) in a **single Parquet/CSV table**, not embedded in each graph object.
- Add a **content hash** of the preprocessing config (cutoff radius, vacuum cutoff, atom‑feature JSON version). If the hash matches, skip rebuilding. This single change removes most "why did this take 20 minutes again" pain.

### 1.3 Bottleneck 2 — CPU↔GPU overhead inside the active‑learning loop (the real cost)

**Symptom.** This is the dominant runtime cost and the brief under‑states it. Every BO cycle, to score the acquisition function, you must push the **entire unlabelled pool** (up to ~8,000 graphs) through the CGCNN encoder to get embeddings, then through the GP. The naive loop re‑encodes all 8,000 graphs *every cycle*, moving data CPU→GPU repeatedly.

**Fix (three independent levers, combine them):**
1. **Cache pool embeddings.** Encoder weights only change when you *retrain*. Between retrains, the 8,000 pool embeddings are constant — compute them once per retrain, keep them resident on GPU (or pinned host memory), and reuse them across the acquisition step.
2. **Amortise encoder retraining.** [P1] retrains jointly each cycle, but you do **not** have to retrain the expensive CGCNN encoder every single cycle. Retrain the full DKL every *K* cycles (e.g. K=5) and only re‑fit the cheap GP head on the new point in between. Benchmark K as a hyperparameter — it trades search optimality against wall‑clock.
3. **Batch the pool forward pass** under `torch.no_grad()` with a fixed mini‑batch size, and keep graphs on the device in pinned memory. Never loop one graph at a time.

### 1.4 Bottleneck 3 — Exact GP scaling and the OOM cliff

**Symptom.** [P1] trains the GP on the *entire* labelled set with batch size = full dataset, because exact GP inference inverts the N×N kernel matrix — **O(N³) time, O(N²) memory** ([P2] states this explicitly). In a pool‑based loop the labelled set N *grows every cycle*. With small N (a few hundred) the exact GP is fine and is the gold standard for calibration. As N climbs into the thousands during long runs, you hit the OOM cliff the brief warns about.

**Fix — offer two surrogate modes behind one interface:**

| Mode | When to use | Cost | Source |
|---|---|---|---|
| **Exact GP** (Matérn‑5/2) | N ≲ 1–2k labelled points; calibration ground truth | O(N³) | [P1] |
| **SVGP** (inducing points) | N large / long runs / OOM risk | O(N·M²), M = #inducing | [P2] |

[P2]'s central architectural lesson is precisely this: the exact GP "requires storing the entire latent space in memory to construct the kernel matrix", so they swap in a **Sparse Variational GP** that approximates the latent space with a learned, compact set of **inducing points**, trained by maximising the **ELBO** (with KL regularisation). Inducing‑point count M (e.g. 64–512) caps memory regardless of how big N grows. Make the surrogate a pluggable component so you can A/B exact‑GP vs SVGP on identical splits.

### 1.5 Bottleneck 4 — Joint‑training instability (mode collapse)

[P2] warns that training a GNN+GP end‑to‑end suffers from **mode collapse and conflicting optimisation dynamics** between the deep encoder and the GP head. Two cheap, high‑value mitigations from [P2]:
- **Per‑layer feature normalisation** in the encoder so embeddings don't drift in scale (a GP kernel is extremely sensitive to input scaling).
- **Component‑specific learning rates** — a smaller LR for the encoder than for the GP/kernel hyperparameters. ([P1] uses a single Adam LR of 0.01; treat that as a starting point, not a law.)

### 1.6 Reference hyperparameters to anchor your defaults (from [P1])

So you are not guessing: [P1]'s CGCNN uses atom‑feature dim **90**, bond‑feature dim **10**, post‑conv node dim **32**, **3** convolution layers, **1** pooling layer, **1** fully‑connected layer, **attention pooling** (not average — average "dilutes information depending on the number of atoms in the unit cell", which is exactly the variable‑cell problem in C2DB), **Matérn‑5/2** kernel, Adam **lr 0.01**, optimise by maximising **marginal log‑likelihood**. For pre‑training they used MSE loss and a pre‑train batch = ⅓ of the dataset. These are sane defaults; put them in config, not in code.

---

## 2. Step‑by‑Step Benchmarking Plan

**Golden rule (from [P1], which ran 50 trials):** every BO result must be reported as a **distribution over ≥20–50 random seeds**, not a single run. Initial‑set randomness causes huge run‑to‑run variance; single‑run "convergence curves" are not evidence. Report median + IQR (box plots), as [P1] does.

### Phase 1 — Data loading (`step1` equivalent)

| Metric | Why | Target / check |
|---|---|---|
| Records loaded / filter pass‑rate | Sanity that ASE read worked | matches expected ~17k |
| Label coverage | Confirms the 8,000/17,000 reality | ~47% have `gap` |
| Zero‑gap fraction | The metal spike (pitfall B) | quantify before modelling |
| Split sizes & leakage check | No formula/prototype overlap across splits | 0 overlap |
| Target distribution (hist) | Detect bimodality / skew | inspect visually |

### Phase 2 — Graph construction (`step2` equivalent)

| Metric | Why |
|---|---|
| **Graph‑construction throughput** (graphs/sec) | The benchmark the brief asks for; isolates preprocessing cost |
| Avg / max nodes & edges per graph | Memory budgeting for batching |
| **Fraction of zero‑edge or disconnected graphs** | Catches a *too‑strict* vacuum cutoff silently deleting all bonds — a 2D‑specific failure |
| Edge‑length histogram vs 4.0 Å vacuum cutoff | **Validates the 2D adaptation**: confirm no edges bridge the vacuum gap in c, and coordination numbers match known 2D prototypes (e.g. MoS₂ ≈ 6) |
| Cache size on disk, peak RAM | Capacity planning |

> The vacuum‑aware neighbour search is the single most important *correctness* check in the whole pipeline. A bug here produces plausible‑looking graphs with garbage connectivity and the model will "work" while learning nonsense. Build an explicit unit test on 3–5 known structures.

### Phase 3 — DKL surrogate, **offline** (predictive quality, *no loop yet*)

Train DKL on a fixed train/val/test split and measure surrogate quality before ever running BO. Split predictive vs. calibration metrics:

**Accuracy:** MAE, RMSE, R² on held‑out test (these are [P2]'s accuracy axis).
**Calibration / UQ ([P2]'s toolkit — this is what separates a *Bayesian* model from a regressor):**

| Metric | What it tells you |
|---|---|
| **NLL** (negative log‑likelihood) | Joint accuracy + calibration |
| **ENCE** (expected normalised calibration error) | Are predicted error bars the right size? [P2] target 0.06–0.10 |
| **Miscalibration area** | Area between reliability curve and ideal; [P2] 0.04–0.07 |
| **Spearman(error, uncertainty)** | Does the model know *when* it's wrong? [P2] up to 0.51 |
| **Coverage** of 95% intervals | Should be ≈95% |

**GP/DKL training convergence (the brief's "GP training convergence speed"):**
- MLL/ELBO vs. epoch; epochs‑to‑plateau.
- Wall‑clock per training, **CPU vs CUDA** (brief notes 2–5 min/cycle CPU, 5–10× faster on GPU — *verify this on your hardware, don't assume it*).

### Phase 4 — BO loop, **online** (search efficiency)

| Metric | Definition | Source |
|---|---|---|
| **Best‑found band gap vs cycle** | The canonical convergence curve | [P1] |
| **Cycles to find top‑1 / top‑5 / top‑1% material** | Concrete discovery success | [P1] |
| **Enhancement factor vs random** | best‑found(DKL) / best‑found(random) at fixed budget | [P1] |
| **Cumulative / simple regret** | max_pool − best_found | standard BO |
| **Peak GPU memory vs cycle index** | *The OOM curve the brief worries about* — plot it explicitly | [P2] |
| **Wall‑clock per cycle vs cycle index** | Detects super‑linear blow‑up of exact GP | [P2] |

### Phase 5 — Baselines (non‑negotiable)

Run all on **identical seeds/splits**:

1. **Random search** — the floor. [P1] shows random needs ≈ N/2 cycles; any method not clearly beating this is broken. *Always include this.*
2. **Standard GP on hand‑crafted descriptors** (matminer features → GP). This is [P1]'s `std‑GP` baseline and the headline comparison ("does the learned representation beat hand‑crafted descriptors?"). Note [P1]'s nuance: when a hand‑crafted descriptor is *strongly correlated* with the target, std‑GP can *win* — so report honestly.
3. **Greedy / exploitation‑only** (pure predicted mean, no uncertainty) — isolates the value of the UQ.
4. (Optional) **Deep ensemble UQ** — [P2]'s strongest accuracy baseline, useful if you want to defend the GP's calibration advantage.

### Acquisition‑function sweep

[P1] tested UCB(β=1.0), PI, and EI and found the best choice is task‑dependent (≈10–20% swing in cycles). [P3] uses **LCB with β=3** (LCB for *minimisation* = UCB for *maximisation*). Benchmark **UCB β ∈ {0.5, 1, 2, 3}** as a first‑class axis, not an afterthought.

---

## 3. Robust Handling of Common Pitfalls

### Pitfall A — OOM in GPyTorch/BoTorch in later cycles

Root causes, in order of impact, and concrete mitigations:

1. **Re‑encoding the whole pool on GPU every cycle** (Section 1.3). Cache embeddings; batch under `torch.no_grad()`; move the pool to CPU between acquisitions.
2. **Exact‑GP kernel matrix growing with N** (Section 1.4). Switch to **SVGP with M inducing points** ([P2]) to cap memory; or use GPyTorch's structured/lazy kernels and fast predictive variances; the brief's own hint — `gpytorch` interpolation approximations — is the same idea.
3. **Reduce `N_INIT`.** [P1]'s finding is counter‑intuitive but strong: **smaller initial sets are better**, and a large N_init wastes the BO budget. So shrinking N_init both saves memory *and* improves efficiency. Win‑win.
4. **Hygiene every cycle:** `del` large tensors, `gc.collect()`, `torch.cuda.empty_cache()`; cap the candidate batch size; keep the GP in `float64` (stability) but the encoder forward in `float32`.

Add an automated guard: log peak memory each cycle and **fail loudly with a clear message** ("OOM at cycle N, labelled set = X; switch surrogate=svgp or lower n_init") rather than a raw CUDA stack trace.

### Pitfall B — The metal / zero‑gap spike (C2DB‑specific, *not* in the brief)

Your ~8,000 band‑gap labels include a large pile of 0 eV metals. A single GP over raw gap is heteroscedastic and poorly calibrated. Three defensible options — **pick one and document it**:

- **(a) Filter to gap > 0** and state the model only addresses semiconductors. Simplest, cleanest for a "high‑band‑gap discovery" objective.
- **(b) Two‑stage**: a metal/non‑metal classifier, then a regressor on non‑metals. Most accurate, more moving parts.
- **(c) Model raw gap** but report calibration *separately* for the zero and non‑zero populations so the spike doesn't hide miscalibration.

For a *maximise band gap* objective, (a) is usually the right call and removes a whole class of artefacts.

### Pitfall C — Severe sparsity for `gap_gw` / `eps_ionic` (< 500 entries)

This is where [P3] and the brief's `PRETRAIN=True` come in. Note clearly: **this does NOT apply to your primary band‑gap target (8,000 points is plenty).** It applies to the sparse properties.

- **Transfer learning**: pre‑train the CGCNN encoder on abundant `Eform` (~4,000), then transfer to the sparse target — [P1] confirms this helps even though formation energy seems uncorrelated with band gap, and [P3] is built entirely on this "pre‑train on heterogeneous datasets, transfer to new task" principle.
- **[P1]'s critical caveat — pre‑training can *hurt* on noisy data.** Their experimental (noisy) datasets got *worse* with `PRETRAIN=True`, because pre‑training assumes noise‑free targets and then overfits the wrong distribution. So: treat **noise as a learnable parameter**, and for noisy/experimental targets prefer **larger weight decay** and *no* pre‑training. C2DB DFT gaps are relatively clean, so pre‑training is more likely to help here than it would on experimental data.
- **Partial transfer**: freeze lower encoder layers, fine‑tune the top + GP head — cheaper and less prone to catastrophic forgetting than re‑optimising everything.
- [P3]'s **mix‑up initialisation** is relevant only if you transfer across genuinely different *input feature spaces*; for same‑graph/different‑property transfer within C2DB it's not needed.

### Pitfall D — Data leakage across splits

2D databases contain near‑duplicate prototypes (same structure, different decoration). Random splitting leaks information and inflates test metrics. **Split by structural prototype / composition family, not by random row.** Verify zero cross‑split overlap in Phase 1.

---

## 4. Flexible, Production‑Ready Repository Layout

Replace the flat `step1…step4.py` scripts with a proper `src/` package + Hydra configs + separate eval entry points. This keeps experiments reproducible and lets you sweep configs without editing code.

```text
dkl-bo-c2db/
├── README.md
├── pyproject.toml                # pinned deps: torch>=2.0, ase>=3.22, pymatgen>=2024,
│                                 # gpytorch>=1.11, botorch, pandas, scikit-learn ...
├── configs/                      # Hydra config groups (compose, don't edit code)
│   ├── config.yaml               # top-level defaults
│   ├── data/
│   │   ├── c2db_gap.yaml         # target=gap, filters, vacuum_cutoff=4.0, radius
│   │   └── c2db_gap_gw.yaml      # sparse target → pretrain=true
│   ├── model/
│   │   ├── cgcnn.yaml            # atom_dim=90, bond_dim=10, n_conv=3, pooling=attention
│   │   ├── gp_exact.yaml         # Matern 5/2, exact
│   │   └── gp_svgp.yaml          # inducing points M, ELBO  (scaling mode, [P2])
│   ├── bo/
│   │   └── ucb.yaml              # acquisition=ucb, beta, n_init, n_cycles, retrain_every_K
│   └── experiment/               # full named experiments for the paper
│       ├── gap_dkl_vs_random.yaml
│       └── gap_dkl_vs_stdgp.yaml
├── src/dklbo/
│   ├── data/
│   │   ├── c2db_loader.py        # ASE db → metadata table + split (prototype-aware)
│   │   ├── graph_builder.py      # vacuum-aware neighbour search (the 2D core)
│   │   └── cache.py              # LMDB / sharded store + config-hash invalidation
│   ├── models/
│   │   ├── cgcnn_encoder.py      # encoder w/ per-layer norm + attention pooling
│   │   ├── surrogate.py          # ABC: ExactGPSurrogate | SVGPSurrogate (swap freely)
│   │   └── dkl.py                # joint encoder+GP, component-specific LRs
│   ├── bo/
│   │   ├── acquisition.py        # UCB/EI/PI/LCB
│   │   ├── loop.py               # pool-based AL loop, embedding cache, mem guard
│   │   └── baselines.py          # random search, std-GP, greedy
│   ├── eval/
│   │   ├── metrics_accuracy.py   # MAE/RMSE/R2
│   │   ├── metrics_calibration.py# NLL, ENCE, miscalibration area, Spearman, coverage
│   │   └── metrics_search.py     # best-found, cycles-to-target, regret, mem/time curves
│   └── utils/
│       ├── seed.py               # full determinism control
│       └── profiling.py          # per-cycle peak-mem + wall-clock logging
├── scripts/                      # thin CLIs — all logic lives in src/
│   ├── 01_build_cache.py
│   ├── 02_eval_surrogate.py      # OFFLINE predictive+calibration (Phase 3)
│   ├── 03_run_bo.py              # ONLINE loop (Phase 4), reads configs/experiment/*
│   └── 04_analyze.py             # convergence curves, Pareto (UQ vs cost, [P2])
├── tests/
│   ├── test_vacuum_cutoff.py     # no edges across the c-vacuum gap (correctness!)
│   └── test_surrogate_swap.py    # exact-GP and SVGP obey the same interface
└── results/                      # seeded runs, one dir per experiment (git-ignored)
```

**Why this is better than the flat scripts:**
- **Config‑driven, not code‑driven.** Changing target property, switching exact‑GP ↔ SVGP, or sweeping β is a CLI override (`model=gp_svgp bo.beta=2`), never an edit to a `step3.py` constant. This is what makes a 50‑seed × 4‑acquisition × 2‑surrogate benchmark tractable.
- **Surrogate behind an ABC** so the exact‑GP→SVGP migration (Section 1.4) is a one‑line config change, and your benchmark compares them on identical splits.
- **Offline eval (`02`) is separate from the loop (`03`)** — enforcing the Section‑2 rule that predictive quality and search efficiency are measured independently.
- **`tests/` encodes the two correctness traps** (vacuum cutoff, surrogate interface) so they can't silently regress.

---

## 5. Recommended Phasing (what to actually do, in order)

| Phase | Goal | Done when |
|---|---|---|
| **0. Env + data audit** | Reproduce brief env; quantify label coverage & the zero‑gap spike | You can state exact #materials, #with gap, %metals, and have prototype‑aware splits |
| **1. Preprocessing + graph cache** | Build the cache once; validate vacuum cutoff | Throughput measured; `test_vacuum_cutoff` passes; coordination numbers sane |
| **2. Offline surrogate validation** | DKL predictive + calibration on band gap, **no loop** | MAE/R² + full [P2] calibration suite reported; exact‑GP vs SVGP compared |
| **3. BO on band gap + baselines** | The headline result | DKL vs Random vs std‑GP over ≥20 seeds, box plots, mem/time curves |
| **4. Scaling + transfer** | SVGP for long runs; `Eform→gap_gw` transfer | OOM curve flat under SVGP; transfer benchmarked with [P1]'s pre‑training caveat tested |
| **5. Analysis + write‑up** | Convergence, Pareto (UQ vs cost), top discoveries | Figures reproduce from `results/` via `04_analyze.py` |

**The one‑sentence summary for your professor:** keep [P1]'s architecture as the default, but make the GP head swappable so [P2]'s SVGP can be dropped in when the exact GP hits the memory cliff, benchmark predictive‑calibration and search‑efficiency *separately* over many seeds against a random‑search and a hand‑crafted‑descriptor std‑GP baseline, and treat the C2DB zero‑gap metal spike as a first‑class modelling decision rather than ignoring it.