# DKL-BO for 2D Materials — Full Project Explanation (Rebuild)

**C2DB → CGCNN crystal graphs → Deep Kernel Learning → Bayesian Optimisation**

*One clean, self-contained walkthrough of the whole rebuilt project: the motivation, the
key innovation, the reference paper we follow, every step we took, how we cached the graphs,
every question we asked, every experiment we ran, and — above all — **where and why DKL wins.***

*Written for a reader starting fresh. Companion to the per-phase notes
(`rebuild_phase1..5_explanation.md`, `window_min/max_explanation.md`). Where this document and
the original planning PDF (`dkl_bo_2d_pipeline.pdf`) disagree on a number, **this document
reflects the code we actually ran**; the PDF was the up-front design sketch.*

---

## 1. What the project is and why it exists

In materials science, measuring one property of one material — say the band gap of a new 2D
crystal — means running a **DFT (Density Functional Theory) calculation** that can cost hours
to days of compute, or a physical experiment that costs far more. We cannot afford to test
every candidate. We want to find the **best** materials while testing as **few** as possible.

This project builds and tests a search engine for exactly that job:

> Predict a material's property *cheaply* from its crystal structure, use that prediction —
> **and its uncertainty** — to choose the single most promising material to test next, test it,
> learn from the answer, and repeat. This loop is **Bayesian Optimisation (BO)**.

We use the **C2DB** (Computational 2D Materials Database), where ~2,000–4,000 2D materials
already have DFT-computed properties. Because the answers are already known, we can *simulate*
the discovery campaign honestly: we hide the property values, let the algorithm request them
one at a time, and measure how quickly it finds the rare, valuable materials. This turns an
open-ended scientific search into a **reproducible, measurable benchmark**.

**Properties we search for** (the "treasures"):

| Target | Meaning | Why it matters |
|---|---|---|
| **Band gap** (`gap`, HSE06) | energy gap between filled and empty electron states | sets whether a material is a metal, semiconductor or insulator → electronics, optics, photovoltaics |
| **Effective mass** (`emass`) | how "heavy" charge carriers behave | low effective mass → high carrier mobility → faster transistors |

For each property we run the search in **both directions** (maximise and minimise), because
different applications want different extremes.

---

## 2. The key innovation (the heart of the whole project)

A standard Gaussian-Process Bayesian Optimiser (**Std-GP-BO**) needs a human to hand it
**descriptors** — a fixed list of numbers describing each material (composition statistics,
geometry, etc.). The quality of the whole search is then capped by how good those handcrafted
numbers happen to be for the property you care about.

> **The key innovation of Deep Kernel Learning (DKL) over standard GP-BO is this: instead of
> requiring handcrafted descriptors, a Crystal Graph Convolutional Neural Network (CGCNN)
> *automatically learns* the feature representation directly from the atomic structure, and it
> learns those features jointly with the Gaussian Process that uses them.** The network is
> trained to produce exactly the embedding that makes the GP predict and quantify uncertainty
> best. This matters most for 2D materials, where the property depends on complex, non-linear
> structural factors that no fixed list of descriptors captures cleanly.

In one line: **Std-GP reads a recipe card someone else wrote; DKL looks at the raw ingredients
and learns what matters.** The entire project is a fair, statistically-tested contest between
these two philosophies — plus a **Random** floor that both must beat — to find *where the
learned representation actually wins, and where it does not.*

---

## 3. The reference paper — what it covered, what we took, what we added

**Kiyohara, S. & Kumagai, Y. — "Bayesian Optimization with Gaussian Processes Assisted by Deep
Learning for Material Designs", *J. Phys. Chem. Lett.* 2025, 16, 5244–5251.** (Code:
`github.com/skiyohara/dklbo`.) This is our direct template; our architecture reproduces theirs.

### 3.1 What the paper covered

- **Model:** DKL = CGCNN encoder + GP with a **Matérn-5/2** kernel, **attention pooling**,
  atom features `a`=90, bond features `u`=10, crystal embedding `v`=32, 3 convolution layers —
  trained **jointly** by maximising the GP marginal log-likelihood. (Their Figure 1.)
- **Contest:** **DKL-BO vs Std-GP vs Random**, acquisition = **UCB** with β=0.2, small start
  `n_init`=10.
- **Std-GP features:** 170 matminer descriptors filtered by random-forest importance down to
  the top `nfea` (3/10/20) each cycle. `nfea` matters a lot for Std-GP, barely for DKL.
- **Tasks (922 oxides):** max band gap, max ionic dielectric constant, **min electron effective
  mass**, and a band-gap-*range* task (find oxides with gap 2.8–3.2 eV). Plus noisy experimental
  datasets (perovskite gaps, alloy Curie temperatures).
- **Headline results we expect to echo:**
  - DKL is **~2× more sample-efficient** than Std-GP for band gap and dielectric constant.
  - DKL's **biggest win is minimum effective mass** (~50 cycles vs Std-GP's 371) — *despite a
    poor pointwise R²*. This is their famous lesson: **"even a less accurate predictive model
    can still effectively identify target materials" → search ≠ accuracy.**
  - **Std-GP wins** when a single descriptor strongly correlates with the target (e.g. Curie
    temperature ∝ Co content) and for the narrow band-gap-*range* task.
  - DKL is **robust to hyperparameters** (a practical advantage).
  - **Pretraining / transfer learning** (e.g. pretrain the encoder on abundant formation
    energy, then adapt) helps on clean computational data, but is *detrimental* on noisy
    experimental data.
- **Validation-set practice:** **no validation set inside the BO loop** — GP parameters are fit
  purely by maximising marginal log-likelihood on the acquired data. Their prediction-accuracy
  figure (their Fig. 5) is computed separately by **5-fold cross-validation**.

### 3.2 What we adopted vs. what we added

| Aspect | Paper (Kiyohara & Kumagai) | Our rebuild |
|---|---|---|
| Database | 922 oxides (3D) | **C2DB 2D materials** (vacuum-aware graphs) |
| Encoder / kernel / pooling | CGCNN + Matérn-5/2 + attention | **same** |
| Methods | DKL-BO, Std-GP, Random | **same three** |
| Acquisition | UCB (β=0.2) | **EI (Expected Improvement)** for the main contest (parameter-free → fairer); window-constrained acquisitions for extensions |
| Targets | gap, dielectric, **min emass**, gap-range | **gap (max/min) + emass (max/min)**, on **one shared dataset**; plus gap-window tasks |
| Validation inside BO | none | **none** (we follow them); a tiny val split is carved from *train* only for encoder pretraining |
| Statistics | efficiency curves | **paired Wilcoxon tests + bootstrap 95% CIs over 10 and 30 seeds** (added rigour) |
| Pretraining study | yes (formation energy) | **frozen vs fine-tuned vs cold-start** ablation |

**What we add over the paper:** (1) the 2D vacuum-aware graph construction; (2) a single shared
"intersection" dataset so all studies are directly comparable; (3) formal significance testing
so we never overclaim; (4) a clean **frozen / fine-tuned / cold** pretraining ablation; and (5)
new **gap-window** search tasks (find materials inside a target band).

### 3.3 Which plots are important (from the paper, and for us)

The paper's persuasive figures — and therefore the ones our write-up must contain — are:

1. **Model architecture diagram** (their Fig. 1): CGCNN encoder → embedding → GP. One picture
   that explains the whole method. *(We reproduce this conceptually.)*
2. **Convergence curve — best-found value vs. BO cycle**, DKL vs Std-GP vs Random. This is
   *the* money plot: it shows sample-efficiency (who reaches good materials in fewer tests).
   → our `{task}_curves.png` (top panel).
3. **Discovery-breadth curve — cumulative count of rare (top-k) materials vs. cycle.** Shows who
   *harvests many* good materials, not just the single best. → our `{task}_curves.png` (lower
   panel, cumulative top-10%).
4. **Prediction-accuracy plot** (their Fig. 5, via cross-validation): parity / R² per method.
   Used to make the "search ≠ accuracy" point. → our `accuracy.png`.
5. **Final bar chart** — best / top-50 / top-10% per method at the end of the budget. The
   podium. → our `{task}_bars.png`.

**For our story specifically, the three plots that carry the argument are:** the **gap_max
curves** (DKL's clean breadth win), the **accuracy plot** (emass R²≈0 for everyone — sets up
search≠accuracy), and the **window-min plots** (DKL's strongest, cleanest all-round win).
Everything is shown with **95% confidence-interval bands** so the reader sees signal vs. noise.

---

## 4. Why we rebuilt, and the clean design

The first pass at this project worked but grew messy: 21 numbered scripts, many competing BO
variants, ~14 loosely-related plots, and several different data subsets — making "did DKL win?"
hard to answer cleanly. The **rebuild** keeps the proven engine (`src/dklbo/`) but throws away
the tangle and re-runs the science on **one fair playground** with **one clear question**.

### 4.1 The 3×3 grid (the whole project on one page)

**Three methods:**
- **Random** — picks the next material at random. The floor everyone must beat.
- **Std-GP** — GP on 43 handcrafted descriptors (the traditional approach).
- **DKL-BO** — GP on a CGCNN-learned 32-dim embedding (the key innovation).

**Three studies:**

| Study | Search tasks | Prediction check |
|---|---|---|
| A. Band gap | maximise gap, minimise gap | gap accuracy: Std-GP vs DKL |
| B. Effective mass | maximise emass, minimise emass | emass accuracy: Std-GP vs DKL |
| C. Combined / windowed | gap inside a target window (max/min variants) | (uses A & B models) |

### 4.2 The single most important design decision: the intersection dataset

C2DB gives a band gap for ~3,351 non-metal materials, but a valid effective mass for only a
subset (~2,667). We keep **only the materials that have BOTH** a valid gap and a valid
emass → **2,667 materials**. Reasons:

- A **combined gap+emass** study can only use materials that have both numbers, so we are forced
  into the intersection anyway.
- If all three studies search the **identical** 2,667-material pool with the identical split,
  their results are directly comparable — no apples-vs-oranges. This single choice is the
  biggest cure for the old project's messiness.

Cost: we drop ~684 gap-only materials. Worth it for one clean, shared arena.

### 4.3 The fairness rule (the backbone)

For the contest to mean anything, **Std-GP's descriptors and DKL's graphs must describe the
exact same materials in the exact same order**, both must use the **same GP engine** and the
**same acquisition rule**, and for a given random seed both get the **same starting materials**.
Then any difference in results is caused *only by the features* — handcrafted vs learned —
which is precisely the question. Phase 1 ends by *verifying* this alignment before anything else
runs. The descriptors deliberately **exclude** electronic DFT outputs (like the gap itself) so
the traditional chef cannot cheat by reading the answer.

---

## 5. The engine: technical architecture

```
Crystal structure (atoms + bonds, from C2DB)
        │
        ▼
CGCNN encoder            src/dklbo/models/cgcnn_encoder.py
  • atom features: 90-dim one-hot (element Z)
  • bond features: 10-dim Gaussian basis of bond distance
  • 3 graph-convolution layers, hidden dim 32
  • attention pooling  → one 32-dim crystal "fingerprint"
        │  (32-dim embedding)
        ▼
Gaussian Process         src/dklbo/models/surrogate.py
  • ExactGP, Matérn-5/2 kernel, ARD (per-feature length scales)
  • outputs mean μ (predicted property) + std σ (uncertainty)
        │
        ▼
Acquisition function     src/dklbo/bo/acquisition.py
  • Expected Improvement (main contest); UCB / window variants available
        │
        ▼
BO loop                  src/dklbo/bo/loop.py
  pick best-scoring material → reveal its true value (oracle lookup)
  → add to training set → retrain → repeat
```

**Std-GP uses the *same* GP and acquisition** — it simply replaces the 32-dim learned embedding
with the 43 handcrafted descriptors. **DKL = encoder + this GP, trained jointly.** That joint
training is what makes the embedding "GP-aware": the network learns features that make the GP's
*uncertainty* trustworthy, not just its point predictions.

---

## 6. How we cached the crystal graphs (Step: data → graphs)

The CGCNN cannot read a crystal directly; it reads a **graph** (atoms = nodes, bonds = edges).
Building that graph for thousands of materials is slow, and we re-read graphs thousands of times
across all the BO runs — so we **build each graph once and cache it**. This is one of the most
important engineering steps, so it gets its own section.

### 6.1 Structure → graph (`src/dklbo/data/graph_builder.py`, `atoms_to_graph`)

For each material's unit cell (an ASE `Atoms` object):

1. **Neighbour search.** Use ASE's `neighbor_list` with a **3D radius of 8.0 Å** to find all
   atom pairs within range, returning their displacement vectors and distances (correctly
   accounting for periodic boundary conditions).
2. **The critical 2D fix — the vacuum filter.** A 2D material is a thin sheet (~3–6 Å thick)
   sitting in a tall box with ~15–20 Å of empty **vacuum** added along the c-axis (z). A naive
   neighbour search can create **false bonds across that vacuum gap**, which are physically
   meaningless and would poison the graph. We therefore **drop any edge whose z-displacement
   `|Δz|` exceeds `vacuum_cutoff = 4.0 Å`.** This single rule is the project's most important
   correctness check, and it is unit-tested.
3. **Neighbour cap.** Keep at most **12 nearest neighbours** per atom (standard CGCNN), so every
   atom has a comparable, bounded local environment.
4. **Features.**
   - *Atom features* `x`: a **90-dim one-hot** of the element (Z=1..90 → index Z−1).
   - *Bond features* `edge_attr`: each bond distance expanded into a **10-dim Gaussian radial
     basis** (centres evenly spaced 0→8 Å). This soft-encodes distance so the network can learn
     smooth distance-dependence instead of a single raw number.
5. Return a PyTorch-Geometric `Data` object: `x [N,90]`, `edge_index [2,E]`, `edge_attr [E,10]`,
   plus `n_atoms`, `n_edges`.

*(Note: the original planning PDF wrote 92 atom features / 40 bond bins / a separate CIF
intermediate; the code we actually run uses **90 / 10** and reads structures straight from the
ASE database — no loose CIF or `.pt` files. This document follows the code.)*

### 6.2 Caching in LMDB (`src/dklbo/data/cache.py`, `GraphCache`)

Rather than scatter ~3,351 loose `.pt` files on disk, every graph is stored in a single
**LMDB** key-value database, keyed by the material's `uid`, value = the pickled graph.

Why LMDB:
- **One file** on disk → no filesystem inode pressure from thousands of small files.
- **Memory-mapped reads** → the OS page cache is shared across all processes, so the many
  parallel BO runs read graphs fast and without re-loading.
- **Transactional writes** → safe to interrupt a build mid-way and resume.
- **~10–100× faster random access** than loading scattered files.

**Config-hash invalidation (the clever bit).** The cache file is named
`graphs_<hash>.lmdb`, where `<hash>` is an 8-char MD5 of the preprocessing config
(`radius`, `vacuum_cutoff`, `max_neighbors`, …). If anyone changes a preprocessing parameter,
the hash changes, so a *different* cache file is used — you can **never silently train on stale
graphs built with different settings**, and multiple configs coexist in one directory. Our cache
is `data/cache/graphs_e98e27ea.lmdb`.

### 6.3 Why the rebuild *reused* the cache instead of rebuilding it

A crystal's **structure does not depend on which property you study** — a structure is a
structure. The old project had already built graphs for all 3,351 gap materials, and our 2,667
intersection materials are a subset of those, with the **identical** preprocessing config (same
hash `e98e27ea`). So Phase 1 does not waste ~10 minutes rebuilding identical files; it **verifies
all 2,667 uids are present in the existing cache** and reuses it. The pantry already has every
ingredient — we just count to confirm nothing is missing.

---

## 7. The rebuild, phase by phase

The rebuild is a left-to-right pipeline. Each phase produces the ingredients the next consumes.
New scripts are plain Python, numbered `01`–`07`, outputs under `results/rebuild/`.

```
Phase 1  Build ONE clean dataset (data + descriptors + graphs + split + fairness check)
Phase 2  Can the models PREDICT?           (offline accuracy: Std-GP vs DKL)
Phase 3  The BO contest (frozen)           (Std-GP vs DKL vs Random — the main result)
Phase 4  Live fine-tuning + cold control   (the pretraining story)
Phase 5  Statistics + plots                (separate real wins from luck)
Phase 6  Extensions                        (30 seeds, epoch sweep, window search)
```

### Phase 1 — Build one clean dataset · `scripts/01_build_dataset.py`

**Goal:** prepare identical ingredients for both chefs and *prove* they are identical.

What it does:
- `build_master_table()` — scan C2DB once, keep only materials with **both** a valid gap
  (>0.01 eV, i.e. not a metal) **and** a finite positive emass → 2,667 materials, with columns
  `id, uid, formula, prototype, gap, emass, n_atoms`.
- `add_train_pool_split()` — split into **train = 1,901** (models may learn from this) and
  **pool = 766** (the held-out hunting ground / exam). The split is **prototype-aware** (no
  structural "twin" appears on both sides → no leakage) and **gap-stratified** (rare extremes are
  deliberately spread into the pool).
- Build the **43 handcrafted descriptors** (35 composition + 8 structural) for Std-GP, aligned
  row-for-row to the master table; deliberately excludes electronic DFT outputs.
- `verify_alignment()` — the fairness police: confirm descriptors and the graph cache cover the
  same materials in the same order, and all 2,667 graphs exist in the cache.

**Question answered — does the pool actually contain rare targets for all four searches?**
Yes: the pool holds 7 high-gap, 10 low-gap, 17 high-emass and 8 low-emass extreme materials (the
1.5% tails), so none of the four searches is pointless.

**The golden rule established here:** *never look at the pool to make a model decision.* The pool
is the exam. (Consequently we add **no** top-level validation split — following both reference
papers; a tiny val set is carved from *train* only, and only for encoder pretraining.)

Outputs: `data/cache/master.parquet`, `data/cache/descriptors.parquet`, reused
`graphs_e98e27ea.lmdb`.

### Phase 2 — Can the models predict? · `02_pretrain_encoder.py`, `03_eval_accuracy.py`, `03b_emass_scale_compare.py`

**Question:** if shown a material it has never seen, how accurately can each method predict its
property? Both chefs feed their features into the **same** ARD ExactGP and predict the 766-pool;
only the features differ. We grade with **MAE, RMSE, R²** and an honesty check **coverage@95**.

- `02_pretrain_encoder.py --target {gap,emass} [--log]` trains a CGCNN encoder (+GP head) on
  *train* for 100 epochs, freezes it, and writes a 32-dim embedding for every material. We train
  **one encoder per property** (features good for gap differ from features good for emass) →
  `encoder_gap.pt`, `encoder_emass_log.pt`, `embeddings_*.parquet`. Method follows the paper:
  ExactGP, fixed epoch budget, **no validation set touching the pool.**
- `03b_emass_scale_compare.py` answers a sub-question: **emass spans 0.001–136 (5 orders of
  magnitude)** — should we model it on a log scale? Yes — `log10` helps both methods and is what
  we adopt for all emass work.

**Results (held-out 766-pool):**

| Target | Std-GP | DKL |
|---|---|---|
| **gap** (R² / coverage) | **0.604** / 0.948 | 0.546 / 0.898 |
| **emass log10** (R²_log) | 0.125 | **0.136** |

**Reading:** for band gap, Std-GP is *slightly* the better, better-calibrated predictor —
descriptors already encode gap chemistry. For effective mass, **neither** predicts well (R²≈0);
emass is genuinely hard and heavy-tailed, but in log space **DKL edges ahead** — the first hint
that *where descriptors are physically weak, learned features start to win.*

**The cliff-hanger:** Phase 2 measures *tasting*, not *hunting*. The reference paper's biggest
*hunting* victory (min emass) came **despite** poor prediction R². So a weak accuracy result does
**not** mean a weak search result — that is exactly what Phase 3 tests. **Search ≠ accuracy.**

### Phase 3 — The BO contest (frozen DKL) · `scripts/04_run_bo.py`

**Question — the one the whole project exists for:** given a small budget of experiments, who
finds the best materials fastest? Setup: 766-material pool, **10 starter + 100 guided digs**,
**Expected Improvement** acquisition (parameter-free → fair), **4 tasks × 3 methods × 10 seeds**.
Searching for a *minimum* reuses the maximise machinery via a sign flip (`y_internal = −y`).
We score two ways: **best** (the single champion) and **top-50 / top-10%** (discovery breadth).

**Results (mean over 10 seeds):**

| Task | Method | best | top-50 | top-10% |
|---|---|---|---|---|
| **gap_max** | Std-GP | 8.671 | 21.3 | 29.4 |
| | **DKL** | 8.638 | **32.0** | **42.4** |
| | Random | 7.332 | 6.4 | 9.8 |
| **gap_min** | Std-GP | 0.022 | 19.4 | 28.7 |
| | DKL | 0.023 | 19.8 | 27.1 |
| **emass_min** | Std-GP | **−1.999** | 16.5 | 21.8 |
| | DKL | −1.714 | **18.9** | **24.1** |
| **emass_max** | **Std-GP** | **1.696** | **16.2** | **21.0** |
| | DKL | 1.455 | 12.6 | 16.8 |

**Honest reading (this is the core result of the rebuild):**
1. **DKL's superpower is harvesting MANY rare materials, not finding the single best.** Clearest
   on **gap_max**: DKL collects **42 top-10% materials vs Std-GP's 29** — a big breadth win —
   while the single best is a tie. Ideal for "give me a *batch* of strong candidates."
2. **gap_min is a tie** — descriptors handle band gap fine in both directions.
3. **emass_min — mixed, slight DKL edge** on breadth; Std-GP reaches a lower single champion.
4. **emass_max — Std-GP wins clearly**; the heaviest-mass extremes are DKL's honest weak spot.
5. **Everyone crushes Random everywhere** — both real methods are doing real work.

These are means only — which differences are *real* is decided in Phase 5.

### Phase 4 — Live fine-tuning + cold control · `scripts/05_run_bo_finetune.py`

**Question:** in Phase 3 the DKL encoder was **frozen** after pretraining. What if it keeps
**learning during the hunt**? And how much was the pretraining actually worth? Two switches:
(1) start pre-trained or cold, (2) during the hunt freeze or fine-tune.

| Variant | Start | During hunt |
|---|---|---|
| `dkl_frozen` (Phase 3) | pre-trained | frozen |
| `dkl_finetune` | pre-trained | fine-tuned (full retrain every 5th cycle, GP-only refit between) |
| `dkl_cold_live` (`--cold`) | random init | fine-tuned (the exact paper setup) |

**Key intuition — why fine-tuning can backfire:** during the hunt you have only ~10–110 labels,
but the encoder has ~18,000 parameters. Training such a big net on so few points **overfits** —
it distorts the embedding to memorise those few materials and *loses* the broad knowledge from
the 1,901-material pretraining. A chef who rewrites his whole palate after 3 dishes gets great at
*those 3* and worse at everything else.

**Results (means) and the clean trade-off:**
- **Fine-tuning helps the single champion but hurts breadth** (esp. on emass: it found lower
  single champions but its top-10% dropped vs frozen).
- **Frozen stays the breadth king.**
- **Cold (no pretraining) collapses on gap** — proof the pretraining does real work on this
  harder 2D dataset (unlike the paper's easier oxides). It still reaches a low single emass_min
  champion — echoing the paper — *but Phase 5 shows that edge is within noise.*

Takeaway: there is no single "best DKL" — there is a trade-off. Want **many** good materials →
**frozen pre-trained**. Want **the one** best fast → **fine-tuned**. **No pretraining → worse.**

### Phase 5 — Statistics + plots · `scripts/06_stats.py`, `scripts/07_plot.py`

**Question:** which differences are *real* and which are luck? With only 10 seeds, a mean can be
tipped by one lucky starter set. So:
- **Paired Wilcoxon signed-rank test** per (task, metric): because every method shares the same
  starter materials per seed, we compare DKL-on-seed-k directly against Std-GP-on-seed-k. Returns
  a **p-value**; p<0.05 = "unlikely to be luck."
- **Bootstrap 95% confidence intervals** — the shaded bands on the curves; narrow = trustworthy.
- **Regret-AUC** — a "how fast did you converge" score (lower = found great materials sooner).
- **A negative result is a result:** if not significant, we say "tied," not "won by a little."

**The verdict (p<0.05, excluding trivial Random losses):**

| Task | Finding | p |
|---|---|---|
| gap_max | **DKL-frozen beats Std-GP on top-10%** (42 vs 29) | **0.002** |
| gap_max | DKL-frozen beats Std-GP on top-50 | 0.002 |
| gap_max | DKL-finetune beats Std-GP on top-10% / top-50 | 0.016 / 0.012 |
| gap_max | cold+live **loses** on best & regret-AUC | 0.031 / 0.037 |
| gap_min | cold+live: better single best, worse breadth | 0.004 / 0.043 |

**Everything else — gap_min (frozen/finetune) and BOTH emass tasks — is statistically TIED with
Std-GP at 10 seeds.** The honest headline: **a pre-trained deep-kernel representation
significantly improves discovery of rare high-band-gap materials over handcrafted descriptors,
and that gain depends on pretraining** — clean, defensible, significant. The exciting-looking
emass means did **not** survive the test, so we claim nothing on emass yet. (Recommended fix:
more seeds → Phase 6.)

---

## 8. Phase 6 — Extensions (the follow-up experiments)

| Sub | Question | Outcome |
|---|---|---|
| **6a 30 seeds** | Do the 10-seed verdicts hold with more power? | **Yes.** DKL wins gap_max top-10% **42.7 vs 29.8, p≈2e-6**; everything else still tied. Plots in `results/rebuild/plots_30seed/`. |
| **6b epoch sweep** | Does training the encoder longer help real accuracy? | No — pool MAE plateaus ~0.73 eV after ~50 epochs while train MAE keeps falling (early overfitting). **100 epochs is plenty.** |
| **6c window (in-band)** | Find gaps inside 0.7–3.0 eV (a "range" task). | On a crowded window Std-GP ≈ DKL-frozen (tie); fine-tuning *hurts*. (Partial wide-window run remains to finish.) |
| **6d window-MAX** | Maximise gap *subject to* a 0.7–3.0 eV ceiling. | Std-GP edges the single top, but **DKL-finetune wins broad top-50 (11.6 vs 8.3, p=0.001)** from cycle 37. |
| **6e window-MIN** | Lowest gap *above* a 0.7 eV floor. | **DKL's strongest, cleanest win of the whole project — DKL wins BOTH best AND breadth, frozen AND fine-tuned, all significant** (best 0.710 vs 0.739; top-50 16.1 vs 9.9, p≈2e-5). |

**Why window-min is the showcase:** descriptors can pin the single material closest to a ceiling
(one descriptor tracks gap), but they **cannot harvest the whole band**; DKL re-focuses its
learned features onto the target region and wins breadth. The low-gap band is poorly served by
descriptors, so **DKL wins everything there** — the purest demonstration of the key innovation.

---

## 9. Where DKL wins — the bottom line

Stated honestly, with the statistics behind it:

1. **DKL's signature strength is discovery breadth — harvesting *many* rare materials.** It
   **significantly** out-discovers descriptors for **maximum band gap** (top-10% 42 vs 30,
   p≈0.002, confirmed at 30 seeds). This is the rock-solid headline.
2. **DKL's cleanest all-round victory is the low-gap window (6e):** it wins both the single best
   *and* breadth, frozen *and* fine-tuned, all statistically significant — exactly where
   handcrafted descriptors are weak.
3. **Pretraining is essential.** Strip it out (cold start) and DKL significantly *loses* on band
   gap — the win is caused by the learned representation built from 1,901 materials, not by
   architecture alone.
4. **Search ≠ accuracy.** DKL beats descriptors at *finding* high-gap and low-gap-window
   materials even though its pointwise prediction R² is no better — the project's central thesis,
   matching the reference paper.
5. **Honest limits.** Std-GP stays competitive or better for single-champion hunts and for
   maximum effective mass; gap-minimisation and raw effective-mass tasks are statistical ties at
   our seed counts. We do **not** claim "DKL wins everywhere" — we claim *where* and *why*.

**Frozen vs fine-tuned, in one line:** frozen = best **breadth**; fine-tuned = best **single
champion / sample-efficiency** (but overfits the few labels and loses breadth).

---

## 10. File & plot map (where everything lives)

**Code (reused engine):** `src/dklbo/` — `data/graph_builder.py` (graphs), `data/cache.py`
(LMDB cache), `models/cgcnn_encoder.py`, `models/surrogate.py` (GP), `models/dkl.py`,
`bo/acquisition.py`, `bo/loop.py`, `baselines/feature_bo_loop.py`.

**Rebuild scripts:** `scripts/01_build_dataset.py` → `07_plot.py`, plus `exp_epoch_sweep.py`,
`exp_window_bo.py`, `exp_window_max_bo.py`, `exp_window_min_bo.py`.

**Data:** `data/cache/master.parquet`, `descriptors.parquet`, `graphs_e98e27ea.lmdb`.

**Results:** `results/rebuild/` — `accuracy.csv`, `bo_summary.csv`, `per_run_metrics.csv`,
`summary_stats.csv`, `stats_pairs.csv`, `winmin_stats.csv`, `winmax_stats.csv`, run CSVs.

**The plots that carry the story** (priority order):

| Plot | File | What it proves |
|---|---|---|
| gap_max curves (best + cumulative top-10% vs cycle, CI bands) | `plots_30seed/gap_max_curves.png` | DKL's significant breadth win — *the headline* |
| window-min best & top-50 vs cycle | `plots/winmin_best_gap.png`, `winmin_top50.png` | DKL's cleanest all-round win |
| window-min/max crossover | `plots/winmin_crossover.png`, `winmax_crossover.png` | the cycle DKL overtakes Std-GP |
| prediction accuracy | `plots/accuracy.png` | emass R²≈0 for all → sets up search≠accuracy |
| per-task final bars | `plots/{task}_bars.png` | the final podium per task |
| gap_max/emass curves | `plots_30seed/{task}_curves.png` | full per-task convergence + breadth |

---

## 11. One-paragraph summary

We rebuilt a Deep-Kernel-Learning Bayesian Optimiser for discovering 2D materials, following
Kiyohara & Kumagai (2025) but on the C2DB with vacuum-aware crystal graphs, one shared
2,667-material intersection dataset, formal significance testing, and a clean pretraining
ablation. The **key innovation** is that DKL lets a CGCNN *learn* the feature representation from
structure instead of relying on handcrafted descriptors, jointly with the GP that uses it. Graphs
are built once (atom one-hots + Gaussian-basis bonds, with a 4 Å vacuum filter that kills false
across-vacuum bonds) and cached in a config-hashed LMDB so the contest is fast and reproducible.
Across a fair, statistically-tested contest, **DKL significantly out-discovers handcrafted
descriptors at finding rare high-band-gap materials (breadth) and dominates the low-gap window
search outright — provided the encoder is pre-trained — even where its raw prediction accuracy is
no better, confirming that search skill is not the same as prediction accuracy.**
