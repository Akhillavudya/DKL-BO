# Rebuild — Phase 2 Explained (can the chefs *predict*?)

*A beginner-friendly walkthrough: the big picture, then file by file, with analogies.*
Continues the cooking-competition story from `rebuild_phase1_explanation.md`.

---

## 0. Where Phase 2 sits

```
   Phase 1  — Build ONE clean dataset
>> Phase 2  — Check the models can PREDICT            <-- YOU ARE HERE
   Phase 3  — The BO contest (search)
   Phase 4  — The "well-pretrained DKL" story
   Phase 5  — Plots + write-up
```

Phase 1 prepared the ingredients. Phase 2 asks a simple question about each chef:
**"If I show you a material you've never tasted, can you guess its property?"**

---

## 1. The big picture (one analogy)

Before the real cooking contest (Phase 3), we give each chef a **tasting exam**.

- We let each chef study the **train** materials (1,901 of them) — they may learn all they want.
- Then we hand them **pool** materials they've never seen (766 of them) and ask them to
  *predict* the band gap (and the effective mass) of each.
- We grade the guesses with three report-card numbers (**MAE, RMSE, R²**) plus an honesty
  check (**coverage@95** — "when you said you were 95% sure, were you right 95% of the time?").

**The golden rule from Phase 1 still holds:** the pool is the exam. Nobody studies it.

Important twist you'll see: **being a good taster (accurate prediction) is NOT the same as
being a good hunter (finding great materials).** Phase 2 measures tasting. Phase 3 measures
hunting. A chef can be a mediocre taster but a brilliant hunter — that is, in fact, the
whole point of the project.

---

## 2. The two key ideas in Phase 2

### Idea A — Each chef predicts the same way, only the "features" differ
Both chefs feed their numbers into the **same** Gaussian Process (GP) and predict the pool.
The ONLY difference is what describes each material:

- **Std-GP** (traditional chef): 43 hand-measured descriptors (`descriptors.parquet`).
- **DKL** (modern chef): 32 learned "embedding" numbers from a neural network that looked at
  the crystal structure.

Same oven, different ingredients → a fair test of *features*.

### Idea B — One encoder per property
The neural network ("encoder") that turns a structure into 32 numbers is trained to predict
**one** property. Features good for guessing band gap aren't the same as features good for
effective mass — so we train **two encoders**: one for gap, one for emass.

**Analogy:** a wine taster trained to judge *sweetness* builds different instincts than one
trained to judge *acidity*. Same nose, different training.

---

## 3. File by file

### 3a. `scripts/02_pretrain_encoder.py` — train the modern chef's instincts
Trains the CGCNN encoder (+ a GP head) on the **train** split to predict one target, then
freezes it and turns **every** material into a 32-number embedding.

| Piece | What it does | Analogy |
|---|---|---|
| `--target gap` / `--target emass` | which property to learn | pick the wine attribute to study |
| `--log` flag | train on `log10(target)` (used for emass) | use a "stretchy ruler" for a property that spans huge ranges |
| `DKLModel.fit(...)` | jointly trains encoder + GP for 100 epochs | the chef practising on the train dishes |
| `dkl.encode(...)` | converts all 2,667 materials → embeddings | writing down a 32-number "flavour fingerprint" for every dish |
| saves `encoder_{t}.pt`, `embeddings_{t}.parquet` | reusable outputs | the trained palate + the fingerprint book |

Method note: we use an **ExactGP** head and a **fixed training budget with no validation
set**, following the Kiyohara & Kumagai DKL-BO paper (our template). The pool is never used
to decide when to stop — it stays a clean exam.

### 3b. `scripts/03_eval_accuracy.py` — grade both chefs on both properties
Fits the **same** ARD GP on train features, predicts the pool, and prints the report card
(MAE, RMSE, R², coverage@95) for `std_gp` and `dkl`, for both gap and emass.

| Piece | What it does | Analogy |
|---|---|---|
| `evaluate(...)` | standardize features + target by *train* stats, fit GP, predict pool | give both chefs the same exam under the same rules |
| `compute_accuracy_metrics` | MAE / RMSE / R² | the accuracy part of the report card |
| `compute_calibration_metrics` | coverage@95 | the honesty part ("were your error bars truthful?") |
| saves `accuracy.csv`, `plots/accuracy.png` | the results | the graded report cards |

### 3c. `scripts/03b_emass_scale_compare.py` — should emass use a "stretchy ruler"?
Effective mass ranges from 0.001 to 136 — five orders of magnitude. A few giant values
dominate the score and hide the real signal. This script compares modelling emass **raw**
vs **log10**, reporting both log-space scores and back-transformed raw-scale scores (so raw
and log models are judged on identical units).

**Analogy:** measuring both ants and elephants with a millimetre ruler is silly — the
elephants drown out the ants. `log10` is a "stretchy ruler" that fits both on one page.

### 3d. The reused engine (`src/dklbo/`)
| Module | Role | Analogy |
|---|---|---|
| `models/cgcnn_encoder.py` | the neural net that reads structures → 32 numbers | the modern chef's trained nose |
| `models/dkl.py` | glues encoder + GP, trains them together | the chef's brain coordinating taste + memory |
| `models/surrogate.py` | the ExactGP (Matérn-5/2, ARD) used by BOTH chefs | the shared oven |
| `eval/metrics_accuracy.py` / `metrics_calibration.py` | the grading formulas | the exam rubric |

---

## 4. What Phase 2 found (the report cards)

**Band gap (predicting the 766-material pool):**

| Method | MAE | RMSE | R² | coverage@95 |
|---|---|---|---|---|
| Standard GP | 0.745 | 0.995 | 0.604 | 0.948 |
| DKL | 0.759 | 1.065 | 0.546 | 0.898 |

→ Std-GP is *slightly* the better taster on gap, and more honest (coverage ≈ 0.95). DKL is
a touch over-confident. Expected: descriptors already encode gap chemistry well.

**Effective mass — raw vs log10 (raw units; log predictions back-transformed):**

| Method | scale | MAE | RMSE | R²(raw) | R²(log) |
|---|---|---|---|---|---|
| Standard GP | raw | 1.855 | 6.156 | −0.095 | — |
| DKL | raw | 1.964 | 6.029 | −0.051 | — |
| Standard GP | log10 | 1.594 | 5.946 | −0.022 | 0.125 |
| DKL | log10 | 1.579 | 5.844 | **0.013** | **0.136** |

→ Two lessons:
1. **Nobody predicts emass well** (R² near zero). It is genuinely hard, and heavy-tailed.
2. **The stretchy ruler (log10) helps everyone**, and in log space **DKL edges ahead** of
   Std-GP (R²_log 0.136 vs 0.125) — the first sign that *where descriptors are physically
   weak, learned features start to win.* We adopted log10 for all emass work.

---

## 5. The cliff-hanger into Phase 3

The headline of Phase 2 is almost a *non-result*: emass prediction is poor for both chefs.
But remember Idea A's twist — **tasting ≠ hunting.** The reference paper found DKL's *biggest
hunting victory* was minimum effective mass, **despite** terrible prediction R². Phase 2
confirmed the "bad taster" half. Phase 3 tests whether DKL is still a good hunter anyway.

---

## 6. One-paragraph summary

Phase 2 gave both chefs a tasting exam on 766 unseen materials. Std-GP predicts the band gap
a little better and more honestly; for effective mass *neither* chef predicts well, but a
log10 "stretchy ruler" helps both and lets DKL nose ahead. The deeper message — set up
deliberately here — is that prediction skill and search skill are different things, which is
exactly what the Phase 3 contest puts to the test.
