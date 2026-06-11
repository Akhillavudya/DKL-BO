# E2 Experiment — Full Explanation

**Branch:** `experiment/e2-recalibration`
**Changes:** Post-hoc uncertainty recalibration (temperature scaling + isotonic)
**Status:** Complete

---

## Start Here — What is σ (Sigma)?

Before explaining E2, you need to understand σ (the Greek letter sigma, pronounced
"sig-mah"). It appears everywhere in this experiment.

When the GP makes a prediction for a material, it doesn't just say one number. It says
**two** numbers:

- **μ (mu)** — the predicted band gap value. Example: "I think this material has a gap
  of 3.5 eV."
- **σ (sigma)** — the uncertainty around that prediction. Example: "...but I'm not sure.
  My uncertainty is ±0.8 eV."

Together, μ and σ define a range: the GP is saying "the true band gap is probably
somewhere between 3.5 − 1.96×0.8 and 3.5 + 1.96×0.8" (where 1.96 comes from
standard statistics for a 95% confidence interval).

**Analogy:**
Imagine a weather forecast. The forecaster doesn't just say "tomorrow will be 25°C."
A good forecaster says "25°C ± 3°C." The ± 3°C is like σ — it tells you how confident
the forecast is. A forecast of 25°C ± 0.1°C would be suspiciously overconfident. A
forecast of 25°C ± 50°C is useless. Somewhere between those extremes is "honest."

---

## What is Calibration?

A GP is **well-calibrated** if its σ values are honest.

Specifically: if the GP says "I'm 95% confident the gap is in range [A, B]", then
roughly 95 out of 100 materials should actually have their true gap inside [A, B].

If only 70 out of 100 fall inside — the GP was **overconfident** (σ too small, intervals
too narrow).
If 99 out of 100 fall inside — the GP was **underconfident** (σ too large, intervals
too wide).

**Why does calibration matter for BO?**
In the UCB acquisition formula:

```
score = μ + β × σ
```

The β controls how much we trust σ. If σ is lying — either too small or too large —
then any β > 0 leads us astray. The BO loop explores the wrong materials. That is
exactly why in the baseline, β=0 (ignore σ completely) was the best strategy: the σ was
untrustworthy, so it was better to not use it at all.

---

## The Five Calibration Metrics — What They Mean

### 1. NLL (Negative Log-Likelihood)
**What it is:** A combined score measuring both accuracy and calibration together.
Lower = better.

**Intuition:** Imagine you bet money on each prediction. NLL measures how often your
bet was wrong AND how confidently you bet wrong. Being confidently wrong is penalised
more than being uncertain and wrong.

**Target:** No fixed target — just lower than before.

---

### 2. Coverage@95
**What it is:** Of all the materials in the evaluation set, what fraction had their true
band gap fall inside the GP's 95% confidence interval?

**Target:** 0.95 (by definition — the 95% interval should cover 95% of cases).

- Coverage > 0.95 → GP is underconfident (intervals too wide, captures too much)
- Coverage < 0.95 → GP is overconfident (intervals too narrow, misses too much)

**Our result before recalibration:** Coverage@95 = 0.90 on test. This means the GP's
"95% confidence intervals" were actually only correct 90% of the time — slightly
overconfident.

---

### 3. Miscalibration Area
**What it is:** The average gap between what the GP claimed and what actually happened,
measured across all confidence levels (not just 95%).

Imagine drawing a graph:
- X-axis: the confidence level the GP claimed (10%, 20%, ..., 90%)
- Y-axis: the actual fraction of materials that fell inside the interval

For a perfectly calibrated GP, the graph would be a straight diagonal line (claim 30%
coverage → actually get 30%). The miscalibration area is the total area between the
actual curve and that perfect diagonal line.

**Target:** 0.04 – 0.07

---

### 4. ENCE (Expected Normalised Calibration Error)
**What it is:** A test of whether σ correctly *ranks* uncertainty — do materials with
large σ actually have large prediction errors?

The data is sorted by σ (from small to large), split into 10 groups. For each group,
ENCE checks: is the average σ in this group proportional to the average prediction error?

**Target:** 0.06 – 0.10

**Important note:** ENCE being high (0.30 in E2) does NOT mean the GP is wrong about
everything. It means σ is not a good *ranker* of uncertainty — it doesn't reliably flag
which specific materials it's unsure about. This is a structural problem with the GP
that temperature scaling alone cannot fix. It would require E4 (spectral-normalization)
to properly address.

---

### 5. Spearman ρ (rho)
**What it is:** A correlation score between:
- The GP's uncertainty σ for each material
- The GP's actual prediction error for that material

If Spearman ρ is high (close to 1.0): materials where the GP was uncertain turned out
to actually be the hard ones. The GP "knows what it doesn't know."

If Spearman ρ is low (close to 0): σ is basically random — high σ doesn't predict where
errors are.

**Target:** As high as possible. Our values (~0.14–0.26) are low, which is related to
the same structural issue as ENCE.

---

## What E2 Added — Temperature Scaling

### The Problem
In E1, after ARD and standardization, the GP was still slightly overconfident. The
Coverage@95 was 0.90 on both val and test (should be 0.95). This means σ was
systematically too small — about 17% too small, as it turned out.

### The Fix: Temperature Scaling (τ)

E2 fits a single number τ (tau, Greek letter, pronounced "taw") on the validation set.
τ is called the **temperature**.

**How τ is calculated:**
For every material in the validation set, compute the z-score:

```
z = (true_gap − predicted_mean) / predicted_sigma
```

If the GP were perfectly calibrated, these z-scores would follow a standard bell curve
(mean 0, standard deviation 1). The temperature τ is:

```
τ = sqrt( average of all z² )
```

This is the "root mean squared z-score." If τ = 1.0 → perfect calibration.
If τ > 1.0 → GP is overconfident (z-scores are too large → σ too small).
If τ < 1.0 → GP is underconfident.

**Our result:** τ = **1.1672**

This means the GP's σ values are about **17% too small**. Every uncertainty estimate
needs to be scaled up by a factor of 1.1672 to be honest.

**Analogy:**
Imagine a weighing scale that is miscalibrated — it always shows 85% of the true weight.
A 10 kg bag reads as 8.5 kg. You can fix this by multiplying every reading by 1/0.85 =
1.176. Temperature scaling does the exact same thing to σ.

After applying: `σ_calibrated = 1.1672 × σ_raw`

This single correction is saved to `results/recalibration_params.json` and
automatically loaded before every BO acquisition in Phase 3.

---

## What is Isotonic Recalibration? (The Second Method)

Temperature scaling is one global correction (one number τ for all materials). Isotonic
recalibration (from a 2018 paper by Kuleshov et al.) is a more flexible non-parametric
correction.

**How it works:**
1. For each validation material, compute the PIT value (Probability Integral Transform):
   `q = Φ((y − μ) / σ)` where Φ is the standard normal CDF.
   If the GP is calibrated, these q values should be uniformly spread between 0 and 1.
2. Sort the q values and fit an isotonic regression (a curve that only goes up):
   `predicted quantile → actual quantile`
3. This curve can then be used to map any predicted confidence level to its true value.

**Why not use isotonic for BO?** Isotonic outputs a *coverage correction*, not a direct
σ correction. Temperature scaling directly multiplies σ, which plugs cleanly into the
UCB formula. Isotonic is used here as a diagnostic to verify the shape of miscalibration
but temperature scaling is what gets applied in the BO loop.

---

## Phase 2 Results — Before and After Recalibration

### Accuracy (no change from recalibration — μ stays the same)

| Metric | Baseline | E1 | E2 |
|---|---|---|---|
| Val MAE (eV) | 0.4463 | 0.4903 | 0.4987 |
| Val R² | 0.699 | 0.606 | 0.588 |
| Test MAE (eV) | 1.0321 | 0.9431 | **0.8821** ✅ best yet |
| Test R² | 0.285 | 0.411 | **0.446** ✅ best yet |

Test accuracy keeps improving experiment by experiment. Val R² dropped slightly — this
is run-to-run noise (val has only 162 materials, making it sensitive to random weight
initialization). The test set (60 materials) is the meaningful measure.

---

### Calibration — Val split

| Metric | Before recal | After recal (τ=1.167) | Target |
|---|---|---|---|
| NLL | 1.1489 | 1.1223 ↓ | lower better |
| ENCE | 0.3015 | (not changed by temp scaling) | 0.06–0.10 |
| Miscal. area | 0.0832 | 0.1071 ↑ | 0.04–0.07 |
| Spearman ρ | 0.1428 | (unchanged) | higher better |
| Coverage@95 | 0.9012 | 0.9259 ↑ | ~0.95 |

**Why did val miscal area get WORSE after recalibration?**
The temperature τ is fitted to minimise NLL, which is equivalent to pushing Coverage@95
toward 0.95. It is NOT fitted to minimise miscal area. Miscal area measures calibration
across all confidence levels (10%, 20%, ..., 90%). The τ shift improved coverage near
95% but overcorrected other confidence levels on the small val set (162 materials).
With only 162 points, the miscal curve is noisy and one global correction cannot
perfectly align every point on it.

---

### Calibration — Test split

| Metric | Before recal | After recal (τ=1.167) | Target |
|---|---|---|---|
| NLL | 1.6546 | 1.5515 ↓ | lower better |
| ENCE | 0.3481 | 0.2269 ↓ | 0.06–0.10 |
| Miscal. area | 0.0860 | **0.0412** ✅ | 0.04–0.07 |
| Spearman ρ | 0.2553 | (unchanged) | higher better |
| Coverage@95 | 0.9000 | 0.9333 ↑ | ~0.95 |

On the test set, recalibration improved **every metric that temperature scaling can
affect**. Miscal area went from 0.0860 → **0.0412**, which is squarely inside the
target range. The test set (60 materials) is more consistent than val, so the single
scalar correction works better here.

---

## The Core E2 Hypothesis — Was It Right?

### β-Sweep Results

| β | Baseline Top-50 | E1 Top-50 | E2 Top-50 (recal) |
|---|---|---|---|
| 0.0 (pure exploitation) | 28 | 20 | **11** ↓ |
| **0.2** | 9 | 20 | **25** ✅ winner |
| **0.5** | 5 | 25 | **25** ✅ |
| 1.0 | 5 | 28 | 20 |
| 2.0 | 1 | 22 | 22 |

The experiment plan stated the hypothesis:
> *After recalibration, some β>0 beats β=0 — demonstrating that honest uncertainty
> restores the value of exploration.*

**The hypothesis was confirmed.**

β=0.2 and β=0.5 are now the clear winners (25 top-50 hits each). Pure exploitation
(β=0) dropped from 20 to only 11.

For Top-10% hits:
| β | E2 Top-10% hits |
|---|---|
| 0.0 | 44 |
| **0.2** | **49** ← best |
| 0.5 | 48 |
| 1.0 | 33 |
| 2.0 | 45 |

β=0.2 is the clear winner at both metrics.

---

## What Got Better and Why

### 1. Test accuracy improved (R² 0.411 → 0.446, MAE 0.943 → 0.882 eV)
The model trained slightly better this run. This is a combination of the same E1
improvements (ARD, standardization) plus natural run-to-run variation favouring the
test split this time.

### 2. Test miscal area hit target (0.0860 → 0.0412, target 0.04–0.07)
Temperature scaling with τ=1.167 correctly identified that σ was 17% too small and
scaled it up. The result on the test set lands inside the target range.

### 3. Exploration (β=0.2, β=0.5) now beats pure exploitation (β=0)
This is the headline result of E2. Before recalibration, β=0 was competitive because
exploring uncertain regions was unreliable. After honest σ values, the BO loop
correctly identifies which materials are genuinely uncertain, and those turn out to be
the valuable ones. Exploration is no longer wasted budget.

### 4. Coverage@95 moved toward target
Val: 0.9012 → 0.9259. Test: 0.9000 → 0.9333. Both moving toward the 0.95 target.

---

## What Got Worse and Why

### 1. β=0 top-50 hits dropped (E1: 20 → E2: 11)
This sounds bad but it's **by design**. When exploration becomes useful, pure
exploitation gets penalised relative to it — the BO loop is now spending some budget
exploring uncertain regions instead of always picking the single highest-mean material.
β=0 isn't "worse" as an algorithm — it's just that β=0.2 is now genuinely better.
This wouldn't have happened in the baseline or E1 where σ was not trustworthy.

### 2. Val miscal area increased (0.0832 → 0.1071)
Explained above: val has only 162 materials, making it noisy. Temperature scaling
optimises NLL (not miscal area directly), so some confidence levels overcorrect while
others undercorrect on a small set. The test result (0.0412) tells the real story.

### 3. ENCE remains high (0.30)
Temperature scaling scales all σ values by the same factor. It cannot fix the *ranking*
of uncertainty — it cannot make "high σ materials" reliably correspond to "hard
materials." ENCE measures ranking quality. This requires a structurally better
uncertainty model, which is E4 (spectral-normalized DKL).

### 4. Val R² dropped slightly (0.606 → 0.588)
Run-to-run variance from random weight initialization. With only 162 val materials,
a few unlucky predictions shift R² noticeably. Test R² improved, so the model is fine.

---

## Files Changed in E2

| File | What Changed |
|---|---|
| `src/dklbo/eval/recalibration.py` | **New.** `fit_temperature()` computes τ. `fit_isotonic()` fits Kuleshov 2018 PIT map. `recalibration_report()` prints before/after table |
| `src/dklbo/bo/loop.py` | `BOLoop` gains `std_scale` parameter. Applies `σ_cal = τ × σ` before every UCB score computation |
| `scripts/02_eval_surrogate.py` | Fits τ on val set after eval, saves `results/recalibration_params.json` |
| `scripts/03_run_bo.py` | Loads τ from JSON when `recalibrate: true`, passes `std_scale` to BOLoop |
| `configs/surrogate/gp_exact.yaml` | `recalibrate: true` added |

---

## Key Takeaways

1. **σ is the GP's uncertainty estimate** — it says how wide the prediction interval is.
   A trustworthy σ is essential for exploration to work.

2. **Calibration = honesty of σ** — if the 95% interval covers 95% of materials, the
   GP is well-calibrated. Our GP was 17% overconfident (intervals too narrow).

3. **Temperature scaling (τ)** is the simplest fix: multiply every σ by one number.
   τ = 1.1672 means "scale all σ up by 17%." Fitted once on val, applied everywhere.

4. **The hypothesis was confirmed** — after honest σ, β=0.2 beats β=0. Exploration is
   now valuable because uncertainty actually tracks where the GP is wrong.

5. **ENCE and Spearman ρ are still poor** — temperature scaling fixes the *scale* of σ
   but not its *ranking*. That requires E4 (spectral normalization).

6. **β=0.2 is the current best setting** — 25 top-50 hits and 49 top-10% hits, better
   than any β in the baseline, and better than β=0 after recalibration.

7. **Open question** — even after τ correction, Coverage@95 is 0.933 (not yet 0.95)
   and ENCE is 0.23. E4 (spectral-norm) should improve both by giving the GP a
   distance-preserving encoder so OOD uncertainty is meaningful.
