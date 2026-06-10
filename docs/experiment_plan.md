# DKL-BO Experiment Plan

**Project:** Deep Kernel Learning + Bayesian Optimisation for 2D materials discovery
**Goal:** Improve model accuracy, uncertainty calibration, and BO sample efficiency
**Baseline:** ExactGP, β=0.2, seed=42, 10 init + 100 cycles

---

## Diagnosis: Where the Pipeline Is Leaking Performance

| Symptom | Evidence | Root cause in code |
|---|---|---|
| Severe generalization gap | Val R²=0.70 → Test R²=0.28; MAE 0.45→1.03 (2.3×) | `dkl.py:fit()` trains fixed epochs with no val-based early stopping, no weight decay, no dropout. Encoder overfits train split. |
| OOD overconfidence | Coverage@95 = 0.75 on test (should be ~0.95 → intervals too narrow) | Classic DKL feature collapse: encoder squeezes far-apart crystals into nearby embeddings, so GP is confident where it shouldn't be. |
| β=0 beats β>0 | Exp 1: pure exploitation wins | Suspicious — usually means σ is not trustworthy, so spending budget on "high-σ" picks is wasted. Calibration and BO are coupled. |
| Weak GP head | `surrogate.py:79` `MaternKernel(nu=2.5)` | Single isotropic lengthscale over 32 dims, no ARD. Embeddings are all-positive (softplus) and never standardized before the GP. |
| Unstable/wasteful refits | `surrogate.py:111-112` | GP rebuilt from scratch every refit — lengthscale & noise re-initialized to defaults each cycle, discarding learned hyperparameters. |
| One acquisition, one seed | UCB only; β=0 conclusion from seed=42 alone | No EI/Thompson; conclusions not seed-averaged. |
| Cheap labels unused | DB has `gap` (PBE) ×8,699, `hform` ×16,905 vs `gap_hse` ×3,363 | No transfer/multi-fidelity pretraining — biggest sample-efficiency lever untouched. |

---

## Recommended Sequencing

```
E1 (ARD + standardize + warm-start)  ──►  better base model
        │
        ├──►  E2 (recalibrate, re-sweep β)  ──┐
        │                                      ├──►  E3 (acquisition bake-off, multi-seed)
        └──►  E4 (spectral-norm DKL)  ─────────┘
E5 (multi-fidelity) — independent; biggest sample-efficiency upside
```

Do **E1 first** — everything else inherits a stronger surrogate. Then **E2/E4** to fix uncertainty. Then **E3** to exploit honest uncertainty. **E5** is runnable in parallel.

---

## E1 — ARD Kernel + Embedding Standardization + GP Warm-Start

**Status:** ⏳ Pending
**Priority:** High — do this first, everything else builds on it
**Targets:** Accuracy + calibration
**Effort:** Low

### Motivation
The GP uses one lengthscale for all 32 embedding dimensions and never standardizes inputs.
With ARD (Automatic Relevance Determination), the kernel learns per-dimension relevance,
downweighting noisy/dead embedding dims. Standardizing embeddings (zero-mean/unit-var,
fit on labelled set) fixes the all-positive softplus distribution that distorts an
isotropic kernel. Warm-starting GP hyperparameters across refits removes a real
instability source — currently GP is rebuilt from scratch every cycle.

### Implementation
- `src/dklbo/models/surrogate.py`: change `MaternKernel(nu=2.5)` →
  `MaternKernel(nu=2.5, ard_num_dims=D)` where D=encoder.out_dim; plumb `ard: true`
  through `configs/surrogate/gp_exact.yaml`
- Add a standardization transform on embeddings inside `DKLModel` — fit on labelled set,
  apply to pool embeddings before GP predict
- In `ExactGPSurrogate.fit()`, on refit reuse previous `model.state_dict()` instead of
  re-instantiating when dims match (warm-start lengthscale/noise/mean)

### Metrics to Measure
Val/test: MAE, RMSE, R², NLL, ENCE, Coverage@95 — compare against current baseline.

### Expected Outcome
R²↑, NLL↓, Coverage@95 → closer to 0.95.
Likely the best effort-to-payoff ratio of all five experiments.

---

## E2 — Post-Hoc Uncertainty Recalibration + Re-Run β Sweep

**Status:** ⏳ Pending
**Priority:** High — reframes the headline β=0 finding
**Targets:** Calibration → BO sample efficiency
**Effort:** Low–medium

### Motivation
Coverage@95=0.75 means the GP is overconfident. The "β=0 is best" result is exactly
what you'd expect from *untrustworthy* σ — exploration is wasted because σ doesn't track
real error. Fit a recalibration map on the validation set and the β conclusion may flip,
producing a genuinely publishable narrative.

### Implementation
- New `src/dklbo/eval/recalibration.py`:
  - **σ-temperature scaling**: single scalar τ minimizing val NLL → σ′ = τσ
  - **Isotonic recalibration** (Kuleshov 2018): non-parametric map on
    probability-integral-transform; fit on val split
- Apply σ′ in `BOLoop` before the acquisition score is computed
- Re-run β ∈ {0, 0.2, 0.5, 1.0, 2.0} sweep **with recalibrated σ**

### Metrics to Measure
Calibration suite (NLL, ENCE, miscal_area, Spearman ρ, Coverage@95) before and after
recalibration. Then best-gap / top-50 / top-10% per β value.

### Expected Outcome (the interesting hypothesis)
After recalibration, some β>0 beats β=0 — demonstrating that *honest uncertainty
restores the value of exploration*. This would be the key narrative result for the
write-up.

---

## E3 — Acquisition Function Bake-Off: EI, log-EI, Thompson vs UCB/Greedy

**Status:** ⏳ Pending
**Priority:** High — directly improves the headline sample-efficiency metric
**Targets:** BO sample efficiency
**Effort:** Low

### Motivation
UCB requires hand-tuning β. **Expected Improvement (EI)** is parameter-free and
purpose-built for max-finding. **log-EI** (Ament 2023) fixes EI's vanishing-gradient
pathology that bites in late cycles when the incumbent is high. **Thompson sampling**
(draw a GP posterior sample, take argmax) gives principled exploration and natural
diversity without a parameter. All three are known to outperform UCB in standard BO
benchmarks; applying them here costs only an afternoon.

### Implementation
- `src/dklbo/bo/acquisition.py`: add:
  - `ei(mean, std, incumbent)` — `(incumbent - mean) * Φ(z) + std * φ(z)` where
    `z = (incumbent - mean) / std`
  - `logei(mean, std, incumbent)` — numerically stable log-space EI (Ament 2023)
  - `thompson(mean, std)` — sample from GP posterior:
    `torch.distributions.Normal(mean, std).rsample()`
- Thread `incumbent` (current best observed gap) from `BOLoop` into the acquisition
  function call for EI/log-EI
- Register all three in `_REGISTRY`

### Metrics to Measure
Best-gap-vs-cycle curve, cycle-to-first-top-50, cumulative top-10% — averaged over ≥5
seeds, with mean ± std bands.

### Expected Outcome
log-EI matches/beats β=0 greedy while being parameter-free and robust across seeds.
Thompson finds rare materials earlier. Strong, low-risk win for the headline metric.

---

## E4 — Spectral-Normalized DKL (SNGP-style) to Fix Feature Collapse

**Status:** ⏳ Pending
**Priority:** Medium — root cause of the val→test calibration cliff
**Targets:** Accuracy + calibration + BO robustness (OOD)
**Effort:** Medium

### Motivation
The test-set overconfidence (Coverage@95=0.75) is the textbook DKL failure mode
(van Amersfoort 2021, Liu SNGP 2020): an unconstrained encoder collapses distances, so
the GP's distance-based uncertainty becomes meaningless OOD. This is exactly the
val→test accuracy cliff (R² 0.70→0.28). Enforcing a bi-Lipschitz (distance-preserving)
encoder makes σ meaningful in unseen chemical families.

### Implementation
- Wrap encoder `nn.Linear` layers in `cgcnn_encoder.py` (`embed`, `lin_gate`, `lin_msg`,
  FC layers) with `torch.nn.utils.parametrizations.spectral_norm(module, name='weight')`
- Sweep spectral norm coefficient `c` ∈ {0.95, 1.0, 1.5, 3.0}
- **Comparison arm (fallback):** deep ensemble of K=3–5 encoders; ensemble disagreement
  adds epistemic variance to GP σ — heavier but strong calibration baseline
- Config flag: `model.spectral_norm: true` with `model.sn_coeff: 1.0`

### Metrics to Measure
Test R² and Coverage@95 (the two broken numbers), NLL, size of val→test gap. Then feed
recalibrated SN-DKL into the BO loop.

### Expected Outcome
Test calibration and OOD accuracy improve; combined with E2, exploration in BO becomes
genuinely useful. The val→test gap should shrink measurably.

---

## E5 — Multi-Fidelity Transfer: Pretrain Encoder on PBE Gaps, BO on HSE

**Status:** ⏳ Pending
**Priority:** Medium — biggest ceiling for sample efficiency
**Targets:** Sample efficiency at small label budgets
**Effort:** Medium

### Motivation
The database holds **8,699 PBE `gap`** and **16,905 `hform`** labels vs only **3,363
`gap_hse`** labels. PBE gap is a strong, cheap proxy for HSE gap. Pretraining the
CGCNN encoder on the abundant PBE labels gives a representation that already separates
wide- vs narrow-gap crystals, so BO on the expensive HSE target starts with an informed
prior. This is realistic: in a real pipeline PBE costs ~100× less than HSE.

This is NOT the unfair warm-start (using Phase-2 HSE weights would cheat); this uses a
genuinely cheaper fidelity as an auxiliary signal.

### Implementation
- Build a second LMDB cache for `gap` (PBE); `configs/data/c2db_gap.yaml` already
  supports `target` override
- New script `scripts/00_pretrain_pbe.py`: train the CGCNN encoder on PBE gap regression
  (no GP, just MSE/MAE head), save encoder weights
- In `scripts/03_run_bo.py`: add a `--pretrain_encoder` flag that loads PBE weights
  before BO starts (encoder init, not GP init)
- **Ambitious extension:** true multi-fidelity GP with a PBE task and an HSE task sharing
  a latent process (GPyTorch `MultitaskGP`)

### Metrics to Measure
Best-gap and top-50 hits at **small** budgets: 10, 25, 50 cycles (where transfer
advantage is largest). Report cycles-to-first-top-50. Compare: BO-from-scratch vs
BO-from-PBE-pretrain.

### Expected Outcome
Large early-cycle gains at budget ≤ 50. The clearest sample-efficiency story for a
final write-up or paper. Synergizes with E1 (better GP head extracts more from the
pretrained representation).

---

## Cross-Cutting Methodology Note

**All BO comparisons must use ≥5 seeds.** The β=0 conclusion currently rests on seed=42
alone. Run each BO experiment over seeds {42, 0, 1, 2, 3} and report mean ± std.
Without this, no result is statistically defensible.

---

## Results Table (fill in as experiments complete)

| Experiment | Val MAE | Test MAE | Val R² | Test R² | Coverage@95 (test) | BO best gap | Top-50 | Top-10% | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (E0)** | 0.4463 | 1.0321 | 0.699 | 0.285 | 0.750 | 9.582 eV | 9 | 47 | 1 |
| E1: ARD + std + warm-start | — | — | — | — | — | — | — | — | — |
| E2: Recalibration + β sweep | — | — | — | — | — | — | — | — | — |
| E3: EI / log-EI / Thompson | — | — | — | — | — | — | — | — | — |
| E4: Spectral-norm DKL | — | — | — | — | — | — | — | — | — |
| E5: PBE pretrain transfer | — | — | — | — | — | — | — | — | — |
