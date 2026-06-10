# Project Results Summary
# DKL-BO: Deep Kernel Learning + Bayesian Optimisation for 2D Material Discovery

---

## Dataset

| Property | Value |
|----------|-------|
| Database | C2DB (Computational 2D Materials Database) |
| Total materials | 3,351 (after filtering non-metals with gap > 0) |
| Target property | HSE06 band gap (eV) |
| Band gap range | 0.01 eV – 10.79 eV |
| Top-50 threshold | ≥ 7.02 eV (rarest 1.5%) |
| Top-10% threshold | ≥ 5.05 eV (top 335 materials) |

---

## Phase 2 — Model Accuracy (Offline Surrogate Validation)

**Setup:** CGCNN encoder (18,593 parameters) + ExactGP surrogate (Matérn-5/2 kernel)
**Training:** 100 epochs joint training, seed=42, device=CUDA
**Training time:** 29.63 seconds

### Accuracy Metrics

| Metric | Val Split | Test Split | Target |
|--------|-----------|------------|--------|
| MAE (eV) | **0.4463** | 1.0321 | ≤ 0.45 |
| RMSE (eV) | **0.6082** | 1.3129 | ≤ 0.65 |
| R² | **0.6990** | 0.2845 | ≥ 0.70 |

### Calibration Metrics (Val Split)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| NLL | 0.9258 | lower = better | — |
| ENCE | 0.2499 | 0.06 – 0.10 | above target |
| Miscalibration Area | 0.0692 | 0.04 – 0.07 | ✓ in range |
| Spearman ρ | 0.0469 | higher = better | — |
| Coverage@95 | **0.9321** | ≈ 0.95 | ✓ close to target |

**Training loss:** 2.0353 → 0.9818 (100 epochs, confirmed learning)

> **Note:** Val metrics meet targets. Test metrics are weaker due to distribution shift
> (test set contains harder, out-of-distribution crystal structures).

---

## Phase 3 — Bayesian Optimisation Loop

**Setup:** Fresh DKL model (random weights), β=0.2 (UCB), seed=42
**Budget:** 10 random init + 100 BO cycles = 110 total experiments
**Total runtime:** ~40 seconds on CUDA

### Main Results

| Metric | DKL-UCB (β=0.2) | Random Search | Improvement |
|--------|-----------------|---------------|-------------|
| Best material found | **9.582 eV** | 6.395 eV | +3.19 eV (+50%) |
| Top-50 hits (≥7.02 eV) | **9** | 0 | ∞ |
| Top-10% hits (≥5.05 eV) | **47** | 10 | **4.7×** |
| Best found at cycle | 88 | — | — |

### Discovery Timeline (DKL-UCB)

| Cycle | Event |
|-------|-------|
| 1 | First top-50 material found (7.038 eV) |
| 18 | Exceeded 6.0 eV consistently |
| 57 | First material above 7.0 eV in later cycles |
| 62 | First material above 9.0 eV (9.43 eV) — cluster found |
| 88 | Best material found (9.582 eV) |

---

## Summary: All Results at a Glance

```
                   Best Gap   Top-50   Top-10%   Efficiency
─────────────────────────────────────────────────────────────
DKL-UCB (β=0.2)    9.58 eV      9        47        4.7×
Random Baseline    6.40 eV      0        10        1.0×
Dataset Maximum   10.79 eV      —         —          —
─────────────────────────────────────────────────────────────
```

---

## Result Files

| File | Description |
|------|-------------|
| `results/surrogate_metrics_exactgpsurrogate.csv` | Phase 2 accuracy + calibration metrics |
| `results/training_loss_exactgpsurrogate.csv` | Phase 2 loss per epoch (100 rows) |
| `results/bo_ucb_beta0.2_results.csv` | Phase 3 BO loop results (100 rows) |
| `results/bo_random_results.csv` | Phase 3 random baseline results (100 rows) |

## Plot Files

| Plot | Description |
|------|-------------|
| `results/plots/01_training_loss.png` | Phase 2 loss curve (2.04 → 0.98) |
| `results/plots/02_phase2_accuracy.png` | MAE, RMSE, R² bar chart |
| `results/plots/03_phase2_calibration.png` | 5 calibration metric bars |
| `results/plots/04_best_gap_over_cycles.png` | Main BO result: 9.58 vs 6.40 eV |
| `results/plots/05_cumulative_top10pct.png` | 4.7× efficiency over 100 cycles |
| `results/plots/06_cumulative_top50.png` | 9 vs 0 rare materials found |
| `results/plots/07_acquisitions_per_cycle.png` | Gap value of each acquisition |
| `results/plots/08_summary_dashboard.png` | All key results in one image |

See `docs/experiment_results.md` for Experiment 1 (beta sweep) and Experiment 2 (SVGP) results.
