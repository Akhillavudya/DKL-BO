# SNUMAT Side-Test — Phase 2 Explained (prediction accuracy)

## The question
Phase 2 asks: **"How well does each chef PREDICT the band gap of crystals it has never seen?"**
This is a *different* question from "who searches better" (that's Phase 3). Here we just test raw
predictive skill on the held-out pool of 3,131 crystals.

## The two chefs, same kitchen
Both chefs use the **identical** statistical model (a Gaussian Process) and the identical
fairness rules (learn only from the 7,228 training crystals; standardise everything on training
statistics). The **only** difference is what each one looks at:
- **Standard GP** — reads the 42 handcrafted numbers (composition + size/shape).
- **DKL** — reads a 32-number "fingerprint" that a small neural network (CGCNN) *learned* by
  studying the crystal structures during training.

We first train DKL's network on the training crystals to predict the HSE gap (~90 s), freeze it,
and turn every crystal into its 32-number fingerprint. Then both chefs predict the pool.

## Results (held-out pool, 3,131 crystals)
| Chef | MAE ↓ | RMSE ↓ | R² ↑ | 95% coverage |
|------|------|------|------|------|
| Standard GP | 0.669 | 0.982 | 0.627 | 0.975 |
| **DKL** | **0.569** | **0.962** | **0.642** | 0.974 |

- **DKL is the more accurate predictor** here: it is off by ~0.57 eV on average vs ~0.67 eV for
  the descriptor GP, and explains slightly more of the variance (R² 0.642 vs 0.627).
- Both are **well-calibrated**: their 95% confidence intervals actually contain the truth ~97.5%
  of the time (close to the ideal 95%), so neither is over-confident.

## Why this is interesting for the paper
On the old **2D (C2DB)** dataset, the handcrafted descriptors were *slightly more accurate* than
DKL for the gap — the descriptors already captured 2D gap chemistry well. On this **3D (SNUMAT)**
dataset the learned fingerprints **win even on plain accuracy**. That is a first hint that the
DKL representation is doing real, transferable work on a structurally different problem — exactly
the kind of evidence a "does it generalize?" section needs.

Remember the project's recurring theme, though: **good prediction ≠ good search.** Phase 3 is the
real test — does DKL's representation help *discover* the rare extreme-gap crystals faster?

## Outputs
```
results/encoder_gap.pt          the trained CGCNN encoder
results/embeddings_gap.parquet  10,359 x 32 learned fingerprints
results/accuracy.csv            the table above
results/plots/accuracy.png      bar chart (MAE / RMSE / R²)
```
Next: **Phase 3** — the head-to-head Bayesian-optimization contest (Random vs Std-GP vs DKL) to
find the highest- and lowest-gap crystals in the pool.
