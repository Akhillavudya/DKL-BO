# Rebuild — Phase 6 Explained (the follow-up experiments)

*A beginner-friendly walkthrough of three experiments we ran AFTER the core 5 phases:
more seeds, an overfitting check, and a brand-new "target-window" search. Same plain-English,
analogy-first style as the earlier phase docs.*

---

## 0. Where Phase 6 sits

Phases 1–5 built the clean dataset, checked the models can predict, ran the BO contest
(Std-GP vs DKL vs Random), told the "well-pretrained DKL" story, and made the plots.

Phase 6 is **three follow-up questions** we asked once the core results were in:

```
   Phase 1–5  — the core story (done)
>> Phase 6a   — Re-run everything on 30 seeds instead of 10 (more trustworthy)
   Phase 6b   — Does MORE training lower the band-gap error, or cause overfitting?
   Phase 6c   — A different kind of search: find materials INSIDE a band-gap window
```

Nothing here changes the earlier conclusions — it stress-tests and extends them.

---

## 1. Phase 6a — From 10 seeds to 30 seeds

### What a "seed" is (the analogy)
A **seed** is one repeat of the whole experiment with different random luck (which 10
materials you happen to start with, etc.). Running more seeds is like **polling more people**:
30 opinions give a more trustworthy average than 10, and the error bars shrink.

### What we did
The pipeline (scripts `04`–`07`) takes a `--seeds N` flag and is **resumable** — it skips any
per-seed file it already has. So `--seeds 30` kept seeds 0–9 and only computed the 20 new ones
(10–29). We re-ran:

1. `04_run_bo.py --seeds 30`   (std_gp, dkl-frozen, random)
2. `05_run_bo_finetune.py --seeds 30`        (dkl_finetune)
3. `05_run_bo_finetune.py --seeds 30 --cold` (dkl_cold_live)
4. `06_stats.py`  → recomputed Wilcoxon tests + CIs over 30 seeds
5. `07_plot.py --outdir results/rebuild/plots_30seed`  (new `--outdir` flag we added so the
   10-seed plots in `plots/` stay intact for comparison)

### The 30-seed verdict (paired Wilcoxon vs Std-GP, ★ = p<0.05)

| Task | Winner | Note |
|------|--------|------|
| **gap_max** | 🏆 **DKL** | DKL-frozen top-10% **42.7** vs Std-GP 29.8 (p=0.000); fine-tune 35.4 (p=0.001) |
| gap_min | tie | DKL ≈ Std-GP |
| emass_min | tie | all roughly equal |
| emass_max | Std-GP slightly ahead | dkl_cold_live significantly loses |

The payoff of 30 seeds: the **gap_max** DKL win that merely *looked* promising at 10 seeds is
now **statistically solid**, and the confidence bands in the curves got noticeably thinner.

**Files:** `results/rebuild/plots_30seed/*.png`, `summary_stats.csv`, `stats_pairs.csv`,
`per_run_metrics.csv`. (The 10-seed plots stay in `results/rebuild/plots/`.)

---

## 2. Phase 6b — Does more training help, or just overfit?

### The question
Comparing two old accuracy plots, the band-gap MAE looked like it had dropped (0.85 → ~0.76).
Was that because the model **trained longer**? We had to be careful: those two plots used
**different test sets** (1032 vs 766 materials), so they can't be compared directly. To answer
honestly we ran a **controlled experiment**: keep the dataset fixed, change ONLY the training
length.

### Key vocabulary (simple)
- **Epoch** = one full pass through the training materials. 100 epochs = studied them 100 times.
- **MAE** = Mean Absolute Error = how far off the prediction is, on average (in eV).
- **Overfitting** = studying so long you **memorize** the training materials instead of learning
  general patterns → you do *worse* on new materials. The signature: **train error keeps
  dropping while test error flattens then rises**, so you must measure BOTH.

The dial that can overfit is the **CGCNN encoder** (the neural net with many parameters), so we
swept its epochs. (The GP head only tunes a few numbers, so it barely overfits.)

### What we did
New script `scripts/exp_epoch_sweep.py`: for encoder epochs **{25, 50, 75, 100, 150, 200, 300,
400}** × 5 seeds, train the encoder, freeze it, fit the same GP, and measure MAE on **both** the
train materials and the held-out pool. Standard GP is a flat reference (it has no epoch dial).

### The result

| Encoder epochs | Train MAE | Pool MAE (real test) |
|---|---|---|
| 25  | 0.117 | 0.776 |
| 50  | 0.116 | **0.736** |
| 100 *(current)* | 0.096 | 0.739 |
| 200 | 0.087 | 0.729 |
| 300 | 0.084 | **0.728 (best)** |
| 400 | 0.074 | 0.731 |

Standard-GP reference: pool MAE = 0.745.

### What it means
- **Train error keeps dropping** (0.117 → 0.074) — the model fits training data better and better.
- **Pool error drops fast then goes FLAT** — it improves only up to ~50 epochs, then barely moves
  (300 is "best" but only 0.008 eV below 50 — within the noise).
- The **widening gap** between the two lines (train ≈ 0.08 vs pool ≈ 0.73) is the **early sign of
  overfitting** — the model predicts training materials ~9× more accurately than new ones, and
  that gap grows with epochs. It just hasn't pushed pool error back *up* within 400 epochs.

**Direct answers:**
- *Does more training lower the MAE?* Only up to ~50 epochs. Beyond that it lowers only the
  *train* MAE (memorizing), not the real-world pool MAE.
- *Was the earlier 0.85→0.76 drop from more training?* **No — confirmed.** Holding the dataset
  fixed, longer training never reaches 0.76; it plateaus at ~0.73. The earlier difference was the
  **different test set**.
- *Practical takeaway:* 100 epochs is already plenty; ~50 would lose nothing.

**Files:** `results/rebuild/epoch_sweep.csv`, `results/rebuild/plots/epoch_sweep_gap.png`.

---

## 3. Phase 6c — A different search: "find materials in a band-gap WINDOW"

### Why (the science)
Band gaps roughly in **0.7–3.0 eV** absorb visible / near-IR light — useful for solar cells,
photodetectors, LEDs. So instead of maximizing or minimizing the gap, we want gaps that **land
inside a band**. This is a genuinely different objective ("target-window" / level-set search).

### The new math (the acquisition)
Earlier tasks used **Expected Improvement** (chase the extreme). For a window we instead pick the
material **most likely to be inside the band**, using the GP's predicted mean μ and uncertainty σ:

```
a(x) = P(lo <= gap <= hi) = Φ((hi - μ)/σ) - Φ((lo - μ)/σ)
```

(Φ = normal CDF.) We added this as a new `window` acquisition in `bo/acquisition.py`.

### The crucial design insight
We checked the data first. **496 of 766 pool materials (64.8%) are already inside 0.7–3.0 eV.**
That makes the search *easy* — like the crowded `gap_min` task — so we expected a **tie**.

To get a *discriminating* test (where a smarter model can win), the target must be **rare**. The
clean way is to **move the window**, NOT delete materials (deleting = cherry-picking + shrinks the
pool). But there's a trap: the model's error is ~0.73 eV, so a window **narrower than that** can't
be resolved by anyone → tie again. The sweet spot is **rare but wider than ~1 eV**. So we ran two
windows:

| Window | In-window % of pool | Width | Expectation |
|---|---|---|---|
| `visible_0p7_3p0` (0.7–3.0 eV) | 64.8% | 2.3 eV | easy → tie (sanity check) |
| `wide_3p0_4p5` (3.0–4.5 eV) | 15.3% | 1.5 eV | rare + resolvable → discriminating |

### What we did
New script `scripts/exp_window_bo.py` runs **3 methods + Random floor** — Std-GP, frozen DKL,
live fine-tuned DKL, Random — across both windows, 30 seeds, 100 cycles. Success metric =
cumulative count of acquired materials whose true gap is in-window. (Small code change: both BO
loops now pass `window_lo`/`window_hi` from config so they can use the new acquisition; everything
else is unchanged. Confirmed predictions are in raw eV, so the eV bounds are correct for all
methods.)

### Results so far — visible window 0.7–3.0 eV (COMPLETE, 30 seeds)

Metric = in-window materials found out of 100 acquired:

| Method | In-window found | vs Std-GP |
|---|---|---|
| **Standard GP** | **93.1** | — |
| **DKL frozen** | **92.3** | tie (p=0.204) |
| DKL fine-tuned | 82.5 | ★ worse (p<0.001) |
| Random | 63.9 | ★ worse (p<0.001) |

**Reading it:**
1. **Std-GP ≈ DKL-frozen → tie**, exactly as predicted for a crowded target.
2. **Random ≈ 63.9 ≈ the 64.8% base rate** — a beautiful sanity check: blind sampling finds
   in-window materials at the pool's natural fraction, confirming the metric works.
3. Both smart methods clearly beat Random (93 vs 64) — modeling lifts you from the 65% base rate
   to ~93%.
4. **Surprise: live fine-tuning HURT** (82.5, below frozen). On a crowded/easy task, fine-tuning
   the encoder on a handful of live labels distorts the pre-trained map — echoing the earlier
   finding that the live/cold DKL variants underperform. **Frozen pre-trained embeddings win.**

### Incomplete: wide window 3.0–4.5 eV
This is the *interesting* discriminating test, but the run was **stopped early**. Only Std-GP
(25/30 seeds) finished; no DKL ran yet, so **no comparison exists** for the wide window. The
experiment is resumable (completed files are skipped), so re-running computes only the wide window.

**Files:** `results/rebuild/window_runs/*.csv` (raw, resumable), `window_summary.csv`,
`window_stats.csv`, `results/rebuild/plots/window_*.png`.

---

## 4. New / changed files this phase

| File | What |
|---|---|
| `scripts/exp_epoch_sweep.py` | NEW — encoder-epoch sweep (overfitting check) |
| `scripts/exp_window_bo.py` | NEW — target-window BO experiment |
| `src/dklbo/bo/acquisition.py` | added the `window` acquisition + registry entry |
| `src/dklbo/baselines/feature_bo_loop.py` | pass `window_lo`/`window_hi` to acquisition |
| `src/dklbo/bo/loop.py` | pass `window_lo`/`window_hi` to acquisition |
| `scripts/07_plot.py` | added `--outdir` flag (so 30-seed plots don't overwrite 10-seed) |
| `results/rebuild/plots_30seed/` | NEW — 30-seed figures |
| `results/rebuild/epoch_sweep.{csv,png}` | epoch-sweep result |
| `results/rebuild/window_*` | window-search results (visible done, wide partial) |

---

## 5. One-paragraph summary

Re-running on **30 seeds** confirmed and sharpened the core finding: **DKL clearly wins `gap_max`**
and ties elsewhere. A controlled **epoch sweep** showed that **more training does NOT lower the
real-world band-gap error beyond ~50 epochs** — it only lowers the *training* error (the start of
overfitting), so the earlier MAE change was a different-test-set artifact, not more training.
Finally, a new **target-window search** confirmed that on a **crowded** window (0.7–3.0 eV, 65% of
the pool) Std-GP and frozen-DKL **tie** (and both beat Random, which sits exactly at the base
rate), while **live fine-tuning hurts**. The **rare** window (3.0–4.5 eV), where DKL might actually
win, is still unfinished and waiting to be resumed.
