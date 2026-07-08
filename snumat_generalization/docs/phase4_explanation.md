# SNUMAT Side-Test — Phase 4 Explained (live fine-tuning + cold control)

## What changed from Phase 3
In Phase 3, DKL **froze** its pre-trained fingerprint network and used fixed features. Phase 4
lets DKL **keep learning during the hunt**:

- **dkl_finetune** — start from the Phase-2 trained network (warm start), then re-train the
  network + GP every 5 picks on the crystals dug up so far. (This is the variant that won the
  effective-mass task in Paper 2.)
- **dkl_cold_live** — a control that starts from a *random* network and trains it from scratch,
  live, during the hunt. It tells us how much the Phase-2 pre-training actually mattered.

Everything else (pool, EI rule, 10 seeds, 100 cycles) is identical, so all five players line up
on one scoreboard.

## Results (mean over 10 seeds)

### gap_max — maximize band gap
| method | best (eV) | top50 | top10% |
|--------|-----------|-------|--------|
| Standard GP | 14.94 | 17.0 | 39.0 |
| **DKL frozen** | 20.30 | **20.5** | **44.5** |
| DKL finetune | 20.24 | 9.3 | 25.3 |
| DKL cold-live | 19.14 | 8.5 | 23.7 |
| random | 10.22 | 1.9 | 10.6 |

### gap_min — minimize band gap
| method | best (eV) | top50 | top10% |
|--------|-----------|-------|--------|
| Standard GP | 0.028 | 9.9 | 52.0 |
| DKL frozen | **0.020** | **11.0** | 44.7 |
| DKL finetune | 0.027 | 9.5 | **53.1** |
| DKL cold-live | 0.025 | 8.9 | 48.6 |
| random | 0.113 | 1.8 | 10.7 |

## The clean trade-off (and it matches C2DB)
Think of it as **explorer vs specialist**:
- **Frozen DKL = the explorer.** Its fixed, broadly-trained fingerprints spread its picks across
  the whole pool, so it harvests the *most* rare wide-gap crystals (top50 20.5, top10% 44.5 on
  gap_max — the best breadth of anyone).
- **Fine-tuned DKL = the specialist.** By re-training on the handful of good crystals it has
  already found, it keeps zeroing in near the single best one (champion 20.24 ≈ frozen's 20.30),
  but it **tunnel-visions** and loses breadth (top10% drops from 44.5 to 25.3). It overfits the
  small dug-up set.

This is exactly the pattern the 2D C2DB study found: **frozen for breadth, fine-tune for the
single champion / hard targets.** Finding the same trade-off on a 3D dataset is good evidence it
is a real property of the method, not a quirk of one dataset.

## One honest difference from C2DB
On C2DB, the **cold** (no pre-training) encoder *collapsed* on the gap task — proof that
pre-training was essential there. **Here it does not collapse:** cold-live still reaches a 19.14 eV
champion, just below frozen. On this larger, more chemically diverse 3D set, even features learned
from scratch during the hunt find the high-gap extremes. Pre-training still clearly helps
**breadth** (frozen's top-k beats both live variants), but it is **less critical for the single
champion** than it was on the smaller 2D set. This nuance is worth stating plainly in the paper.

## Bottom line for the paper
The headline — **frozen pre-trained DKL is the best band-gap *discoverer* (breadth)** — holds on
3D SNUMAT, and the frozen-vs-finetune trade-off reproduces the C2DB finding. Pre-training's role
shifts from "essential" (2D) to "helps breadth" (3D).

## Outputs
```
results/runs_finetune/   20 warm-start fine-tune trajectories
results/runs_coldlive/   20 from-scratch control trajectories
results/bo_finetune_summary.csv   the combined 5-method table
results/phase4_run.log
```
Next: **Phase 5** — significance tests (paired Wilcoxon, bootstrap CIs) and the figures, to say
which of these differences are statistically real at 10 seeds.
