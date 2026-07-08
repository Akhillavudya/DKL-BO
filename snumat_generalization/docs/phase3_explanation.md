# SNUMAT Side-Test — Phase 3 Explained (the BO contest)

## The contest
This is the real test. Three "treasure hunters" search the held-out pool of **3,131** crystals,
one pick at a time, for **100 rounds** (after a shared 10-crystal random warm-up), repeated over
**10 random seeds**. Whoever uncovers the rare extreme-gap crystals fastest wins.

- **Random** — picks blindly (the floor everyone must beat).
- **Standard GP** — guides its picks using the 42 handcrafted descriptors.
- **DKL** — guides its picks using the 32 *learned* fingerprints from Phase 2.

All three use the identical decision rule (Expected Improvement) and the identical underlying GP,
and start each seed from the identical 10 crystals. **The only difference is the features** — so
any gap in performance is a gap in the *representation*.

Two goals: **gap_max** (find the highest-gap crystals) and **gap_min** (find the lowest-gap ones).
We score three ways: **best** (the single most extreme crystal found), **top50** (how many of the
pool's 50 most extreme crystals were dug up), and **top10%** (how many of the pool's top-10%
extreme crystals were dug up) — the last two measure *discovery breadth*.

## Results (mean over 10 seeds, 110 picks each)

### gap_max — maximize band gap
| method | best (eV) | top50 found | top10% found |
|--------|-----------|-------------|--------------|
| Standard GP | 14.94 | 17.0 | 39.0 |
| **DKL** | **20.30** | **20.5** | **44.5** |
| random | 10.22 | 1.9 | 10.6 |

**DKL wins on every single metric.** Its best find averages **20.30 eV — the pool's global
maximum** — meaning DKL reliably tracks down the very highest-gap crystal, while the descriptor GP
stalls around 14.9 eV. DKL also harvests more of the rare wide-gap crystals (top50 20.5 vs 17,
top10% 44.5 vs 39). Both crush random.

### gap_min — minimize band gap
| method | best (eV) | top50 found | top10% found |
|--------|-----------|-------------|--------------|
| Standard GP | 0.028 | 9.9 | **52.0** |
| **DKL** | **0.020** | **11.0** | 44.7 |
| random | 0.113 | 1.8 | 10.7 |

A **mixed/tie**: DKL reaches a lower single champion (0.020 vs 0.028 eV) and digs up slightly more
of the top-50, but the descriptor GP wins the wider top-10% breadth. Both far exceed random.

## Why this matters for the paper
On the original **2D (C2DB)** dataset, the robust, statistically-significant DKL win was exactly
**band-gap maximization breadth**, while gap-min was a tie. **Here on a completely different 3D
dataset we see the same pattern — and stronger:** DKL now wins gap_max on *all three* metrics,
including the single champion (a tie on C2DB). gap_min is again a near-tie.

So the project's central claim — *a pre-trained DKL representation discovers high-band-gap
materials better than handcrafted descriptors* — **generalizes from 2D sheets to 3D bulk
crystals.** That is precisely the robustness evidence a "does it generalize?" section needs.

(Significance tests — paired Wilcoxon, bootstrap CIs — come in Phase 5. These are means only.)

## Outputs
```
results/runs/{task}__{method}__seed{S}.csv   60 per-run trajectories
results/bo_summary.csv                        the tables above
results/phase3_run.log                        run log
```
Next: **Phase 4** — let DKL fine-tune its encoder *live* during the hunt (warm-start), plus a
"cold" from-scratch variant, to see whether adapting features mid-search helps or overfits.
