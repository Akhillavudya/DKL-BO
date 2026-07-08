# Rebuild — Phase 5 Explained (which wins are REAL?)

*A beginner-friendly walkthrough: the big picture, then file by file, with analogies.*
Continues the story from Phases 1–4.

---

## 0. Where Phase 5 sits

```
   Phase 1  — Build ONE clean dataset
   Phase 2  — Check the models can predict
   Phase 3  — The BO contest (frozen)
   Phase 4  — Live fine-tuning + cold control
>> Phase 5  — Statistics + plots: separate signal from luck   <-- YOU ARE HERE
```

Phases 3–4 gave us **averages**. Averages can lie: one lucky starter set can tip a 10-run
mean. Phase 5 asks the only question that matters for a paper: **which differences are real,
and which are just noise?**

---

## 1. The big picture (one analogy)

Imagine two basketball players each take 10 free throws and one sinks 6, the other 5. Is the
first *truly* better, or did he just get lucky on 10 shots? You can't tell from "6 vs 5"
alone — you need a **statistical test** that accounts for how noisy 10 shots are.

That is Phase 5. We have 10 hunts (seeds) per method. For each pair (DKL vs Std-GP) we ask:
*across the matched seeds, does DKL reliably beat Std-GP, or could this gap happen by chance?*

**Why "matched/paired":** seed 3 gives *every* method the **same** 10 starter materials. So we
compare DKL-on-seed-3 directly against Std-GP-on-seed-3, 10 times. Comparing matched pairs is
far more sensitive than comparing two unpaired piles of numbers.

---

## 2. The key ideas in Phase 5

### Idea A — The paired Wilcoxon test → a p-value
For each metric we take the 10 paired differences (DKL − Std-GP) and run a **Wilcoxon
signed-rank test**. It returns a **p-value**: the probability of seeing a gap this big *if the
two methods were actually equal*. **p < 0.05** = "unlikely to be luck" = a real difference.

### Idea B — Bootstrap confidence intervals (CIs)
A single mean (e.g. "top-10% = 42.4") hides its wobble. We **resample** the 10 seeds many
times to get a **95% CI** — a range we're 95% confident the true mean sits in. Narrow CI =
trustworthy; wide CI = shaky. The shaded bands on the curve plots are these CIs.

### Idea C — Regret-AUC (a "how fast did you converge" score)
Besides best/top-k, we compute **regret-AUC**: how far below the *perfect* answer the hunter
stayed, averaged over all cycles. **Lower = found great materials sooner.** It rewards
*sample-efficiency*, complementing the top-k "breadth" metrics.

### Idea D — A negative result is still a result
If a difference is *not* significant, the honest conclusion is "tied," not "we won by a little."
Phase 5's most important job is stopping us from over-claiming.

---

## 3. File by file

### 3a. `scripts/06_stats.py` — the judge
Reads all runs (the three folders), computes per-run metrics, CIs, and paired Wilcoxon tests.

| Piece | What it does | Analogy |
|---|---|---|
| `task_optimum(...)` | the best possible value in the pool (per task; log10 for emass) | the perfect score to measure regret against |
| `per_run_metrics(...)` | one row per (task, method, seed): regret-AUC, best, top-50, top-10% | each player's scorecard |
| `bootstrap_ci(...)` | 95% CI for each mean | the "wobble" around each average |
| `paired_vs_stdgp(...)` | Wilcoxon DKL-variant vs Std-GP, per task & metric | the referee's verdict |
| outputs | `per_run_metrics.csv`, `summary_stats.csv`, `stats_pairs.csv` | the official results sheets |

### 3b. `scripts/07_plot.py` — the artist
Turns the runs into figures, per task.

| Output | What it shows | Analogy |
|---|---|---|
| `{task}_curves.png` | best-so-far vs cycle + cumulative top-10% vs cycle, with 95% CI bands and the pool-optimum dashed line | the "race footage" over time |
| `{task}_bars.png` | final best / top-50 / top-10% bars (mean over seeds) | the final podium |
| (`accuracy.png` from Phase 2) | prediction report cards | the tasting exam scores |

Run: `python scripts/06_stats.py` then `python scripts/07_plot.py`.

---

## 4. What Phase 5 found (the verdict)

**Statistically significant (p < 0.05), excluding the trivial random losses:**

| Task | Finding | p |
|---|---|---|
| gap_max | **DKL-frozen beats Std-GP on top-10%** (42.4 vs 29.4) | **0.002** |
| gap_max | DKL-frozen beats Std-GP on top-50 | 0.002 |
| gap_max | DKL-finetune beats Std-GP on top-10% / top-50 | 0.016 / 0.012 |
| gap_max | cold+live *loses* on best & regret-AUC | 0.031 / 0.037 |
| gap_min | cold+live: better single best, worse breadth | 0.004 / 0.043 |

**Everything else — gap_min (frozen/finetune) and BOTH emass tasks — is statistically TIED
with Std-GP at 10 seeds.**

### How to read it
1. **The one rock-solid win:** pre-trained DKL **significantly out-discovers** descriptors for
   **maximum band gap** (top-k), frozen strongest. This is the headline result.
2. **Pre-training is required:** the cold (no pre-training) variant *significantly loses* on
   gap_max — so the win is caused by the 1,901-material pre-training, not by DKL magic.
3. **The honest correction:** the exciting-looking emass means from Phase 4 (e.g. cold+live
   finding the lowest effective mass) **do NOT survive the test** — they're within noise. We
   therefore claim *nothing* on effective mass yet. This is Idea D in action.

---

## 5. What this means going forward

- **Write-up-ready claim:** "A pre-trained deep-kernel representation significantly improves
  rare high-band-gap material discovery over handcrafted descriptors, and the gain depends on
  pre-training." Clean, defensible, significant.
- **Underpowered but promising:** the emass trade-offs look real in the means but n=10 can't
  confirm them. Bumping emass to **30 seeds** is the natural next step to resolve them.

---

## 6. One-paragraph summary

Phase 5 replaced "looks better" with "is better." Paired Wilcoxon tests + bootstrap CIs show
exactly one robust result — pre-trained DKL significantly beats descriptors at discovering
many high-band-gap materials, and that edge vanishes without pre-training — while every other
task (low band gap, both effective-mass directions) is a statistical tie at 10 seeds. The
figures (per-task curves with CI bands and final bars) visualise all of it. The most valuable
outcome is discipline: the test caught the emass means as noise and stopped an overclaim,
leaving you with one clean, citable finding and a clear, well-motivated next experiment.
