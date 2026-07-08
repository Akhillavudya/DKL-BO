# Rebuild — Phase 3 Explained (the hunting contest)

*A beginner-friendly walkthrough: the big picture, then file by file, with analogies.*
Continues the cooking-competition story from Phases 1–2.

---

## 0. Where Phase 3 sits

```
   Phase 1  — Build ONE clean dataset
   Phase 2  — Check the models can predict (tasting exam)
>> Phase 3  — The BO contest: who HUNTS best?           <-- YOU ARE HERE
   Phase 4  — The "well-pretrained DKL" story
   Phase 5  — Plots + write-up
```

Phase 2 asked "can you *predict*?" Phase 3 asks the question the whole project exists for:
**"Given a small budget of experiments, who *finds the best materials fastest*?"**

---

## 1. The big picture (one analogy)

Imagine a **treasure hunt** across 766 islands (the pool). Buried treasure = rare,
valuable materials (e.g. the highest band gaps). Digging on an island = running one
expensive DFT experiment. You only get **110 digs total** (10 free starter digs + 100
guided digs).

Each hunter uses a different strategy to choose where to dig next:

- **Random** — throws a dart at the map. The floor everyone must beat.
- **Std-GP** — the traditional chef, reasoning from his 43 recipe-card numbers.
- **DKL** — the modern chef, reasoning from his 32 learned fingerprint numbers.

After each dig, the hunter *learns* the treasure value there, updates his beliefs, and picks
the next island. This loop — predict → pick the most promising → dig → update — is
**Bayesian Optimization (BO)**.

We score hunters two ways:
- **best** = the single most valuable treasure found (did you find *the* champion?).
- **top-50 / top-10%** = how many rare treasures you collected (did you find *many*?).

---

## 2. The key ideas in Phase 3

### Idea A — A fair race: only the strategy differs
All three hunters share the **same** map (pool), the **same** acquisition rule (how to turn
predictions into a "where to dig next" score), the **same** GP engine, and — crucially — the
**same starting digs** for a given seed. So any difference in results comes purely from the
*features* each uses. This is the fairness backbone inherited from Phase 1.

### Idea B — Expected Improvement (EI), the "where to dig" rule
After fitting the GP, every un-dug island has a predicted value (μ) and an uncertainty (σ).
**EI** scores each island by how much it might *beat the best treasure found so far* — it
balances "probably good" (high μ) against "worth a gamble" (high σ). The island with the
highest EI gets dug next. EI is parameter-free, so both chefs compete on identical terms.

### Idea C — The sign trick (searching for *minimum* too)
BO always *maximizes*. To hunt for the **smallest** value (e.g. lowest band gap, lowest
effective mass), the loop secretly flips the sign of the target (`y_internal = −y_true`),
maximizes that, then flips the answer back. So "find the minimum" reuses the exact same
machinery. **Analogy:** to find the deepest valley, turn the map upside-down and look for the
highest peak.

### Idea D — Many seeds
Treasure hunts are luck-sensitive (your free starter digs are random). So we repeat each
hunt **10 times** with different random starts (seeds) and average. One lucky run proves
nothing; a consistent average means something.

---

## 3. File by file

### 3a. `scripts/04_run_bo.py` — the contest organiser
Runs every (task × method × seed) combination and writes one CSV per run.

| Piece | What it does | Analogy |
|---|---|---|
| `TASKS` dict | the 4 hunts: gap_max, gap_min, emass_min, emass_max (emass in log10) | the four treasure types |
| `make_target(...)` | builds the property column (applies log10 for emass) | choosing which treasure to value, on which ruler |
| `features(...)` | loads descriptors or embeddings, aligned to the pool | handing each hunter his map-reading toolkit |
| `FeatureBOLoop` (std_gp, dkl) | the BO loop over fixed features | a chef hunting |
| `RandomBaseline` (random) | dart-throwing | the floor |
| idempotent `if out.exists(): skip` | resume without redoing work | pause/resume the tournament safely |
| `summarize()` | averages final best / top-k over seeds → `bo_summary.csv` | the final scoreboard |

Run it: `python scripts/04_run_bo.py` (10 seeds × 100 cycles). A quick check:
`python scripts/04_run_bo.py --seeds 1 --cycles 5`.

### 3b. The reused engine (`src/dklbo/`)
| Module | Role | Analogy |
|---|---|---|
| `baselines/feature_bo_loop.py` | the BO loop over a fixed feature matrix (used by BOTH chefs) | the hunting rulebook |
| `bo/baselines.py` (`RandomBaseline`) | random search, same columns/seed as the chefs | the dart-thrower, same rules |
| `bo/acquisition.py` (`ei`) | Expected Improvement scoring | the "where to dig next" compass |
| `bo/loop.py` (`CycleRecord`) | the shared per-dig record format | one identical logbook page everyone fills in |
| `models/surrogate.py` (ExactGP) | the shared GP engine | the shared oven |

Because all three write the **same** `CycleRecord` columns, their CSVs stack together for
plotting in Phase 5.

---

## 4. What Phase 3 found (the scoreboard)

Mean over 10 seeds. **best** = best value found; **top-50 / top-10%** = rare materials collected.
(For emass, `best` is in log10 units; top-k counts are scale-free.)

| Task | Method | best | top-50 | top-10% |
|---|---|---|---|---|
| **gap_max** | Std-GP | 8.671 | 21.3 | 29.4 |
| | **DKL** | 8.638 | **32.0** | **42.4** |
| | random | 7.332 | 6.4 | 9.8 |
| **gap_min** | Std-GP | 0.022 | 19.4 | 28.7 |
| | DKL | 0.023 | 19.8 | 27.1 |
| | random | 0.072 | 6.4 | 10.5 |
| **emass_min** | Std-GP | **−1.999** | 16.5 | 21.8 |
| | **DKL** | −1.714 | **18.9** | **24.1** |
| | random | −1.451 | 7.7 | 10.5 |
| **emass_max** | **Std-GP** | **1.696** | **16.2** | **21.0** |
| | DKL | 1.455 | 12.6 | 16.8 |
| | random | 1.512 | 6.1 | 10.0 |

### How to read it (the honest story)
1. **DKL's superpower is collecting MANY rare materials, not finding the single best.**
   Clearest in **gap_max**: DKL found 42 top-10% materials vs Std-GP's 29 — a big win on
   *breadth* — while the single best was a tie. Great for "give me a batch of candidates."
2. **gap_min is a tie** — descriptors handle band gap fine in both directions.
3. **emass_min — mixed, slight DKL edge:** DKL collects more rare low-emass materials, but
   Std-GP reaches a lower single champion.
4. **emass_max — Std-GP wins clearly,** DKL is weakest (the heaviest-mass extremes are poorly
   represented by the encoder). An honest weak spot.
5. **Everyone beats random everywhere** — both chefs are doing real work.

### Two caveats (important, not yet fixed)
- These are **averages only** — no significance tests yet. Some gaps (gap_min) are likely
  within noise. Wilcoxon tests + confidence intervals (a later step) decide which wins are
  *real* vs lucky.
- **No plots yet** — that is Phase 5.

---

## 5. The tie-back to Phase 2 (the project's whole point)

Phase 2 said both chefs are *poor tasters* of effective mass (R² near zero). Yet here in
Phase 3, DKL still *hunts* low-emass materials better than its tasting score would suggest.
That is the project's thesis in action: **search skill ≠ prediction accuracy.** A model can
mis-estimate exact values yet still *rank* candidates well enough to find treasure.

---

## 6. One-paragraph summary

Phase 3 ran a fair treasure hunt: three strategies, four targets, ten seeds, identical map
and rules, only the features differing. The result is honest and nuanced — DKL excels at
harvesting *many* rare materials (especially maximum band gap), descriptors stay competitive
for the single champion and win outright for maximum effective mass, and both crush random.
The wins still need significance testing (later) and plots (Phase 5), but the core finding of
the rebuild is now reproduced cleanly on one fair dataset.
