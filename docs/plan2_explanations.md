# Plan 2 Explained — Standard GP-BO vs DKL-BO (Beginner's Guide)

> This document explains the *whole* benchmarking study in plain language, with
> analogies for every technical term, and walks through what each new file does.
> No prior knowledge assumed. If you read only one doc, read this one.

---

## 1. The big picture — what are we even trying to find out?

Imagine you run a kitchen, and **cooking one dish takes a full day** (that's like
running one expensive DFT physics simulation on a material). You have a menu of
**3,351 possible dishes** and you want to find the *tastiest* one — but you can't
cook all of them; that would take 9 years.

So instead you hire a **food critic who can guess** how tasty a dish will be just by
*reading the recipe*, without cooking it. The critic isn't perfect, but they learn:
every time you actually cook a dish and taste it, the critic updates their guesses.
Your strategy becomes: *let the critic point at the most promising recipe, cook only
that one, taste it, let the critic learn, repeat.* After ~100 cooks you've found a
near-best dish having cooked 100 instead of 3,351. **That whole loop is Bayesian
Optimization (BO).**

Now the real question of Plan 2:

> The food critic needs to "read the recipe" somehow. There are **two ways** to turn
> a recipe into something the critic understands. **Which way makes the critic better?**

- **Way A — Standard GP-BO:** *you* hand-write a checklist describing each recipe
  ("how many ingredients, total calories, is it spicy, ..."). Simple, human-designed.
- **Way B — DKL-BO (your existing method):** a **neural network reads the raw recipe
  itself** and *learns* its own way of describing it. Fancier, self-taught.

**Plan 2 is a fair race between Way A and Way B.** That's the entire study.

---

## 2. The key idea: a *fair* race

A race is only meaningful if both runners use the same track. So we force everything
to be identical **except the one thing we're testing** (how the material is described):

| Ingredient | Both methods use... |
|------------|---------------------|
| The prediction engine (the "critic's brain") | the **same** Gaussian Process |
| The "what should I cook next?" rule | the **same** Expected Improvement |
| The starting dishes | the **same** random first picks (same seed) |
| The menu, the budget | the **same** 3,351 materials, same number of cooks |

The **only** difference: Way A feeds the GP *handcrafted descriptors*; Way B feeds it
the *neural-network's learned description*. So whoever wins, wins **because of the
description** — which is exactly what we wanted to measure. This is the heart of the
whole plan.

---

## 3. The vocabulary, decoded with analogies

**Material / 2D material** — a crystal (like a super-thin sheet of atoms). Our "dish".

**C2DB** — the database (cookbook) of ~3,351 materials with known properties.

**Band gap (`gap_hse`)** — an electronic property of a material (measured in eV,
"electron-volts"). Roughly: how much energy it takes to make the material conduct
electricity. High gap = insulator, zero gap = metal, in-between = semiconductor (the
useful ones for chips and solar cells). This is the "tastiness score" we optimize.

**Effective mass (`emass_cbm`)** — how "heavy" an electron behaves inside the material.
*Low* effective mass = electrons zip around easily = good for fast electronics.
> ⚠️ Your original brief called this `emass_cb_1`, but **that column doesn't exist** in
> the database — the real name is `emass_cbm`. We used the real one. 2,667 materials
> have a valid value.

**Surrogate model** — the "food critic": a cheap stand-in that *predicts* the
expensive property so we don't have to run the real physics every time.

**Gaussian Process (GP)** — the specific kind of critic we use. Its superpower:
it gives you a **guess *and* a confidence**. Instead of just "this gap is 4 eV", it
says "**4 eV, give or take 1 eV**". That "give or take" (the uncertainty) is gold for
deciding what to try next. Analogy: a weather forecaster who says "70% chance of rain"
instead of just "rain" — the confidence matters.

**Matérn-5/2 kernel** — a setting inside the GP that controls how it assumes "similar
recipes give similar scores". Just think: the GP's notion of "similarity". Both methods
use the same one.

**CGCNN (Crystal Graph Convolutional Neural Network)** — the neural network in Way B.
It looks at the crystal as a **graph**: atoms are dots, bonds are strings connecting
them. It learns to squeeze each crystal into a list of 32 numbers (an "embedding") that
captures what matters. Analogy: a sommelier who, after tasting thousands of wines,
develops their own private 32-point scoring system that you could never write down by
hand.

**Embedding** — that list of 32 learned numbers describing a material (Way B's
"description").

**Descriptor** — a hand-written number describing a material (Way A's "description"),
e.g. "average atomic weight = 31.2". We built **43** of them.

**Deep Kernel Learning (DKL)** — the combo of "CGCNN + GP" = Way B. "Deep" because of
the neural network, "Kernel Learning" because the network learns how to feed the GP.

**Acquisition function** — the **"what should I cook next?"** rule. It scores every
un-cooked dish and picks the highest. Two flavors:
- **UCB (Upper Confidence Bound)** = `guess + β × confidence`. A dial (β) between
  "play it safe" and "gamble on uncertain ones".
- **Expected Improvement (EI)** = a smarter, dial-free rule: "how much do I *expect*
  this dish to beat my current best?" It automatically balances safe vs. risky. **We
  use EI for the benchmark** because it's parameter-free, so neither method gets to
  tune a knob the other doesn't.

**Seed** — a fixed starting number for the random picks, so the experiment is
**repeatable**. Same seed → same "random" starting dishes. We run **10 different seeds**
so results aren't a fluke (like running a coin-flip experiment 10 times, not once).

**Regret** — how far your best-found dish is from the true best on the whole menu.
Starts high, shrinks as you discover better dishes. **Lower = better.**

**Regret-AUC** — the *area under the regret curve* over all cooks. One number that
captures "how fast did you close the gap to the best?" **This is our headline score for
sample-efficiency** — lower means the method found great materials sooner.

**Top-50 / Top-10%** — "how many of the genuinely rare-great materials did you
discover?" Top-50 = the 50 best on the whole menu; Top-10% = the best 335. More hits =
better hunter.

**Calibration** — is the critic's "give or take" *honest*? If it says "4 ± 1 eV" 100
times, are the true answers within ±1 about 68% of the time? A well-calibrated critic
knows what it doesn't know. We measure this with NLL, Coverage@95, etc.

**Wilcoxon signed-rank test** — a statistics referee. After 10 seeds we have 10
head-to-head matchups (Way A vs Way B on identical starting conditions). This test asks:
"is one method **reliably** better, or is the difference just luck?" It outputs a
**p-value**: small p (< 0.05) = "this is a real, trustworthy difference, not noise."
Because the two methods start from *identical* conditions each seed, they're a **paired**
comparison — which is the strongest, fairest kind of statistical test.

**Bootstrap 95% confidence interval** — a way to put error bars on an average by
re-sampling the data thousands of times. "The true average is very likely in this range."

---

## 4. The four tasks (what we search for)

You asked for all four. The same machinery handles them via a **"sign trick"** (more
below):

| Task | Property | Goal | Why you'd want it |
|------|----------|------|-------------------|
| `gap_max`   | band gap        | **highest** | strong insulators |
| `gap_min`   | band gap        | **lowest** (but > 0) | narrow-gap semiconductors |
| `emass_min` | effective mass  | **lowest**  | fast electronics (electrons zip) |
| `emass_max` | effective mass  | **highest** | heavy-carrier / flat-band materials |

**The sign trick (finding the *lowest* instead of *highest*):** all our machinery is
built to find the *biggest* number. To find the *smallest* instead, we just flip every
score's sign (multiply by −1), find the biggest of *those*, then flip back when
reporting. Analogy: to find the **shortest** person in a line-up using a "find the
tallest" machine, stand everyone in a pit of the same depth flipped upside-down — the
"tallest" upside-down person is the shortest right-side-up. One trick, works everywhere.

---

## 5. Each file, and what it does (with analogies)

### Configuration files (the "settings")
- **`configs/bo/ei.yaml`** — switches the chooser to Expected Improvement and sets the
  budget (100 cooks, 10 starting dishes). *The race rulebook.*
- **`configs/data/c2db_emass.yaml`** — tells the pipeline "this task is about effective
  mass, and reuse the existing crystal graphs." *A new recipe-source card.*
- **`configs/benchmark/bench.yaml`** — the master matrix: which 4 tasks, which 10 seeds,
  which 3 methods. *The tournament bracket.*

### The "describe a material" code (Way A's brain)
- **`src/dklbo/baselines/descriptors.py`** — turns each material into 43 hand-written
  numbers: composition stats (from the chemistry library `pymatgen`) + simple geometry
  (thickness, formation energy, symmetry). **Crucially, it refuses to use any
  electronic-structure shortcuts** (like a cheaper estimate of the band gap) — that would
  be *cheating*, because it's basically peeking at the answer. *The honest hand-written
  checklist.*

### The "search" loops (the actual hunts)
- **`src/dklbo/baselines/feature_bo_loop.py`** (`FeatureBOLoop`) — Way A's full search:
  fit the GP on the descriptors of dishes cooked so far → predict the rest → pick the
  best by EI → cook it → repeat. *Way A's hunting expedition.*
- **`src/dklbo/bo/loop.py`** (existing `BOLoop`, upgraded) — Way B's search (the
  neural-network one). We added the **direction** option (max or min) and taught it to
  use EI. *Way B's hunting expedition.*
- **`src/dklbo/bo/baselines.py`** (`RandomBaseline`, upgraded) — the "monkey throwing
  darts" comparison: pick dishes at random. If a smart method can't beat random, it's
  worthless. *The control group / sanity check.*

### The shared engine pieces (used by both)
- **`src/dklbo/bo/acquisition.py`** — added **`ei()`**, the Expected Improvement rule.
  We verified it mathematically (it matched a brute-force simulation to 4 decimal
  places). *The "what to cook next" calculator.*
- **`src/dklbo/models/dkl.py`** — taught the neural-network model to accept an explicit
  list of answers to learn from (`train_y`). **Why needed:** the saved crystal graphs
  carry *band-gap* labels, but for the effective-mass tasks we must teach it effective
  mass instead — and for "find the lowest" we feed sign-flipped values. *A switch that
  lets the same model study for different exams.*
- **`src/dklbo/models/surrogate.py`** (existing, **reused unchanged**) — the Gaussian
  Process itself. **Both** methods use this same file. That's what makes the race fair.

### The scripts you actually run (the pipeline, in order)
- **`scripts/06_build_emass_dataset.py`** — builds the effective-mass dataset
  (2,667 materials) with proper train/validation/test splits, and checks all their
  crystal graphs already exist (so we skip rebuilding them). *Prep the new menu.*
- **`scripts/05_build_descriptors.py`** — computes the 43 descriptors for every material
  and saves them. *Pre-write all the checklists.*
- **`scripts/07_run_benchmark.py`** — the tournament runner: 4 tasks × 3 methods
  × 10 seeds = **120 searches**. Saves each result. It's **resumable** — if it stops, it
  skips finished runs and continues. *The referee running every match.*
- **`scripts/08_benchmark_stats.py`** — crunches all 120 results into scores
  (regret-AUC, top-K), runs the Wilcoxon test (Way A vs Way B), and computes confidence
  intervals + the offline accuracy/calibration table. *The scoreboard + statistics
  referee.*
- **`scripts/09_plot_benchmark.py`** — draws the publication figures: best-found curves,
  regret curves, rare-material discovery, and a summary bar chart. *The poster maker.*

### Quality control
- **`tests/test_benchmark.py`** — 20 automated checks (EI math is correct, the sign
  trick really finds minimums, descriptors never leak the answer, etc.). Combined with
  the old tests, **62 tests all pass**. *The taste-testers who catch mistakes before they
  matter.*
- **`docs/benchmark_design.md`** — the technical design doc (this file's nerdier sibling).

---

## 6. How it all flows (one picture)

```
        06  build effective-mass dataset  ─┐
        05  write 43 descriptors  ─────────┤→ inputs ready
                                            │
        07  RUN THE TOURNAMENT             ▼
            for each task (4):
              for each method (DKL, Std-GP, Random):
                for each seed (10):
                   start from same random dishes
                   loop 100 times: predict → pick best (EI) → "cook" → learn
                   save the trail of best-found-so-far
                                            │
        08  score everything  ─────────────┤→ regret-AUC, top-K, Wilcoxon p-values, CIs
        09  draw the figures  ─────────────┘→ curves + bars in results/benchmark/plots/
```

---

## 7. How to read the result (what "winning" means)

When the tournament finishes, look at **regret-AUC** per task:
- **DKL-BO wins** if its regret-AUC is **lower** *and* the Wilcoxon **p-value < 0.05**
  (so it's a real difference, not luck). → The neural network's self-taught description
  genuinely helps; the extra complexity is worth it.
- **Standard GP-BO wins or ties** → the simple hand-written checklist already captures
  what matters here; the neural network isn't pulling its weight *for that task*.

**Both answers are good science.** We deliberately built the benchmark to report the
*truth*, not to make DKL look good. An honest "the simple method is just as good" is a
publishable, valuable result.

---

## 8. One-glance run guide

```bash
# 1. build the effective-mass dataset (once)
python scripts/06_build_emass_dataset.py data=c2db_emass data.db_path=data/raw/c2db.db
# 2. write descriptors for both targets (once)
python scripts/05_build_descriptors.py data.db_path=data/raw/c2db.db
python scripts/05_build_descriptors.py data=c2db_emass data.db_path=data/raw/c2db.db
# 3. run the tournament (~40 min, resumable)
python scripts/07_run_benchmark.py bo=ei data.db_path=data/raw/c2db.db
# 4. scores + statistics
python scripts/08_benchmark_stats.py --db_path data/raw/c2db.db
# 5. figures
python scripts/09_plot_benchmark.py
```
Everything lands in **`results/benchmark/`** (one CSV per match in `runs/`, summary
tables, and figures in `plots/`).
