# Phase 3 — Bayesian Optimisation Loop
# Complete Beginner Explanation

---

## 1. Big Picture First — Where Does Phase 3 Fit?

```
┌──────────────────────────────────────────────────────────────────┐
│                    Full Project Pipeline                         │
│                                                                  │
│  PHASE 1 │ Read C2DB → filter metals → build crystal graphs     │
│          │ Save 3351 graphs to LMDB cache              ✅ DONE   │
│          │                                                       │
│  PHASE 2 │ Train CGCNN encoder + Gaussian Process               │
│          │ Measure accuracy (MAE=0.45 eV, R²=0.70)     ✅ DONE   │
│          │ Measure calibration (Coverage@95=0.93)               │
│          │                                                       │
│  PHASE 3 │ Run the smart search loop          ◄── THIS PHASE     │
│          │ Use the model to GUIDE which material to test next    │
│          │ Compare against random guessing                       │
│          │                                                       │
│  PHASE 4 │ Scaling + transfer learning (pending)                │
│          │                                                       │
│  PHASE 5 │ Analysis + write-up (pending)                        │
└──────────────────────────────────────────────────────────────────┘
```

**Phase 1** built the library of crystal graphs.
**Phase 2** built and tested the prediction brain.
**Phase 3** puts the brain to work — it now GUIDES which materials to test,
instead of just predicting them.

---

## 2. The Problem Phase 3 Solves

We have **3351 materials** in our database, each with a known band gap.

In the **real world**, we wouldn't know the band gap of any material until we run an expensive DFT (quantum chemistry) calculation. Each calculation costs real time and money.

The question Phase 3 answers:
> "If we can only afford 100 DFT calculations, which 100 materials should we test to find the one with the highest band gap?"

**Naive approach — Random Search:**
Pick 100 materials at random. Some will be good, most won't.
Like throwing darts blindfolded.

**Smart approach — Bayesian Optimisation:**
After each experiment, update the model with the new label.
Use the model to predict which UNEXPLORED material looks most promising.
Go test that one. Repeat.
Like throwing darts, but you check where they landed before each throw.

---

## 3. The Core Concept — Bayesian Optimisation (BO)

### The Treasure Hunt Analogy

Imagine you're hunting for treasure buried on a beach. The beach is 1 km long.
You have a metal detector (your model). You can only dig 100 holes.

**Random strategy:** Dig 100 holes randomly spread across the beach.
You'll cover the whole beach but waste most digs on empty spots.

**BO strategy:**
- Dig a few random holes first (initialisation)
- When you find something, dig nearby (exploitation — there might be more)
- But also occasionally dig in places you haven't explored (exploration — the big treasure might be somewhere else)
- After each dig, update your mental map of where treasure probably is

BO balances two competing instincts:
```
EXPLOITATION  ←────────────────────────────────→  EXPLORATION
"Mine what we know is good"             "Explore unknown territory"
(pick highest predicted band gap)       (pick most uncertain material)
```

The formula that balances them is called **UCB** (Upper Confidence Bound):

```
UCB score = predicted_mean + β × predicted_uncertainty

β = 0   → pure exploitation (only care about mean)
β = 0.2 → our setting (slight exploration, mostly exploit)
β = 5   → heavy exploration (mostly go to unknown places)
```

---

## 4. How the Loop Actually Runs — Step by Step

```
START
  │
  │  Pick n_init=10 random materials
  │  "Label" them (look up their true gap_hse)
  │
  ▼
CYCLE 1 (of 100)
  │
  ├── Every 5th cycle: Full DKL retrain
  │     Encode all labelled graphs → CGCNN → fingerprints
  │     Jointly train encoder + GP for 50 epochs
  │
  ├── Other cycles: GP-only refit (fast)
  │     Encoder weights unchanged → fingerprints still valid (cached)
  │     Just update GP on new labels (1 second instead of 30 seconds)
  │
  ├── Predict ALL unlabelled materials
  │     pool_embs = encode(remaining 3341 materials)
  │     mean, std = GP.predict(pool_embs)
  │
  ├── Score each material
  │     score = mean + β × std   (UCB formula)
  │
  ├── Pick the material with highest score
  │     selected_uid = pool_uids[argmax(scores)]
  │
  ├── "Label" it
  │     true_gap = oracle[selected_uid]   (look up in database)
  │
  ├── Record results
  │     best_so_far, top-50 hit?, top-10% hit?
  │
  └── Move selected material from POOL → LABELLED
        Pool shrinks by 1, labelled grows by 1
  │
  ▼
CYCLE 2 → CYCLE 3 → ... → CYCLE 100
  │
  ▼
END: Save results CSV, print summary table
```

---

## 5. Files in Phase 3 — One by One

---

### File 1: `configs/bo/ucb.yaml` — The Settings Card

#### What it does
Stores all the numbers that control how the BO loop behaves.
No numbers are hardcoded in Python — they all live here.

#### Why it exists
If you want to test a different β value or run 200 cycles instead of 100,
you change one line in this YAML file, not the Python code.
This is the Hydra config system from Phase 1 — same idea.

#### The file content explained

```yaml
acquisition: ucb        ← which scoring formula to use
beta: 0.2               ← β in "mean + β × std"

n_init: 10              ← how many random materials to start with
n_cycles: 100           ← how many BO acquisitions to run
retrain_every_k: 5      ← full DKL retrain every 5 cycles; GP-only refit otherwise

n_joint_epochs: 50      ← how many epochs per full retrain (50, not 100 like Phase 2)
n_pretrain_epochs: 20   ← GP warmup before joint training
gp_refit_epochs: 50     ← GP-only refit epochs (no encoder update)
```

**Why n_joint_epochs=50 instead of 100?**
In Phase 2 we trained once for 100 epochs. In Phase 3 we retrain 20 times
(once every 5 cycles over 100 cycles). If each retrain took 30 seconds,
total training would be 10 minutes. With 50 epochs, each retrain takes ~0.5 seconds.

#### Analogy
Like a recipe card. The recipe (Python code) stays the same.
You just change the ingredient amounts (YAML numbers) to make a different dish.

---

### File 2: `src/dklbo/bo/acquisition.py` — The Scoring Function

#### What it does
Defines the formula that scores each material.
The material with the highest score gets selected next.

#### Why it exists
There are multiple possible scoring strategies (UCB, greedy, random).
Having them all in one file with a `get_acquisition(name)` function means
switching strategy is one config change, not a code rewrite.

#### Analogy
Like different methods for picking the next song in a playlist:
- **UCB**: "Play songs I like AND some I haven't heard yet"
- **Greedy**: "Always play my current favourite"
- **Random**: "Shuffle everything"

All three are song-picking strategies. The music app (BO loop) doesn't care which one —
it just calls `next_song()`.

#### Key functions

```python
def ucb(mean, std, beta=0.2):
    return mean + beta * std
    # high mean = predicted to be good (exploit)
    # high std  = we're uncertain about it (explore)
    # beta balances how much we care about each

def greedy(mean, std, **_):
    return mean.clone()
    # ignores uncertainty entirely — pure exploitation

def random_acquisition(mean, std, **_):
    return torch.rand_like(mean)
    # ignores everything — pure random baseline in acquisition form

def get_acquisition(name):
    return _REGISTRY[name]
    # returns the function by name — used by BOLoop
```

**Why `**_` in greedy and random?**
The BO loop always calls `acq_fn(mean, std, beta=β)`.
Greedy and random don't use beta, but they still receive it.
`**_` means "accept any keyword arguments I don't use, ignore them."
This lets all three functions be called identically.

**The key insight about UCB:**
```
Material A: mean=5.0, std=0.1  → score = 5.0 + 0.2×0.1 = 5.02
Material B: mean=3.0, std=8.0  → score = 3.0 + 0.2×8.0 = 4.60
Material C: mean=4.8, std=2.0  → score = 4.8 + 0.2×2.0 = 5.20  ← SELECTED

C wins because it combines a good mean AND decent uncertainty.
A has the best mean but we're already very sure about it.
B has high uncertainty but its mean is low.
```

---

### File 3: `src/dklbo/bo/loop.py` — The Main Engine

#### What it does
This is the heart of Phase 3. The `BOLoop` class runs the full simulation:
initialise → retrain → predict → score → select → label → repeat.

#### Why it exists
The BO logic needs to coordinate the DKL model, the LMDB cache, the acquisition function,
and the oracle (true labels). Putting all of this in one class keeps it testable and clean.

#### Analogy
Like the manager of a mining expedition:
- Starts with a map and 10 random test digs (initialisation)
- After each dig, updates the geological model (retrain)
- Uses the model to decide where to dig next (predict + score)
- Digs, records what was found (label + record)
- Keeps the best-found-so-far on a leaderboard

#### Important class: `CycleRecord` (the scoreboard row)

```python
@dataclass
class CycleRecord:
    cycle:          int     # which cycle number (1–100)
    uid:            str     # which material was selected
    gap_acquired:   float   # its true band gap (eV)
    acquisition_fn: str     # "ucb"
    best_so_far:    float   # best gap found up to this cycle
    n_labelled:     int     # total labelled materials now
    is_top50:       bool    # is this material in the top-50 rarest?
    is_top10pct:    bool    # is this material in the top-10%?
    cumul_top50:    int     # running count of top-50 hits
    cumul_top10pct: int     # running count of top-10% hits
    train_time_s:   float   # seconds spent training this cycle
    predict_time_s: float   # seconds spent predicting this cycle
```

One `CycleRecord` is created per cycle. After 100 cycles, 100 records → one CSV file.

#### Important class: `BOLoop`

```python
class BOLoop:
    def __init__(self, dkl, cache, meta_df, cfg, seed):
        self.oracle = dict(zip(meta_df["uid"], meta_df["target"]))
        # oracle = the "cheat sheet" — all true band gaps
        # In real research, this doesn't exist — you'd actually run DFT
        # In simulation, we just look up the answer
        
        self.top50_threshold    = ...  # 7.02 eV — top 50 materials
        self.top10pct_threshold = ...  # 5.05 eV — top 10% materials
```

#### The embedding cache — why it exists

```
PROBLEM: In each cycle, we need to predict band gaps for ALL unlabelled materials.
         With 3300 materials and n_cycles=100, that's 330,000 encoder forward passes.
         Each pass through the CGCNN takes a tiny amount of time.
         330,000 × tiny = slow.

SOLUTION: The encoder only changes on full-retrain cycles (every 5th cycle).
          Between retrains, the fingerprints don't change.
          So we compute fingerprints ONCE per full-retrain, then REUSE them.

Cycle 1  (full retrain): encode all 3341 pool materials → cache [3341, 32]
Cycle 2  (GP refit):    use cached embeddings → just remove the selected one → [3340, 32]
Cycle 3  (GP refit):    use cached embeddings → remove one more            → [3339, 32]
Cycle 4  (GP refit):    use cached embeddings                              → [3338, 32]
Cycle 5  (GP refit):    use cached embeddings                              → [3337, 32]
Cycle 6  (full retrain): re-encode all remaining → new cache [3336, 32]
...
```

Instead of encoding 3337 graphs 5 times (cycles 2-5), we encode once and reuse.
This is the `pool_embs` tensor that shrinks by 1 each cycle when a material is labelled.

#### How the pool and labelled sets update each cycle

```
BEFORE cycle 5:
  pool_uids    = [A, B, C, D, E, F, ...]   (3337 materials)
  pool_embs    = tensor of shape [3337, 32]
  labelled_uids = [u1..u14]                 (14 materials so far)
  labelled_embs = tensor of shape [14, 32]

CYCLE 5 selects material C (index=2):
  → append C to labelled_uids         → labelled_uids = [u1..u14, C]
  → append pool_embs[2] to lab_embs   → labelled_embs shape = [15, 32]
  → pop C from pool_uids              → pool_uids = [A, B, D, E, F, ...]
  → delete row 2 from pool_embs       → pool_embs shape = [3336, 32]

AFTER cycle 5:
  pool_uids    = [A, B, D, E, F, ...]  (3336 materials — C is gone)
  labelled_uids = [u1..u14, C]         (15 materials)
```

#### The full/GP retrain decision

```python
full_retrain = (cycle % retrain_every_k == 0)

if full_retrain:
    # cycle 1, 6, 11, 16, ... 96
    dkl.fit(labelled_graphs, n_epochs=50, gp_pretrain_epochs=20)
    pool_embs = None        # invalidate cache — encoder changed
    labelled_embs = None
else:
    # cycle 2,3,4,5, 7,8,9,10, ...
    surrogate.fit(labelled_embs, train_y, n_epochs=50)
    # pool_embs still valid — just shrinks by 1
```

**Why does full retrain set pool_embs=None?**
After the encoder is retrained, the fingerprints it produces have changed.
The cached fingerprints from before the retrain are now stale (wrong).
Setting to None forces a re-encoding on the same cycle.

---

### File 4: `src/dklbo/bo/baselines.py` — The Fair Comparison

#### What it does
Implements `RandomBaseline` — a simulated random search that picks materials
uniformly at random each cycle, with no model guidance at all.

#### Why it exists
Without a comparison, there's no way to know if BO is actually useful.
Maybe random search would find the same materials just by luck?
The baseline answers: "What would happen if we just guessed randomly?"

**The answer from our results:**
- Random found best gap = 6.40 eV
- DKL-BO found best gap = 9.58 eV
- Random found 0 top-50 materials
- DKL-BO found 9 top-50 materials
→ BO is clearly, significantly better.

#### Analogy
The baseline is the control group in a science experiment.
You can't say your medicine works if you don't also test patients who got no medicine.
RandomBaseline is the "no medicine" group.

#### Fairness — same seed, same initialisation

```python
# BOLoop and RandomBaseline both use seed=42
# This means they start with the EXACT SAME 10 random initial materials

init_uids = random.sample(self.all_uids, n_init)
# With seed=42, this produces the same list for both
# So any advantage BO has is purely from smarter acquisition, not lucky initialisation
```

#### Key class: `RandomBaseline`

```python
class RandomBaseline:
    def run(self) -> pd.DataFrame:
        # No model. No prediction. No GP.
        for cycle in range(n_cycles):
            idx = random.randrange(len(pool_uids))   # just pick randomly
            selected_uid = pool_uids[idx]
            true_gap = self.oracle[selected_uid]
            # record results, update pool
        return pd.DataFrame(records)
```

**Same return format as BOLoop:**
Both `BOLoop.run()` and `RandomBaseline.run()` return a DataFrame with
identical column names. This means you can plot them together, compare them
row by row, or concatenate them — without any conversion.
The `CycleRecord` dataclass (defined in `loop.py`) is imported by `baselines.py`
to guarantee this.

---

### File 5: `scripts/03_run_bo.py` — The Conductor

#### What it does
The main script that orchestrates everything in the right order:
1. Load metadata + LMDB cache
2. Create a fresh DKL model (random weights — no Phase 2 knowledge)
3. Run BOLoop (100 cycles of smart search)
4. Run RandomBaseline (100 cycles of random search)
5. Save both result tables to CSVs
6. Print the final comparison summary

#### Why it exists
The `BOLoop`, `RandomBaseline`, `DKLModel`, `GraphCache` are all separate modules.
This script is the recipe that uses them in the right sequence.

#### Analogy
Like a race organiser:
- Sets up the track (loads data + builds model)
- Starts both runners (BOLoop and RandomBaseline)
- Records both finish times (saves CSVs)
- Announces the winner (prints summary)

#### Important design decision: fresh model

```python
# Build fresh DKL model — start from random weights
# The BO simulation has no prior knowledge
encoder   = CGCNNEncoder(...)     # random weights
surrogate = build_surrogate(cfg)  # untrained GP
dkl = DKLModel(encoder, surrogate, ...)
```

We do NOT load the Phase 2 trained weights here.
**Why?** In real research, you'd start BO with no prior knowledge.
If we loaded Phase 2 weights (trained on a specific train split), the simulation
would be unfair — the model already "knows" those materials. Phase 4 will explore
transfer learning where we DO use Phase 2 weights as a starting point.

#### How to run it

```bash
# Default: 100 cycles, beta=0.2, ExactGP
python scripts/03_run_bo.py data.db_path=data/raw/c2db.db

# Fewer cycles (quick test)
python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.n_cycles=20

# More exploration (higher beta)
python scripts/03_run_bo.py data.db_path=data/raw/c2db.db bo.beta=1.0

# Switch GP backend
python scripts/03_run_bo.py data.db_path=data/raw/c2db.db surrogate=gp_svgp
```

#### What it prints at the end

```
============================================================
  FINAL SUMMARY  (100 cycles, 10 init)
============================================================
Metric               DKL-UCB      Random
─────────────────────────────────────────────────────────
Best gap found        9.582 eV     6.395 eV
Top-50 hits               9           0
Top-10% hits             47          10
============================================================
```

---

### File 6: `tests/test_bo_loop.py` — The Safety Net

#### What it does
18 automated tests that verify the Phase 3 components work correctly
before you run the full 100-cycle experiment.

#### Why it exists
The BO loop runs for ~40 seconds total. If there's a bug in the acquisition function
or the RandomBaseline, you only find out at the end — after wasting 40 seconds.
The tests catch bugs in 5 seconds using tiny fake data.

#### Analogy
Like testing a fire extinguisher before a fire happens.
You don't want to discover the extinguisher is broken when the kitchen is burning.

#### The 3 groups of tests

**Group 1: Acquisition function tests (7 tests)**
```
test_ucb_output_shape               → scores must be the same shape as inputs
test_ucb_beta_zero_equals_greedy    → β=0 UCB must give same result as greedy
test_ucb_higher_beta_favours_uncertainty → with high β, uncertain materials win
test_greedy_ignores_std             → greedy score must not change if std changes
test_random_acquisition_is_non_deterministic → random must give different scores each call
test_get_acquisition_returns_callable  → "ucb", "greedy", "random" all work
test_get_acquisition_raises_on_unknown → "banana" must raise ValueError
```

**Group 2: RandomBaseline tests (6 tests)**
```
test_random_baseline_returns_dataframe        → returns a DataFrame
test_random_baseline_required_columns         → has all needed columns
test_random_baseline_best_so_far_nondecreasing → best only goes up, never down
test_random_baseline_no_duplicate_selections   → same material never picked twice
test_random_baseline_cumul_top50_nondecreasing → hit count only goes up
test_random_baseline_n_labelled_grows          → labelled set grows by 1 each cycle
```

**Group 3: BOLoop integration tests (5 tests)**
```
test_bo_loop_returns_dataframe         → loop returns a DataFrame
test_bo_loop_required_columns          → has all needed columns
test_bo_loop_best_so_far_nondecreasing → best only goes up
test_bo_loop_no_duplicate_selections   → same material never picked twice
test_bo_loop_cycle_numbers_sequential  → cycles are [1,2,3,...,n_cycles]
```

**How fake data is used in tests:**
The integration test can't use the real 3351-material dataset (too slow).
Instead it creates 25 fake materials with random PyG graphs:

```python
def _fake_graph(uid, gap):
    n = random.randint(2, 5)        # 2-5 atoms
    e = random.randint(n, 2*n)      # n to 2n bonds
    return Data(
        x          = torch.randn(n, 90),    # 90-dim atom features
        edge_index = torch.randint(0, n, (2, e)),
        edge_attr  = torch.randn(e, 10),    # 10-dim bond features
        y          = torch.tensor([[gap]]),
        uid        = uid,
    )
```

The fake graphs have the same structure as real C2DB graphs (90-dim atom features,
10-dim bond features) so the real CGCNN encoder works on them without modification.
Only 3 cycles are run (not 100) with n_epochs=3 (not 50) so the test finishes in ~5 seconds.

---

## 6. What We Achieved at the End of Phase 3

### The Numbers

```
Starting point:   3351 materials, band gaps unknown
Budget:           110 total experiments (10 initial + 100 BO cycles)
Dataset maximum:  10.79 eV

──────────────────────────────────────────────────────────
              DKL-UCB     Random    Improvement
──────────────────────────────────────────────────────────
Best gap:      9.58 eV    6.40 eV      +50%
Top-50 hits:       9          0      ∞ (vs 0)
Top-10% hits:     47         10       4.7×
──────────────────────────────────────────────────────────
```

### In Simple Words

**What random search does in 100 experiments:**
Imagine randomly testing 100 people out of 3351 to find the tallest person.
You'll probably find some tall people but likely miss the really tall ones.
Random found best gap = **6.40 eV**. It found **0 of the 50 rarest** materials.

**What DKL-BO does in 100 experiments:**
After each test, it thinks: "That material had a high gap. Similar materials
(in fingerprint space) probably also have high gaps. Let me test those next."
It builds a map of the chemical space and hunts in the right regions.
DKL-BO found best gap = **9.58 eV** — within 1.2 eV of the theoretical maximum.
It found **9 of the 50 rarest** materials.

### The Discovery Timeline

```
Cycle  18: First exceeded 6.0 eV  (random never got here reliably)
Cycle  57: First exceeded 7.0 eV  (top-50 region starts here)
Cycle  62: Discovered 9.43 eV     (a huge jump — the model found a rich region)
Cycle  88: Discovered 9.58 eV     (the best material found in the whole run)
```

The jump from 6.88 eV (cycle 50) to 9.43 eV (cycle 62) happened because the model
discovered that fluoride materials (like SrP₂F₁₂, BaP₂F₁₂) form a "cluster" of
high-gap materials in fingerprint space. Once it found one, UCB directed it to
explore the whole cluster.

### The 4.7× Efficiency Gain — What It Means

Out of 3351 materials, 335 are in the top-10% (gap > 5.05 eV).
If you had infinite budget, random search would find them all eventually.
But with only 100 experiments:

```
Random:   found 10/335 top-10% materials  (10% hit rate = same as random chance)
DKL-BO:   found 47/335 top-10% materials  (47% hit rate — 4.7× better)
```

**This means:** If a real research lab needed to find 47 high-band-gap materials
and could only afford DFT on N materials, DKL-BO would need ~100 experiments
while random search would need ~470. That's 4.7× fewer experiments, 4.7× less time
and money spent in the lab.

### 8 Graphs Generated

```
results/plots/
  01_training_loss.png        ← loss curve going from 2.04 to 0.98 (learning confirmed)
  02_phase2_accuracy.png      ← MAE=0.45, RMSE=0.61, R²=0.70 bar chart
  03_phase2_calibration.png   ← 5 calibration metrics vs targets
  04_best_gap_over_cycles.png ← 9.58 vs 6.40 eV over 100 cycles (THE main result)
  05_cumulative_top10pct.png  ← 47 vs 10 hits, 4.7× efficiency shown
  06_cumulative_top50.png     ← 9 vs 0 rare materials found
  07_acquisitions_per_cycle.png ← colour-coded scatter of what was found each cycle
  08_summary_dashboard.png    ← all key results in one overview image
```

### How All Phase 3 Files Connect

```
configs/bo/ucb.yaml
  └─ β, n_cycles, n_init, retrain settings
       │ loaded by Hydra into cfg.bo
       ▼
scripts/03_run_bo.py
  └─ orchestrates everything
       │
       ├─► BOLoop (loop.py)
       │     │
       │     ├─ acquisition.py    ← scores each material with UCB
       │     ├─ DKLModel          ← trains encoder+GP, produces predictions
       │     ├─ GraphCache        ← fetches graphs for labelled materials
       │     └─ GraphDataset      ← feeds pool into encoder for batch encoding
       │
       └─► RandomBaseline (baselines.py)
             └─ same oracle, same seed, zero model cost

tests/test_bo_loop.py
  └─ 18 tests verifying all components work before running the full experiment
```

---

## 7. Why Phase 3 Comes Before Phase 4

Phase 3 proved the core idea works:
**A DKL surrogate trained jointly on CGCNN fingerprints + GP
can guide material discovery 4.7× more efficiently than random search.**

Phase 4 will ask: "Can we make it even better?"
- Transfer learning: start from Phase 2 weights instead of random
- SVGP: handle larger labelled datasets without memory issues
- β tuning: find the best exploration-exploitation balance

But all of Phase 4's improvements build on the Phase 3 foundation.
If Phase 3 had failed (BO worse than random), there would be no point proceeding.
Since Phase 3 succeeded, Phase 4 has a strong base to build from.
