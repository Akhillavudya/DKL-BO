# Rebuild — Phase 1 Explained (the clean foundation)

*A beginner-friendly walkthrough: the big picture, then file by file, with analogies.*

---

## 0. Where Phase 1 sits in the whole project

We are rebuilding the project cleanly so it answers **one question** with **three methods**
(Random, Std-GP, DKL-BO) across **three studies** (band gap, effective mass, both combined).

```
>> Phase 1  — Build ONE clean dataset            <-- YOU ARE HERE
   Phase 2  — Check the models can predict
   Phase 3  — The BO contest: Std-GP vs DKL-BO vs Random
   Phase 4  — The "well-pretrained DKL" story
   Phase 5  — Plots + write-up
```

Phase 1 builds nothing clever. It just prepares the **ingredients** every later phase eats.

---

## 1. The big picture (in one analogy)

Imagine a **cooking competition**. Two chefs compete:

- **Std-GP** — the *traditional chef*. Works from a written recipe card of 43 hand-measured
  numbers about each dish (its "descriptors").
- **DKL-BO** — the *modern chef*. Doesn't read a recipe card; instead *looks at the raw
  ingredients themselves* (the crystal structure as a graph) and learns what matters.

For the contest to be **fair**, both chefs must cook with the **exact same ingredients**, in
the **same kitchen**, judged by the **same rules**. Phase 1 is the part where we:

1. Decide *which dishes* are even allowed in the competition (the dataset).
2. Hand the traditional chef his recipe cards (descriptors).
3. Hand the modern chef the raw ingredients (graphs).
4. **Double-check both chefs got the identical set of dishes** — no cheating, no mismatch.

If we skip step 4, one chef might quietly get 17 extra dishes and "win" unfairly. That check
is the heart of Phase 1.

---

## 2. The one big decision: the "intersection" dataset

The database has two properties we care about:

- **band gap** (`gap`) — an electronic property, known for ~3,351 materials.
- **effective mass** (`emass`) — only meaningful for semiconductors, known for ~2,667.

**Analogy:** think of two guest lists for a party. List A (gap) has 3,351 names. List B
(emass) has 2,667 names, and *everyone on list B is also on list A* (emass is a subset).

We chose to keep only the people on **both** lists → **2,667 materials**. Why?

- The **combined study (gap + emass together)** can *only* use materials that have *both*
  numbers. So we are forced into the intersection anyway.
- If all three studies use the *same* 2,667 materials, they become directly comparable —
  no "apples vs oranges". This is the single biggest cure for the old project feeling messy.

Cost: we drop ~684 gap-only materials. Worth it for one clean, shared playground.

---

## 3. File by file

### 3a. The one new script — `scripts/01_build_dataset.py`

This is the only script Phase 1 runs. It is plain Python (run with
`python scripts/01_build_dataset.py`). Inside, it has four clearly-named steps:

| Function | What it does | Analogy |
|---|---|---|
| `build_master_table(db_path)` | Scans the C2DB database once, keeps only materials with **both** a valid gap (>0.01 eV, i.e. not a metal) **and** a finite positive emass. Returns a table with columns `id, uid, formula, prototype, gap, emass, n_atoms`. | The bouncer at the door: only guests on *both* lists get in. |
| `add_train_pool_split(df)` | Splits the 2,667 materials into **train** (what models may learn from) and **pool** (the held-out hunting ground). Splits by *structural prototype*, stratified by gap, so rare materials are spread into the pool. | Dealing cards into two piles, making sure the rare "good cards" land in the pile we'll search. |
| `verify_alignment(df, desc, db_path)` | The fairness police. Confirms the descriptor table and the graph cache cover the **same materials in the same order**. | Step 4 of the cooking analogy — both chefs got identical ingredients. |
| `main()` | Glues it together: build table -> split -> save -> build descriptors -> verify. | The head chef running the prep line. |

### 3b. The helper modules it reuses (in `src/dklbo/`)

Phase 1 does **not** reinvent the wheel — it imports proven code from the old project:

| Module | Role in Phase 1 | Analogy |
|---|---|---|
| `data/c2db_loader.py` | We borrow `verify_no_split_leakage()` — it proves no structural "twin" appears in both train and pool. | Making sure identical-twin dishes don't sit on both sides (which would let a chef peek at the answer). |
| `baselines/descriptors.py` | `build_descriptors()` turns each material's formula + structure into **43 hand-crafted numbers** (composition statistics + geometry/stability). It deliberately **excludes** electronic-structure DFT outputs (like the gap itself) so the traditional chef can't cheat. | Writing the recipe card — but banned from writing down the final taste-score. |
| `data/cache.py` | `GraphCache` opens the LMDB file of pre-built crystal **graphs** (atoms + bonds). | The pantry of raw ingredients, already washed and chopped. |

### 3c. Why we did NOT rebuild the graphs

A crystal's **structure** does not depend on which property you study — a structure is a
structure. The old project already built graphs for all 3,351 gap materials, and our 2,667
are a subset of those. So instead of spending ~10 minutes rebuilding identical files, Phase 1
just **verifies** all 2,667 are already in the cache and reuses it.

**Analogy:** the pantry already has every ingredient we need — no need to go shopping again.
We just count to confirm nothing is missing.

---

## 4. What Phase 1 produced (the outputs)

| File | What's inside | Used by |
|---|---|---|
| `data/cache/master.parquet` | The 2,667-material table: `id, uid, formula, prototype, gap, emass, n_atoms, split`. **This is "the dataset".** | Everyone, every phase |
| `data/cache/descriptors.parquet` | 2,667 rows x 43 hand-crafted features, same order as master. | **Std-GP** (the traditional chef) |
| `data/cache/graphs_e98e27ea.lmdb` *(reused)* | Crystal graphs for all materials. | **DKL-BO** (the modern chef) |

### The actual numbers from the run

```
Master dataset : 2,667 materials
  gap   range  : 0.010 - 10.792 eV
  emass range  : 0.001 - 136.193
Split          : train = 1,901   |   pool = 766
Descriptors    : 2,667 x 43 features (35 composition + 8 structural)
Graphs         : all 2,667 present in existing cache (which holds 3,351)  OK
Fairness check : descriptors aligned + all graphs present  OK
```

### Does the pool contain rare targets for all four searches?

Our four search tasks hunt the extremes: highest gap, lowest gap, highest emass, lowest emass.
A split is only useful if those rare materials actually landed in the **pool**. They did:

```
pool has  7 of the high-gap   extreme (1.5% tail)
pool has 10 of the low-gap    extreme
pool has 17 of the high-emass extreme
pool has  8 of the low-emass  extreme
```

So none of the four searches is pointless. (Note: the split is *stratified by gap*, so the gap
tails are deliberately balanced; the emass tails came along naturally and look fine. If the
emass search ever feels off, we can revisit this — a known, deliberate choice.)

---

## 5. The "train vs pool" idea (and the validation-set question)

- **train (1,901)** = what a model is allowed to learn from / pre-train on.
- **pool (766)** = the held-out "exam". BO searches it; prediction accuracy is measured on it.

**The golden rule:** *never look at the pool to make a decision about the model.* The pool is
the exam — peeking inflates your score and a reviewer will catch it.

This is also why we will **not** add a third top-level "validation" split (it would shrink the
pool). Following our two reference papers:

- **During BO search** -> *no* validation set (the GP tunes itself via marginal likelihood).
  *(This is what the Kiyohara & Kumagai DKL-BO paper does.)*
- **During encoder pre-training** -> a small validation set **carved out of `train`** for
  early stopping, so the pool is never touched. *(This is what the DGKL paper does.)*

We settle that in Phase 2. Phase 1's `train / pool` design stays exactly as built.

---

## 6. One-paragraph summary

Phase 1 turned a messy database into **one clean, shared playground of 2,667 materials** that
both competitors (Std-GP and DKL-BO) will fight over fairly. It produced a master table, a
recipe-card table (descriptors) for the traditional chef, reused the pantry of crystal graphs
for the modern chef, split the materials into a learn-from `train` and a held-out `pool`, and —
most importantly — **proved both chefs got the identical ingredients.** Everything later phases
do stands on this foundation.
