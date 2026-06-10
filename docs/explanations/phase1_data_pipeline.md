# Phase 1 — Data Pipeline: Preprocessing + Graph Cache

## What Phase 1 Does (One Line)
Converts the raw C2DB crystal database into a fast-readable cache of crystal graphs that the neural network can understand.

---

## The Big Picture First

Think of the whole project like this:

```
C2DB Database (17,000 materials)
        ↓
  [Phase 1] Convert raw data into a format the AI can read
        ↓
  [Phase 2] Train the AI model (CGCNN + GP) — no loop yet
        ↓
  [Phase 3] Run the Bayesian Optimisation loop
        ↓
  [Phase 4] Scale up + transfer learning
        ↓
  [Phase 5] Analyse results and write up
```

Phase 1 is purely **data preparation**. The AI model can't read a crystal structure directly — we need to convert it into numbers (a "graph") it can understand. That's all Phase 1 does.

---

## Core Concept: What is a Crystal Graph?

A crystal (like MoS₂) is a collection of atoms connected by chemical bonds. We represent it as a **graph**:

```
        S
       / \
      Mo  Mo       ←— This is the "graph"
       \ /
        S
```

- **Nodes** = atoms (each atom gets a feature vector describing what element it is)
- **Edges** = bonds between atoms (each bond gets a feature describing how long it is)

The CGCNN neural network reads these graphs and learns to predict band gap from them.

---

## File-by-File Explanation

### 1. `pyproject.toml` — The Shopping List

**Analogy:** Imagine you're cooking a recipe. Before you start, you write down all the ingredients you need to buy. `pyproject.toml` is that shopping list — but for Python libraries.

```toml
dependencies = [
    "torch",           # PyTorch — the deep learning engine
    "torch-geometric", # For working with graph neural networks
    "ase",             # Atomic Simulation Environment — reads crystal structures
    "gpytorch",        # Gaussian Process library
    "lmdb",            # Fast database for storing graphs
    "hydra-core",      # Config management system
    ...
]
```

When you run `pip install -e .`, Python reads this file and installs everything. You never have to install things one by one.

---

### 2. `configs/` folder — The Settings Panel

**Analogy:** Think of a TV remote. You don't open the TV to change the channel — you use the remote. The `configs/` folder is the "remote" for the entire project. All settings live here, never inside the Python code.

**Why this matters:** If you want to change the band gap target from `gap_hse` to `gap_gw`, you change **one line in a YAML file**, not dig through Python code.

Here's what each config file does:

#### `configs/config.yaml` — The Master Switch
```yaml
defaults:
  - data: c2db_gap    # which dataset config to use
  - model: cgcnn      # which model config to use
  - bo: ucb           # which acquisition function to use
seed: 42              # random seed for reproducibility
```
This is the top-level config. It points to the other config files.

#### `configs/data/c2db_gap.yaml` — Dataset Settings
```yaml
target: gap_hse        # what we're trying to predict
gap_min: 0.01          # ignore metals (their gap = 0 eV)
radius: 8.0            # how far to look for neighbours (in Ångströms)
vacuum_cutoff: 4.0     # don't bond atoms across the vacuum gap
max_neighbors: 12      # each atom can have at most 12 bonds
```

#### `configs/model/cgcnn.yaml` — Neural Network Architecture
```yaml
atom_dim: 90    # each atom is represented by 90 numbers
bond_dim: 10    # each bond is represented by 10 numbers
n_conv: 3       # how many "message passing" layers in the network
pooling: attention  # how to combine atom info into one crystal-level vector
```
These exact numbers come from Paper [P1] — they're not guesses.

#### `configs/model/gp_exact.yaml` vs `gp_svgp.yaml` — Two Modes of the Same Thing
```
gp_exact  → the "Ferrari": very accurate, but runs out of memory with large datasets
gp_svgp   → the "Toyota": slightly less accurate, but handles any dataset size
```
You can switch between them with just a command-line flag: `model=gp_svgp`

#### `configs/bo/ucb.yaml` — Bayesian Optimisation Settings
```yaml
beta: 0.2       # exploration vs exploitation tradeoff (professor's value)
n_init: 10      # start with 10 randomly chosen materials
n_cycles: 100   # then run 100 rounds of smart selection
```

---

### 3. `src/dklbo/data/c2db_loader.py` — The Librarian

**What it does:** Opens the C2DB database, reads all materials, filters out metals, and organises them into train/validation/test groups.

**Analogy:** Imagine a librarian who goes through 17,000 books, removes the ones with missing chapters (no band gap label), removes ones you don't care about (metals), and then splits the rest into three piles: books for studying (train), books for checking your progress (val), and books for the final exam (test).

---

#### Why We Filter Out Metals (gap_min = 0.01 eV)

**First, what is a band gap physically?**

```
SEMICONDUCTOR (gap > 0):          METAL (gap = 0):

Conduction band  ─────────         Conduction band  ─────────
                                                     ← electrons flow freely
      GAP  (e.g. 2.5 eV)           Valence band     ─────────
                                   (overlaps with conduction band)
Valence band     ─────────
```

- **Semiconductors** have an energy gap. Electrons need energy to jump across it. This gap is what we are trying to predict and maximise.
- **Metals** have **zero gap** — their bands overlap. There is literally no band gap to predict. The number is not approximately zero — it is **exactly zero by definition** of being a metal.

**The core problem: what does the data look like?**

If you plot all 8699 band gap values as a histogram, it looks like this:

```
Count
  │
  │  ████  ← SPIKE: ~4132 metals all exactly at 0 eV
  │  ████
  │  ████
  │  ████
  │  ████                  semiconductors spread out here
  │  ████  ██
  │  ████  ████ ███
  │  ████  ████ ████ ██
  │  ████  ████ ████ ████ ██ ██
  └──────────────────────────────────── Band gap (eV)
     0    1    2    3    4    5    6
```

This shape is called **bimodal** — two separate humps. One huge spike at 0, then a smooth distribution of semiconductors.

**Why this breaks the Gaussian Process**

The GP surrogate assumes data comes from one **smooth, continuous** process. It works on the principle:

> "Points that are similar in structure should have similar band gaps."

But metals and semiconductors are **not** similar — they are fundamentally different states of matter. A material with gap = 0.001 eV is a semiconductor. A material with gap = 0.000 eV is a metal. That tiny numerical difference represents a **complete change in physical behaviour**.

If you train the GP on both:

```
WHAT THE GP SEES (with metals):
  4132 examples with gap=0  → "many materials cluster at zero"
  4567 examples with gap>0  → "many others are spread above zero"

WHAT THE GP TRIES TO DO:
  Fit one smooth function over this bimodal distribution

RESULT:
  ✗ High uncertainty everywhere near gap=0 (boundary is confusing)
  ✗ Calibration metrics (ENCE, NLL) become unreliable
  ✗ Model wastes capacity learning "is this metal or not?"
    instead of "how large is the gap for semiconductors?"
```

**What about: "will it still predict metals anyway?"**

Yes — if you feed a metal's graph into the trained model, it outputs a number. But:
1. The model was trained only on semiconductors — it has never seen a metal during training
2. It is **extrapolating outside its training distribution** — like asking a model trained on adults to predict a baby's weight. It gives a number, but it is wrong.
3. More importantly: metals do not have a band gap to predict. Gap = 0 is not a target value, it is a categorical statement — *"this material is metallic, the concept of band gap does not apply."*

**The goal makes the filter obvious**

Our goal is: **"Find materials with the HIGHEST possible band gap."**

Metals have gap = 0. They are the **worst possible** candidates. Including them is like adding toddlers to a search for the world's tallest person — they can never be the answer and only confuse the predictor.

```
WITH metals in the pool:
  BO searches 8699 materials
  4132 of them can NEVER be the answer (gap = 0)
  Model is confused by bimodal distribution
  → Slower, miscalibrated, wastes cycles on useless candidates

WITHOUT metals (our approach):
  BO searches only 4567 materials
  ALL of them are valid candidates
  Model learns a clean, unimodal distribution
  → Faster, better calibrated, every cycle is meaningful
```

**The three options the plan considered (Section 3, Pitfall B)**

```
(a) Filter to gap > 0   ← what we do — simplest, cleanest for high-gap discovery
(b) Two-stage model     ← classify metal/semiconductor first, then regress on non-metals
(c) Model raw gap       ← keep metals but report calibration separately for the two groups
```

We chose **(a)** because our objective is purely "maximise band gap" — metals are irrelevant to that objective, so we remove them before the model ever sees them.

**One-sentence summary:**
We filter metals because a GP trained on a mix of "gap=0 metals" and "gap>0 semiconductors" produces a poorly calibrated, confused model — and since metals can never be the answer to "find the highest band gap material", including them wastes both computation and model capacity.

---

**The critical design choice — Prototype-Aware Splitting:**

In C2DB, many materials are nearly identical (same structure, slightly different atoms). For example, MoS₂ and MoSe₂ have the same crystal structure. If we randomly split them, MoS₂ might go to train and MoSe₂ to test — but the model already "knows" MoS₂ so it cheats when predicting MoSe₂. This is called **data leakage**.

The solution: all materials with the same structural prototype (blueprint) go into the **same split**. No cheating.

```
Random split (BAD):             Prototype-aware split (GOOD):
  Train: MoS₂, WS₂               Train: all MX₂-type structures
  Test:  MoSe₂, WSe₂              Test:  all different prototypes
         ↑ leakage!                       ↑ no leakage
```

**Key functions:**
- `load_c2db(...)` — the main function; returns a table (DataFrame) with one row per material
- `verify_no_split_leakage(...)` — a safety check that screams at you if there's accidental overlap

---

### 4. `src/dklbo/data/graph_builder.py` — The Translator

**What it does:** Takes an ASE Atoms object (a crystal structure) and converts it into a PyTorch graph that the neural network can read.

**This is the most important file for 2D materials correctness.**

**Analogy:** Imagine converting a 3D model of a building into a blueprint. You need rules for what counts as a "connection" between rooms. For a normal building, two rooms are connected if they share a wall. For a 2D material, you have an additional rule: rooms on different floors separated by a huge gap (the vacuum) are NOT connected.

---

#### Why the Vacuum Filter is Designed the Way It Is

**The problem: What is a 2D material in a database?**

A 2D material like MoS₂ is physically just a single sheet. But computers need **3D periodic boxes** to simulate it. So databases like C2DB add fake empty space (vacuum) above and below the layer:

```
┌─────────────────────────────────┐  ← top of simulation box (z = 25 Å)
│                                 │
│         V A C U U M             │  ~20 Å of empty space
│                                 │
│  ──── S  S  S  S  S  S ────   │  ← top sulfur layer    (z ≈ 5 Å)
│  ──── Mo Mo Mo Mo Mo Mo ────   │  ← molybdenum layer    (z ≈ 3 Å)
│  ──── S  S  S  S  S  S ────   │  ← bottom sulfur layer (z ≈ 1 Å)
│                                 │
│         V A C U U M             │  ~0 Å (bottom of box)
└─────────────────────────────────┘  ← z = 0 Å
```

The total box height is ~25 Å, but the actual material is only ~4 Å thick. The rest is vacuum — just empty space.

**The dangerous part: Periodic Boundary Conditions (PBC)**

The simulation box repeats infinitely in all directions (like tiling a floor). So stacked copies look like this:

```
┌─────────────────┐  z = 50 Å
│    VACUUM       │
│  S  Mo  S       │  ← COPY of the layer   (z ≈ 26–30 Å)
│    VACUUM       │
├─────────────────┤  z = 25 Å  (boundary between copies)
│    VACUUM       │
│  S  Mo  S       │  ← ORIGINAL layer      (z ≈ 1–5 Å)
│    VACUUM       │
└─────────────────┘  z = 0 Å
```

A neighbour search with `radius=8 Å` asks: "find all atoms within 8 Å of each atom."  
With 20 Å vacuum, the gap between layers is 20 Å — too large to bridge with radius=8. Safe.  
But with a poorly-prepared structure (only 3 Å vacuum), gap = 7 Å < 8 Å → **a fake bond gets created!**

**The filter:**
```python
z_disp = np.abs(D[:, 2])          # z-component of actual displacement vector
valid  = z_disp <= vacuum_cutoff   # vacuum_cutoff = 4.0 Å
```

We check **only z** (not the full 3D distance) because the vacuum is only in the z-direction. In-plane bonds (x-y) should always be kept.

```
IN-PLANE bond (x-y plane):          CROSS-VACUUM bond (z direction):
Mo ────── Mo                         S (layer 1, z=5 Å)
Δx = 3.2 Å, Δz = 0                  │
→ KEEP ✅                            │ Δz = 21 Å  → DELETE ❌
                                      S (layer 2, z=26 Å)
```

The 4.0 Å cutoff is generous enough to include real intra-layer bonds (Mo–S Δz ≈ 1.6 Å, S–S Δz ≈ 3.2 Å) but strict enough to block any cross-vacuum bond.

---

**Step-by-step of `atoms_to_graph()`:**

```
1. ASE finds all atom pairs within 8 Å radius (including periodic images)
2. For each pair: compute the actual 3D displacement vector D
3. Filter: remove any pair where |D_z| > 4.0 Å   ← the 2D fix
4. For each atom: keep only the 12 nearest neighbours
5. Build atom features: 90-number one-hot vector per atom
6. Build bond features: 10-number Gaussian encoding of distance per bond
7. Package everything into a PyG Data object
```

**What are the features?**

*Atom features (90 numbers):* A one-hot encoding. Only one slot is "1", everything else is "0". The slot position tells you the element.
```
H:   [1, 0, 0, 0, 0, ..., 0]    (position 1)
He:  [0, 1, 0, 0, 0, ..., 0]    (position 2)
Mo:  [0, 0, ..., 1, ..., 0, 0]  (position 42)
```

*Bond features (10 numbers):* Gaussian basis encoding of bond length.
```
distance = 3.2 Å → [0.1, 0.8, 0.9, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
                          ↑ the peak is near the Gaussian centre at ~3 Å
```

---

### 5. `src/dklbo/data/cache.py` — The Smart Filing Cabinet

**What it does:** Stores all the graphs we built so we never have to rebuild them.

**The problem it solves:** Building all 8,000 graphs takes several minutes. If we rebuild them every time we run an experiment (100+ times), that's hours wasted. The cache builds once, stores everything in a fast database (LMDB), and any future run just reads from it.

**The config hash — why it's clever:**

If you change `radius=8.0` to `radius=6.0`, the graphs are completely different. The cache filename includes a fingerprint of the config:

```
radius=8.0, vacuum_cutoff=4.0  →  graphs_a3f2b1c4.lmdb
radius=6.0, vacuum_cutoff=4.0  →  graphs_9d7e2a11.lmdb  ← different file!
```

Changing any setting automatically creates a new cache. No stale data.

**Structure of the LMDB database:**
```
Key:   "MoS2_uid_001"       (material ID string)
Value: <serialized PyG graph object>
```

It's like a dictionary: you give it a material ID, it gives you back the graph.

---

### 6. `src/dklbo/utils/seed.py` — The Reproducibility Pin

**What it does:** One tiny but crucial function: `seed_everything(42)`.

Machine learning involves randomness. Setting the seed to the same number makes everything deterministic — you can run the code a month later and get identical results.

```python
seed_everything(42)
# Now: numpy, pytorch, python's random module all start from the same state
```

---

### 7. `src/dklbo/utils/profiling.py` — The Stopwatch + Memory Meter

**What it does:** Wraps each BO cycle to measure how long it took and how much GPU memory it used.

The plan warns about an "OOM cliff" — as more materials get labelled, the GP's memory usage grows. If we plot memory vs. cycle number and see it rising steeply, we know we need to switch to SVGP.

```python
with profile_cycle(cycle=5, stats_list):
    # ... run one BO cycle ...
# → logs: "Cycle 5: 2.3s  GPU 1240MB  RAM 4200MB"
```

---

### 8. `scripts/01_build_cache.py` — The Builder Script

**What it does:** This is the script you actually **run**. It orchestrates everything above:

```
1. Load configs via Hydra
2. Call c2db_loader.py → get metadata DataFrame
3. Save metadata to metadata.parquet (a fast table format)
4. For each material: call graph_builder.py → build graph
5. Store each graph via cache.py → LMDB database
6. Report throughput: how many graphs/second
```

**How to run it:**
```bash
python scripts/01_build_cache.py data.db_path=/path/to/c2db.db
```

The output tells you:
- How many materials have the target label
- What % are metals (filtered out)
- How many zero-edge graphs (a warning — means vacuum filter might be too strict)
- Graphs per second (the Phase 1 benchmark metric)

---

### 9. `tests/test_vacuum_cutoff.py` — The Safety Net

**What it does:** 7 automated tests that verify the graph builder is correct.

Before a bridge opens, engineers test it with known weights. These tests use structures where we *know* the correct answer and verify the code gives that answer.

The tests check:
1. **Vacuum filter removes cross-vacuum edges** — uses a synthetic 2-atom structure with a known vacuum gap
2. **Intra-layer bonds are kept** — the filter shouldn't remove bonds within the layer
3. **No edges cross vacuum in MoS₂** — uses a real structure with 20 Å vacuum
4. **MoS₂ coordination number** — Mo should have 4–8 S neighbours (ideal = 6 in 2H phase)
5. **Atom feature dimension = 90** — shape check
6. **Bond feature dimension = 10** — shape check
7. **Max neighbours cap works** — set max=4 and verify no atom has more than 4 bonds

Run with: `pytest tests/test_vacuum_cutoff.py -v`

---

## What a PyG Data Object Looks Like

### What is PyG?
PyG = **PyTorch Geometric** — a library for deep learning on graphs. It gives us a container called `Data` to store a graph.

### The simplest possible example

Imagine a tiny MoS₂ fragment: 1 Mo atom bonded to 2 S atoms.

```
     S (atom 0)
    /
   Mo (atom 1)
    \
     S (atom 2)
```

Here's what the PyG `Data` object stores:

```python
Data(
    x          = tensor([[...]],   # atom features — shape [3, 90]
                         [...]],   #   3 atoms, 90 features each
                         [...]]),

    edge_index = tensor([[0, 1, 1, 2],   # source atoms
                         [1, 0, 2, 1]]), # destination atoms — shape [2, 4]

    edge_attr  = tensor([[...],   # bond features — shape [4, 10]
                          [...],  #   4 edges, 10 features each
                          [...],
                          [...]]),

    y          = tensor([3.5]),   # band gap label: 3.5 eV — shape [1]

    n_atoms    = 3,
    n_edges    = 4,
    uid        = "MoS2_001",
    split      = "train"
)
```

### Breaking down each field

#### `x` — Atom Features `[N_atoms, 90]`

A 2D table. One **row per atom**, 90 columns per row.

```
         Element →  H  He Li  ...  Mo  ...  S   ...  Th
         index   →  0   1  2  ...  41  ...  15  ...  89

Atom 0 (S):    [ 0  0  0  ... 0  ...  1   ...  0 ]
                                       ↑
                                  position 15 = S = 1.0

Atom 1 (Mo):   [ 0  0  0  ... 1  ...  0   ...  0 ]
                                  ↑
                             position 41 = Mo = 1.0

Atom 2 (S):    [ 0  0  0  ... 0  ...  1   ...  0 ]
```

Each row is a one-hot vector — all zeros except one "1" that marks the element.

---

#### `edge_index` — Who is bonded to whom `[2, N_edges]`

A pair of lists: `[source_atoms, destination_atoms]`.

```
Bonds in the structure:
  S(0) ↔ Mo(1)   →  two directed edges: 0→1 and 1→0
  Mo(1) ↔ S(2)   →  two directed edges: 1→2 and 2→1

edge_index = [
  [0, 1, 1, 2],   ← source atoms
  [1, 0, 2, 1]    ← destination atoms
]
```

Why **two directions** for each bond? Because in a graph neural network, information flows along edges. Mo needs to receive a message from S (0→1), and S also needs to receive from Mo (1→0). Same physical bond, two communication channels.

---

#### `edge_attr` — Bond Features `[N_edges, 10]`

One **row per edge**, 10 columns = Gaussian encoding of the bond length.

```
Bond length = 2.4 Å
Gaussian centres: [0.0, 0.89, 1.78, 2.67, 3.56, 4.44, 5.33, 6.22, 7.11, 8.0]

edge_attr = [0.00, 0.02, 0.34, 0.97, 0.34, 0.02, 0.00, 0.00, 0.00, 0.00]
                               ↑           ↑
                          peaked near 3rd and 4th centers (≈ 2.67 Å)
```

Instead of just "2.4 Å", we spread that information across 10 soft bumps. This makes it smooth and differentiable for the neural network.

---

#### `y` — The Target Label `[1]`

```python
y = tensor([3.5])   # band gap = 3.5 eV
```

This is what the model must learn to predict from `x`, `edge_index`, and `edge_attr`.

---

## How Everything Connects

```
C2DB database (.db file)
        │
        ▼
c2db_loader.py
  → reads ASE rows
  → filters metals (gap < 0.01 eV removed)
  → prototype-aware split → train/val/test
  → saves metadata.parquet
        │
        ▼ (for each material)
graph_builder.py
  → ASE Atoms object
  → neighbor_list (3D, radius=8 Å)
  → vacuum filter (remove |Δz| > 4 Å)    ← the 2D fix
  → atom features [N, 90]
  → bond features [E, 10]
  → PyG Data object
        │
        ▼
cache.py (LMDB)
  → stores all graphs in graphs_<hash>.lmdb
  → key = material UID, value = graph
        │
        ▼ (Phase 2 reads from here)
CGCNN encoder + GP surrogate
```

---

## Summary Table

| File | Role | Analogy |
|------|------|---------|
| `pyproject.toml` | Lists all library dependencies | Shopping list |
| `configs/*.yaml` | All hyperparameters, no hardcoding | TV remote |
| `c2db_loader.py` | Loads + filters + splits the database | Librarian |
| `graph_builder.py` | Converts crystal → graph (with 2D vacuum fix) | Translator |
| `cache.py` | Stores graphs in a fast database | Filing cabinet |
| `seed.py` | Makes experiments reproducible | Reproducibility pin |
| `profiling.py` | Measures time + memory per cycle | Stopwatch + meter |
| `01_build_cache.py` | The script you actually run | The cook using all ingredients |
| `test_vacuum_cutoff.py` | Verifies correctness with known answers | Safety net |

## PyG Data Object Field Reference

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `[N_atoms, 90]` | What element each atom is (one-hot) |
| `edge_index` | `[2, N_edges]` | Which atoms are bonded (both directions) |
| `edge_attr` | `[N_edges, 10]` | How long each bond is (Gaussian encoded) |
| `y` | `[1]` | The band gap we want to predict (eV) |
| `uid` | string | Material's unique ID |
| `split` | string | `"train"`, `"val"`, or `"test"` |

## How to Run Phase 1

```bash
# Install all dependencies
pip install -e .

# Build the graph cache (point to your C2DB file)
python scripts/01_build_cache.py data.db_path=/path/to/c2db.db

# Run correctness tests
pytest tests/test_vacuum_cutoff.py -v
```
