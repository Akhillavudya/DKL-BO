# Concepts Reference — Background for the DKL-BO Rebuild

> **What this file is.** A beginner-friendly glossary of the *timeless* ideas behind the
> project — crystal graphs, CGCNN, Gaussian Processes, Deep Kernel Learning, Bayesian
> Optimisation, and pre-training. These concepts do **not** change between versions of the
> project, so they live here once, as shared background for all the `rebuild_phase*`
> explanation files.
>
> The concrete numbers below are the **rebuild's** numbers (the clean intersection dataset),
> not the old pipeline's. If you ever see a doc quoting "3351 materials" or "MAE 0.45",
> that is the *old* pipeline — see `_archive_old/docs/explanations/` for those.

## Rebuild facts to anchor on

| Thing | Rebuild value |
|-------|---------------|
| Dataset | **2,667 materials** that have BOTH a valid band gap AND a valid effective mass (the *intersection*) |
| Split | **train = 1901 / pool = 766**, prototype-aware + gap-stratified, SEED=42 |
| Graphs | reused from the existing `data/cache/graphs_e98e27ea.lmdb` (structure-only, identical) |
| Surrogate | **ExactGP** only (ARD Matérn kernel), following Paper 2 — **no validation set** |
| Handcrafted rival | **Std-GP**: 43 handcrafted descriptors + the same ExactGP |
| Acquisition | **Expected Improvement (EI)** |
| emass target | modelled in **log10** space (heavy-tailed, ~5 orders of magnitude) |
| Search tasks | gap_max, gap_min, emass_min, emass_max |

---

## 1. What is a crystal graph?

A crystal (like MoS₂) is atoms connected by chemical bonds. We represent it as a **graph**:

- **Nodes** = atoms (each gets a feature vector saying which element it is)
- **Edges** = bonds (each gets a feature describing the bond length)

```
        S
       / \
      Mo  Mo       ←— the "graph"
       \ /
        S
```

The neural network (CGCNN) reads these graphs and learns to predict a property (band gap
or effective mass) from them. The graphs are built once and cached in an LMDB database so
we never rebuild them.

### The 2D vacuum filter (why the graph builder is special)

C2DB materials are 2D sheets, but simulations need a 3D periodic box, so databases add
fake empty space ("vacuum") above and below the sheet:

```
┌─────────────────────────────────┐  ← top of box (~25 Å)
│         V A C U U M             │  ~20 Å empty space
│  ──── S  Mo  S ────             │  ← the actual 2D layer (~4 Å thick)
│         V A C U U M             │
└─────────────────────────────────┘  ← z = 0
```

Because the box repeats infinitely (periodic boundary conditions), a naive neighbour
search with `radius=8 Å` can accidentally bond an atom to a **copy of itself in the layer
above**, creating a fake bond. The fix: after finding neighbours in 3D, delete any pair
whose **z-displacement** exceeds a cutoff (`vacuum_cutoff = 4.0 Å`). We check only *z*
because the vacuum is only in the z-direction; in-plane (x-y) bonds are always kept.

```
IN-PLANE bond:  Mo ── Mo   Δz≈0    → KEEP ✅
CROSS-VACUUM:   S (z=5) ┆ S (z=26) Δz≈21 → DELETE ❌
```

The rebuild **reuses the exact same cached graphs** the old pipeline built, so this filter
still describes how those graphs were made.

### What a PyG `Data` object stores

PyG = PyTorch Geometric. Each graph is a `Data` object:

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `[N_atoms, 90]` | which element each atom is (one-hot) |
| `edge_index` | `[2, N_edges]` | which atoms are bonded (both directions) |
| `edge_attr` | `[N_edges, 10]` | bond length (Gaussian-basis encoded) |
| `y` | `[1]` | the property label |
| `uid` | string | material's unique ID |

Bonds are stored in **both directions** (0→1 and 1→0) because messages flow along edges;
both atoms need to receive from each other. Bond length is spread across 10 soft Gaussian
"bumps" instead of a single number so it is smooth for the network.

---

## 2. Band-gap flavours: gap vs gap_hse vs gap_gw

C2DB stores the band gap computed at three levels of physics theory:

| Method | Key | Accuracy | Cost | Coverage |
|--------|-----|----------|------|----------|
| PBE (DFT) | `gap` | low (underestimates by 30–50%) | fast | most materials |
| HSE06 (hybrid) | `gap_hse` | good | ~10× | ~3363 materials ← **we use this** |
| GW | `gap_gw` | best | ~100× | ~200–500 (rare) |

We use **HSE06** because it is close enough to experiment that discoveries are physically
meaningful, without the extreme cost of GW. (This is why a naive "gap > 0" count of the raw
`gap` field gives a *different, larger* number than the HSE06 set — they are different
columns.)

---

## 3. Prototype-aware splitting (no cheating)

Many C2DB materials are near-identical (same structure, swapped atoms — e.g. MoS₂ vs
MoSe₂). A random train/pool split could put MoS₂ in train and MoSe₂ in the pool; the model
would "recognise" the twin and effectively cheat. This is **data leakage**.

Fix: every material sharing a structural **prototype** goes into the **same** split. The
rebuild adds gap-quartile **stratification** on top, so rare high/low-gap families are
shared across train and pool instead of all landing on one side (otherwise there would be
nothing worth discovering in the pool).

---

## 4. CGCNN — the fingerprint maker

A Gaussian Process needs **fixed-size** input, but crystals have variable numbers of
atoms. The **Crystal Graph Convolutional Neural Network (CGCNN)** is the converter:

```
crystal graph (variable size) → fixed-size fingerprint (32 numbers)
```

Think of it like a fingerprint scanner: any crystal in, always a 32-number "template" out.

How it builds the fingerprint:

1. **Embed** — squash each atom's 90-number one-hot into 32 numbers.
2. **Convolve ×3** — each atom updates its numbers based on its neighbours (a "message"
   from each bonded atom, weighted by a learned sigmoid **gate**). After 3 rounds each
   atom encodes its full local chemical environment, not just its element.
3. **Attention pool** — collapse the N atom vectors into ONE crystal vector, weighting the
   atoms that matter most for the property (mean pooling would dilute the important atom).
4. **Final layer** — one linear transform → the 32-number crystal fingerprint.

---

## 5. Gaussian Process — the predictor with honest uncertainty

A GP predicts not just a value but a **range**:

> "band gap 2.3 ± 0.1 eV" (confident) vs "1.8 ± 0.9 eV" (uncertain)

**House-price analogy:** to price a house it has never seen, a GP finds the most *similar*
houses it *has* seen. Many similar examples → narrow range (confident); few or distant
examples → wide range (uncertain).

- "Houses" = materials, "price" = the property, "neighbourhood" = position in the
  32-dimensional fingerprint space.
- "Similarity" is measured by a **kernel** (the rebuild uses an **ARD Matérn** kernel —
  "ARD" = each fingerprint dimension gets its own length scale).

The ± (uncertainty) is what makes Bayesian Optimisation possible. Without it, we'd just be
guessing.

> **Rebuild note:** we use **ExactGP** only (exact `O(N³)` kernel inversion), because the
> pool is small (766). The old pipeline also had an approximate **SVGP** for large sets —
> not needed here. Following Paper 2, the rebuild uses **no validation set**: the pool stays
> pristine for the search.

---

## 6. Deep Kernel Learning (DKL) — why combine CGCNN + GP

- Train **CGCNN alone** to predict the number → its fingerprints aren't shaped for a GP.
- Train **GP alone** on raw atom vectors → no structural information to work with.
- **DKL** trains them **together**: the CGCNN learns fingerprints that make the GP's
  predictions *and its uncertainty* informative; the GP adapts its kernel to those
  fingerprints. They improve each other via joint backpropagation.

```
crystal graph ─► CGCNN encoder ─► GP kernel (Matérn) ─► mean ± std
        ◄──────── joint backprop: "make fingerprints the GP loves" ────────►
```

The rebuild's DKL comes in flavours the phases compare: **frozen** (encoder pre-trained
then locked), **fine-tuned** (pre-trained then keeps adapting during the hunt), and **cold**
(random init, trained from scratch during the hunt — the Paper-2-exact control).

---

## 7. Accuracy ≠ calibration ≠ discovery (three different skills)

- **Accuracy** — is the predicted *number* right? (MAE, RMSE, R²)
- **Calibration** — is the predicted *uncertainty* honest? (e.g. do 95% of true values
  land inside the 95% interval? — "Coverage@95")
- **Discovery** — does the model rank and explore the *top region* well? (this is what BO
  actually rewards)

A model can predict poorly on average yet be an excellent *discoverer*, because BO cares
about **ranking the best candidates and exploring smartly**, not about getting every
material's value right.

> **This is the rebuild's headline theme.** For band gap, handcrafted descriptors are
> already accurate, yet **pre-trained DKL still discovers a wider set of rare high-gap
> materials**. For effective mass, *both* models have near-zero R² (it is heavy-tailed),
> yet search still works — proving search ≠ accuracy.

---

## 8. Bayesian Optimisation — the smart search

**The problem:** with 766 pool materials but a budget of only ~100 "experiments" (in real
life each is an expensive DFT calculation), which ones do we test to find the best?

**Treasure-hunt analogy:** you have a metal detector and can dig only 100 holes. Random
digging wastes most holes. BO digs a few random holes, then after each dig updates its map
and digs where treasure is most likely — balancing:

```
EXPLOITATION  ◄──────────────────────────►  EXPLORATION
"mine what looks good"          "probe uncertain, unexplored regions"
```

**Acquisition functions** turn (mean, std) into a single score:

- **UCB** (Upper Confidence Bound): `score = mean + β·std`. β=0 is pure exploitation; large
  β is heavy exploration.
- **Expected Improvement (EI)** — **the rebuild's choice** — scores how much a material is
  *expected to beat the best found so far*, automatically blending "probably good" with
  "uncertain enough to maybe be great."

For minimisation tasks (gap_min, emass_min) the same machinery runs on the negated target.

### How one BO cycle runs

```
Start: pick n_init random materials, look up their true values.
Repeat for ~100 cycles:
  1. (periodically) retrain the surrogate on everything labelled so far
  2. predict (mean, std) for every material still in the pool
  3. score them with the acquisition function (EI)
  4. pick the highest-scoring material
  5. "label" it (look up its true value — the "oracle")
  6. record best-so-far, top-50 hits, top-10% hits; move it out of the pool
```

The **oracle** is the cheat-sheet of true values that only exists because this is a
*simulation* of discovery; in a real lab step 5 would be an actual DFT run. To stay fast,
pool embeddings are **cached** and only recomputed when the encoder is retrained.

**Fairness rule:** every method (Std-GP, DKL, Random) uses the **same seed**, so they start
from the **same** initial materials and hunt the **same** pool. Any advantage is from
*smarter features/acquisition*, not luck.

---

## 9. Pre-training / warm-start (why "studying first" matters)

- **Cold DKL** = a blank network that must learn features from scratch using only the
  ~100 examples it digs up during one hunt.
- **Pre-trained (warm-start) DKL** = the same network, but it first **studies the train
  materials** (learning good features), *then* hunts.

**Exam analogy:** the handcrafted **Std-GP** is a student who memorised a good textbook —
solid, but the textbook never improves. Cold DKL walked into the exam having studied
nothing. Pre-trained DKL is the same student after studying thousands of past questions.

**Avoiding the "study the test answers" trap:** the model studies **train** materials, but
everyone hunts the held-out **pool** whose answers are hidden — so a win reflects *better
learned features*, not memorisation. Running **cold DKL as a control** proves the win comes
from the studying (data), not some other quirk: if pre-trained wins but cold doesn't, the
cause is unmistakably the training data.

> **Rebuild finding (the honest headline).** Pre-training does real work: on **band-gap
> maximisation**, pre-trained DKL beats Std-GP on discovery breadth (statistically
> significant), and **cold DKL collapses** — so pre-training is *necessary*, not optional.
> The other tasks (gap_min, both emass) are statistically tied with descriptors at 10 seeds.
> "Learned features overtake handcrafted ones once they have enough data" — but only where
> the data actually teaches something the descriptors miss.

---

## Where to go next

- Per-phase walkthroughs: `rebuild_phase1_explanation.md` … `rebuild_phase5_explanation.md`
- Extensions (epoch sweep, window search): `rebuild_phase6_extensions_explanation.md`,
  `window_max_explanation.md`, `window_min_explanation.md`
- Whole-project narrative: `DKL-BO_full_project_explanation.md`
- The old-pipeline explanations these concepts were distilled from:
  `_archive_old/docs/explanations/`
