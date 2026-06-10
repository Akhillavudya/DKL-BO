# Phase 2 — Offline Surrogate Validation
# Complete Beginner Explanation

---

## 1. Big Picture First — Where Does Phase 2 Fit?

Before diving into any file, understand where Phase 2 sits in the entire project.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DKL-BO Full Pipeline                         │
│                                                                 │
│  PHASE 0 │ Set up folder structure                             │
│          │                                                      │
│  PHASE 1 │ Read C2DB database                                  │
│          │ Filter metals, build crystal graphs                  │
│          │ Save 3351 graphs to LMDB cache  ◄── DONE            │
│          │                                                      │
│  PHASE 2 │ Build the prediction brain ◄── WE ARE HERE          │
│          │ Train CGCNN + Gaussian Process                       │
│          │ Check: does the model predict well?                  │
│          │ Check: is the uncertainty honest?                    │
│          │                                                      │
│  PHASE 3 │ Run the Bayesian Optimisation search loop           │
│          │ (model guides which material to label next)          │
│          │                                                      │
│  PHASE 4 │ Scaling + transfer learning                         │
│          │                                                      │
│  PHASE 5 │ Analysis + write-up                                 │
└─────────────────────────────────────────────────────────────────┘
```

**In one sentence:**
Phase 1 built a library. Phase 2 builds and tests the brain that reads that library.

**Why test before searching?**
Phase 3 (the BO loop) lets the model *guide* which materials to test next.
If we skip Phase 2 and go straight to Phase 3, and the model is broken,
we'll waste 100 search cycles and never know if the problem is the model or the strategy.
Phase 2 separates model quality from search quality.

---

## 2. The Two Questions Phase 2 Answers

Phase 2 answers exactly two questions about the model before we trust it to guide anything:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Question 1: ACCURACY                               │
│  "Is the predicted band gap number correct?"        │
│                                                     │
│  Question 2: CALIBRATION                            │
│  "Is the predicted uncertainty honest?"             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

These are two completely different things. Here is an analogy:

Imagine a weather forecast:
- **Accuracy**: "It says 25°C tomorrow. It was actually 26°C." → Good accuracy.
- **Calibration**: "It says 90% chance of rain." → Was it actually raining 90% of the time when it said that? If yes, well-calibrated. If it only rained 40% of the time, it's over-confident (badly calibrated).

A model can be accurate but badly calibrated (right numbers, wrong confidence).
A model can be well-calibrated but inaccurate (honest uncertainty, but predictions are off).
Bayesian Optimisation needs **both** to work correctly.

---

## 3. What is CGCNN? (The Fingerprint Maker)

### Start with the problem

We have 3351 crystal materials. Each material is a graph:
- Different number of atoms (MoS₂ has 3, some oxides have 15+)
- Different bonds between atoms
- Different bond lengths and angles

A Gaussian Process (coming next) needs **fixed-size vectors** as input.
You cannot feed a graph of 3 nodes and a graph of 15 nodes to the same GP —
the input sizes are different.

**We need a converter:**
```
Crystal graph (variable size) → Fixed-size number list (always 32 numbers)
```

That converter is the **CGCNN — Crystal Graph Convolutional Neural Network**.

### The fingerprint analogy

Think of each material like a person's fingerprint.
Every fingerprint looks different, but a fingerprint scanner converts it to a
fixed-size number code (a "fingerprint template") that can be compared and stored.

CGCNN is our fingerprint scanner for crystals.
Input: a graph with N atoms and E bonds.
Output: always exactly 32 numbers = the "fingerprint" of that crystal.

### How CGCNN creates the fingerprint — step by step

```
STEP 1: EMBED
Each atom starts as 90 numbers (one-hot encoding, from Phase 1)
A linear layer squashes each atom to 32 numbers

  Atom 1: [90 numbers] → [32 numbers]
  Atom 2: [90 numbers] → [32 numbers]
  Atom 3: [90 numbers] → [32 numbers]

STEP 2: CONVOLUTION × 3 (atoms talk to neighbours)
Each atom looks at its bonds and updates its 32 numbers
based on what its neighbours look like.

  Round 1:
    Atom 1 learns from Atom 2 and Atom 3
    Atom 2 learns from Atom 1
    Atom 3 learns from Atom 1 and Atom 2

  Round 2: atoms learn from their 2-hop neighbours
  Round 3: atoms learn from their 3-hop neighbours

  After 3 rounds: each atom's 32 numbers encode not just
  "what element am I?" but "what is my full chemical environment?"

STEP 3: POOLING (collapse N atom vectors → 1 crystal vector)
We still have N separate atom vectors. We need ONE vector for the whole crystal.
Attention pooling: each atom gets a score (how important is this atom for band gap?)
The final crystal vector = weighted sum of all atom vectors.

STEP 4: FC LAYER (final transform)
One linear layer: 32 → 32
Output: one 32-number "fingerprint" for the whole crystal ✓
```

---

## 4. What is a Gaussian Process? (The Predictor with Uncertainty)

### Start with the problem

We have fingerprints for 3351 materials, but only 162 have been "labelled" by our
experiments (the val/test splits). The rest we need to predict.

We need a predictor that says not just:
> "This material has band gap 2.3 eV"

But:
> "This material has band gap 2.3 ± 0.1 eV"  (confident)
> "This material has band gap 1.8 ± 0.9 eV"  (uncertain)

The ± part (the uncertainty) is what drives Bayesian Optimisation. Without it, we're guessing.

### The house price analogy

Imagine you're estimating house prices.
You've seen prices for houses in neighbourhood A and B.
Now you need to estimate a house in neighbourhood C (never seen).

A Gaussian Process does this:
1. It remembers all the houses it has seen (training data)
2. To estimate a new house, it finds the most *similar* houses it has seen
3. It returns a **range** based on how similar and how many similar houses exist:
   - Many similar houses → narrow range (confident)
   - Few or distant similar houses → wide range (uncertain)

In our project:
- "Houses" = materials
- "Price" = band gap (eV)
- "Neighbourhood" = position in 32-dimensional fingerprint space
- "Similarity" = how close two fingerprints are (measured by the **kernel function**)

### The kernel function (similarity measure)

The kernel decides: "how similar are these two fingerprints?"

We use **Matérn-5/2**, a standard kernel for physical systems.
It assumes: if two fingerprints are close → similar band gaps.
If they are far apart → their band gaps can be very different.

```
Same fingerprint → similarity = 1.0 → GP predicts same value
Similar fingerprint → similarity = 0.8 → GP predicts similar value, low uncertainty
Different fingerprint → similarity = 0.1 → GP predicts something, high uncertainty
Very different → similarity ≈ 0.0 → GP just predicts the mean, max uncertainty
```

---

## 5. What is Deep Kernel Learning (DKL)? (Why Combine Both)

### The problem with training them separately

**Approach A (train CGCNN alone):**
Train CGCNN to predict band gap directly (minimise MAE).
Then take those fingerprints and hand them to a GP.
Problem: CGCNN optimised for "predict the correct number."
But GP needs: "similar materials → close fingerprints." Different objective.

**Approach B (train GP alone, raw features):**
Feed atom one-hot vectors directly to GP without CGCNN.
Problem: the 90-dim one-hot vectors have no structural information.
The GP can't tell MoS₂ with tensile strain from relaxed MoS₂.

**DKL (train them together):**
Let the CGCNN learn fingerprints that make the GP work well.
The GP adjusts its kernel to fit the fingerprints.
They improve each other simultaneously.

```
┌──────────────────────────────────────────────────────────────┐
│                  Deep Kernel Learning                        │
│                                                              │
│   Crystal       CGCNN          GP Kernel       GP Output    │
│   Graph    ──►  Encoder   ──►  (Matérn-5/2) ──► mean ± std │
│  (graph)       (neural       (similarity on                  │
│                network)       fingerprints)                  │
│                                                              │
│   ◄────────── Joint backpropagation ──────────────────────► │
│   "Make fingerprints that maximise GP likelihood"            │
└──────────────────────────────────────────────────────────────┘
```

The key insight: the CGCNN doesn't just learn to predict band gap —
it learns to create fingerprints in a space where **the GP's uncertainty is informative**.

---

## 6. Full Data Flow — Crystal Graph to Prediction

This is the complete journey of one material through Phase 2:

```
C2DB Database
  MoS₂ (uid = "MoS2_1T_123")
  gap_hse = 1.85 eV
      │
      │  (done in Phase 1)
      ▼
LMDB Cache
  PyG Data object:
    x          = [3 atoms × 90 features]   ← atom types
    edge_index = [2 × 12 edges]            ← which atoms are bonded
    edge_attr  = [12 edges × 10 features]  ← bond distances
    y          = [1.85]                    ← band gap label
      │
      │  dataset.py wraps cache into DataLoader
      ▼
DataLoader (batches of 64 graphs at a time)
  Batch of 64 materials:
    x          = [~192 atoms × 90 features]   ← all atoms from all 64 graphs
    edge_index = [2 × ~768 edges]
    edge_attr  = [~768 × 10 features]
    batch      = [~192]  ← tells which atom belongs to which graph
      │
      │  cgcnn_encoder.py
      ▼
CGCNNEncoder
  embed:   [~192 × 90]  →  [~192 × 32]
  conv 1:  [~192 × 32]  →  [~192 × 32]   (atoms talk to neighbours)
  conv 2:  [~192 × 32]  →  [~192 × 32]
  conv 3:  [~192 × 32]  →  [~192 × 32]
  pool:    [~192 × 32]  →  [64 × 32]     ← one vector per graph
  fc:      [64 × 32]    →  [64 × 32]
      │
      │  output: 64 fingerprint vectors, each 32 numbers
      ▼
Embeddings: [64 × 32]  (float32)
      │
      │  convert to float64 for GP stability
      ▼
Gaussian Process (Matérn-5/2 kernel)
  Compares each of the 64 test fingerprints
  against all 3129 training fingerprints
      │
      ▼
Output for each material:
  mean  = 1.82 eV   ← predicted band gap
  std   = 0.08 eV   ← predicted uncertainty
      │
      ▼
Metrics
  Accuracy:    |1.82 - 1.85| = 0.03 eV error  ← small, good
  Calibration: was std=0.08 honest?
               did 95% of true values land inside mean ± 1.96*std?
```

---

## 7. Why Uncertainty Matters for Bayesian Optimisation

### The exploration vs exploitation dilemma

Imagine you're searching for the highest-band-gap material among 3351 candidates.
You can only run expensive DFT calculations on a few at a time.

**Greedy strategy (no uncertainty):**
Always pick the material with the highest predicted band gap.
Problem: the model might be confidently wrong about one material.
You keep testing it and its neighbours, ignoring huge unexplored regions.

**Random strategy:**
Pick randomly. You'll explore everything but waste most tests on obvious low-gap materials.

**Bayesian Optimisation strategy (needs uncertainty):**
Pick materials that are either:
- Predicted to have a high gap (exploitation — mine the good regions)
- OR have high uncertainty (exploration — learn about unknown regions)

The formula: `score = mean + β × std`
(UCB: Upper Confidence Bound, β=0.2 from our config)

```
Material A:  mean=3.5, std=0.1  → score = 3.5 + 0.2×0.1 = 3.52  (confident, high gap)
Material B:  mean=2.0, std=3.0  → score = 2.0 + 0.2×3.0 = 2.60  (uncertain, maybe amazing)
Material C:  mean=3.4, std=0.5  → score = 3.4 + 0.2×0.5 = 3.50  (good gap, moderate uncertainty)

Pick A first → test it, update model
```

If `std` is always wrong (too small or too large), the score formula breaks.
The model either never explores (over-confident) or always explores (under-confident).

**This is why Phase 2 measures calibration. If calibration is bad, Phase 3 will fail.**

---

## 8. Files in Phase 2 — One by One

---

### File 1: `src/dklbo/data/dataset.py`

#### What it does
Wraps the LMDB graph cache (from Phase 1) so that PyTorch's DataLoader
can iterate over it in batches.

#### Why it exists
Phase 1's `GraphCache` is like a dictionary — you look up a material by its ID string.
PyTorch's DataLoader expects a different interface: give me item 0, item 1, item 2...
`GraphDataset` is the translator between these two interfaces.

#### Analogy
The LMDB cache is a library where books are stored by ISBN number.
PyTorch's DataLoader is a reader who asks: "give me book 1, book 2, book 3..."
`GraphDataset` is the librarian who knows: "book 1 = ISBN 978-..., here it is."

#### Important class: `GraphDataset`

```python
class GraphDataset(Dataset):
    def __init__(self, cache: GraphCache, uids: List[str]):
        self.cache = cache    # the LMDB cache from Phase 1
        self.uids  = uids     # list of material IDs we want (train / val / test)

    def __len__(self):
        return len(self.uids)       # "how many items do you have?"

    def __getitem__(self, idx):
        return self.cache[self.uids[idx]]   # "give me item number idx"
```

`uids` is what makes this flexible. You pass in the train IDs → training dataset.
Pass in val IDs → validation dataset. Same class, different scope.

---

### File 2: `src/dklbo/models/cgcnn_encoder.py`

#### What it does
Implements the Crystal Graph Convolutional Neural Network.
Takes a batch of graphs (variable sizes) and returns one 32-number fingerprint per graph.

#### Why it exists
The GP needs fixed-size input. Materials have variable-size graphs.
This file is the converter: graph → fingerprint.

#### Analogy
Like a music app that analyses any song (short or long, any genre)
and produces a fixed-size "audio fingerprint" for copyright detection.
No matter the song length, output is always the same format.

#### Important class 1: `CGCNNConv` (one convolution layer)

This is the most important mathematical piece. The formula:

```
For each atom i, for each neighbour j:
    combined = [atom_i features, atom_j features, bond_ij features]
    gate    = sigmoid(W_gate × combined)   → 0 to 1, "how relevant is j?"
    message = softplus(W_msg  × combined)  → positive number, "what does j say?"
    m_ij    = gate × message               → gated message

New atom_i = LayerNorm(atom_i + sum of all m_ij)
```

```python
class CGCNNConv(MessagePassing):
    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)   # combine everything
        return torch.sigmoid(self.lin_gate(z)) * F.softplus(self.lin_msg(z))

    def forward(self, x, edge_index, edge_attr):
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + aggr)   # residual + LayerNorm
```

**Why sigmoid gate?** Different neighbours have different importance.
A sulphur neighbour of molybdenum matters more than a distant oxygen.
The gate (0–1) learns to weight them automatically.

**Why LayerNorm + residual?** Without LayerNorm, numbers drift after 3 layers —
some atoms end up with values in the thousands, others near zero.
The GP kernel assumes fingerprints are in a reasonable range; LayerNorm enforces this.

#### Important class 2: `CGCNNEncoder` (full encoder)

```python
class CGCNNEncoder(nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):
        h = F.softplus(self.embed(x))          # 90 → 32, all atoms
        for conv in self.convs:                # 3 rounds of message passing
            h = conv(h, edge_index, edge_attr)
        # Attention pooling: N atom vectors → B graph vectors
        attn = pyg_softmax(self.attn_lin(h), batch)   # score per atom (within-graph)
        h = global_add_pool(attn * h, batch)          # weighted sum per graph
        for fc in self.fc_layers:                     # 1 FC layer
            h = F.softplus(fc(h))
        return h   # [B, 32]
```

**Why attention pooling over mean pooling?**
Mean pooling divides by number of atoms. A unit cell with 12 atoms would dilute each
atom's contribution by 12×. Some atoms matter much more than others for band gaps
(e.g., the transition metal in MoS₂ drives the electronic structure).
Attention learns which atoms to weight more heavily.

---

### File 3: `src/dklbo/models/surrogate.py`

#### What it does
Defines the Gaussian Process models. Two options: ExactGP and SVGP.
Both sit behind a shared interface (BaseSurrogate) so the rest of the code
doesn't need to know which one is active.

#### Why it exists
The GP is the uncertainty-producing part of DKL.
It takes fingerprints and returns (predicted band gap, uncertainty).

#### Analogy
Two doctors with different examination methods:
- **ExactGP doctor**: reads every patient's chart before diagnosing you. Thorough, slow for large hospitals.
- **SVGP doctor**: keeps 128 representative "case summaries" and compares against those. Faster, slightly approximate.
Both give you the same format of diagnosis: "likely has condition X, confidence: 85%."

#### Important class: `BaseSurrogate` (the contract)

```python
class BaseSurrogate(ABC):
    def fit(X, y)        → list of losses   # train the GP
    def predict(X)       → (mean, std)      # make predictions
    def joint_loss(X, y) → scalar tensor    # loss for backprop through encoder
    def to(device)                          # move to GPU/CPU
    def train_mode()                        # switch to training mode
    def eval_mode()                         # switch to eval mode
```

This is an **Abstract Base Class** (ABC).
Think of it as a legal contract: "If you want to be a surrogate, you MUST implement all six methods."
If you forget to implement any, Python raises an error immediately.

Why? The BO loop (Phase 3) will call `surrogate.predict(X)` and `surrogate.joint_loss(X, y)`.
It doesn't care if it's ExactGP or SVGP. The ABC guarantees both will respond correctly.

#### Important class: `ExactGPSurrogate`

```
How it works:
1. fit(X_train, y_train):
   - Build the GP model (Matérn-5/2 kernel, ConstantMean prior)
   - Maximise MLL (Marginal Log Likelihood)
     MLL = "how probable is the training data given these GP parameters?"
   - Adam optimiser updates: kernel length scale, output scale, noise
   - Early stopping: stop if MLL doesn't improve for 20 epochs

2. predict(X_test):
   - For each test fingerprint, find similar training fingerprints
   - Return mean (weighted average of training labels) and std (how spread out they are)
   - Output: float32 on CPU (always, regardless of training device)

3. joint_loss(X, y):
   - Compute MLL with gradients ON so backprop can flow to the encoder
```

**Why float64 inside the GP?**
The GP's core operation is inverting an N×N matrix (3000×3000 = 9 million entries).
Matrix inversion amplifies tiny numerical errors. Float32 has 7 significant digits.
For a 3000×3000 inversion, that's not enough — results become NaN or garbage.
Float64 (double precision) has 15 significant digits. Stable.
The encoder still runs in float32 (fast); conversion happens at the boundary.

**Cost: O(N³).**
3000 training points → 3000³ = 27 billion operations per training epoch.
On GPU this is manageable. Above ~3000 points, memory runs out.

#### Important class: `SVGPSurrogate`

Instead of the full N×N kernel matrix, SVGP uses M "inducing points" (M=128 by default).
These are M representative materials that summarise the whole training set.

```
Cost comparison:
  ExactGP: O(N³)       = 3000³   = 27,000,000,000 operations
  SVGP:    O(N × M²)   = 3000 × 128² = 49,152,000 operations
  →  550× cheaper
```

Inducing point locations are learned during training (not fixed).
They start as a random subset of training embeddings and move to where they're most useful.

Training uses **ELBO** (Evidence Lower BOund) instead of MLL —
an approximation to the exact likelihood that works with inducing points.

#### Important function: `build_surrogate(cfg)`

```python
def build_surrogate(cfg):
    if cfg.surrogate_type == "svgp":
        return SVGPSurrogate(n_inducing=cfg.n_inducing, ...)
    return ExactGPSurrogate(lr=cfg.lr, ...)
```

The factory function. You never call `ExactGPSurrogate()` or `SVGPSurrogate()` directly
outside this file. Everything goes through `build_surrogate(cfg)`.
Change the config → different surrogate. No code changes needed.

---

### File 4: `src/dklbo/models/dkl.py`

#### What it does
Combines the CGCNN encoder and the GP surrogate into one DKL model.
Manages the joint training process (warmup + joint, two learning rates).
Provides `encode()` for inference and `cache_pool_embeddings()` for Phase 3.

#### Why it exists
The encoder and surrogate need to be trained together but they're kept in separate files
(different concerns). `DKLModel` is the manager that orchestrates them.

#### Analogy
The encoder is a translator (crystal → fingerprint).
The GP is a predictor (fingerprint → prediction).
`DKLModel` is the project manager who says:
"First let the predictor learn on rough translations (warmup).
Then let both improve together (joint training).
The translator should improve slowly (encoder_lr=0.001),
the predictor can update faster (gp_lr=0.01)."

#### Important class: `DKLModel`

```
fit(train_graphs, n_epochs=100, gp_pretrain_epochs=50)
│
├── PHASE A: GP WARMUP (50 epochs)
│     encode all train graphs (no gradient — encoder frozen conceptually)
│     fit GP on these initial embeddings
│     goal: give GP sensible starting parameters before chaotic joint training
│
└── PHASE B: JOINT TRAINING (100 epochs)
      for each epoch:
        1. forward: all graphs → encoder → embeddings [N, 32]  (float32, grad=True)
        2. loss:    embeddings → GP → MLL or ELBO              (converts to float64)
        3. backward: gradients flow through GP → embeddings → encoder
        4. update:  encoder params (lr=0.001) AND GP params (lr=0.01)
      
      After all epochs:
        final refit: encode all graphs once more (clean), refit GP on final embeddings
```

**Why two phases?**
At epoch 0, the encoder outputs nearly random fingerprints.
If you do joint training immediately, the GP loss is huge and noisy — gradients are chaotic.
The warmup gives the GP a stable starting point. After 50 warmup epochs, joint training
is stable because the GP already understands the rough structure of the fingerprint space.

**Why two learning rates?**

```python
optimizer = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 0.001},   # slow
    {"params": gp_params,            "lr": 0.010},   # fast
])
```

The encoder has millions of parameters and complex non-linear interactions.
If you update it too fast, all materials collapse to the same fingerprint
(called "mode collapse" — the encoder finds a trivial solution that satisfies the GP loss
but destroys all structural information).

The GP has only a few parameters (kernel length scale, output scale, noise level).
These can be updated quickly without instability.

#### Important method: `encode(loader)`

```python
@torch.no_grad()
def encode(self, loader):
    """Encode all graphs → (embeddings [N, D], targets [N]).
    Used for inference (no gradient computation)."""
    for batch in loader:
        e = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        ...
    return torch.cat(embs), torch.cat(targets)
```

Two modes:
- `encode()` — for inference. `@torch.no_grad()` means no gradient tracking. Fast, memory efficient.
- `_encode_with_grad()` — for training. Gradients tracked so backprop can reach the encoder.

#### Important method: `cache_pool_embeddings()` (used in Phase 3)

```python
def cache_pool_embeddings(self, pool_loader):
    """Pre-compute all pool embeddings and store them."""
    emb, _ = self.encode(pool_loader)
    self._embedding_cache = emb
```

In Phase 3, the BO loop runs 100 cycles. In each cycle, it needs to predict band gaps
for all ~3300 remaining unlabelled materials. If we re-encoded all 3300 graphs every cycle,
that's 3300 × 100 = 330,000 forward passes through the encoder — very slow.

The encoder weights only change every K cycles (retrain_every_k=5 from config).
So we encode once, cache the result, reuse it for 5 cycles, then re-encode.

---

### File 5: `src/dklbo/eval/metrics_accuracy.py`

#### What it does
Computes three accuracy metrics: MAE, RMSE, R².
Measures whether the predicted band gap numbers are correct.

#### Why it exists
Accuracy and calibration are separate concerns — separate files, separate functions.
This file only cares about "was the mean prediction right?"
It completely ignores the uncertainty (std) — that is metrics_calibration.py's job.

#### Analogy
Like checking if a scale gives the right weight reading.
You put a 1 kg weight on it. Did it say 1.00 kg? 0.98 kg?
This file measures the scale's accuracy. It doesn't care about the scale's
"I'm 95% confident this is the right weight" — that's calibration.

#### Important dataclass: `AccuracyMetrics`

```python
@dataclass
class AccuracyMetrics:
    mae:  float   # Mean Absolute Error — average |predicted - true| in eV
    rmse: float   # Root Mean Squared Error — like MAE but big errors count more
    r2:   float   # Coefficient of determination — 1.0=perfect, 0.0=no better than mean
```

#### Important function: `compute_accuracy_metrics(y_true, y_pred)`

```
MAE  = mean( |predicted - true| )
       average error in eV
       target: 0.3–0.5 eV (CGCNN on C2DB literature range)

RMSE = sqrt( mean( (predicted - true)² ) )
       like MAE but squares errors first → big errors matter more
       RMSE ≥ MAE always. If RMSE >> MAE: a few materials have huge errors.

R²   = 1 - (sum of squared residuals) / (total variance)
       1.0 = perfect, 0.0 = model just predicts the mean every time
       <0.0 = worse than just predicting the mean
       target: R² > 0.7
```

---

### File 6: `src/dklbo/eval/metrics_calibration.py`

#### What it does
Computes five calibration metrics from research paper [P2]:
NLL, ENCE, Miscalibration Area, Spearman ρ, Coverage@95.
Measures whether the predicted uncertainty (std) is honest and useful.

#### Why it exists
The most critical file for Phase 2. If these metrics fail, the BO loop in Phase 3
will make systematically wrong decisions about what to explore next.

#### Analogy
Like auditing a hospital's diagnosis confidence scores:
"When the doctor says 95% sure, is the patient actually sick 95% of the time?
When the doctor says 60% sure, is it actually 60%?"
If the doctor always says 99% sure but is right only 70% of the time → over-confident.
That doctor's confidence scores cannot be used to prioritise which patients to see next.

Our GP's uncertainty is that doctor. These metrics audit it.

#### Important dataclass: `CalibrationMetrics`

```python
@dataclass
class CalibrationMetrics:
    nll:          float   # Negative log-likelihood
    ence:         float   # Expected Normalised Calibration Error
    miscal_area:  float   # Area between reliability curve and diagonal
    spearman_rho: float   # Spearman correlation(|error|, std)
    coverage_95:  float   # Fraction of true values inside 95% interval
```

#### The five metrics explained

**Metric 1: NLL — Negative Log Likelihood**

```
NLL = -mean( log( P(true_gap | predicted_mean, predicted_std) ) )

Each prediction is a Gaussian: N(mean, std²)
NLL measures: "how probable was the true value under this Gaussian?"

If mean=2.3, std=0.1 and true=2.31:  very probable → low NLL (good)
If mean=2.3, std=0.1 and true=3.80:  very improbable → high NLL (bad)
If mean=2.3, std=2.0 and true=3.80:  moderate probability → medium NLL
   (the wide std saves you but inflates NLL even for correct predictions)

NLL penalises both wrong predictions AND wrong uncertainty sizes.
Lower is better. No specific target — compare ExactGP vs SVGP.
```

**Metric 2: ENCE — Expected Normalised Calibration Error (target: 0.06–0.10)**

```
Question: "Is the predicted std consistently the right SIZE?"
A model might have correct relative ordering of uncertainties (uncertain materials are
more uncertain than confident ones) but the actual scale might be off by 3×.

Method:
  1. Sort all test materials by predicted std (low to high)
  2. Split into 10 equal bins (bin 1 = most confident predictions)
  3. In each bin, compare:
       RMV  = sqrt(mean(std²))     ← model's claimed average uncertainty
       RMSE = sqrt(mean(error²))   ← actual average error
  4. ENCE = mean over bins of |RMV - RMSE| / RMV

┌──────────────────────────────────────────────────────┐
│  Bin 1 (most confident):  RMV=0.10, RMSE=0.11  ✓    │
│  Bin 2:                   RMV=0.15, RMSE=0.16  ✓    │
│  ...                                                 │
│  Bin 10 (least confident): RMV=1.20, RMSE=1.18 ✓    │
│  ENCE = 0.07  (well-calibrated, within target)       │
└──────────────────────────────────────────────────────┘

ENCE=0.0: perfect scaling. Target: 0.06–0.10.
```

**Metric 3: Miscalibration Area (target: 0.04–0.07)**

This draws the "reliability curve" and measures how far from perfect it is.

```
For each confidence level p (5%, 10%, 15%, ..., 95%):
  Expected: p% of materials should have true value inside the p% interval
  Actual:   count what fraction actually do

Plot actual vs expected → should be the diagonal line (y=x)

       1.0 ┤                              ╱ ← perfect (y=x)
           │                          ╱
  Actual   │                  ╱
  fraction │            ╱
           │      ╱
       0.0 └──────────────────────────────── Expected fraction
           0.0                          1.0

Over-confident → curve BELOW diagonal (claiming 95% but only 70% actually inside)
Under-confident → curve ABOVE diagonal (claiming 95% but 99% inside — too wide)

Miscalibration area = mean( |actual - expected| ) over all 19 confidence levels
Target: 0.04–0.07. Perfect: 0.0.
```

**Metric 4: Spearman ρ — Does the model know when it's wrong? (higher = better)**

```
Question: "When the model is uncertain, is it actually more wrong?"
This is critical for Bayesian Optimisation.

Spearman rank correlation between:
  |error_i| = |predicted_i - true_i|   (how wrong was this prediction?)
  std_i                                 (how uncertain was the model?)

ρ near 1.0: uncertain predictions are indeed wrong ones ← ideal for BO
ρ near 0.0: uncertainty has no relationship to actual errors ← useless for BO
ρ negative: model is MORE uncertain when it's actually RIGHT ← dangerous

Example:
  Material A: std=0.8 (uncertain), error=0.9 → ρ contribution: positive ✓
  Material B: std=0.1 (confident), error=0.1 → ρ contribution: positive ✓
  Material C: std=0.8 (uncertain), error=0.0 → ρ contribution: negative ✗
```

**Metric 5: Coverage@95 (target: ≈ 0.95)**

```
Simplest calibration check.

For each test material:
  95% interval = [mean - 1.96×std,  mean + 1.96×std]
  Is the true band gap inside this interval?

Coverage@95 = fraction of materials where true value is inside 95% interval

Target: 0.95
  If coverage = 0.70: model is over-confident (intervals too narrow)
  If coverage = 0.99: model is under-confident (intervals too wide)
  If coverage = 0.95: perfectly calibrated on this metric ✓
```

#### Important function: `print_metrics_table(acc, cal, split, surrogate)`

Prints a formatted summary table to the terminal when you run `02_eval_surrogate.py`:

```
=======================================================
  Surrogate: ExactGPSurrogate   Split: test
=======================================================
  ACCURACY
    MAE          0.3821 eV
    RMSE         0.5114 eV
    R²           0.7233
  CALIBRATION  (targets from [P2])
    NLL          0.8431        (lower = better)
    ENCE         0.0812        (target 0.06–0.10)
    Miscal. area 0.0531        (target 0.04–0.07)
    Spearman ρ   0.4123        (higher = better)
    Coverage@95  0.9417        (target ≈0.95)
=======================================================
```

---

### File 7: `scripts/02_eval_surrogate.py`

#### What it does
The main script for Phase 2. Orchestrates everything:
loads data → builds model → trains → evaluates → saves results.

#### Why it exists
Each component (encoder, GP, metrics) is in its own module.
This script is the conductor that calls them in the right order.

#### Analogy
Like a recipe card. The ingredients (encoder.py, surrogate.py, metrics.py)
are in different cabinets. The recipe (02_eval_surrogate.py) says exactly:
"First take this, then mix with that, then measure the result."

#### How to run it

```bash
# ExactGP (default):
python scripts/02_eval_surrogate.py data.db_path=data/raw/c2db.db

# SVGP (swap surrogate):
python scripts/02_eval_surrogate.py data.db_path=data/raw/c2db.db model=gp_svgp
```

#### Step-by-step execution

```
Step 1: Load metadata.parquet
  → knows 3351 materials, which are train/val/test, what their gap_hse is
  → log: "Loaded metadata: 3351 materials. Split: train=3129, val=162, test=60"

Step 2: Open LMDB cache (read-only)
  → same cache built in Phase 1
  → no graph-building happens here — already done

Step 3: Build DataLoaders
  → train: 3129 graphs, batch_size=64, shuffle=True
  → val:   162 graphs, batch_size=64, shuffle=False
  → test:  60 graphs, batch_size=64, shuffle=False

Step 4: Build model
  → CGCNNEncoder(atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1, pooling=attention)
  → build_surrogate(cfg) → ExactGPSurrogate or SVGPSurrogate
  → DKLModel(encoder, surrogate, encoder_lr=0.001, gp_lr=0.01)

Step 5: Train
  → dkl.fit(train_graphs, n_epochs=100, gp_pretrain_epochs=50)
  → logs every 20 epochs: "Epoch 20/100  loss=-2.3411"

Step 6: Evaluate on val
  → dkl.encode(val_loader) → embeddings [162, 32], targets [162]
  → surrogate.predict(embeddings) → mean [162], std [162]
  → compute_accuracy_metrics(targets, mean) → AccuracyMetrics
  → compute_calibration_metrics(targets, mean, std) → CalibrationMetrics
  → print_metrics_table(...)

Step 7: Evaluate on test (same as Step 6)

Step 8: Save results
  → results/surrogate_metrics_exactgpsurrogate.csv
  → results/training_loss_exactgpsurrogate.csv
```

---

### File 8: `tests/test_surrogate_swap.py`

#### What it does
Runs 7 automated tests (14 total — each runs for both ExactGP and SVGP)
that verify the surrogate interface is correctly implemented.

#### Why it exists
The BO loop in Phase 3 calls the surrogate's methods in a specific way.
If any method returns the wrong shape, wrong dtype, or missing gradients,
the BO loop will crash — often with a confusing error far from the actual bug.

These tests catch interface bugs before they become BO bugs.

#### Analogy
Before a new aircraft engine goes in the plane, engineers test:
- Does it have the right bolt pattern? (can it attach at all?)
- Does the fuel connector fit? (will data flow correctly?)
- Does it produce the right thrust range? (are the outputs in range?)

These tests don't check "is it the best engine?" — they check "does it fit?"

#### The 7 tests

```
Test 1: test_is_base_surrogate
  Both ExactGP and SVGP must be instances of BaseSurrogate.
  If this fails: the ABC contract wasn't followed. Phase 3 will type-error.

Test 2: test_fit_returns_loss_list
  fit() must return a non-empty list of float values.
  Why? dkl.py reads losses[-1] and logs the curve. If it returns None → AttributeError.

Test 3: test_predict_output_shapes
  predict(X_test) must return (mean [N], std [N]).
  If std returns shape [N, 1] instead of [N], acquisition function breaks silently.

Test 4: test_predict_std_is_positive
  std > 0 everywhere.
  If std has zeros: log(0) = -∞ in NLL. If std has negatives: sqrt fails.

Test 5: test_predict_output_is_cpu_float32
  Output must be float32 on CPU.
  Why float32? Acquisition function runs in float32. Mixed dtypes → runtime error.
  Why CPU? Metrics computation uses numpy, which doesn't handle CUDA tensors.

Test 6: test_to_device_runs_without_error
  .to('cpu') must work before AND after fit().
  Before fit: model is None → must not crash.
  After fit: model must actually move to device.

Test 7: test_joint_loss_is_scalar
  joint_loss(X, y) must return a 0-dim Tensor with requires_grad=True.
  If shape is [N] instead of []: loss.backward() picks wrong reduction → wrong gradients.
  If requires_grad=False: encoder never gets gradients → encoder never trains.
```

---

## 9. How All 8 Files Connect — The Full Picture

```
                    PHASE 1 OUTPUT
                    ┌────────────┐
                    │  LMDB      │  metadata.parquet
                    │  Cache     │  3351 graphs
                    └────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   dataset.py        │
               │   GraphDataset      │
               │   wraps cache for   │
               │   DataLoader        │
               └─────────────────────┘
                          │
                          │  batches: (x, edge_index, edge_attr, batch, y)
                          ▼
               ┌─────────────────────┐
               │  cgcnn_encoder.py   │
               │  CGCNNEncoder       │
               │  3 conv layers      │
               │  attention pool     │
               │  → [B, 32] embeds   │
               └─────────────────────┘
                          │
                          │  embeddings [N, 32]  (float32 → float64 at GP boundary)
                          ▼
               ┌─────────────────────────────────────┐
               │  surrogate.py                        │
               │                                     │
               │   BaseSurrogate (ABC contract)       │
               │        │            │               │
               │  ExactGPSurrogate  SVGPSurrogate    │
               │  (O(N³), exact)   (O(N·M²), fast)  │
               │        │            │               │
               │        └─────┬──────┘               │
               │              ▼                      │
               │  predict() → (mean[N], std[N])       │
               └─────────────────────────────────────┘
                     │               │
           ┌─────────┘               └──────────┐
           ▼                                    ▼
┌─────────────────────┐            ┌─────────────────────────┐
│ metrics_accuracy.py │            │ metrics_calibration.py   │
│ MAE, RMSE, R²        │            │ NLL, ENCE, Miscal. area  │
│ "was it right?"     │            │ Spearman ρ, Coverage@95  │
└─────────────────────┘            │ "was uncertainty honest?"│
                                   └─────────────────────────┘

All of the above is orchestrated by:
┌─────────────────────────────────────────────────────────────┐
│  dkl.py  — DKLModel                                         │
│  "manager" connecting encoder + surrogate + joint training  │
└─────────────────────────────────────────────────────────────┘

All of the above is run by:
┌─────────────────────────────────────────────────────────────┐
│  scripts/02_eval_surrogate.py  — the conductor/recipe card  │
│  Hydra loads configs, runs all steps, saves CSVs            │
└─────────────────────────────────────────────────────────────┘

And verified by:
┌─────────────────────────────────────────────────────────────┐
│  tests/test_surrogate_swap.py  — the safety net             │
│  7 × 2 = 14 tests ensuring ExactGP and SVGP are            │
│  truly interchangeable                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. What "Good Results" Look Like After Running Phase 2

After running `python scripts/02_eval_surrogate.py data.db_path=data/raw/c2db.db`,
you should see a table like:

```
=======================================================
  Surrogate: ExactGPSurrogate   Split: test
=======================================================
  ACCURACY
    MAE          ~0.35–0.55 eV    ← acceptable for CGCNN on C2DB
    RMSE         ~0.50–0.70 eV
    R²           >0.65            ← model explains majority of variance
  CALIBRATION  (targets from [P2])
    NLL          ~0.5–1.5         ← lower is better
    ENCE         0.06–0.10        ← well-sized uncertainty
    Miscal. area 0.04–0.07        ← reliability curve near diagonal
    Spearman ρ   >0.3             ← positive = model knows when wrong
    Coverage@95  0.92–0.97        ← intervals contain right fraction of truths
=======================================================
```

If these targets are met, Phase 2 is complete and Phase 3 (the BO loop) can begin.
The model predicts well AND the uncertainty is honest — both requirements for
Bayesian Optimisation to function correctly.

---

## 11. Summary in One Page

| What | Why |
|------|-----|
| `dataset.py` | Bridge between LMDB cache and PyTorch DataLoader |
| `cgcnn_encoder.py` | Convert variable-size crystal graphs → fixed 32-dim fingerprints |
| `surrogate.py` | Gaussian Process: predict (band gap, uncertainty) from fingerprints |
| `dkl.py` | Joint training manager: encoder + GP improve each other |
| `metrics_accuracy.py` | Measure whether predictions are numerically correct |
| `metrics_calibration.py` | Measure whether uncertainty estimates are trustworthy |
| `02_eval_surrogate.py` | The script: runs everything end-to-end, saves CSV results |
| `test_surrogate_swap.py` | Safety net: proves ExactGP and SVGP are interchangeable |

**The two things Phase 2 proves before Phase 3 can start:**
1. The model predicts band gaps with reasonable accuracy (MAE, R²)
2. The model's uncertainty is honest and correlates with actual errors (Spearman ρ, Coverage@95)

Without both, Bayesian Optimisation cannot work correctly.
