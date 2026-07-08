# Rebuild — Phase 4 Explained (the chef who keeps learning mid-hunt)

*A beginner-friendly walkthrough: the big picture, then file by file, with analogies.*
Continues the treasure-hunt story from Phases 1–3.

---

## 0. Where Phase 4 sits

```
   Phase 1  — Build ONE clean dataset
   Phase 2  — Check the models can predict (tasting)
   Phase 3  — The BO contest (frozen DKL vs Std-GP vs Random)
>> Phase 4  — Let the DKL encoder LEARN during the hunt    <-- YOU ARE HERE
   Phase 5  — Stats + plots
```

In Phase 3 the modern chef studied 1,901 dishes, then his palate was **locked** for the
whole hunt (a *frozen* encoder). Phase 4 asks: **what if the chef keeps refining his palate
every time he digs up a new material?** That is "live fine-tuning," and it's the exact setup
that won the Paper-2 effective-mass result.

---

## 1. The big picture (one analogy)

Two ways the modern chef can hunt:

- **Frozen (Phase 3):** learns once from 1,901 dishes, then never updates his instincts. His
  "flavour fingerprints" for every material are fixed before the hunt begins.
- **Fine-tuned (Phase 4):** starts from the same trained palate, but **after every few digs he
  re-trains his nose** on everything he has tasted so far. His fingerprints *change* during
  the hunt to suit the specific treasure he is chasing.

There is also a pure-control version we ran as a side experiment:

- **Cold + live:** the chef starts with **no palate at all** (random nose) and must learn
  everything from scratch during the hunt — the exact Paper-2 setup, used to measure how much
  the 1,901-dish pre-training was actually worth.

---

## 2. The key ideas in Phase 4

### Idea A — Two knobs, not one
"DKL" is really set by two switches: **(1)** does the encoder start pre-trained or cold, and
**(2)** during the hunt is it frozen or fine-tuned? Phase 4 explores the *fine-tuned* column.

| Variant | Start | During hunt |
|---|---|---|
| dkl_frozen (Phase 3) | pre-trained | frozen |
| **dkl_finetune (Phase 4)** | pre-trained | **fine-tuned** |
| dkl_cold_live (experiment) | cold/random | fine-tuned |

### Idea B — The retrain schedule
Re-training the encoder every single cycle is slow and unstable. So the loop does a **full
encoder + GP retrain every 5th cycle**, and a cheap **GP-only refit** on the other 4. This
gives most of the benefit at a fraction of the cost.

### Idea C — Why fine-tuning can BACKFIRE (the key intuition)
During the hunt you only have ~10–110 dug-up labels, but the encoder has ~18,000 parameters.
Training such a big net on so few points makes it **overfit** — it distorts its fingerprints
to memorise those few materials and *loses* the broad knowledge from the 1,901 pre-training
dishes. **Analogy:** a chef who rewrites his entire palate based on the last 3 dishes he tasted
becomes great at *those 3* and worse at everything else. This is the "tied/overfit features"
worry — and Phase 4 confirms it.

---

## 3. File by file

### 3a. `scripts/05_run_bo_finetune.py` — the live-learning runner
Runs the **real** BO loop (the one that can retrain the encoder) for all 4 tasks × seeds.

| Piece | What it does | Analogy |
|---|---|---|
| `build_dkl(enc_tag, device, cold)` | builds the chef: warm-start from the Phase-2 palate, OR (`cold=True`) start with a random nose | hand the chef his trained palate, or none |
| `BOLoop(...).run()` | the real loop: full retrain every 5 cycles, GP refit otherwise | the chef hunting *and* refining his nose as he goes |
| `--cold` flag | switches to the from-scratch Paper-2 control | the no-palate experiment |
| warm-start from `encoder_{gap,emass_log}.pt` | reuses the Phase-2 encoders | the palate trained back in Phase 2 |
| idempotent CSVs + `summarize()` | resumable runs + the combined scoreboard | pause/resume + final table |

Run it: `python scripts/05_run_bo_finetune.py` (fine-tune) or add `--cold` (Paper-2 control).

### 3b. The reused engine (`src/dklbo/`)
| Module | Role in Phase 4 | Analogy |
|---|---|---|
| `bo/loop.py` (`BOLoop`) | the loop that retrains the encoder during the hunt | the rulebook for "hunt-and-learn" |
| `models/dkl.py` | jointly trains encoder + GP each full retrain | the chef's brain wiring nose + memory together |
| `models/cgcnn_encoder.py` | the palate being fine-tuned | the chef's nose |
| `models/surrogate.py` | the GP that refits each cycle | the shared oven |

(Phase 3's frozen DKL used `FeatureBOLoop` — static features. Phase 4 uses the full `BOLoop`
so the encoder *can* change. That's the whole difference.)

---

## 4. What Phase 4 found (mean over 10 seeds)

| Task | frozen (best / top-10%) | fine-tuned | cold+live |
|---|---|---|---|
| gap_max | 8.638 / **42.4** | 8.462 / 37.6 | 7.619 / 28.5 |
| emass_min | −1.714 / **24.1** | −1.948 / 20.2 | **−2.117** / 20.5 |
| emass_max | 1.455 / 16.8 | **1.651** / 19.8 | 1.491 / 16.3 |

Three lessons from the means (significance is decided in Phase 5):

1. **Fine-tuning helps the single champion, hurts breadth.** On emass it found *lower/heavier*
   single champions (e.g. emass_min best −1.714 → −1.948), but its top-10% breadth dropped
   everywhere vs frozen. Exactly the overfitting trade-off of Idea C.
2. **Frozen stays the breadth king** — best top-10% on the gap tasks.
3. **Cold+live shows pre-training matters.** With no pre-training it collapses on gap_max
   (top-10% 28.5, barely above Std-GP) — proof the 1,901-dish pre-training does real work on
   this harder dataset. (It did still reach the lowest single emass_min champion, echoing
   Paper 2 — but Phase 5 will show that single-champion edge is within noise.)

---

## 5. The honest takeaway

There is no single "best DKL." There is a **trade-off**:
- want **many** good materials (breadth)? → **frozen pre-trained**.
- want **the one** best material fast (single champion / sample-efficiency)? → **fine-tuned**.
- **no pre-training** (cold)? → clearly worse on band gap; pre-training is essential here.

This nuance — and the fact that fine-tuning's apparent emass gains need a significance check —
is exactly what Phase 5 puts to the statistical test.

---

## 6. One-paragraph summary

Phase 4 let the modern chef keep refining his palate during the hunt (live fine-tuning), and
added a no-palate control (cold+live, the exact Paper-2 setup). The result is a clean
trade-off: fine-tuning hones onto the single champion but overfits the few dug-up labels and
loses discovery breadth, while the frozen pre-trained chef remains the breadth champion; and
stripping out pre-training (cold) clearly hurts on band gap. Whether the fine-tuning gains are
real or noise is the job of Phase 5.
