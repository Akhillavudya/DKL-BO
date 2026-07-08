# Window-MIN Experiment Explained (find the dimmest light *above* the floor)

*A beginner-friendly walkthrough: the goal, the trick that makes it work, the four
contestants, what we measure, what we found — and why DKL wins here but tied on the plain
min-search.*
Script: `scripts/exp_window_min_bo.py` · Acquisition: `src/dklbo/bo/acquisition.py:window_min`

This is the **mirror image** of the [window-MAX experiment](window_max_explanation.md). Read
that one first if you haven't — everything here is the same idea, flipped upside down.

---

## 0. What question is this experiment asking?

> "Find the material with the **lowest** band gap **that still stays at or above 0.7 eV**."

So we have a **hard floor** at 0.7 eV and a soft ceiling at 3.0 eV. We want to go as low as
we can *without falling through the floor*. The single best material in the pool is the one
whose gap is **closest to 0.7 from above** (≈ 0.700 eV). Anything below 0.7 eV is
**rejected** — it scores zero, even though it's "smaller."

```
reward(x) = (3.0 − g)   if  0.7 ≤ g ≤ 3.0     (inside the window — lower g = bigger reward)
          = REJECT        otherwise             (below floor or above ceiling)
```

We reward the **headroom below the ceiling** (`3.0 − g`), which is biggest for the lowest
in-window gap. As the model becomes certain, this picks exactly the material with the lowest
gap that's still inside the box.

---

## 1. The analogy

Same storage room as before, but now you want the **shortest** treasure chest that is
**still at least 0.7 m tall** — say, because anything shorter than 0.7 m falls through a grate
in the floor and is lost. You don't want the shortest chest in the world (those fall through);
you want the shortest chest that *survives the grate*. The perfect find is a chest of 0.71 m.

A naive "always grab the smallest" strategy keeps grabbing tiny chests that fall through the
grate — useless. We need a smarter rule that stops right at the floor.

---

## 2. The clever bit — the `window_min` acquisition

We score each material by the **expected headroom** it has under the ceiling, counting only
the part of the prediction inside the window:

```
score(x) = E[ (3.0 − g) · 1{0.7 ≤ g ≤ 3.0} ]
         = (3.0 − μ)·(Φ(βʰ) − Φ(αˡ)) + σ·(φ(βʰ) − φ(αˡ))
```

In words: **"how *low* a gap do I expect this material to have, counting only the part of the
prediction that stays inside the box?"** It rewards a low predicted gap but throws away any
probability that the gap leaks *below* 0.7 (likely to be rejected). As the model becomes
certain (σ → 0), it collapses to "pick the lowest gap still inside the box" — the true
constrained minimum.

> Just like window-MAX, the model is fed the **raw band gap in eV** (loop runs in "maximise"
> mode); the `window_min` acquisition does the flipping toward *low* gaps internally, so no
> sign trick is applied to the targets.

---

## 3. The four contestants

Identical setup to window-MAX — same starting samples (per seed), same GP back-end, same
`window_min` rule. Only the **features** differ:

| Method | How it "sees" a material | One-line idea |
|---|---|---|
| **Standard GP** | 43 hand-crafted descriptors | the classic chemist's checklist |
| **DKL (frozen)** | 32-d pre-trained embeddings | a learned palate, locked before the hunt |
| **DKL (fine-tuned, live)** | encoder keeps **learning during** the hunt | a chef refining his palate every dig |
| **Random** | nothing | the floor / sanity baseline |

---

## 4. What we measure (the metrics)

Run setup: **30 random seeds × 100 BO cycles**, starting from 10 random labels.

1. **best_inwin_gap** — the lowest in-window gap found so far (closeness to the 0.7 eV floor).
   Here **LOWER is better**.
2. **cumul_top50** — how many of the **50 lowest in-window gaps** in the pool you've found so
   far. Measures *efficiency*. Higher = better.

---

## 5. The results (30 seeds, ★ = significant, p < 0.05)

**best_inwin_gap (closeness to the 0.7 eV floor, LOWER is better):**

| Method | Final mean | vs Std-GP |
|---|---|---|
| **DKL (fine-tuned, live)** | **0.710** | **−0.028 ★ (p=0.002)** |
| DKL (frozen) | 0.718 | −0.020 ★ (p=0.031) |
| Random | 0.730 | −0.009 (n.s.) |
| Standard GP | 0.739 | — |

**cumul_top50 (how many of the 50 lowest you harvested, HIGHER is better):**

| Method | Final mean | vs Std-GP |
|---|---|---|
| **DKL (fine-tuned, live)** | **16.13** | **+6.20 ★ (p<0.001)** |
| DKL (frozen) | 12.40 | +2.47 ★ (p=0.016) |
| Standard GP | 9.93 | — |
| Random | 6.37 | −3.57 ★ (worse) |

**Crossover** (cycle the DKL mean overtakes Std-GP and stays better):

| Metric | DKL frozen | DKL fine-tuned |
|---|---|---|
| best_inwin_gap | cycle 26 | **cycle 1** |
| cumul_top50 | cycle 8 | **cycle 6** |

→ Unlike window-MAX, here DKL wins on **both** metrics, and the fine-tuned version is the
**clear winner** — better single best point *and* far more of the top-50, overtaking Std-GP
within the first handful of cycles.

---

## 6. The key question: why does DKL win here, but *tied* on the plain min-search?

This is the subtle, important part. On the earlier **unconstrained** `gap_min` task ("find the
absolute lowest gap, target ≈ 0"), Standard GP and DKL fine-tuned **tied** (both reached
≈ 0.024–0.028 eV; the difference was not significant). Yet on this **windowed** min task, DKL
is a clear winner. Same goal ("minimise the gap") — opposite verdict. Why?

**Because the two tasks have completely different *difficulty*:**

**Unconstrained min is an *easy* problem.** "Find the smallest gap" means walking to the very
edge of the distribution, toward 0. Low-gap materials are **abundant** (the pool is full of
near-metals), and they all cluster at one extreme. The surrogate just has to point "downhill,"
and there's a whole crowd of winners at the bottom. Every method bottoms out at ~0.02 eV — the
global floor. When a task is this easy, **both models saturate it**, leaving no headroom for a
smarter model to win. A better map doesn't help when the destination is "just go to the edge,
you can't miss it."

**Windowed min is a *hard, localized* problem.** Now the target is a narrow shelf: as low as
possible **but not below 0.7**. The desirable region is squeezed between two *bad* regions:
- below 0.7 → **rejected** — and that's exactly where all those abundant low-gap materials
  live, so they flip from being prizes to being **traps**;
- well above 0.7 → allowed but suboptimal.

So the search must **thread a thin band and stop right at its lower edge**. That requires
*accurately knowing each material's gap near a specific value* — to tell a 0.72 eV material
(great) from a 0.65 eV one (rejected) from a 1.5 eV one (meh). This fine, in-the-middle
discrimination is precisely where DKL's **learned gap-embeddings** beat 43 hand-crafted
descriptors: the encoder was pre-trained to predict gap, so it organises materials by gap
exactly where the resolution is needed.

**Library analogy:**
- *Unconstrained min* = "bring me a book from the bottom shelf." Anyone can grab one — knowing
  the catalogue perfectly is no advantage.
- *Windowed min* = "bring me the book closest to call-number 0.7, but nothing below 0.7." Now
  you need someone who knows precisely where every book sits. The expert (DKL) nails the slice
  fast; the amateur (Std-GP) wanders and keeps grabbing rejects.

**One honest nuance:** the "clear winner" is mostly about **efficiency / top-50**, not the
single best point. Even Std-GP *eventually* lands one material near 0.7 (0.739 vs DKL's 0.710
— real but small). The dramatic win is that DKL **populates** the near-floor region far
faster (16 vs 10 of the top-50, crossing ahead by cycle 6).

---

## 7. The take-away

- For the **constrained band-targeting** task, **live fine-tuned DKL is the clear winner** —
  better best point *and* far more sample-efficient (top-50 +6.2, p<0.001).
- The reason it wins here but *tied* on plain min-search: **the windowed task is harder and
  more localized**, and that's where a richer learned representation earns its keep. The
  unconstrained extreme is so easy that both models solve it to saturation.
- This is consistent with the rest of the project: DKL's advantage shows up on the **hard,
  precise** targeting problems, not the easy "just go to the edge" ones.

---

## 8. Files produced

```
results/rebuild/
  winmin_runs/{method}__seed{S}.csv   raw per-cycle records (resumable)
  winmin_summary.csv                  per (method, seed) finals
  winmin_stats.csv                    paired Wilcoxon vs Std-GP
  winmin_crossover.csv                cycle where each DKL overtakes Std-GP
  plots/winmin_best_gap.png           best (lowest) in-window gap vs cycle
  plots/winmin_top50.png              cumulative top-50 found vs cycle
  plots/winmin_crossover.png          2 panels: best-gap + top50 crossover
```

Re-run: `python scripts/exp_window_min_bo.py` (30 seeds, 100 cycles).
Smoke test: `python scripts/exp_window_min_bo.py --seeds 2 --cycles 8`.
