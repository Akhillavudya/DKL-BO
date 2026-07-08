# SNUMAT Side-Test — Phase 1 Explained (the dataset)

## The big picture
We already showed, on the **C2DB** database (2D materials), that a pre-trained DKL-BO finds
high-band-gap materials faster than a traditional descriptor GP. A reviewer's fair question is:
*"Did that only work because of something special about 2D materials?"* To answer it we repeat
the **exact same experiment** on a completely different database — **SNUMAT**, which is ~10,000
ordinary **3D bulk crystals** (think normal salt-like / oxide / semiconductor crystals from the
ICSD). If DKL-BO wins here too, the result **generalizes**.

## What Phase 1 does (analogy)
Before any contest you need a clean, fair playing field. Phase 1 builds that field once, so all
three players (Random, Std-GP, DKL-BO) search the **identical** list of materials.

Think of each material as a player card with three things printed on it:
1. **The structure** (where the atoms sit) → turned into a *graph* the DKL "modern chef" reads.
2. **Handcrafted numbers** (composition + size/shape) → the *descriptors* the Std-GP "classic
   chef" reads.
3. **The answer** (the HSE band gap) → hidden during the hunt; only revealed when a material is
   "picked".

## Key differences from the C2DB version
- **3D, not 2D.** A 2D material is a single sheet floating in vacuum, so the old graph builder
  deleted any "bond" that jumped across the empty vacuum gap. A 3D crystal is solid in every
  direction — there is no vacuum — so that deletion rule is **switched off** here
  (`lib/graph_builder_3d.py`). Leaving it on would have erased real bonds.
- **Band gap only.** SNUMAT has no "effective mass", so unlike C2DB there is just one property.
  The contest later has two goals: find the **highest**-gap and the **lowest**-gap crystals.
- **Source format.** SNUMAT ships as one JSON file per material containing the HSE gap and the
  crystal structure as a text block; we read those directly.
- **Fairness rule kept identical.** The descriptor player is *not* allowed to peek at any
  DFT-computed electronic property (the GGA gap, the direct/indirect label) — those are excluded,
  exactly as in the C2DB study, so neither player cheats.

## What came out (results)
- Scanned **10,477** materials, kept **10,359** (dropped 19 with no gap, 5 metals with gap ≤ 0.01,
  93 with an unreadable/missing structure, 1 duplicate id).
- Band gap range **0.012 – 20.30 eV** (the very top end is rare wide-gap insulators such as solid
  noble gases — physically real).
- **Train 7,228 / Pool 3,131** (~30 % held out). The split is by *structure family* (prototype =
  anonymised formula + space group) so near-identical crystals can't leak across the divide, and
  it is balanced across gap quartiles so the rare extreme-gap crystals are spread into the pool.
- The held-out pool contains **48** of the highest-gap and **53** of the lowest-gap crystals, so
  both search goals have real targets to find.
- Built **10,359** crystal graphs (cache `graphs_324e189e.lmdb`) and a **42-feature** descriptor
  table (35 composition + 4 geometry + 3 categorical), perfectly aligned so both chefs see the
  same materials in the same order.

## Outputs
```
snumat_generalization/data/cache/master.parquet        10,359 rows (uid, formula, prototype, gap, split, ...)
snumat_generalization/data/cache/descriptors.parquet   10,359 x 42  (Std-GP features)
snumat_generalization/data/cache/graphs_324e189e.lmdb  10,359 3D crystal graphs (DKL)
```
Next: **Phase 2** — pre-train the CGCNN encoder on the train split and measure how accurately
Std-GP vs DKL predict the HSE gap on the untouched pool.
