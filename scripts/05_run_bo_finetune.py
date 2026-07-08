"""Phase 4 (clean rebuild) — live fine-tuning DKL (Paper-2 style warm-start).

Phase 3's DKL FROZE the pre-trained encoder and treated its embeddings as static features.
Here we run the REAL BO loop: the encoder is warm-started from the Phase-2 checkpoint and
then keeps fine-tuning (encoder + GP jointly) on the materials it digs up during the hunt —
a full retrain every `retrain_every_k` cycles. This is the variant that won Paper 2's
effective-mass task.

Same 4 tasks, same pool, same EI/seeds/n_init as Phase 3, so results stack directly against
the frozen DKL, Std-GP and Random runs in results/rebuild/runs/.

  gap tasks   warm-start from results/rebuild/encoder_gap.pt
  emass tasks warm-start from results/rebuild/encoder_emass_log.pt   (log10 target)

Outputs (idempotent / resumable)
  results/rebuild/runs_finetune/{task}__dkl_finetune__seed{S}.csv
  results/rebuild/bo_finetune_summary.csv   (combined frozen-vs-finetune-vs-stdgp table)

Usage
    python scripts/05_run_bo_finetune.py                  # 10 seeds, 100 cycles
    python scripts/05_run_bo_finetune.py --seeds 1 --cycles 8   # smoke test
"""

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from dklbo.bo.loop import BOLoop
from dklbo.data.cache import GraphCache
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.models.dkl import DKLModel
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO / "data" / "cache"
OUT = REPO / "results" / "rebuild"
RUNS_FT = OUT / "runs_finetune"
RUNS_CL = OUT / "runs_coldlive"
RUNS_FROZEN = OUT / "runs"

ENCODER_KW = dict(atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1,
                  pooling="attention")
GRAPH_PREPROC = {"radius": 8.0, "vacuum_cutoff": 4.0, "max_neighbors": 12,
                 "target": "gap_hse", "gap_min": 0.01}

# task -> (target column, transform, direction, encoder-checkpoint tag)
TASKS = {
    "gap_max":   ("gap",   "none",  "max", "gap"),
    "gap_min":   ("gap",   "none",  "min", "gap"),
    "emass_min": ("emass", "log10", "min", "emass_log"),
    "emass_max": ("emass", "log10", "max", "emass_log"),
}


def make_target(df, col, transform):
    s = df[col].astype("float64")
    return np.log10(s) if transform == "log10" else s


def build_dkl(enc_tag: str, device: str, cold: bool) -> DKLModel:
    """Build a DKLModel. cold=False warm-starts the encoder from the Phase-2 checkpoint
    (fine-tune); cold=True leaves it randomly initialised (Paper-2 from-scratch control)."""
    enc = CGCNNEncoder(**ENCODER_KW)
    if not cold:
        ckpt = OUT / f"encoder_{enc_tag}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing — run scripts/02_pretrain_encoder.py first.")
        enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DKLModel(encoder=enc, surrogate=ExactGPSurrogate(lr=0.01, n_epochs=100, ard=False),
                    encoder_lr=0.001, gp_lr=0.01, device=device, standardize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--cold", action="store_true",
                   help="random-init encoder, trained from scratch live (Paper-2 exact)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = "dkl_cold_live" if args.cold else "dkl_finetune"
    runs_dir = RUNS_CL if args.cold else RUNS_FT

    cfg = OmegaConf.create({
        "acquisition": "ei", "beta": 0.0, "xi": 0.01,
        "n_init": args.n_init, "n_cycles": args.cycles,
        "retrain_every_k": 5, "n_joint_epochs": 50,
        "n_pretrain_epochs": 20, "gp_refit_epochs": 50,
    })

    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    pool_base = df[df.split == "pool"].reset_index(drop=True)
    cache = GraphCache(str(CACHE_DIR), GRAPH_PREPROC, readonly=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pool={len(pool_base)} | seeds={args.seeds} cycles={args.cycles} "
                f"device={device} | method={method}")

    n_done = n_skip = 0
    for task, (col, transform, direction, enc_tag) in TASKS.items():
        pool = pool_base.copy()
        pool["target"] = make_target(pool, col, transform).to_numpy()
        for seed in seeds:
            out = runs_dir / f"{task}__{method}__seed{seed}.csv"
            if out.exists():
                n_skip += 1
                continue
            seed_everything(seed)
            dkl = build_dkl(enc_tag, device, cold=args.cold)   # fresh per seed (no leakage)
            res = BOLoop(dkl=dkl, cache=cache, meta_df=pool, cfg=cfg,
                         seed=seed, direction=direction).run()
            res.insert(0, "task", task)
            res.insert(1, "method", method)
            res.insert(2, "seed", seed)
            res.to_csv(out, index=False)
            n_done += 1
            logger.info(f"  {task}/seed{seed} → best={res.best_so_far.iloc[-1]:.3f} "
                        f"top50={res.cumul_top50.iloc[-1]} top10%={res.cumul_top10pct.iloc[-1]}")
        logger.info(f"  [{task}] done")
    cache.close()
    logger.info(f"Fine-tune runs written={n_done} skipped={n_skip}. Summarizing…")
    summarize()


def _finals(files):
    rows = []
    for f in files:
        d = pd.read_csv(f)
        last = d.iloc[-1]
        rows.append({"task": last.task, "method": last.method, "seed": last.seed,
                     "best": last.best_so_far, "top50": last.cumul_top50,
                     "top10pct": last.cumul_top10pct})
    return rows


def summarize() -> None:
    """Combined table: Std-GP vs DKL-frozen vs DKL-finetune vs DKL-cold-live vs Random."""
    rows = (_finals(sorted(RUNS_FROZEN.glob("*.csv")))
            + _finals(sorted(RUNS_FT.glob("*.csv")))
            + _finals(sorted(RUNS_CL.glob("*.csv"))))
    fin = pd.DataFrame(rows)
    # Phase-3 "dkl" is the frozen variant — rename for clarity.
    fin["method"] = fin["method"].replace({"dkl": "dkl_frozen"})
    summary = (fin.groupby(["task", "method"])[["best", "top50", "top10pct"]]
               .mean().round(3).reset_index())
    summary.to_csv(OUT / "bo_finetune_summary.csv", index=False)

    order = {"std_gp": 0, "dkl_frozen": 1, "dkl_finetune": 2, "dkl_cold_live": 3, "random": 4}
    print("\n" + "=" * 70)
    print("DKL variants vs Std-GP vs Random (mean over seeds)")
    print("(emass 'best' in log10 units; top-k counts are scale-free)")
    print("=" * 70)
    for task in TASKS:
        sub = summary[summary.task == task].copy()
        sub["o"] = sub.method.map(order)
        sub = sub.sort_values("o")
        print(f"\n  {task}")
        print(f"    {'method':13s} | {'best':>8s} {'top50':>7s} {'top10%':>7s}")
        print("    " + "-" * 42)
        for _, r in sub.iterrows():
            print(f"    {r.method:13s} | {r.best:8.3f} {r.top50:7.1f} {r.top10pct:7.1f}")
    print(f"\nSaved → {OUT/'bo_finetune_summary.csv'}")


if __name__ == "__main__":
    main()
