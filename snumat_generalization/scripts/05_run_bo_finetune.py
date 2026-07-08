"""Phase 4 (SNUMAT side-test) — live fine-tuning DKL (Paper-2 style warm-start).

Phase 3's DKL FROZE the pre-trained encoder and treated its embeddings as static features.
Here we run the REAL BO loop: the encoder is warm-started from the Phase-2 checkpoint and keeps
fine-tuning (encoder + GP jointly) on the materials it digs up during the hunt — a full retrain
every `retrain_every_k` cycles. `--cold` instead starts from a random encoder trained live from
scratch (the Paper-2 control).

Same gap tasks, pool, EI/seeds/n_init as Phase 3, so results stack directly against the frozen
DKL, Std-GP and Random runs in results/runs/.

Outputs (idempotent / resumable)
  results/runs_finetune/{task}__dkl_finetune__seed{S}.csv
  results/runs_coldlive/{task}__dkl_cold_live__seed{S}.csv
  results/bo_finetune_summary.csv   (combined Std-GP / frozen / finetune / cold / random table)

Usage
    python snumat_generalization/scripts/05_run_bo_finetune.py            # warm-start finetune
    python snumat_generalization/scripts/05_run_bo_finetune.py --cold     # from-scratch control
"""

import argparse
import logging
import sys
from pathlib import Path

SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))

import pandas as pd
import torch
from omegaconf import OmegaConf

from lib import config
from dklbo.bo.loop import BOLoop
from dklbo.data.cache import GraphCache
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.models.dkl import DKLModel
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUNS_FT = config.RESULTS_DIR / "runs_finetune"
RUNS_CL = config.RESULTS_DIR / "runs_coldlive"
RUNS_FROZEN = config.RESULTS_DIR / "runs"

ENCODER_KW = dict(atom_dim=config.GRAPH_PREPROC["atom_feat_dim"], bond_dim=10,
                  hidden_dim=32, n_conv=3, n_fc=1, pooling="attention")

# task -> direction. Single gap encoder checkpoint (encoder_gap.pt).
TASKS = {"gap_max": "max", "gap_min": "min"}


def build_dkl(device: str, cold: bool) -> DKLModel:
    """cold=False warm-starts the encoder from the Phase-2 checkpoint (fine-tune);
    cold=True leaves it randomly initialised (Paper-2 from-scratch control)."""
    enc = CGCNNEncoder(**ENCODER_KW)
    if not cold:
        ckpt = config.RESULTS_DIR / "encoder_gap.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing — run script 02 first.")
        enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DKLModel(encoder=enc, surrogate=ExactGPSurrogate(lr=0.01, n_epochs=100, ard=False),
                    encoder_lr=0.001, gp_lr=0.01, device=device, standardize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--cycles", type=int, default=config.N_CYCLES)
    ap.add_argument("--n_init", type=int, default=config.N_INIT)
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

    df = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    pool_base = df[df.split == "pool"].reset_index(drop=True)
    cache = GraphCache(str(config.CACHE_DIR), config.GRAPH_PREPROC, readonly=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pool={len(pool_base)} | seeds={args.seeds} cycles={args.cycles} "
                f"device={device} | method={method}")

    n_done = n_skip = 0
    for task, direction in TASKS.items():
        pool = pool_base.copy()
        pool["target"] = pool["gap"].astype("float64").to_numpy()
        for seed in seeds:
            out = runs_dir / f"{task}__{method}__seed{seed}.csv"
            if out.exists():
                n_skip += 1
                continue
            seed_everything(seed)
            dkl = build_dkl(device, cold=args.cold)   # fresh per seed (no leakage)
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
    logger.info(f"Runs written={n_done} skipped={n_skip}. Summarizing…")
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
    fin["method"] = fin["method"].replace({"dkl": "dkl_frozen"})  # Phase-3 "dkl" = frozen
    summary = (fin.groupby(["task", "method"])[["best", "top50", "top10pct"]]
               .mean().round(3).reset_index())
    summary.to_csv(config.RESULTS_DIR / "bo_finetune_summary.csv", index=False)

    order = {"std_gp": 0, "dkl_frozen": 1, "dkl_finetune": 2, "dkl_cold_live": 3, "random": 4}
    print("\n" + "=" * 70)
    print("SNUMAT — DKL variants vs Std-GP vs Random (mean over seeds)")
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
    print(f"\nSaved → {config.RESULTS_DIR/'bo_finetune_summary.csv'}")


if __name__ == "__main__":
    main()
