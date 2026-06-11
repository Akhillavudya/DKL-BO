"""Phase 3: Bayesian Optimisation loop + random baseline.

Runs DKL-UCB for n_cycles acquisitions, then runs random search with the
same seed for comparison. Saves both result tables to results/.

Usage
-----
    python scripts/03_run_bo.py data.db_path=data/raw/c2db.db

Override cycles or beta:
    python scripts/03_run_bo.py data.db_path=... bo.n_cycles=50 bo.beta=0.5

Switch GP backend:
    python scripts/03_run_bo.py data.db_path=... surrogate=gp_svgp

Note: a fresh DKL model is created here (not the Phase 2 weights).
The BO simulation starts with no prior knowledge — labels are revealed
one at a time by the oracle (metadata.parquet target column).
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from dklbo.bo.baselines import RandomBaseline
from dklbo.bo.loop import BOLoop
from dklbo.data.cache import GraphCache
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import build_surrogate
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load metadata ──────────────────────────────────────────────────
    cache_dir = Path(cfg.data.cache_dir)
    meta_path = cache_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.parquet not found at {meta_path}. "
            "Run 01_build_cache.py first."
        )
    df = pd.read_parquet(meta_path)
    logger.info(
        f"Pool: {len(df)} materials  |  "
        f"target range: {df['target'].min():.2f}–{df['target'].max():.2f} eV"
    )

    # ── Open LMDB cache (read-only) ────────────────────────────────────
    preproc_cfg = {
        "radius":       cfg.data.radius,
        "vacuum_cutoff":cfg.data.vacuum_cutoff,
        "max_neighbors":cfg.data.max_neighbors,
        "target":       cfg.data.target,
        "gap_min":      cfg.data.gap_min,
    }
    cache = GraphCache(cfg.data.cache_dir, preproc_cfg, readonly=True)

    # ── Build fresh DKL model ──────────────────────────────────────────
    # Start from random weights — BO simulation has no prior knowledge.
    encoder = CGCNNEncoder(
        atom_dim  = cfg.model.atom_dim,
        bond_dim  = cfg.model.bond_dim,
        hidden_dim= cfg.model.hidden_dim,
        n_conv    = cfg.model.n_conv,
        n_fc      = cfg.model.n_fc,
        pooling   = cfg.model.pooling,
    )
    surrogate = build_surrogate(cfg.surrogate)
    dkl = DKLModel(
        encoder      = encoder,
        surrogate    = surrogate,
        encoder_lr   = float(cfg.model.encoder_lr),
        gp_lr        = float(cfg.surrogate.lr),
        device       = device,
        standardize  = cfg.model.get("standardize_embeddings", False),
    )

    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder parameters: {n_params:,}  |  Surrogate: {type(surrogate).__name__}")

    # ── E2: Load recalibration temperature τ (if enabled) ─────────────
    std_scale = 1.0
    if cfg.surrogate.get("recalibrate", False):
        recal_path = Path(cfg.results_dir) / "recalibration_params.json"
        if recal_path.exists():
            with open(recal_path) as fh:
                std_scale = float(json.load(fh).get("tau", 1.0))
            logger.info(f"Recalibration: τ = {std_scale:.4f} loaded from {recal_path}")
        else:
            logger.warning(
                f"recalibrate=true but {recal_path} not found. "
                "Run 02_eval_surrogate.py first to fit τ. Using τ=1.0."
            )

    # ── Run DKL-BO loop ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        f"DKL-BO  |  {cfg.bo.n_cycles} cycles  |  "
        f"acq={cfg.bo.acquisition}  β={cfg.bo.beta}  "
        f"retrain_every={cfg.bo.retrain_every_k}"
    )
    logger.info("=" * 60)

    bo_loop    = BOLoop(dkl=dkl, cache=cache, meta_df=df, cfg=cfg.bo, seed=cfg.seed,
                       std_scale=std_scale)
    bo_results = bo_loop.run()

    # ── Run random baseline ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Random baseline")
    logger.info("=" * 60)

    rand_baseline = RandomBaseline(meta_df=df, cfg=cfg.bo, seed=cfg.seed)
    rand_results  = rand_baseline.run()

    # ── Save results ───────────────────────────────────────────────────
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    surrogate_tag = "" if cfg.surrogate.surrogate_type == "exact" else f"_{cfg.surrogate.surrogate_type}"
    bo_path   = results_dir / f"bo_ucb_beta{cfg.bo.beta}{surrogate_tag}_results.csv"
    rand_path = results_dir / "bo_random_results.csv"

    bo_results.to_csv(bo_path,   index=False)
    rand_results.to_csv(rand_path, index=False)

    logger.info(f"DKL-BO results   → {bo_path}")
    logger.info(f"Random baseline  → {rand_path}")

    # ── Summary comparison ─────────────────────────────────────────────
    final_bo   = bo_results.iloc[-1]
    final_rand = rand_results.iloc[-1]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  FINAL SUMMARY  ({cfg.bo.n_cycles} cycles, {cfg.bo.n_init} init)")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'DKL-UCB':>10}  {'Random':>10}")
    logger.info("-" * 45)
    logger.info(
        f"{'Best gap found':<20} "
        f"{final_bo.best_so_far:>9.3f} eV  "
        f"{final_rand.best_so_far:>9.3f} eV"
    )
    logger.info(
        f"{'Top-50 hits':<20} "
        f"{final_bo.cumul_top50:>10}  "
        f"{final_rand.cumul_top50:>10}"
    )
    logger.info(
        f"{'Top-10% hits':<20} "
        f"{final_bo.cumul_top10pct:>10}  "
        f"{final_rand.cumul_top10pct:>10}"
    )
    logger.info("=" * 60)

    cache.close()


if __name__ == "__main__":
    main()
