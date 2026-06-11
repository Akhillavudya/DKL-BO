"""Offline surrogate evaluation — Phase 2.

Train DKL on the train split and measure predictive + calibration quality
on the test split.  No BO loop — pure model quality assessment.

Why offline first (plan Section 2, Phase 3):
  Predictive accuracy and search efficiency are TWO DIFFERENT things.
  A model can have mediocre MAE but still be a great BO surrogate
  (because BO cares about *ranking* not absolute accuracy).
  We measure them independently so we don't conflate them.

Usage:
    python scripts/02_eval_surrogate.py data.db_path=data/raw/c2db.db

Switch to SVGP:
    python scripts/02_eval_surrogate.py data.db_path=... surrogate=gp_svgp
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from dklbo.data.cache import GraphCache, config_hash
from dklbo.data.dataset import GraphDataset
from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.eval.metrics_calibration import compute_calibration_metrics, print_metrics_table
from dklbo.eval.recalibration import (
    fit_temperature,
    recalibration_report,
    print_recalibration_report,
)
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

    # ------------------------------------------------------------------ #
    # Load metadata + open cache
    # ------------------------------------------------------------------ #
    cache_dir = Path(cfg.data.cache_dir)
    meta_path = cache_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.parquet not found at {meta_path}. Run 01_build_cache.py first."
        )
    df = pd.read_parquet(meta_path)
    logger.info(f"Loaded metadata: {len(df)} materials")
    logger.info(f"  Split counts: {df['split'].value_counts().to_dict()}")

    preproc_cfg = {
        "radius": cfg.data.radius,
        "vacuum_cutoff": cfg.data.vacuum_cutoff,
        "max_neighbors": cfg.data.max_neighbors,
        "target": cfg.data.target,
        "gap_min": cfg.data.gap_min,
    }
    cache = GraphCache(cfg.data.cache_dir, preproc_cfg, readonly=True)

    # ------------------------------------------------------------------ #
    # Build DataLoaders for each split
    # ------------------------------------------------------------------ #
    batch_size = cfg.model.get("batch_size", 64)

    def make_loader(split: str, shuffle: bool = False) -> DataLoader:
        uids = df[df["split"] == split]["uid"].tolist()
        dataset = GraphDataset(cache, uids)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    train_uids = df[df["split"] == "train"]["uid"].tolist()
    train_graphs = [cache[uid] for uid in train_uids]

    logger.info(
        f"DataLoaders ready — "
        f"train={len(train_uids)}  val={len(df[df['split']=='val'])}  "
        f"test={len(df[df['split']=='test'])}"
    )

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    encoder = CGCNNEncoder(
        atom_dim=cfg.model.atom_dim,
        bond_dim=cfg.model.bond_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_conv=cfg.model.n_conv,
        n_fc=cfg.model.n_fc,
        pooling=cfg.model.pooling,
    )

    # Access surrogate config — try dedicated surrogate key, fallback to model
    gp_cfg = cfg.get("surrogate", cfg.model)
    surrogate = build_surrogate(gp_cfg)

    surrogate_name = type(surrogate).__name__

    dkl = DKLModel(
        encoder=encoder,
        surrogate=surrogate,
        encoder_lr=cfg.model.get("encoder_lr", 0.001),
        gp_lr=gp_cfg.get("lr", 0.01),
        device=device,
        standardize=cfg.model.get("standardize_embeddings", False),
    )

    logger.info(f"Model: CGCNN encoder + {surrogate_name}")
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"  Encoder parameters: {n_params:,}")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    logger.info("=" * 55)
    logger.info("Training DKL (offline, no BO loop)")
    logger.info("=" * 55)

    t_train = time.perf_counter()
    results = dkl.fit(
        train_graphs=train_graphs,
        n_epochs=gp_cfg.get("n_train_epochs", 100),
        gp_pretrain_epochs=50,
        gp_final_epochs=gp_cfg.get("n_train_epochs", 300),
    )
    train_time = time.perf_counter() - t_train
    logger.info(f"Training wall-clock: {train_time:.1f}s  ({train_time/60:.1f} min)")

    # ------------------------------------------------------------------ #
    # Evaluate on val split (sanity check)
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on validation split...")
    val_emb, val_y = dkl.encode(val_loader)
    # Use dkl.predict() so the E1 scaler (if enabled) is applied consistently
    val_mean, val_std = dkl.surrogate.predict(dkl._scale(val_emb))
    acc_val = compute_accuracy_metrics(val_y, val_mean)
    cal_val = compute_calibration_metrics(val_y, val_mean, val_std)
    print_metrics_table(acc_val, cal_val, split="val", surrogate=surrogate_name)

    # ------------------------------------------------------------------ #
    # E2: Fit recalibration on val set
    # ------------------------------------------------------------------ #
    do_recal = gp_cfg.get("recalibrate", False)
    tau = 1.0
    if do_recal:
        logger.info("Fitting temperature recalibration on val split...")
        tau = fit_temperature(val_y, val_mean, val_std)
        report = recalibration_report(val_y, val_mean, val_std, tau)
        print_recalibration_report(report)

        recal_path = Path(cfg.results_dir) / "recalibration_params.json"
        recal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(recal_path, "w") as fh:
            json.dump({"tau": tau, "surrogate": surrogate_name}, fh, indent=2)
        logger.info(f"Recalibration params saved → {recal_path}")
    else:
        logger.info("Recalibration disabled (set surrogate.recalibrate=true to enable)")

    # ------------------------------------------------------------------ #
    # Evaluate on test split (final)
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on test split...")
    test_emb, test_y = dkl.encode(test_loader)
    test_mean, test_std = dkl.surrogate.predict(dkl._scale(test_emb))
    acc_test = compute_accuracy_metrics(test_y, test_mean)
    cal_test = compute_calibration_metrics(test_y, test_mean, test_std)
    print_metrics_table(acc_test, cal_test, split="test", surrogate=surrogate_name)

    if do_recal and tau != 1.0:
        logger.info("Test calibration after temperature recalibration:")
        test_std_cal = test_std * tau
        cal_test_recal = compute_calibration_metrics(test_y, test_mean, test_std_cal)
        print_metrics_table(acc_test, cal_test_recal, split="test (recalibrated)", surrogate=surrogate_name)

    # ------------------------------------------------------------------ #
    # Save metrics to CSV
    # ------------------------------------------------------------------ #
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"surrogate_metrics_{surrogate_name.lower()}.csv"

    rows = []
    for split, acc, cal in [("val", acc_val, cal_val), ("test", acc_test, cal_test)]:
        rows.append({
            "surrogate": surrogate_name,
            "split": split,
            "seed": cfg.seed,
            "train_time_s": round(train_time, 2),
            "mae": round(acc.mae, 4),
            "rmse": round(acc.rmse, 4),
            "r2": round(acc.r2, 4),
            "nll": round(cal.nll, 4),
            "ence": round(cal.ence, 4),
            "miscal_area": round(cal.miscal_area, 4),
            "spearman_rho": round(cal.spearman_rho, 4),
            "coverage_95": round(cal.coverage_95, 4),
        })

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved → {metrics_path}")

    # Training loss convergence
    loss_path = results_dir / f"training_loss_{surrogate_name.lower()}.csv"
    pd.DataFrame({"epoch": range(len(results["losses"])), "loss": results["losses"]}).to_csv(
        loss_path, index=False
    )
    logger.info(f"Training loss curve saved → {loss_path}")

    cache.close()


if __name__ == "__main__":
    main()
