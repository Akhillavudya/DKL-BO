"""Experiment — does more encoder training lower band-gap MAE, or cause overfitting?

The question
-----------
The DKL "model" is really a CGCNN **neural network** (the encoder, ~the part with lots of
parameters) followed by a small GP head. The encoder is the thing that can OVERFIT. So we
sweep ONE knob — the number of encoder pre-training epochs — and watch the band-gap error.

For each epoch budget E (and several seeds) we:
  1. train the encoder for E epochs on the `train` split (gap target),
  2. freeze it, turn every material into a 32-d embedding,
  3. fit the SAME ExactGP head on the train embeddings (identical to scripts/03),
  4. measure MAE/RMSE/R² on BOTH the train materials and the held-out pool.

Why measure train AND pool: overfitting is exactly when the TRAIN error keeps dropping while
the POOL (test) error flattens then RISES. One curve can't show that — you need both. The
gap between them (pool − train) widening = overfitting.

Standard GP uses fixed handcrafted descriptors (no epochs), so it appears as a flat
reference line — the bar everyone is measured against.

Outputs (results/rebuild/)
  epoch_sweep.csv                 one row per (model, epochs, seed)
  plots/epoch_sweep_gap.png       MAE vs epochs: train vs pool, with Std-GP reference

Usage
    python scripts/exp_epoch_sweep.py                       # full sweep, 5 seeds
    python scripts/exp_epoch_sweep.py --epochs 25 50 --seeds 2   # quick smoke test
"""

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from dklbo.data.cache import GraphCache
from dklbo.data.dataset import GraphDataset
from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO / "data" / "cache"
OUT = REPO / "results" / "rebuild"
PLOTS = OUT / "plots"
CSV = OUT / "epoch_sweep.csv"

TARGET = "gap"
ENCODER_KW = dict(atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1,
                  pooling="attention")
ENCODER_LR, GP_LR = 0.001, 0.01
GRAPH_PREPROC = {"radius": 8.0, "vacuum_cutoff": 4.0, "max_neighbors": 12,
                 "target": "gap_hse", "gap_min": 0.01}
DEFAULT_EPOCHS = [25, 50, 75, 100, 150, 200, 300, 400]


def gp_metrics(feats: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Fit ExactGP on train embeddings/descriptors, score on train AND pool.
    Identical GP recipe to scripts/03_eval_accuracy.py (ARD Matern, train-standardized)."""
    tr, po = df[df.split == "train"], df[df.split == "pool"]
    Xtr = feats.loc[tr.uid].to_numpy("float64")
    Xpo = feats.loc[po.uid].to_numpy("float64")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr_n = torch.tensor((Xtr - mu) / sd)
    Xpo_n = torch.tensor((Xpo - mu) / sd)

    y = df.set_index("uid")[TARGET]
    ytr_raw = y.loc[tr.uid].to_numpy("float64")
    ypo_raw = y.loc[po.uid].to_numpy("float64")
    ym, ys = ytr_raw.mean(), (ytr_raw.std() or 1.0)

    gp = ExactGPSurrogate(n_epochs=200, ard=True)
    gp.fit(Xtr_n, torch.tensor((ytr_raw - ym) / ys))
    gp.eval_mode()
    pred_tr = gp.predict(Xtr_n)[0] * ys + ym
    pred_po = gp.predict(Xpo_n)[0] * ys + ym

    a_tr = compute_accuracy_metrics(torch.tensor(ytr_raw, dtype=torch.float32), pred_tr.float())
    a_po = compute_accuracy_metrics(torch.tensor(ypo_raw, dtype=torch.float32), pred_po.float())
    return {"train_mae": a_tr.mae, "train_rmse": a_tr.rmse, "train_r2": a_tr.r2,
            "pool_mae": a_po.mae, "pool_rmse": a_po.rmse, "pool_r2": a_po.r2}


def dkl_embeddings(df, cache, train_df, epochs, seed) -> pd.DataFrame:
    """Train the CGCNN encoder for `epochs` on the train split, return embeddings for ALL."""
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = CGCNNEncoder(**ENCODER_KW)
    surrogate = ExactGPSurrogate(lr=GP_LR, n_epochs=100, ard=False)
    dkl = DKLModel(encoder=enc, surrogate=surrogate, encoder_lr=ENCODER_LR,
                   gp_lr=GP_LR, device=device, standardize=True)
    train_graphs = [cache[u] for u in train_df.uid]
    train_y = torch.tensor(train_df[TARGET].to_numpy(), dtype=torch.float32)
    dkl.fit(train_graphs=train_graphs, n_epochs=epochs,
            gp_pretrain_epochs=50, gp_final_epochs=300, train_y=train_y)

    all_uids = df.uid.tolist()
    loader = DataLoader(GraphDataset(cache, all_uids), batch_size=256, shuffle=False)
    emb, _ = dkl.encode(loader)
    feats = pd.DataFrame(emb.numpy())
    feats.insert(0, "uid", all_uids)
    return feats.set_index("uid")


def load_done() -> pd.DataFrame:
    return pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(
        columns=["model", "epochs", "seed"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    PLOTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    train_df = df[df.split == "train"]
    cache = GraphCache(str(CACHE_DIR), GRAPH_PREPROC, readonly=True)
    descriptors = pd.read_parquet(CACHE_DIR / "descriptors.parquet").set_index("uid")
    logger.info(f"train={len(train_df)} pool={int((df.split=='pool').sum())} "
                f"| epochs={args.epochs} seeds={seeds}")

    done = load_done()
    rows = done.to_dict("records")
    have = {(r["model"], int(r["epochs"]), int(r["seed"])) for r in rows}

    for seed in seeds:
        # Std-GP reference: descriptors don't depend on epochs — compute once per seed,
        # store under a sentinel epochs=0 so it's easy to pull out for the flat line.
        if ("std_gp", 0, seed) not in have:
            seed_everything(seed)
            m = gp_metrics(descriptors, df)
            rows.append({"model": "std_gp", "epochs": 0, "seed": seed, **m})
            pd.DataFrame(rows).to_csv(CSV, index=False)
            logger.info(f"  std_gp/seed{seed}  pool_MAE={m['pool_mae']:.3f}")

        for E in args.epochs:
            if ("dkl", E, seed) in have:
                continue
            feats = dkl_embeddings(df, cache, train_df, E, seed)
            m = gp_metrics(feats, df)
            rows.append({"model": "dkl", "epochs": E, "seed": seed, **m})
            pd.DataFrame(rows).to_csv(CSV, index=False)
            logger.info(f"  dkl/epochs{E}/seed{seed}  "
                        f"train_MAE={m['train_mae']:.3f} pool_MAE={m['pool_mae']:.3f}")

    cache.close()
    plot()


def plot() -> None:
    d = pd.read_csv(CSV)
    dkl = d[d.model == "dkl"]
    std = d[d.model == "std_gp"]

    def agg(sub, col):
        g = sub.groupby("epochs")[col]
        mean = g.mean()
        n = g.count().clip(lower=1)
        ci = 1.96 * g.std(ddof=1).fillna(0) / np.sqrt(n)
        return mean.index.to_numpy(), mean.to_numpy(), ci.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    for col, color, label in [("train_mae", "#9CA3AF", "DKL — train MAE (fit error)"),
                              ("pool_mae", "#2563EB", "DKL — pool MAE (generalization)")]:
        x, m, ci = agg(dkl, col)
        ax.plot(x, m, "o-", color=color, lw=2.2, label=label)
        ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15)

    if not std.empty:
        ref = std["pool_mae"].mean()
        ax.axhline(ref, ls="--", color="#16A34A", lw=2,
                   label=f"Standard GP — pool MAE ({ref:.3f})")

    # mark the best (lowest pool MAE) epoch
    x, m, _ = agg(dkl, "pool_mae")
    best = int(x[np.argmin(m)])
    ax.axvline(best, ls=":", color="#2563EB", alpha=0.6)
    ax.annotate(f"best pool MAE\n@ {best} epochs", (best, m.min()),
                textcoords="offset points", xytext=(10, 20), fontsize=9, color="#2563EB")

    ax.set_xlabel("encoder pre-training epochs")
    ax.set_ylabel("band-gap MAE (eV) — lower is better")
    ax.set_title("Does more encoder training help? (band gap)\n"
                 "train falls + pool turns up = overfitting", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "epoch_sweep_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {CSV}  and  {PLOTS/'epoch_sweep_gap.png'}")


if __name__ == "__main__":
    main()
