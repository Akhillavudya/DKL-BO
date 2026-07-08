"""Phase 2 (clean rebuild) — pre-train the DKL encoder for ONE target, dump embeddings.

What this does
--------------
The DKL "modern chef" learns features by looking at crystal structures. Here we train
the CGCNN encoder (+ its GP head) on the `train` split to predict ONE target (gap OR
emass), then freeze it and turn EVERY material into a 32-d embedding vector. The held-out
`pool` is never used to train — we only encode it.

Why per-target encoders
-----------------------
Features useful for predicting the band gap are not the same as features useful for
effective mass, so we pre-train one encoder per target. Run this script twice:

    python scripts/02_pretrain_encoder.py --target gap
    python scripts/02_pretrain_encoder.py --target emass

Outputs (results/rebuild/)
--------------------------
  encoder_{target}.pt            the trained encoder weights
  embeddings_{target}.parquet    uid + 32 embedding dims, for ALL master materials

Method note (follows the Kiyohara & Kumagai DKL-BO paper, our template): ExactGP head,
fixed training budget, fit by marginal likelihood. No validation set is carved out — the
pool stays a pristine test set.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from dklbo.data.cache import GraphCache
from dklbo.data.dataset import GraphDataset
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO / "data" / "cache"
OUT_DIR = REPO / "results" / "rebuild"

# CGCNN encoder architecture (from configs/model/cgcnn.yaml — the Paper-2 design).
ENCODER_KW = dict(atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1,
                  pooling="attention")
ENCODER_LR = 0.001
GP_LR = 0.01
# Same structure-only graph cache built in Phase 1.
GRAPH_PREPROC = {"radius": 8.0, "vacuum_cutoff": 4.0, "max_neighbors": 12,
                 "target": "gap_hse", "gap_min": 0.01}
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["gap", "emass"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log", action="store_true",
                   help="train on log10(target) (recommended for heavy-tailed emass)")
    args = ap.parse_args()
    tag = f"{args.target}_log" if args.log else args.target

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    train_df = df[df.split == "train"]
    logger.info(f"Target={args.target}  |  train={len(train_df)}  "
                f"pool={int((df.split=='pool').sum())}  device={device}")

    cache = GraphCache(str(CACHE_DIR), GRAPH_PREPROC, readonly=True)

    # Build encoder + ExactGP head, train jointly on the train split.
    enc = CGCNNEncoder(**ENCODER_KW)
    surrogate = ExactGPSurrogate(lr=GP_LR, n_epochs=100, ard=False)
    dkl = DKLModel(encoder=enc, surrogate=surrogate, encoder_lr=ENCODER_LR,
                   gp_lr=GP_LR, device=device, standardize=True)

    train_graphs = [cache[u] for u in train_df.uid]
    # The cached graphs carry gap_hse as .y; override with the chosen target so the
    # encoder learns features for THIS property (needed for emass).
    y_train = train_df[args.target].to_numpy()
    if args.log:
        import numpy as np
        y_train = np.log10(y_train)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    logger.info(f"Training CGCNN encoder to predict {tag} ({args.epochs} epochs)…")
    dkl.fit(train_graphs=train_graphs, n_epochs=args.epochs,
            gp_pretrain_epochs=50, gp_final_epochs=300, train_y=train_y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), OUT_DIR / f"encoder_{tag}.pt")

    # Encode EVERY material (train + pool) in master order.
    all_uids = df.uid.tolist()
    loader = DataLoader(GraphDataset(cache, all_uids), batch_size=256, shuffle=False)
    emb, _ = dkl.encode(loader)                     # raw embeddings, [N, 32]
    out = pd.concat([pd.DataFrame({"uid": all_uids}),
                     pd.DataFrame(emb.numpy())], axis=1)
    out.to_parquet(OUT_DIR / f"embeddings_{tag}.parquet", index=False)
    cache.close()

    logger.info(f"Saved encoder_{tag}.pt and embeddings_{tag}.parquet "
                f"({emb.shape[0]} × {emb.shape[1]}) → {OUT_DIR}")


if __name__ == "__main__":
    main()
