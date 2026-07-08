"""Phase 2 (SNUMAT side-test) — pre-train the DKL encoder for the HSE band gap.

Mirror of the C2DB rebuild's `scripts/02_pretrain_encoder.py`, simplified to the single
band-gap target (SNUMAT has no effective mass, so no per-target loop and no --log step).

The CGCNN encoder (+ its GP head) is trained on the `train` split to predict the HSE gap,
then frozen and used to turn EVERY material into a 32-d embedding. The held-out `pool` is
never trained on — only encoded.

Method follows Kiyohara & Kumagai (Paper 2): ExactGP head, fixed training budget, no
validation set (the pool stays a pristine test set).

Outputs (snumat_generalization/results/)
    encoder_gap.pt          trained encoder weights
    embeddings_gap.parquet  uid + 32 embedding dims, for ALL materials (master order)

Usage
    python snumat_generalization/scripts/02_pretrain_encoder.py
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
from torch_geometric.loader import DataLoader

from lib import config
from dklbo.data.cache import GraphCache
from dklbo.data.dataset import GraphDataset
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CGCNN encoder architecture (Paper-2 design). atom_dim=100 matches the 3D graph one-hot.
ENCODER_KW = dict(atom_dim=config.GRAPH_PREPROC["atom_feat_dim"], bond_dim=10,
                  hidden_dim=32, n_conv=3, n_fc=1, pooling="attention")
ENCODER_LR = 0.001
GP_LR = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    seed_everything(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    train_df = df[df.split == "train"]
    logger.info(f"Target=gap(HSE)  |  train={len(train_df)}  "
                f"pool={int((df.split=='pool').sum())}  device={device}")

    cache = GraphCache(str(config.CACHE_DIR), config.GRAPH_PREPROC, readonly=True)

    enc = CGCNNEncoder(**ENCODER_KW)
    surrogate = ExactGPSurrogate(lr=GP_LR, n_epochs=100, ard=False)
    dkl = DKLModel(encoder=enc, surrogate=surrogate, encoder_lr=ENCODER_LR,
                   gp_lr=GP_LR, device=device, standardize=True)

    train_graphs = [cache[u] for u in train_df.uid]
    train_y = torch.tensor(train_df["gap"].to_numpy(), dtype=torch.float32)
    logger.info(f"Training CGCNN encoder to predict HSE gap ({args.epochs} epochs)…")
    dkl.fit(train_graphs=train_graphs, n_epochs=args.epochs,
            gp_pretrain_epochs=50, gp_final_epochs=300, train_y=train_y)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), config.RESULTS_DIR / "encoder_gap.pt")

    # Encode EVERY material (train + pool) in master order.
    all_uids = df.uid.tolist()
    loader = DataLoader(GraphDataset(cache, all_uids), batch_size=256, shuffle=False)
    emb, _ = dkl.encode(loader)
    out = pd.concat([pd.DataFrame({"uid": all_uids}),
                     pd.DataFrame(emb.numpy())], axis=1)
    out.to_parquet(config.RESULTS_DIR / "embeddings_gap.parquet", index=False)
    cache.close()

    logger.info(f"Saved encoder_gap.pt and embeddings_gap.parquet "
                f"({emb.shape[0]} × {emb.shape[1]}) → {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
