"""Phase 4 — pre-train the DKL encoder on the train split, then dump embeddings.

This gives DKL the "lots of examples" advantage it lacked in the Plan-2 benchmark.
We train the CGCNN encoder (+ GP) on the phase-4 TRAIN split (2,319 band-gap examples),
freeze the encoder, and turn every material into a 32-d embedding. The held-out POOL is
hunted later (scripts 12) using these frozen features — so DKL never sees pool labels.

We also dump embeddings from a fresh RANDOM-init encoder ("cold") as a control: if
DKL-pretrained beats Std-GP but DKL-cold does not, the win is caused by the training data.

Outputs (results/phase4/)
-------------------------
  pretrained_encoder.pt          encoder.state_dict()
  embeddings_pretrained.parquet  uid + 32 dims (frozen pre-trained encoder)
  embeddings_cold.parquet        uid + 32 dims (random-init encoder)

Usage
-----
    python scripts/11_pretrain_encoder.py data.db_path=data/raw/c2db.db
"""

import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from dklbo.data.cache import GraphCache
from dklbo.data.dataset import GraphDataset
from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate, build_surrogate
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _make_encoder(cfg) -> CGCNNEncoder:
    return CGCNNEncoder(
        atom_dim=cfg.model.atom_dim,
        bond_dim=cfg.model.bond_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_conv=cfg.model.n_conv,
        n_fc=cfg.model.n_fc,
        pooling=cfg.model.pooling,
        spectral_norm=cfg.model.get("spectral_norm", False),
        sn_coeff=float(cfg.model.get("sn_coeff", 1.0)),
    )


def _encode_all(dkl: DKLModel, cache, uids) -> torch.Tensor:
    loader = DataLoader(GraphDataset(cache, list(uids)), batch_size=256, shuffle=False)
    emb, _ = dkl.encode(loader)            # [N, D], order matches `uids`
    return emb


def _pool_r2(emb: torch.Tensor, df: pd.DataFrame, uids) -> float:
    """Fit a GP on TRAIN embeddings, predict POOL — a quick feature-quality score."""
    idx = {u: i for i, u in enumerate(uids)}
    tr = df[df.split4 == "train"]; po = df[df.split4 == "pool"]
    Xtr = emb[[idx[u] for u in tr.uid]].double()
    Xpo = emb[[idx[u] for u in po.uid]].double()
    # standardize features by train stats
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    Xtr, Xpo = (Xtr - mu) / sd, (Xpo - mu) / sd
    ytr = torch.tensor(tr.target.to_numpy())
    gp = ExactGPSurrogate(n_epochs=150, ard=False)
    gp.fit(Xtr, ytr); gp.eval_mode()
    mean, _ = gp.predict(Xpo)
    ypo = torch.tensor(po.target.to_numpy(), dtype=torch.float32)
    return compute_accuracy_metrics(ypo, mean.float()).r2


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(REPO / "data" / "cache" / "metadata_phase4.parquet")
    train_df = df[df.split4 == "train"]
    logger.info(f"Phase-4: train={len(train_df)}  pool={int((df.split4=='pool').sum())}")

    preproc = {
        "radius": cfg.data.radius, "vacuum_cutoff": cfg.data.vacuum_cutoff,
        "max_neighbors": cfg.data.max_neighbors,
        "target": cfg.data.get("graph_cache_target", "gap_hse"),
        "gap_min": cfg.data.gap_min,
    }
    cache = GraphCache(str(REPO / cfg.data.cache_dir), preproc, readonly=True)

    out_dir = REPO / cfg.results_dir / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_uids = df.uid.tolist()

    # ── Pre-trained encoder ───────────────────────────────────────────
    logger.info("Training DKL encoder on the phase-4 train split…")
    enc = _make_encoder(cfg)
    surrogate = build_surrogate(cfg.get("surrogate", cfg.model))
    dkl = DKLModel(encoder=enc, surrogate=surrogate,
                   encoder_lr=cfg.model.get("encoder_lr", 0.001),
                   gp_lr=cfg.get("surrogate", cfg.model).get("lr", 0.01),
                   device=device, standardize=cfg.model.get("standardize_embeddings", False))
    train_graphs = [cache[u] for u in train_df.uid]
    dkl.fit(train_graphs=train_graphs,
            n_epochs=cfg.get("surrogate", cfg.model).get("n_train_epochs", 100),
            gp_pretrain_epochs=50,
            gp_final_epochs=cfg.get("surrogate", cfg.model).get("n_train_epochs", 300))

    torch.save(enc.state_dict(), out_dir / "pretrained_encoder.pt")
    emb_pre = _encode_all(dkl, cache, all_uids)
    pd.concat([pd.DataFrame({"uid": all_uids}),
               pd.DataFrame(emb_pre.numpy())], axis=1).to_parquet(
        out_dir / "embeddings_pretrained.parquet", index=False)

    # ── Cold (random) encoder control ─────────────────────────────────
    cold_enc = _make_encoder(cfg)
    cold_dkl = DKLModel(encoder=cold_enc, surrogate=build_surrogate(cfg.get("surrogate", cfg.model)),
                        device=device)
    emb_cold = _encode_all(cold_dkl, cache, all_uids)
    pd.concat([pd.DataFrame({"uid": all_uids}),
               pd.DataFrame(emb_cold.numpy())], axis=1).to_parquet(
        out_dir / "embeddings_cold.parquet", index=False)

    # ── Sanity: feature quality on the held-out pool ──────────────────
    r2_pre = _pool_r2(emb_pre, df, all_uids)
    r2_cold = _pool_r2(emb_cold, df, all_uids)
    logger.info("=" * 56)
    logger.info(f"Pool R²  |  pre-trained encoder = {r2_pre:.3f}   "
                f"cold encoder = {r2_cold:.3f}")
    logger.info("(pre-trained should be much higher — confirms the encoder learned useful features)")
    logger.info("=" * 56)
    logger.info(f"Embeddings + encoder saved → {out_dir}")
    cache.close()


if __name__ == "__main__":
    main()
