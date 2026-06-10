"""Deep Kernel Learning model: CGCNN encoder + GP surrogate, trained jointly.

Training flow (per epoch):
  1. Forward-pass ALL training graphs through the encoder → embeddings [N, D]
  2. Feed embeddings into GP → compute MLL (ExactGP) or ELBO (SVGP)
  3. Backprop through both GP and encoder simultaneously
  4. Update with component-specific LRs (encoder_lr < gp_lr) to prevent
     mode collapse ([P2] recommendation)

Float precision:
  Encoder runs in float32 (fast, standard for NNs)
  GP runs in float64 (stable matrix inversions — [P1/P2] practice)
  Conversion: embeddings.double() at the boundary

Embedding standardization (E1):
  When standardize=True, embeddings are zero-centred and unit-scaled using
  statistics from the labelled set.  The same scaler is applied to pool
  embeddings so the GP sees consistent input ranges.  This corrects the
  all-positive softplus output distribution that distorts an isotropic kernel.

Embedding cache (used in BO loop, Phase 3):
  Between GP-only refits (retrain_every_k > 1), the encoder weights
  don't change so we cache embeddings and skip re-encoding.
"""

import logging
import time
from typing import List

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .cgcnn_encoder import CGCNNEncoder
from .surrogate import BaseSurrogate

logger = logging.getLogger(__name__)


class DKLModel:
    """Manages the encoder, surrogate, and their joint training.

    Parameters
    ----------
    encoder      : CGCNNEncoder
    surrogate    : BaseSurrogate (ExactGPSurrogate or SVGPSurrogate)
    encoder_lr   : smaller LR for the encoder (prevents mode collapse)
    gp_lr        : larger LR for GP hyperparameters
    device       : 'cuda' | 'cpu' | 'auto'
    standardize  : E1 — if True, standardize embeddings to zero-mean/unit-var
                   before passing to the GP (scaler fit on labelled embeddings)
    """

    def __init__(
        self,
        encoder: CGCNNEncoder,
        surrogate: BaseSurrogate,
        encoder_lr: float = 0.001,
        gp_lr: float = 0.01,
        device: str = "auto",
        standardize: bool = False,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.encoder = encoder.to(self.device)
        self.surrogate = surrogate.to(self.device)
        self.encoder_lr = encoder_lr
        self.gp_lr = gp_lr
        self.standardize = standardize

        # E1: scaler params — fit on labelled embeddings, applied to pool
        self._emb_mean: Tensor | None = None  # [D]
        self._emb_std:  Tensor | None = None  # [D]

        self._embedding_cache: Tensor | None = None  # [N_pool, D], used in BO loop

    # ------------------------------------------------------------------
    # E1: Embedding standardization helpers
    # ------------------------------------------------------------------

    def _fit_scaler(self, emb: Tensor) -> None:
        """Compute mean and std from labelled embeddings (fit step)."""
        self._emb_mean = emb.mean(dim=0)
        self._emb_std  = emb.std(dim=0).clamp(min=1e-6)  # avoid div-by-zero

    def _scale(self, emb: Tensor) -> Tensor:
        """Apply the fitted scaler. No-op when standardize=False or not yet fit."""
        if not self.standardize or self._emb_mean is None:
            return emb
        return (emb - self._emb_mean) / self._emb_std

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        """Encode all graphs in loader → (embeddings [N, D], targets [N]).

        Uses no_grad — for inference only.  For training, use _encode_with_grad.
        """
        self.encoder.eval()
        embs, targets = [], []
        for batch in loader:
            batch = batch.to(self.device)
            e = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(e.cpu())
            targets.append(batch.y.squeeze(-1).cpu())
        return torch.cat(embs), torch.cat(targets)

    def _encode_with_grad(self, all_batch: Batch) -> tuple[Tensor, Tensor]:
        """Encode in training mode (gradients enabled). Used for joint training."""
        self.encoder.train()
        all_batch = all_batch.to(self.device)
        emb = self.encoder(
            all_batch.x, all_batch.edge_index, all_batch.edge_attr, all_batch.batch
        )  # [N, D] — float32, gradients tracked
        y = all_batch.y.squeeze(-1).to(self.device)
        return emb, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_graphs: List,
        n_epochs: int = 100,
        gp_pretrain_epochs: int = 50,
        gp_final_epochs: int = 300,
    ) -> dict:
        """Joint DKL training on all training graphs.

        Parameters
        ----------
        train_graphs      : list of PyG Data objects (all training materials)
        n_epochs          : joint training epochs (encoder + GP together)
        gp_pretrain_epochs: GP-only warmup before joint training starts

        Returns
        -------
        dict with 'losses' (per-epoch MLL/ELBO) and 'time_s' (wall clock)
        """
        t0 = time.perf_counter()

        # Build one big batch of ALL training graphs
        all_batch = Batch.from_data_list(train_graphs)

        # ---- Phase A: GP warmup (encoder frozen, fit GP on initial embeddings) ----
        logger.info(f"GP warmup: {gp_pretrain_epochs} epochs on initial embeddings")
        with torch.no_grad():
            init_emb, train_y = self._encode_with_grad.__func__(self, all_batch)
            # Can't use _encode_with_grad under no_grad; use encode instead
        init_loader = DataLoader(
            _SimpleGraphList(train_graphs), batch_size=len(train_graphs)
        )
        init_emb, train_y = self.encode(init_loader)
        init_emb = init_emb.to(self.device)
        train_y = train_y.to(self.device)
        self.surrogate.fit(init_emb.cpu(), train_y.cpu(), n_epochs=gp_pretrain_epochs)
        self.surrogate.to(self.device)
        self.surrogate.train_mode()

        # ---- Phase B: Joint training (encoder + GP together) ----
        logger.info(f"Joint DKL training: {n_epochs} epochs")

        # Component-specific learning rates
        optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self._get_gp_params(), "lr": self.gp_lr},
            ]
        )

        losses = []
        for epoch in range(n_epochs):
            self.encoder.train()
            self.surrogate.train_mode()
            optimizer.zero_grad()

            emb, y = self._encode_with_grad(all_batch)  # float32, has grad
            loss = self.surrogate.joint_loss(emb, y)    # converts to float64 internally
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

        elapsed = time.perf_counter() - t0
        logger.info(f"DKL training done in {elapsed:.1f}s")

        # Refit GP one final time on the trained encoder's embeddings
        final_emb, final_y = self.encode(
            DataLoader(_SimpleGraphList(train_graphs), batch_size=512)
        )
        # E1: fit scaler on labelled embeddings, then standardize before GP
        self._fit_scaler(final_emb)
        self.surrogate.fit(self._scale(final_emb), final_y, n_epochs=gp_final_epochs)

        return {"losses": losses, "time_s": elapsed}

    def _get_gp_params(self):
        """Collect GP model + likelihood parameters for the optimizer.

        GPyTorch registers the likelihood as a sub-module of the GP model,
        so its parameters appear in both model.parameters() and
        likelihood.parameters(). Deduplicate by tensor identity to avoid
        the 'duplicate parameters' warning that would become an error in
        future PyTorch versions.
        """
        seen, params = set(), []
        for src in (self.surrogate.model, self.surrogate.likelihood):
            if src is None:
                continue
            for p in src.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        """Encode graphs and predict (mean, std). Used for offline eval."""
        emb, _ = self.encode(loader)
        return self.surrogate.predict(self._scale(emb))

    # ------------------------------------------------------------------
    # Embedding cache (used in Phase 3 BO loop)
    # ------------------------------------------------------------------

    def cache_pool_embeddings(self, pool_loader: DataLoader) -> None:
        """Pre-compute and store all pool embeddings (Phase 3 optimisation).

        Encoder weights only change every retrain_every_k cycles.
        Between retrains, we reuse these cached embeddings rather than
        re-encoding 3000+ graphs on every acquisition step.
        """
        emb, _ = self.encode(pool_loader)
        self._embedding_cache = emb

    def clear_embedding_cache(self) -> None:
        self._embedding_cache = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class _SimpleGraphList(torch.utils.data.Dataset):
    """Minimal Dataset wrapper around a plain Python list of PyG Data objects."""

    def __init__(self, graphs: List):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]
