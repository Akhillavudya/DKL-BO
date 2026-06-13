"""Bayesian Optimisation loop — Phase 3.

Simulation setup
----------------
All 3351 materials are the candidate pool.
Oracle: look up true gap_hse (target column) from metadata — simulates running DFT.
Start with n_init=10 random labels, then acquire one material per cycle.

Training schedule (retrain_every_k)
-------------------------------------
Every K cycles  : full DKL retrain — encoder + GP trained jointly from scratch
Other cycles    : GP-only refit   — encoder frozen, only GP kernel params updated

Why alternate?
  Full retrain is expensive (~20s) but improves embeddings.
  GP refit is cheap (~1s) and updates the surrogate as new labels come in.
  Alternating gives most of the accuracy gain at a fraction of the compute.

Embedding cache
---------------
The encoder only changes on full-retrain cycles.
Between retrains, pool embeddings [N_pool, 32] are cached and reused.
When a material is labelled, its row is removed from the cache instead of
re-encoding the whole pool. Saves ~3000 forward passes per GP-refit cycle.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ..data.dataset import GraphDataset
from ..models.dkl import DKLModel
from .acquisition import get_acquisition

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    """One row in the BO results DataFrame."""
    cycle:          int
    uid:            str
    gap_acquired:   float    # true band gap of the selected material (eV)
    acquisition_fn: str
    best_so_far:    float    # best gap found up to and including this cycle (eV)
    n_labelled:     int      # total labelled materials after this cycle
    is_top50:       bool     # is this material in the top-50 by gap?
    is_top10pct:    bool     # is this material in the top-10% by gap?
    cumul_top50:    int      # cumulative top-50 hits so far
    cumul_top10pct: int      # cumulative top-10% hits so far
    train_time_s:   float
    predict_time_s: float


class BOLoop:
    """Simulated active-learning loop driven by a DKL surrogate.

    Parameters
    ----------
    dkl     : DKLModel whose weights are mutated in-place each cycle
    cache   : GraphCache (read-only) — fetches graphs for training
    meta_df : DataFrame with columns [uid, target]
    cfg     : Hydra DictConfig with keys: acquisition, beta, n_init,
              n_cycles, retrain_every_k, n_joint_epochs,
              n_pretrain_epochs, gp_refit_epochs
    seed    : reproducibility seed
    """

    def __init__(
        self,
        dkl: DKLModel,
        cache,
        meta_df: pd.DataFrame,
        cfg,
        seed: int = 42,
        std_scale: float = 1.0,
        direction: str = "max",
    ) -> None:
        self.dkl       = dkl
        self.cache     = cache
        self.std_scale = std_scale  # E2: temperature-scaling factor τ; 1.0 = no recal
        self.oracle: Dict[str, float] = dict(zip(meta_df["uid"], meta_df["target"]))
        self.all_uids: List[str]      = meta_df["uid"].tolist()
        self.cfg    = cfg
        self.seed   = seed

        # Optimisation direction. The GP is always trained to maximise an
        # *internal* target y_internal = sign * y_true, so all acquisition /
        # argmax / top-k logic stays in the maximise convention. Reporting is
        # converted back to original units (best_so_far = sign * best_internal).
        if direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
        self.direction = direction
        self.sign = 1.0 if direction == "max" else -1.0

        # Precompute top-k thresholds once (used every cycle for metrics).
        # "Best" always means largest internal value, so for a min task the
        # top-50 are the 50 *lowest* original values.
        internal_vals = sorted((self.sign * v for v in self.oracle.values()),
                               reverse=True)
        n = len(internal_vals)
        self.top50_threshold    = internal_vals[min(49, n - 1)]
        self.top10pct_threshold = internal_vals[max(0, int(0.1 * n) - 1)]

        logger.info(
            f"BOLoop: {n} materials  |  direction={direction}  |  "
            f"top-50 internal threshold={self.top50_threshold:.2f}  "
            f"top-10% internal threshold={self.top10pct_threshold:.2f}"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full BO loop. Returns one-row-per-cycle DataFrame."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        acq_fn   = get_acquisition(self.cfg.acquisition)
        xi       = float(self.cfg.get("xi", 0.01))   # EI exploration jitter
        n_joint  = int(self.cfg.get("n_joint_epochs",    50))
        n_pre    = int(self.cfg.get("n_pretrain_epochs", 20))
        n_refit  = int(self.cfg.get("gp_refit_epochs",   50))

        # ── Initialise pool ───────────────────────────────────────────
        init_uids     = random.sample(self.all_uids, int(self.cfg.n_init))
        labelled_uids: List[str] = list(init_uids)
        pool_uids:     List[str] = [u for u in self.all_uids
                                    if u not in set(labelled_uids)]

        # Track the incumbent in internal (signed) space; report in original units.
        best_internal = max(self.sign * self.oracle[u] for u in labelled_uids)
        best_so_far   = self.sign * best_internal
        logger.info(
            f"Init: {self.cfg.n_init} random labels  |  "
            f"best ({self.direction}) = {best_so_far:.3f}"
        )

        # Caches — invalidated whenever the encoder is retrained
        pool_embs:     Optional[torch.Tensor] = None   # [N_pool, D]  CPU float32
        labelled_embs: Optional[torch.Tensor] = None   # [N_lab,  D]  CPU float32

        cumul_top50    = 0
        cumul_top10pct = 0
        records: List[CycleRecord] = []

        for cycle in range(int(self.cfg.n_cycles)):
            full_retrain = (cycle % int(self.cfg.retrain_every_k) == 0)

            # ── Train ─────────────────────────────────────────────────
            t_train = time.perf_counter()

            if full_retrain:
                logger.info(
                    f"Cycle {cycle + 1:3d}: full DKL retrain  "
                    f"({len(labelled_uids)} labelled)"
                )
                labelled_graphs = [self.cache[uid] for uid in labelled_uids]
                train_y_internal = torch.tensor(
                    [self.sign * self.oracle[u] for u in labelled_uids],
                    dtype=torch.float32,
                )
                self.dkl.fit(
                    train_graphs      = labelled_graphs,
                    n_epochs          = n_joint,
                    gp_pretrain_epochs= n_pre,
                    train_y           = train_y_internal,
                )
                pool_embs     = None   # encoder changed → invalidate
                labelled_embs = None

            else:
                # GP-only refit (encoder unchanged → cached embeddings still valid)
                if labelled_embs is not None:
                    train_y = torch.tensor(
                        [self.sign * self.oracle[u] for u in labelled_uids],
                        dtype=torch.float32,
                    )
                    # E1: re-fit scaler on updated labelled set, then standardize
                    self.dkl._fit_scaler(labelled_embs)
                    self.dkl.surrogate.fit(
                        self.dkl._scale(labelled_embs), train_y, n_epochs=n_refit
                    )
                    logger.info(
                        f"Cycle {cycle + 1:3d}: GP refit  "
                        f"({len(labelled_uids)} labelled)"
                    )

            train_time = time.perf_counter() - t_train

            # ── Encode (or reuse cache) ────────────────────────────────
            if pool_embs is None:
                pool_embs, _     = self.dkl.encode(
                    DataLoader(GraphDataset(self.cache, pool_uids),
                               batch_size=256, shuffle=False)
                )
                labelled_embs, _ = self.dkl.encode(
                    DataLoader(GraphDataset(self.cache, labelled_uids),
                               batch_size=256, shuffle=False)
                )

            # ── Predict & score ───────────────────────────────────────
            t_pred = time.perf_counter()
            self.dkl.surrogate.eval_mode()
            # E1: apply scaler to pool embeddings before GP predict
            mean, std = self.dkl.surrogate.predict(self.dkl._scale(pool_embs))
            # E2: apply temperature-scaling recalibration (τ=1.0 is a no-op)
            std_cal = std * self.std_scale
            # mean/std are in internal (signed) space; best_f is the internal incumbent.
            scores  = acq_fn(mean, std_cal, beta=float(self.cfg.beta),
                             best_f=best_internal, xi=xi)
            predict_time = time.perf_counter() - t_pred

            # ── Select & label ────────────────────────────────────────
            best_idx     = int(scores.argmax().item())
            selected_uid = pool_uids[best_idx]
            true_gap     = self.oracle[selected_uid]          # original units
            internal_gap = self.sign * true_gap

            if internal_gap > best_internal:
                best_internal = internal_gap
            best_so_far = self.sign * best_internal           # back to original units

            is_top50    = internal_gap >= self.top50_threshold
            is_top10pct = internal_gap >= self.top10pct_threshold
            cumul_top50    += int(is_top50)
            cumul_top10pct += int(is_top10pct)

            logger.info(
                f"  → {selected_uid}  gap={true_gap:.3f} eV  "
                f"best={best_so_far:.3f} eV  "
                f"top50={cumul_top50}  top10%={cumul_top10pct}"
            )

            records.append(CycleRecord(
                cycle          = cycle + 1,
                uid            = selected_uid,
                gap_acquired   = true_gap,
                acquisition_fn = self.cfg.acquisition,
                best_so_far    = best_so_far,
                n_labelled     = len(labelled_uids) + 1,
                is_top50       = is_top50,
                is_top10pct    = is_top10pct,
                cumul_top50    = cumul_top50,
                cumul_top10pct = cumul_top10pct,
                train_time_s   = round(train_time,   3),
                predict_time_s = round(predict_time, 3),
            ))

            # ── Update labelled / pool ────────────────────────────────
            labelled_uids.append(selected_uid)
            labelled_embs = torch.cat(
                [labelled_embs, pool_embs[best_idx : best_idx + 1]], dim=0
            )
            pool_uids.pop(best_idx)
            pool_embs = torch.cat(
                [pool_embs[:best_idx], pool_embs[best_idx + 1 :]], dim=0
            )

        return pd.DataFrame([vars(r) for r in records])
