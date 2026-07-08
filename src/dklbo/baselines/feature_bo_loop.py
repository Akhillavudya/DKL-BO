"""Standard GP-BO loop over a static handcrafted-descriptor matrix.

This is the descriptor-based comparator to DKL-BO. It mirrors `bo.loop.BOLoop`
but has *no* encoder: the feature matrix [N, D] is fixed and known for every
candidate up front, so each cycle simply refits the GP on the labelled rows and
scores the pool rows. It emits the exact same `CycleRecord` columns as `BOLoop`
and `RandomBaseline`, so all three concatenate and plot together.

Fairness guarantees vs DKL-BO (see docs/benchmark_design.md):
  * identical GP backend  — `models.surrogate.ExactGPSurrogate` (Matérn-5/2)
  * identical init set     — same `random.seed` + `random.sample(all_uids, n_init)`
  * identical acquisition  — EI/UCB via `bo.acquisition.get_acquisition`
  * identical direction    — the same sign trick (y_internal = sign * y_true)
"""

import logging
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..bo.acquisition import get_acquisition
from ..bo.loop import CycleRecord
from ..models.surrogate import ExactGPSurrogate

logger = logging.getLogger(__name__)


class FeatureBOLoop:
    """BO loop over a precomputed descriptor matrix.

    Parameters
    ----------
    X        : np.ndarray [N, D] descriptor matrix aligned to `uids`
    uids     : list[str] row order of X
    meta_df  : DataFrame with columns [uid, target]; defines the oracle. Its uid
               order MUST match the order used by `BOLoop` so the random init set
               is identical for the same seed.
    cfg      : BO config (acquisition, beta, xi, n_init, n_cycles)
    seed     : reproducibility seed — use the SAME seed as BOLoop for paired runs
    direction: "max" | "min" — sign trick, identical to BOLoop
    ard      : if True, one GP lengthscale per descriptor dim (ARD Matérn)
    gp_epochs: GP fit epochs per cycle
    """

    def __init__(
        self,
        X: np.ndarray,
        uids: List[str],
        meta_df: pd.DataFrame,
        cfg,
        seed: int = 42,
        direction: str = "max",
        ard: bool = False,
        gp_epochs: int = 100,
    ) -> None:
        if direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
        self.direction = direction
        self.sign = 1.0 if direction == "max" else -1.0
        self.cfg = cfg
        self.seed = seed
        self.ard = ard
        self.gp_epochs = gp_epochs

        # Standardize descriptors using full-pool statistics. This is NOT leakage:
        # descriptors are cheap and known for every candidate before any labelling;
        # only the *target* is hidden by the oracle.
        Xf = np.asarray(X, dtype=np.float64)
        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        sd[sd < 1e-9] = 1.0
        self.X = torch.tensor((Xf - mu) / sd, dtype=torch.float64)
        self.uid_to_idx: Dict[str, int] = {u: i for i, u in enumerate(uids)}

        self.oracle: Dict[str, float] = dict(zip(meta_df["uid"], meta_df["target"]))
        self.all_uids: List[str] = meta_df["uid"].tolist()

        internal_vals = sorted((self.sign * v for v in self.oracle.values()),
                               reverse=True)
        n = len(internal_vals)
        self.top50_threshold = internal_vals[min(49, n - 1)]
        self.top10pct_threshold = internal_vals[max(0, int(0.1 * n) - 1)]

        logger.info(
            f"FeatureBOLoop: {n} materials × {self.X.shape[1]} descriptors  |  "
            f"direction={direction}"
        )

    def _rows(self, uids: List[str]) -> torch.Tensor:
        return self.X[[self.uid_to_idx[u] for u in uids]]

    def run(self) -> pd.DataFrame:
        """Execute the standard GP-BO loop. Returns one-row-per-cycle DataFrame."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        acq_fn = get_acquisition(self.cfg.acquisition)
        xi = float(self.cfg.get("xi", 0.01))

        init_uids = random.sample(self.all_uids, int(self.cfg.n_init))
        labelled_uids: List[str] = list(init_uids)
        pool_uids: List[str] = [u for u in self.all_uids
                                if u not in set(labelled_uids)]

        best_internal = max(self.sign * self.oracle[u] for u in labelled_uids)
        best_so_far = self.sign * best_internal
        logger.info(
            f"Init: {self.cfg.n_init} random labels  |  "
            f"best ({self.direction}) = {best_so_far:.3f}"
        )

        surrogate = ExactGPSurrogate(n_epochs=self.gp_epochs, ard=self.ard)
        cumul_top50 = 0
        cumul_top10pct = 0
        records: List[CycleRecord] = []

        for cycle in range(int(self.cfg.n_cycles)):
            # ── Fit GP on labelled descriptors (internal signed targets) ──
            t_train = time.perf_counter()
            X_lab = self._rows(labelled_uids)
            y_lab = torch.tensor(
                [self.sign * self.oracle[u] for u in labelled_uids],
                dtype=torch.float64,
            )
            surrogate.fit(X_lab, y_lab, n_epochs=self.gp_epochs)
            train_time = time.perf_counter() - t_train

            # ── Predict & score pool ──────────────────────────────────
            t_pred = time.perf_counter()
            surrogate.eval_mode()
            mean, std = surrogate.predict(self._rows(pool_uids))
            scores = acq_fn(mean, std, beta=float(self.cfg.beta),
                            best_f=best_internal, xi=xi,
                            window_lo=self.cfg.get("window_lo"),
                            window_hi=self.cfg.get("window_hi"))
            predict_time = time.perf_counter() - t_pred

            # ── Select & label ────────────────────────────────────────
            best_idx = int(scores.argmax().item())
            selected_uid = pool_uids[best_idx]
            true_gap = self.oracle[selected_uid]
            internal_gap = self.sign * true_gap

            if internal_gap > best_internal:
                best_internal = internal_gap
            best_so_far = self.sign * best_internal

            is_top50 = internal_gap >= self.top50_threshold
            is_top10pct = internal_gap >= self.top10pct_threshold
            cumul_top50 += int(is_top50)
            cumul_top10pct += int(is_top10pct)

            records.append(CycleRecord(
                cycle=cycle + 1,
                uid=selected_uid,
                gap_acquired=true_gap,
                acquisition_fn=self.cfg.acquisition,
                best_so_far=best_so_far,
                n_labelled=len(labelled_uids) + 1,
                is_top50=is_top50,
                is_top10pct=is_top10pct,
                cumul_top50=cumul_top50,
                cumul_top10pct=cumul_top10pct,
                train_time_s=round(train_time, 3),
                predict_time_s=round(predict_time, 3),
            ))

            labelled_uids.append(selected_uid)
            pool_uids.pop(best_idx)

        logger.info(
            f"FeatureBOLoop done  |  best={best_so_far:.3f}  "
            f"top50={cumul_top50}  top10%={cumul_top10pct}"
        )
        return pd.DataFrame([vars(r) for r in records])
