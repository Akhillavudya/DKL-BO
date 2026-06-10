"""Baseline comparators for BO evaluation.

RandomBaseline
--------------
Picks uniformly at random from the unlabelled pool every cycle.
Zero compute cost — the null hypothesis.
DKL-BO must beat this to justify its complexity.

Both baselines return a DataFrame with exactly the same columns as BOLoop
so they can be concatenated and plotted together in Phase 5.
"""

import logging
import random
from typing import Dict, List

import pandas as pd

from .loop import CycleRecord   # shared dataclass keeps column names identical

logger = logging.getLogger(__name__)


class RandomBaseline:
    """Pure random search — no model, no predictions.

    Parameters
    ----------
    meta_df : DataFrame with columns [uid, target]
    cfg     : same BO config as BOLoop (only n_init and n_cycles are used)
    seed    : reproducibility seed — use the SAME seed as BOLoop for fair comparison
    """

    def __init__(self, meta_df: pd.DataFrame, cfg, seed: int = 42) -> None:
        self.oracle: Dict[str, float] = dict(zip(meta_df["uid"], meta_df["target"]))
        self.all_uids: List[str]      = meta_df["uid"].tolist()
        self.cfg  = cfg
        self.seed = seed

        sorted_vals = sorted(self.oracle.values(), reverse=True)
        n = len(sorted_vals)
        self.top50_threshold    = sorted_vals[min(49, n - 1)]
        self.top10pct_threshold = sorted_vals[max(0, int(0.1 * n) - 1)]

    def run(self) -> pd.DataFrame:
        """Run random baseline. Returns one-row-per-cycle DataFrame."""
        random.seed(self.seed)

        init_uids     = random.sample(self.all_uids, int(self.cfg.n_init))
        labelled_uids: List[str] = list(init_uids)
        pool_uids:     List[str] = [u for u in self.all_uids
                                    if u not in set(labelled_uids)]

        best_so_far    = max(self.oracle[u] for u in labelled_uids)
        cumul_top50    = 0
        cumul_top10pct = 0
        records: List[CycleRecord] = []

        logger.info(
            f"RandomBaseline: {self.cfg.n_init} init labels  |  "
            f"best gap = {best_so_far:.3f} eV"
        )

        for cycle in range(int(self.cfg.n_cycles)):
            idx          = random.randrange(len(pool_uids))
            selected_uid = pool_uids[idx]
            true_gap     = self.oracle[selected_uid]

            if true_gap > best_so_far:
                best_so_far = true_gap

            is_top50    = true_gap >= self.top50_threshold
            is_top10pct = true_gap >= self.top10pct_threshold
            cumul_top50    += int(is_top50)
            cumul_top10pct += int(is_top10pct)

            records.append(CycleRecord(
                cycle          = cycle + 1,
                uid            = selected_uid,
                gap_acquired   = true_gap,
                acquisition_fn = "random",
                best_so_far    = best_so_far,
                n_labelled     = len(labelled_uids) + 1,
                is_top50       = is_top50,
                is_top10pct    = is_top10pct,
                cumul_top50    = cumul_top50,
                cumul_top10pct = cumul_top10pct,
                train_time_s   = 0.0,
                predict_time_s = 0.0,
            ))

            labelled_uids.append(selected_uid)
            pool_uids.pop(idx)

        logger.info(
            f"RandomBaseline done  |  "
            f"best={best_so_far:.3f} eV  "
            f"top50={cumul_top50}  top10%={cumul_top10pct}"
        )
        return pd.DataFrame([vars(r) for r in records])
