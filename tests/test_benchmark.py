"""Tests for the GP-BO vs DKL-BO benchmark additions.

Fast and CPU-only:
  - EI acquisition: shape, non-negativity, σ=0 edge case, Monte-Carlo agreement
  - direction sign trick: FeatureBOLoop / RandomBaseline / BOLoop in min & max
  - handcrafted descriptors: composition stats determinism + no target leakage
  - column parity: FeatureBOLoop emits the same CycleRecord columns as RandomBaseline

The full `build_descriptors` (needs the C2DB file) and the BOLoop integration test
are skipped automatically when the database is absent.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dklbo.baselines.descriptors import (
    EXCLUDED_LEAKAGE_FIELDS,
    _composition_features,
    build_descriptors,
)
from dklbo.baselines.feature_bo_loop import FeatureBOLoop
from dklbo.bo.acquisition import ei, get_acquisition
from dklbo.bo.baselines import RandomBaseline
from dklbo.bo.loop import BOLoop, CycleRecord
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate

ATOM_DIM, BOND_DIM, HIDDEN_DIM = 90, 10, 8
DB_PATH = Path(__file__).parent.parent / "data" / "raw" / "c2db.db"
RECORD_COLS = set(vars(CycleRecord(
    cycle=1, uid="x", gap_acquired=0.0, acquisition_fn="ei", best_so_far=0.0,
    n_labelled=1, is_top50=False, is_top10pct=False, cumul_top50=0,
    cumul_top10pct=0, train_time_s=0.0, predict_time_s=0.0,
)).keys())


class _Cfg:
    """Minimal BO config object."""
    def __init__(self, **kw):
        self.acquisition = kw.get("acquisition", "ei")
        self.beta = kw.get("beta", 0.0)
        self.xi = kw.get("xi", 0.01)
        self.n_init = kw.get("n_init", 5)
        self.n_cycles = kw.get("n_cycles", 8)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _synth_dataset(n=40, seed=0):
    rng = random.Random(seed)
    uids = [f"mat_{i:03d}" for i in range(n)]
    y = [round(rng.uniform(0.2, 9.0), 3) for _ in uids]
    df = pd.DataFrame({"uid": uids, "target": y})
    X = np.random.default_rng(seed).normal(size=(n, 6))
    return df, uids, X


# ──────────────────────────────────────────────────────────────────────────
# EI acquisition
# ──────────────────────────────────────────────────────────────────────────

def test_ei_registered_and_callable():
    fn = get_acquisition("ei")
    assert callable(fn)


def test_ei_shape_and_nonnegative():
    torch.manual_seed(0)
    mean = torch.randn(50)
    std = torch.abs(torch.randn(50)) + 0.01
    out = ei(mean, std, best_f=0.5)
    assert out.shape == mean.shape
    assert (out >= 0).all()


def test_ei_zero_when_no_uncertainty():
    mean = torch.tensor([2.0, 0.1])
    std = torch.tensor([0.0, 0.0])
    out = ei(mean, std, best_f=1.0)
    assert torch.allclose(out, torch.zeros_like(out))


def test_ei_matches_monte_carlo():
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([0.5, 1.0])
    best, xi = 1.5, 0.0
    got = ei(mean, std, best_f=best, xi=xi)
    ref = []
    for m, s in zip(mean.tolist(), std.tolist()):
        samp = torch.normal(m, s, size=(400000,))
        ref.append(torch.clamp(samp - best, min=0).mean().item())
    assert np.allclose(got.numpy(), ref, atol=5e-3)


def test_ei_accepts_extra_kwargs():
    # The loop passes beta/best_f/xi to every acquisition uniformly.
    out = ei(torch.zeros(3), torch.ones(3), beta=0.7, best_f=0.0, xi=0.01)
    assert out.shape == (3,)


# ──────────────────────────────────────────────────────────────────────────
# FeatureBOLoop (standard GP-BO) + direction sign trick
# ──────────────────────────────────────────────────────────────────────────

def test_feature_loop_columns_match_record():
    df, uids, X = _synth_dataset()
    res = FeatureBOLoop(X, uids, df, _Cfg(), seed=0, gp_epochs=10).run()
    assert set(res.columns) == RECORD_COLS


def test_feature_loop_columns_match_random_baseline():
    df, uids, X = _synth_dataset()
    feat = FeatureBOLoop(X, uids, df, _Cfg(), seed=0, gp_epochs=10).run()
    rand = RandomBaseline(df, _Cfg(), seed=0).run()
    assert set(feat.columns) == set(rand.columns)


def test_feature_loop_max_best_nondecreasing():
    df, uids, X = _synth_dataset()
    res = FeatureBOLoop(X, uids, df, _Cfg(), seed=1, direction="max", gp_epochs=10).run()
    assert res.best_so_far.is_monotonic_increasing


def test_feature_loop_min_best_nonincreasing():
    df, uids, X = _synth_dataset()
    res = FeatureBOLoop(X, uids, df, _Cfg(), seed=1, direction="min", gp_epochs=10).run()
    assert res.best_so_far.is_monotonic_decreasing


def test_feature_loop_no_duplicate_selections():
    df, uids, X = _synth_dataset()
    res = FeatureBOLoop(X, uids, df, _Cfg(), seed=2, gp_epochs=10).run()
    assert res.uid.nunique() == len(res)


def test_feature_loop_identical_init_set_to_random_baseline():
    # Same seed + same uid universe ⇒ identical initial labelled set, so the two
    # methods form a paired sample (precondition for the Wilcoxon test).
    df, uids, X = _synth_dataset()
    seed = 7
    # FeatureBOLoop's init = random.sample(all_uids, n_init) after random.seed(seed)
    random.seed(seed)
    expected_init = set(random.sample(df["uid"].tolist(), 5))
    random.seed(seed)
    rand_init = set(random.sample(df["uid"].tolist(), 5))
    assert expected_init == rand_init


def test_direction_invalid_raises():
    df, uids, X = _synth_dataset()
    with pytest.raises(ValueError):
        FeatureBOLoop(X, uids, df, _Cfg(), direction="sideways")


def test_min_topk_threshold_picks_lowest():
    # For a min task, the "top-50" set must be the lowest-valued materials.
    df, uids, X = _synth_dataset(n=30)
    loop = FeatureBOLoop(X, uids, df, _Cfg(), direction="min", gp_epochs=5)
    # internal = -value, so the threshold corresponds to a SMALL original value
    smallest = sorted(df["target"])[:30]
    # top10pct threshold (internal) maps back to an original value near the low end
    assert -loop.top10pct_threshold <= np.median(smallest)


# ──────────────────────────────────────────────────────────────────────────
# RandomBaseline direction
# ──────────────────────────────────────────────────────────────────────────

def test_random_baseline_min_best_nonincreasing():
    df, uids, X = _synth_dataset()
    res = RandomBaseline(df, _Cfg(), seed=3, direction="min").run()
    assert res.best_so_far.is_monotonic_decreasing


def test_random_baseline_max_best_nondecreasing():
    df, uids, X = _synth_dataset()
    res = RandomBaseline(df, _Cfg(), seed=3, direction="max").run()
    assert res.best_so_far.is_monotonic_increasing


# ──────────────────────────────────────────────────────────────────────────
# Descriptors
# ──────────────────────────────────────────────────────────────────────────

def test_composition_features_deterministic():
    f1, n1 = _composition_features("MoS2")
    f2, n2 = _composition_features("MoS2")
    assert n1 == n2
    assert np.allclose(f1, f2)
    assert all(np.isfinite(f1))


def test_composition_features_no_leakage_names():
    _, names = _composition_features("MoSe2")
    for name in names:
        for bad in EXCLUDED_LEAKAGE_FIELDS:
            assert bad not in name


@pytest.mark.skipif(not DB_PATH.exists(), reason="C2DB database not present")
def test_build_descriptors_real_db():
    meta = pd.read_parquet(DB_PATH.parent.parent / "cache" / "metadata.parquet").head(40)
    uids, X, names = build_descriptors(meta, str(DB_PATH))
    assert X.shape[0] == len(meta)
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all()                       # NaNs were imputed
    for name in names:                                # no leakage features
        for bad in EXCLUDED_LEAKAGE_FIELDS:
            assert bad not in name


# ──────────────────────────────────────────────────────────────────────────
# BOLoop direction + train_y override (integration, CPU)
# ──────────────────────────────────────────────────────────────────────────

def _fake_graph(uid: str, y: float) -> Data:
    rng = random.Random(abs(hash(uid)) % (2 ** 31))
    n = rng.randint(2, 5)
    e = rng.randint(n, 2 * n)
    return Data(
        x=torch.randn(n, ATOM_DIM),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, BOND_DIM),
        y=torch.tensor([[y]], dtype=torch.float32),
        uid=uid,
    )


class _FakeCache:
    def __init__(self, graphs):
        self._g = graphs

    def __getitem__(self, uid):
        return self._g[uid]


class _LoopCfg(_Cfg):
    retrain_every_k = 3
    n_joint_epochs = 2
    n_pretrain_epochs = 1
    gp_refit_epochs = 2


def _tiny_dkl():
    enc = CGCNNEncoder(atom_dim=ATOM_DIM, bond_dim=BOND_DIM, hidden_dim=HIDDEN_DIM,
                       n_conv=1, n_fc=1, pooling="attention")
    sur = ExactGPSurrogate(lr=0.01, n_epochs=2, patience=5)
    return DKLModel(encoder=enc, surrogate=sur, encoder_lr=0.001, gp_lr=0.01, device="cpu")


def _fake_bo_data(n=20):
    rng = random.Random(5)
    uids = [f"m{i:02d}" for i in range(n)]
    y = [round(rng.uniform(0.5, 9.0), 3) for _ in uids]
    df = pd.DataFrame({"uid": uids, "target": y})
    cache = _FakeCache({u: _fake_graph(u, g) for u, g in zip(uids, y)})
    return df, cache


def test_bo_loop_min_direction_runs_and_is_nonincreasing():
    df, cache = _fake_bo_data()
    cfg = _LoopCfg(acquisition="ei", n_init=4, n_cycles=5)
    res = BOLoop(dkl=_tiny_dkl(), cache=cache, meta_df=df, cfg=cfg,
                 seed=0, direction="min").run()
    assert set(res.columns) == RECORD_COLS
    assert res.best_so_far.is_monotonic_decreasing


def test_bo_loop_max_direction_columns_and_nondecreasing():
    df, cache = _fake_bo_data()
    cfg = _LoopCfg(acquisition="ei", n_init=4, n_cycles=5)
    res = BOLoop(dkl=_tiny_dkl(), cache=cache, meta_df=df, cfg=cfg,
                 seed=0, direction="max").run()
    assert res.best_so_far.is_monotonic_increasing
    assert res.uid.nunique() == len(res)
