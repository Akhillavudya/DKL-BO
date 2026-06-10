"""Phase 3 tests — acquisition functions, RandomBaseline, and BOLoop.

Fast tests (no GPU, no real data):
  - All acquisition function tests use synthetic tensors
  - RandomBaseline tests use a tiny fake metadata DataFrame
  - BOLoop integration test uses a fake cache + tiny DKL model (CPU)
"""

import random
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dklbo.bo.acquisition import (
    get_acquisition,
    greedy,
    random_acquisition,
    ucb,
)
from dklbo.bo.baselines import RandomBaseline
from dklbo.bo.loop import BOLoop
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate

# ──────────────────────────────────────────────────────────────────────────────
# Shared constants + helpers
# ──────────────────────────────────────────────────────────────────────────────

ATOM_DIM  = 90   # must match real graph feature dim
BOND_DIM  = 10
HIDDEN_DIM = 8   # small for fast tests
N_MATS    = 25


def _fake_graph(uid: str, gap: float) -> Data:
    """Minimal PyG graph with correct feature dimensions."""
    rng = random.Random(abs(hash(uid)) % (2 ** 31))
    torch.manual_seed(abs(hash(uid)) % (2 ** 31))
    n = rng.randint(2, 5)
    e = rng.randint(n, 2 * n)
    return Data(
        x          = torch.randn(n, ATOM_DIM),
        edge_index = torch.randint(0, n, (2, e)),
        edge_attr  = torch.randn(e, BOND_DIM),
        y          = torch.tensor([[gap]], dtype=torch.float32),
        uid        = uid,
    )


class _FakeCache:
    """Dict-backed stand-in for GraphCache."""
    def __init__(self, graphs: dict):
        self._g = graphs
    def __getitem__(self, uid: str) -> Data:
        return self._g[uid]


def _make_fake_dataset(n: int = N_MATS):
    rng   = random.Random(99)
    uids  = [f"mat_{i:03d}" for i in range(n)]
    gaps  = [round(rng.uniform(0.5, 9.0), 3) for _ in uids]
    df    = pd.DataFrame({"uid": uids, "target": gaps})
    cache = _FakeCache({uid: _fake_graph(uid, gap) for uid, gap in zip(uids, gaps)})
    return df, cache


class _FakeBOCfg:
    """Minimal BO config object for tests."""
    acquisition       = "ucb"
    beta              = 0.2
    n_init            = 4
    n_cycles          = 6
    retrain_every_k   = 3
    n_joint_epochs    = 3    # very few epochs for speed
    n_pretrain_epochs = 2
    gp_refit_epochs   = 3

    def get(self, key, default=None):
        return getattr(self, key, default)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Acquisition function tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synth():
    torch.manual_seed(0)
    mean = torch.randn(50)
    std  = torch.abs(torch.randn(50)) + 0.01
    return mean, std


def test_ucb_output_shape(synth):
    mean, std = synth
    assert ucb(mean, std, beta=0.2).shape == mean.shape


def test_ucb_beta_zero_equals_greedy(synth):
    mean, std = synth
    assert torch.allclose(ucb(mean, std, beta=0.0), greedy(mean, std))


def test_ucb_higher_beta_favours_uncertainty():
    """At β=0 the low-std material wins; at β=3 the high-std one wins."""
    m = torch.tensor([1.0, 0.5])
    s = torch.tensor([0.1, 1.0])
    assert int(ucb(m, s, beta=0.0).argmax()) == 0   # greedy: pick #0 (higher mean)
    assert int(ucb(m, s, beta=3.0).argmax()) == 1   # explore: pick #1 (higher std)


def test_greedy_ignores_std(synth):
    mean, std = synth
    assert torch.allclose(greedy(mean, std), greedy(mean, torch.zeros_like(std)))


def test_random_acquisition_is_non_deterministic():
    mean = torch.zeros(100)
    std  = torch.zeros(100)
    assert not torch.allclose(random_acquisition(mean, std), random_acquisition(mean, std))


def test_get_acquisition_returns_callable():
    for name in ("ucb", "greedy", "random"):
        assert callable(get_acquisition(name))


def test_get_acquisition_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown acquisition"):
        get_acquisition("banana_search")


# ──────────────────────────────────────────────────────────────────────────────
# 2. RandomBaseline tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_data():
    return _make_fake_dataset(N_MATS)


def test_random_baseline_returns_dataframe(fake_data):
    df, _ = fake_data
    result = RandomBaseline(meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == _FakeBOCfg.n_cycles


def test_random_baseline_required_columns(fake_data):
    df, _ = fake_data
    result = RandomBaseline(meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    required = [
        "cycle", "uid", "gap_acquired", "best_so_far",
        "n_labelled", "cumul_top50", "cumul_top10pct", "acquisition_fn",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_random_baseline_best_so_far_nondecreasing(fake_data):
    df, _ = fake_data
    bsf = RandomBaseline(meta_df=df, cfg=_FakeBOCfg(), seed=0).run()["best_so_far"].tolist()
    assert all(bsf[i] <= bsf[i + 1] for i in range(len(bsf) - 1))


def test_random_baseline_no_duplicate_selections(fake_data):
    df, _ = fake_data
    result = RandomBaseline(meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    assert result["uid"].nunique() == len(result), "Same UID selected twice"


def test_random_baseline_cumul_top50_nondecreasing(fake_data):
    df, _ = fake_data
    c50 = RandomBaseline(meta_df=df, cfg=_FakeBOCfg(), seed=0).run()["cumul_top50"].tolist()
    assert all(c50[i] <= c50[i + 1] for i in range(len(c50) - 1))


def test_random_baseline_n_labelled_grows(fake_data):
    df, _ = fake_data
    cfg    = _FakeBOCfg()
    result = RandomBaseline(meta_df=df, cfg=cfg, seed=0).run()
    expected_start = cfg.n_init + 1
    assert result["n_labelled"].iloc[0]  == expected_start
    assert result["n_labelled"].iloc[-1] == expected_start + cfg.n_cycles - 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. BOLoop integration test  (CPU, tiny model, tiny data)
# ──────────────────────────────────────────────────────────────────────────────

def _build_tiny_dkl() -> DKLModel:
    encoder   = CGCNNEncoder(
        atom_dim  = ATOM_DIM,
        bond_dim  = BOND_DIM,
        hidden_dim= HIDDEN_DIM,
        n_conv    = 1,
        n_fc      = 1,
        pooling   = "attention",
    )
    surrogate = ExactGPSurrogate(lr=0.01, n_epochs=3, patience=5)
    return DKLModel(encoder=encoder, surrogate=surrogate,
                    encoder_lr=0.001, gp_lr=0.01, device="cpu")


def test_bo_loop_returns_dataframe(fake_data):
    df, cache = fake_data
    dkl    = _build_tiny_dkl()
    cfg    = _FakeBOCfg()
    result = BOLoop(dkl=dkl, cache=cache, meta_df=df, cfg=cfg, seed=0).run()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == cfg.n_cycles


def test_bo_loop_required_columns(fake_data):
    df, cache = fake_data
    result = BOLoop(dkl=_build_tiny_dkl(), cache=cache,
                    meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    required = [
        "cycle", "uid", "gap_acquired", "best_so_far",
        "n_labelled", "cumul_top50", "cumul_top10pct",
        "acquisition_fn", "train_time_s", "predict_time_s",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_bo_loop_best_so_far_nondecreasing(fake_data):
    df, cache = fake_data
    bsf = BOLoop(dkl=_build_tiny_dkl(), cache=cache,
                 meta_df=df, cfg=_FakeBOCfg(), seed=0).run()["best_so_far"].tolist()
    assert all(bsf[i] <= bsf[i + 1] for i in range(len(bsf) - 1))


def test_bo_loop_no_duplicate_selections(fake_data):
    df, cache = fake_data
    result = BOLoop(dkl=_build_tiny_dkl(), cache=cache,
                    meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    assert result["uid"].nunique() == len(result), "Same UID selected twice"


def test_bo_loop_cycle_numbers_sequential(fake_data):
    df, cache = fake_data
    result = BOLoop(dkl=_build_tiny_dkl(), cache=cache,
                    meta_df=df, cfg=_FakeBOCfg(), seed=0).run()
    assert list(result["cycle"]) == list(range(1, _FakeBOCfg.n_cycles + 1))
