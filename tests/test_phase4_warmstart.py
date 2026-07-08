"""Phase 4 tests — fine-tuning warm-start mechanism + sweep subset logic.

Fast, CPU-only, no real data:
  * encoder state_dict round-trip (the warm-start mechanism scripts 15/17 rely on)
  * BOLoop runs end-to-end when handed a pre-initialised (warm) encoder
  * _stratified_subset returns exactly N rows spanning the gap range
"""

import importlib.util
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from dklbo.bo.loop import BOLoop
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate

ATOM_DIM, BOND_DIM, HIDDEN_DIM = 90, 10, 8


def _fake_graph(uid: str, gap: float) -> Data:
    rng = random.Random(abs(hash(uid)) % (2 ** 31))
    torch.manual_seed(abs(hash(uid)) % (2 ** 31))
    n = rng.randint(2, 5)
    e = rng.randint(n, 2 * n)
    return Data(
        x=torch.randn(n, ATOM_DIM),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, BOND_DIM),
        y=torch.tensor([[gap]], dtype=torch.float32),
        uid=uid,
    )


class _FakeCache:
    def __init__(self, graphs):
        self._g = graphs

    def __getitem__(self, uid):
        return self._g[uid]


class _FakeBOCfg:
    acquisition = "ei"
    beta = 0.0
    xi = 0.01
    n_init = 4
    n_cycles = 4
    retrain_every_k = 2
    n_joint_epochs = 2
    n_pretrain_epochs = 1
    gp_refit_epochs = 2

    def get(self, key, default=None):
        return getattr(self, key, default)


def _encoder():
    return CGCNNEncoder(atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
                        hidden_dim=HIDDEN_DIM, n_conv=2, n_fc=1, pooling="attention")


def test_encoder_state_dict_roundtrip():
    """Loading a saved state_dict reproduces the encoder weights exactly.

    This is the warm-start mechanism: scripts 15/17 build a fresh encoder then
    load pretrained_encoder.pt into it before constructing the DKLModel.
    """
    enc = _encoder()
    state = {k: v.clone() for k, v in enc.state_dict().items()}

    fresh = _encoder()                      # different random init
    diff_before = any(
        not torch.allclose(fresh.state_dict()[k], v) for k, v in state.items()
    )
    assert diff_before, "fresh encoder should differ before loading"

    fresh.load_state_dict(state)
    for k, v in state.items():
        assert torch.allclose(fresh.state_dict()[k], v)


def test_boloop_runs_with_warmstarted_encoder():
    """BOLoop fine-tunes a warm-started DKL and returns a valid results frame."""
    rng = random.Random(7)
    uids = [f"mat_{i:03d}" for i in range(20)]
    gaps = [round(rng.uniform(0.5, 9.0), 3) for _ in uids]
    df = pd.DataFrame({"uid": uids, "target": gaps})
    cache = _FakeCache({u: _fake_graph(u, g) for u, g in zip(uids, gaps)})

    # "Pre-train": save then reload encoder weights into the model BOLoop will use.
    warm_enc = _encoder()
    ckpt = {k: v.clone() for k, v in warm_enc.state_dict().items()}
    enc = _encoder()
    enc.load_state_dict(ckpt)

    dkl = DKLModel(encoder=enc, surrogate=ExactGPSurrogate(n_epochs=3),
                   device="cpu")
    res = BOLoop(dkl=dkl, cache=cache, meta_df=df, cfg=_FakeBOCfg(),
                 seed=0, direction="max").run()

    assert len(res) == _FakeBOCfg.n_cycles
    assert {"best_so_far", "cumul_top10pct"}.issubset(res.columns)
    # best_so_far is monotonic non-decreasing for a max task
    assert (res.best_so_far.diff().fillna(0) >= -1e-9).all()


def _load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stratified_subset_size_and_spread():
    """_stratified_subset returns exactly N rows spanning the gap range."""
    sweep = _load_script("16_pretrain_sweep.py")
    rng = random.Random(1)
    train = pd.DataFrame({
        "uid": [f"m{i}" for i in range(400)],
        "target": [round(rng.uniform(0.0, 10.0), 3) for _ in range(400)],
    })
    sub = sweep._stratified_subset(train, n=120, seed=42)
    assert len(sub) == 120
    # stratified sample should still cover low and high gaps
    assert sub.target.min() < 2.0 and sub.target.max() > 8.0
    # full set returned unchanged when N >= len
    assert len(sweep._stratified_subset(train, n=999, seed=42)) == len(train)
