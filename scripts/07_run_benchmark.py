"""Run the GP-BO vs DKL-BO benchmark: 4 tasks × {dkl, std_gp, random} × N seeds.

For each (task, method, seed) cell this runs one BO simulation and writes a
per-cell CSV to results/benchmark/runs/{task}_{method}_seed{S}.csv. The run is
idempotent: existing CSVs are skipped, so a long sweep is resumable.

Fairness (see docs/benchmark_design.md):
  * all methods share the identical GP backend (ExactGPSurrogate, Matérn-5/2)
  * for a given (task, seed) the random init set is identical across methods
  * all methods use the same acquisition (default EI) and the same sign trick

Usage
-----
    # full sweep (band-gap + effective-mass tasks)
    python scripts/07_run_benchmark.py bo=ei data.db_path=data/raw/c2db.db

    # smoke test: one seed, few cycles, single task
    python scripts/07_run_benchmark.py bo=ei data.db_path=data/raw/c2db.db \
        bo.n_cycles=8 +benchmark.seeds='[0]' +benchmark.tasks_filter='[gap_max]'

Override the matrix by editing configs/benchmark/bench.yaml or passing
`+benchmark.<key>=...` overrides (seeds, methods, tasks_filter).
"""

import logging
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from dklbo.baselines.feature_bo_loop import FeatureBOLoop
from dklbo.bo.baselines import RandomBaseline
from dklbo.bo.loop import BOLoop
from dklbo.data.cache import GraphCache
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import build_surrogate
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _build_dkl(cfg: DictConfig, device: str) -> DKLModel:
    """Fresh DKL model (random-init encoder) — no Phase-2 warm start, as in 03_run_bo."""
    encoder = CGCNNEncoder(
        atom_dim=cfg.model.atom_dim,
        bond_dim=cfg.model.bond_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_conv=cfg.model.n_conv,
        n_fc=cfg.model.n_fc,
        pooling=cfg.model.pooling,
        spectral_norm=cfg.model.get("spectral_norm", False),
        sn_coeff=float(cfg.model.get("sn_coeff", 1.0)),
    )
    surrogate = build_surrogate(cfg.surrogate)
    return DKLModel(
        encoder=encoder,
        surrogate=surrogate,
        encoder_lr=float(cfg.model.encoder_lr),
        gp_lr=float(cfg.surrogate.lr),
        device=device,
        standardize=cfg.model.get("standardize_embeddings", False),
    )


def _load_descriptors(path: Path, meta_df: pd.DataFrame):
    """Load descriptor parquet and align rows to meta_df uid order."""
    d = pd.read_parquet(path).set_index("uid")
    missing = [u for u in meta_df["uid"] if u not in d.index]
    if missing:
        raise AssertionError(
            f"{len(missing)} uids missing from {path.name} (e.g. {missing[:3]}). "
            f"Rebuild with scripts/05_build_descriptors.py."
        )
    d = d.loc[meta_df["uid"]]
    uids = d.index.tolist()
    X = d.to_numpy(dtype="float64")
    return uids, X


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  acquisition={cfg.bo.acquisition}  n_cycles={cfg.bo.n_cycles}")

    bench = OmegaConf.load(REPO / "configs" / "benchmark" / "bench.yaml")
    # Allow CLI overrides of the matrix via benchmark.* (merged if provided)
    if "benchmark" in cfg:
        bench = OmegaConf.merge(bench, cfg.benchmark)

    seeds = list(bench.seeds)
    methods = list(bench.methods)
    task_filter = set(bench.get("tasks_filter", []))  # optional subset by name
    tasks = [t for t in bench.tasks if not task_filter or t.name in task_filter]

    runs_dir = REPO / cfg.results_dir / "benchmark" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO / cfg.data.cache_dir

    # Graphs are structure-only → always read from the band-gap cache.
    graph_preproc = {
        "radius": cfg.data.radius,
        "vacuum_cutoff": cfg.data.vacuum_cutoff,
        "max_neighbors": cfg.data.max_neighbors,
        "target": cfg.data.get("graph_cache_target", "gap_hse"),
        "gap_min": cfg.data.gap_min,
    }
    cache = GraphCache(str(cache_dir), graph_preproc, readonly=True)
    if "dkl" in methods and not cache.exists():
        raise FileNotFoundError(f"Graph cache {cache.db_path} not found (run 01_build_cache.py).")

    n_done = n_skip = 0
    for task in tasks:
        meta_path = cache_dir / task.meta_file
        meta_df = pd.read_parquet(meta_path)
        desc_path = REPO / cfg.results_dir / "benchmark" / task.descriptors
        logger.info(
            f"=== TASK {task.name}  ({task.direction})  |  {len(meta_df)} materials ==="
        )

        for method in methods:
            for seed in seeds:
                out_path = runs_dir / f"{task.name}_{method}_seed{seed}.csv"
                if out_path.exists():
                    n_skip += 1
                    continue

                seed_everything(seed)
                if method == "dkl":
                    dkl = _build_dkl(cfg, device)
                    loop = BOLoop(dkl=dkl, cache=cache, meta_df=meta_df, cfg=cfg.bo,
                                  seed=seed, direction=task.direction)
                    res = loop.run()
                elif method == "std_gp":
                    uids, X = _load_descriptors(desc_path, meta_df)
                    loop = FeatureBOLoop(
                        X=X, uids=uids, meta_df=meta_df, cfg=cfg.bo, seed=seed,
                        direction=task.direction,
                        ard=bool(bench.std_gp.get("ard", True)),
                        gp_epochs=int(bench.std_gp.get("gp_epochs", 100)),
                    )
                    res = loop.run()
                elif method == "random":
                    res = RandomBaseline(meta_df=meta_df, cfg=cfg.bo, seed=seed,
                                         direction=task.direction).run()
                else:
                    raise ValueError(f"Unknown method {method!r}")

                res.insert(0, "task", task.name)
                res.insert(1, "method", method)
                res.insert(2, "seed", seed)
                res.insert(3, "direction", task.direction)
                res.to_csv(out_path, index=False)
                n_done += 1
                logger.info(
                    f"  {task.name}/{method}/seed{seed}  → best={res.best_so_far.iloc[-1]:.3f}  "
                    f"top50={res.cumul_top50.iloc[-1]}  ({out_path.name})"
                )

    cache.close()
    logger.info(f"Benchmark sweep complete: {n_done} runs written, {n_skip} skipped (already done).")


if __name__ == "__main__":
    main()
