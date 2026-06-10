"""Build the LMDB graph cache from the C2DB ASE database.

This script is the Phase 1 entry point.  Run it once; all downstream scripts
(02_eval_surrogate.py, 03_run_bo.py) read from the cache.

Usage:
    python scripts/01_build_cache.py data.db_path=/path/to/c2db.db

Hydra config overrides work as usual:
    python scripts/01_build_cache.py data.db_path=... data.target=gap_gw seed=0
"""

import logging
import sys
import time
from pathlib import Path

# Make the src package importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
from omegaconf import DictConfig

from dklbo.data.c2db_loader import load_c2db, verify_no_split_leakage
from dklbo.data.cache import GraphCache, config_hash
from dklbo.data.graph_builder import atoms_to_graph
from dklbo.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    # The cache hash encodes the full preprocessing config.
    # If any of these change, a new cache is built automatically.
    preproc_cfg = {
        "radius": cfg.data.radius,
        "vacuum_cutoff": cfg.data.vacuum_cutoff,
        "max_neighbors": cfg.data.max_neighbors,
        "target": cfg.data.target,
        "gap_min": cfg.data.gap_min,
    }

    cache = GraphCache(cfg.data.cache_dir, preproc_cfg)

    if cache.exists():
        logger.info(
            f"Cache already exists: {cache.db_path}  (hash={cache.hash})\n"
            f"  {len(cache)} graphs stored — skipping rebuild.\n"
            f"  Delete the file to force a rebuild."
        )
        return

    # ------------------------------------------------------------------ #
    # Step 1 — Load metadata from C2DB
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Step 1: Loading C2DB metadata")
    logger.info("=" * 60)

    df = load_c2db(
        db_path=cfg.data.db_path,
        target=cfg.data.target,
        gap_min=cfg.data.gap_min,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        random_seed=cfg.seed,
    )

    verify_no_split_leakage(df)

    # Save metadata table — used by all downstream scripts
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.parquet"
    df.to_parquet(meta_path, index=False)
    logger.info(f"Metadata saved → {meta_path}  ({len(df)} rows)")

    # ------------------------------------------------------------------ #
    # Step 2 — Build graphs and write to LMDB
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Step 2: Building crystal graphs → LMDB cache")
    logger.info(f"  Config hash: {cache.hash}")
    logger.info(f"  radius={cfg.data.radius}  vacuum_cutoff={cfg.data.vacuum_cutoff}  max_neighbors={cfg.data.max_neighbors}")
    logger.info("=" * 60)

    from ase.db import connect
    db = connect(cfg.data.db_path)

    t_start = time.perf_counter()
    n_ok = 0
    n_fail = 0
    n_zero_edge = 0
    total_edges = 0

    for _, row in df.iterrows():
        try:
            atoms = db.get(id=int(row["id"])).toatoms()
            graph = atoms_to_graph(
                atoms,
                radius=cfg.data.radius,
                vacuum_cutoff=cfg.data.vacuum_cutoff,
                max_neighbors=cfg.data.max_neighbors,
            )

            # Attach the band-gap label and identifiers to the graph object
            graph.y = torch.tensor([float(row["target"])], dtype=torch.float32)
            graph.uid = row["uid"]
            graph.split = row["split"]

            cache.put(row["uid"], graph)

            if graph.n_edges == 0:
                n_zero_edge += 1
                logger.warning(
                    f"Zero-edge graph for uid={row['uid']} formula={row['formula']}"
                )

            total_edges += graph.n_edges
            n_ok += 1

            if n_ok % 500 == 0:
                elapsed = time.perf_counter() - t_start
                throughput = n_ok / elapsed
                logger.info(
                    f"  {n_ok}/{len(df)} graphs  ({throughput:.1f} graphs/sec)"
                    f"  avg_edges={total_edges/n_ok:.1f}"
                )

        except Exception as exc:
            logger.warning(f"Failed: id={row['id']}  formula={row.get('formula', '?')}  {exc}")
            n_fail += 1

    elapsed = time.perf_counter() - t_start
    throughput = n_ok / max(elapsed, 1e-6)

    logger.info("=" * 60)
    logger.info("Build complete")
    logger.info(f"  Graphs built:     {n_ok}/{len(df)}")
    logger.info(f"  Failed:           {n_fail}")
    logger.info(f"  Zero-edge graphs: {n_zero_edge}  (check vacuum_cutoff if > 0)")
    logger.info(f"  Avg edges/graph:  {total_edges/max(n_ok,1):.1f}")
    logger.info(f"  Throughput:       {throughput:.1f} graphs/sec")
    logger.info(f"  Total time:       {elapsed:.1f}s")
    logger.info(f"  Cache:            {cache.db_path}")
    logger.info("=" * 60)

    cache.close()


if __name__ == "__main__":
    main()
