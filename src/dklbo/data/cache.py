"""LMDB-backed graph cache with config-hash invalidation.

Why LMDB instead of 17k loose .pt files:
    - Single file on disk → no filesystem inode pressure
    - Memory-mapped reads → OS page cache shared across processes
    - Transactional writes → safe to interrupt mid-build and resume
    - ~10–100× faster random access than scatter loading from a directory

Why config hash:
    Changing radius, vacuum_cutoff, or max_neighbors produces different graphs.
    The hash encodes the full preprocessing config so cache mismatches are caught
    automatically rather than silently training on stale graphs.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import lmdb
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Reserve 10 GB in the LMDB map — does not allocate disk space upfront on Linux
_LMDB_MAP_SIZE = 10 * 1024 ** 3


def config_hash(config: Dict[str, Any]) -> str:
    """Return an 8-char hex hash of the preprocessing config dict."""
    serialised = json.dumps(config, sort_keys=True).encode()
    return hashlib.md5(serialised).hexdigest()[:8]


class GraphCache:
    """Read/write interface to an LMDB graph store.

    Usage — building:
        cache = GraphCache("data/cache", config_dict)
        if not cache.exists():
            for uid, graph in ...:
                cache.put(uid, graph)
        cache.close()

    Usage — reading:
        cache = GraphCache("data/cache", config_dict, readonly=True)
        graph = cache.get("uid_123")

    The LMDB file is named `graphs_<hash>.lmdb` so different preprocessing
    configs co-exist in the same directory without clobbering each other.
    """

    def __init__(
        self,
        cache_dir: str,
        config: Dict[str, Any],
        readonly: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.hash = config_hash(config)
        self.db_path = self.cache_dir / f"graphs_{self.hash}.lmdb"
        self.readonly = readonly
        self._env: Optional[lmdb.Environment] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self._env is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self.db_path),
            map_size=_LMDB_MAP_SIZE,
            readonly=self.readonly,
            lock=not self.readonly,
            create=not self.readonly,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "GraphCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Existence check (no-open)
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """True if this cache has already been built (db file present)."""
        return self.db_path.exists()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def put(self, uid: str, graph: Data) -> None:
        """Serialise and store one graph under its uid."""
        self._open()
        with self._env.begin(write=True) as txn:
            txn.put(uid.encode(), pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))

    def put_batch(self, items: Iterator[tuple[str, Data]]) -> None:
        """Store many graphs in a single LMDB transaction (faster than many put calls)."""
        self._open()
        with self._env.begin(write=True) as txn:
            for uid, graph in items:
                txn.put(uid.encode(), pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))

    def get(self, uid: str) -> Optional[Data]:
        """Return the graph for uid, or None if not found."""
        self._open()
        with self._env.begin() as txn:
            raw = txn.get(uid.encode())
        return pickle.loads(raw) if raw is not None else None

    def __getitem__(self, uid: str) -> Data:
        graph = self.get(uid)
        if graph is None:
            raise KeyError(uid)
        return graph

    def __contains__(self, uid: str) -> bool:
        self._open()
        with self._env.begin() as txn:
            return txn.get(uid.encode()) is not None

    def __len__(self) -> int:
        self._open()
        with self._env.begin() as txn:
            return txn.stat()["entries"]

    def keys(self) -> list[str]:
        """Return all stored uids."""
        self._open()
        with self._env.begin() as txn:
            return [k.decode() for k, _ in txn.cursor()]

    def __iter__(self) -> Iterator[tuple[str, Data]]:
        """Iterate over (uid, graph) pairs — loads one at a time."""
        self._open()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                yield key.decode(), pickle.loads(value)
