"""PyTorch Dataset backed by the LMDB graph cache.

Wraps GraphCache so it can be fed into a torch_geometric DataLoader,
which handles batching of variable-size graphs automatically.
"""

from typing import List
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .cache import GraphCache


class GraphDataset(Dataset):
    """One material per item, backed by the LMDB graph cache."""

    def __init__(self, cache: GraphCache, uids: List[str]) -> None:
        self.cache = cache
        self.uids = uids

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Data:
        return self.cache[self.uids[idx]]
