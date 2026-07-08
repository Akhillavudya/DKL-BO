"""Convert ASE Atoms to PyG Data graphs for 3D BULK crystals (SNUMAT).

This is a trimmed copy of `dklbo.data.graph_builder.atoms_to_graph` with ONE
deliberate change for 3D materials:

    The C2DB version is for 2D materials and removes any bond whose out-of-plane
    (z) displacement exceeds `vacuum_cutoff`, because a 2D cell has ~15-20 A of
    vacuum in z and a large radius would otherwise create false bonds across it.

    SNUMAT structures are 3D bulk crystals (periodic in x, y AND z, pbc=[T,T,T]).
    There is no vacuum gap, and atoms are genuinely bonded in every direction.
    Applying the z-filter here would DELETE real chemical bonds, so we omit it.

Everything else (neighbour list, Gaussian bond basis, per-atom neighbour cap,
output Data fields) is identical to the C2DB builder so the reused CGCNN encoder
consumes these graphs unchanged.

We widen the atom one-hot to `atom_feat_dim=100` (vs 90 in C2DB) because ICSD/SNUMAT
crystals can contain heavier elements than the 2D set.
"""

from collections import defaultdict

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

ATOM_FEAT_DIM: int = 100  # elements H(1) .. Fm(100) — headroom over C2DB's 90
BOND_FEAT_DIM: int = 10   # Gaussian basis functions for bond distance


def atoms_to_graph_3d(
    atoms: Atoms,
    radius: float = 8.0,
    max_neighbors: int = 12,
    atom_feat_dim: int = ATOM_FEAT_DIM,
) -> Data:
    """Convert a 3D bulk-crystal ASE Atoms object to a PyTorch Geometric Data graph.

    No vacuum/z filtering is applied (3D periodic crystal). Output fields match the
    C2DB builder: x [N, atom_feat_dim], edge_index [2, E], edge_attr [E, 10],
    n_atoms, n_edges.
    """
    # i, j atom-index pairs (both directions); D Cartesian displacement (PBC-aware);
    # d scalar distances in Angstrom.
    idx_i, idx_j, _, d = neighbor_list("ijDd", atoms, cutoff=radius, self_interaction=False)

    # Per-atom neighbour cap: keep the `max_neighbors` nearest j for each source i.
    idx_i, idx_j, d = _cap_neighbors(idx_i, idx_j, d, max_neighbors)

    x = _atom_features(atoms.get_atomic_numbers(), atom_feat_dim)
    edge_attr = _gaussian_basis(d, dmin=0.0, dmax=radius, n_basis=BOND_FEAT_DIM)
    edge_index = torch.tensor(np.stack([idx_i, idx_j], axis=0), dtype=torch.long)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        n_atoms=len(atoms),
        n_edges=edge_index.shape[1],
    )


def _atom_features(atomic_numbers: np.ndarray, atom_feat_dim: int) -> np.ndarray:
    """One-hot encoding: element Z at position Z-1. Z > atom_feat_dim -> all-zeros."""
    n = len(atomic_numbers)
    feat = np.zeros((n, atom_feat_dim), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        if 1 <= int(z) <= atom_feat_dim:
            feat[i, int(z) - 1] = 1.0
    return feat


def _gaussian_basis(distances: np.ndarray, dmin: float, dmax: float, n_basis: int) -> np.ndarray:
    """Expand scalar distances into a Gaussian radial basis (standard CGCNN featurisation)."""
    centers = np.linspace(dmin, dmax, n_basis)
    sigma = (dmax - dmin) / max(n_basis - 1, 1)
    return np.exp(
        -((distances[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2)
    ).astype(np.float32)


def _cap_neighbors(idx_i, idx_j, dists, max_neighbors):
    """For each source atom i, keep only the max_neighbors nearest j atoms."""
    per_node = defaultdict(list)
    for src, dst, dist in zip(idx_i, idx_j, dists):
        per_node[int(src)].append((float(dist), int(dst)))

    kept_i, kept_j, kept_d = [], [], []
    for src, neighbours in per_node.items():
        neighbours.sort(key=lambda x: x[0])
        for dist, dst in neighbours[:max_neighbors]:
            kept_i.append(src)
            kept_j.append(dst)
            kept_d.append(dist)

    if not kept_i:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))
    return (np.array(kept_i, dtype=np.int64), np.array(kept_j, dtype=np.int64),
            np.array(kept_d, dtype=np.float64))
