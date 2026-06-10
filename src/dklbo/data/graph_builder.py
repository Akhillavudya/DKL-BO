"""Convert ASE Atoms to PyG Data graphs with 2D-specific vacuum-aware filtering.

The vacuum_cutoff is the single most important correctness check in the whole
pipeline. C2DB structures are 2D materials with ~15–20 Å of vacuum added in the
c-direction (z-axis). Without this filter, a radius cutoff of 8 Å would never
accidentally span a 20 Å vacuum — but structures with smaller vacuum (or if ASE
returns distance vectors across PBC with unexpected wrapping) can silently create
garbage connectivity. The filter is cheap, explicit, and testable.

Architecture constants from [P1] Table 1:
    ATOM_FEAT_DIM = 90   (one-hot over elements 1–90)
    BOND_FEAT_DIM = 10   (Gaussian basis of interatomic distance)
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

ATOM_FEAT_DIM: int = 90  # elements H(1) through Th(90)
BOND_FEAT_DIM: int = 10  # Gaussian basis functions for bond distance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def atoms_to_graph(
    atoms: Atoms,
    radius: float = 8.0,
    vacuum_cutoff: float = 4.0,
    max_neighbors: int = 12,
) -> Data:
    """Convert an ASE Atoms object to a PyTorch Geometric Data graph.

    Parameters
    ----------
    atoms:
        The 2D material unit cell (should be periodic in x,y; vacuum in z).
    radius:
        3D neighbour search radius in Angstroms.
    vacuum_cutoff:
        Maximum allowed z-component (Å) of the displacement vector between two
        bonded atoms.  Edges where |Δz| > vacuum_cutoff are removed.
        This blocks false bonds that would span the out-of-plane vacuum gap.
    max_neighbors:
        Maximum number of bonds per atom.  The nearest neighbours are kept
        when the raw count exceeds this limit.

    Returns
    -------
    PyG Data with:
        x           : float32 [N, ATOM_FEAT_DIM]  — atom features
        edge_index  : long    [2, E]               — directed bond pairs
        edge_attr   : float32 [E, BOND_FEAT_DIM]  — bond distance features
        n_atoms     : int
        n_edges     : int
    """
    # ASE neighbor_list:
    #   i, j  — atom index pairs (both directions included)
    #   D     — displacement vectors in Cartesian Å (accounts for PBC)
    #   d     — scalar distances in Å
    idx_i, idx_j, D, d = neighbor_list(
        "ijDd", atoms, cutoff=radius, self_interaction=False
    )

    # ---- 2D vacuum filter -----------------------------------------------
    # D[:, 2] is the z-component of the actual Cartesian displacement.
    # For a 2D material the layer is thin (~3–6 Å) while the vacuum is large
    # (~15–20 Å).  Any |Δz| > vacuum_cutoff must cross the vacuum gap and
    # represents an artefact, not a real chemical bond.
    z_disp = np.abs(D[:, 2])
    valid = z_disp <= vacuum_cutoff
    idx_i, idx_j, d = idx_i[valid], idx_j[valid], d[valid]

    # ---- Per-atom neighbour cap ------------------------------------------
    # Keep only the `max_neighbors` nearest neighbours for each atom.
    idx_i, idx_j, d = _cap_neighbors(idx_i, idx_j, d, max_neighbors)

    # ---- Features -------------------------------------------------------
    x = _atom_features(atoms.get_atomic_numbers())      # [N, ATOM_FEAT_DIM]
    edge_attr = _gaussian_basis(d, dmin=0.0, dmax=radius, n_basis=BOND_FEAT_DIM)  # [E, BOND_FEAT_DIM]

    edge_index = torch.tensor(np.stack([idx_i, idx_j], axis=0), dtype=torch.long)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        n_atoms=len(atoms),
        n_edges=edge_index.shape[1],
    )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _atom_features(atomic_numbers: np.ndarray) -> np.ndarray:
    """90-dim one-hot encoding: element Z is at position Z-1 (index 0 = H).

    Elements above Z=90 fall back to an all-zeros vector — they are rare in
    C2DB and this avoids an out-of-range panic.
    """
    n = len(atomic_numbers)
    feat = np.zeros((n, ATOM_FEAT_DIM), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        if 1 <= int(z) <= ATOM_FEAT_DIM:
            feat[i, int(z) - 1] = 1.0
    return feat


def _gaussian_basis(
    distances: np.ndarray,
    dmin: float,
    dmax: float,
    n_basis: int,
) -> np.ndarray:
    """Expand scalar distances into a Gaussian radial basis.

    Centres are evenly spaced from dmin to dmax.  Width σ = spacing between
    centres.  This is the standard CGCNN distance featurisation.
    """
    centers = np.linspace(dmin, dmax, n_basis)
    sigma = (dmax - dmin) / max(n_basis - 1, 1)
    # distances: [E], centers: [B] → output: [E, B]
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Neighbour cap
# ---------------------------------------------------------------------------


def _cap_neighbors(
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    dists: np.ndarray,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each source atom i, keep only the max_neighbors nearest j atoms."""
    per_node: dict[int, list] = defaultdict(list)
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
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    return (
        np.array(kept_i, dtype=np.int64),
        np.array(kept_j, dtype=np.int64),
        np.array(kept_d, dtype=np.float64),
    )
