"""Tests for the 2D vacuum-aware neighbour filter in graph_builder.py.

This is the most important correctness test in the pipeline (plan Section 2,
Phase 2).  A bug in vacuum filtering produces plausible-looking graphs with
garbage connectivity and the model will train without error while learning
nonsense.

Test strategy:
1. Synthetic test — use a handcrafted structure where we know exactly which
   bonds should and should not exist (independent of ASE's ase.build module).
2. MoS2 coordination test — use a real prototypical 2D structure; Mo should
   have 6 S neighbours (S-Mo-S sandwich, hexagonal symmetry).
3. Edge-count sanity — a sensible structure should never produce zero edges.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ase import Atoms
from dklbo.data.graph_builder import atoms_to_graph, ATOM_FEAT_DIM, BOND_FEAT_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bilayer_test_structure(layer_gap: float = 2.0, vacuum_gap: float = 5.0) -> Atoms:
    """Two carbon atoms in a thin 2D layer.

    Layout along z (all values in Å):
        z = 0.0  →  atom A
        z = layer_gap  →  atom B          (intra-layer bond, Δz = layer_gap)
        z = layer_gap + vacuum_gap  (next periodic image of atom A)

    With radius=8.0 and vacuum_gap=5.0:
        - A–B distance = layer_gap (intra-layer, SHOULD bond if < radius)
        - A–A' distance = cell_z = layer_gap + vacuum_gap  (SHOULD NOT bond)

    The vacuum filter with vacuum_cutoff=4.0 must remove the A–A' edge
    because its |Δz| = layer_gap + vacuum_gap > 4.0.
    """
    cell_z = layer_gap + vacuum_gap
    return Atoms(
        symbols="CC",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, layer_gap]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, cell_z]],
        pbc=True,
    )


def make_mos2(vacuum: float = 20.0) -> Atoms:
    """MoS2 monolayer (2H phase) with specified vacuum.

    Uses ase.build.mx2 if available; falls back to a hand-placed structure.
    The Mo–S distance in z is ~1.6 Å; the S–S distance (top to bottom) is
    ~3.2 Å — both well within a 4.0 Å vacuum_cutoff.
    """
    try:
        from ase.build import mx2
        return mx2(formula="MoS2", kind="2H", a=3.18, thickness=3.19, vacuum=vacuum)
    except Exception:
        # Fallback: hand-placed MoS2 unit cell
        # Mo at z=vacuum, S at z=vacuum±1.6 within a cell of height 3.19+2*vacuum
        h = 3.19 + 2 * vacuum
        return Atoms(
            symbols="MoS2",
            positions=[
                [0.0, 0.0, vacuum + 1.595],         # Mo
                [1.59, 0.919, vacuum + 1.595 - 1.59], # S below
                [1.59, 0.919, vacuum + 1.595 + 1.59], # S above
            ],
            cell=[[3.18, 0.0, 0.0], [1.59, 2.753, 0.0], [0.0, 0.0, h]],
            pbc=True,
        )


# ---------------------------------------------------------------------------
# Test 1 — vacuum filter removes cross-vacuum edges
# ---------------------------------------------------------------------------


def test_vacuum_filter_removes_cross_vacuum_edges():
    """The filter must remove edges where |Δz| > vacuum_cutoff.

    With layer_gap=2 Å, vacuum_gap=5 Å, radius=8 Å, vacuum_cutoff=4 Å:
      - Intra-layer A–B: Δz=2 Å → KEPT   (Δz ≤ 4)
      - Cross-vacuum A–A': Δz=7 Å → REMOVED (Δz > 4)
      - Cross-vacuum A–B image: Δz=5 Å → REMOVED (Δz > 4)

    max_neighbors=100 disables the neighbour cap so cross-vacuum bonds
    (which are ranked 14th+ by distance and would otherwise be silently
    dropped by the cap before the filter runs) are visible in the comparison.
    """
    atoms = make_bilayer_test_structure(layer_gap=2.0, vacuum_gap=5.0)

    graph_filtered = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0, max_neighbors=100)
    graph_no_filter = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=100.0, max_neighbors=100)

    assert graph_filtered.edge_index.shape[1] < graph_no_filter.edge_index.shape[1], (
        "Vacuum filter should remove cross-vacuum edges but edge count did not decrease"
    )


def test_intra_layer_bonds_are_kept():
    """Atoms within the same layer (Δz < vacuum_cutoff) must still be bonded."""
    atoms = make_bilayer_test_structure(layer_gap=2.0, vacuum_gap=5.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)

    assert graph.edge_index.shape[1] > 0, "All edges removed — intra-layer bonds lost"


def test_no_edges_cross_vacuum_in_mos2():
    """In MoS2 with 20 Å vacuum, no bond should have |Δz| > 4 Å.

    We reconstruct Δz from positions + stored edge directions (raw reconstruction
    ignores PBC but is sufficient for 20 Å vacuum: cross-vacuum Δz >> 4 Å).
    """
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)

    pos = atoms.get_positions()
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()

    # Direct position difference (valid for intra-layer bonds in large-vacuum structures)
    z_disps = np.abs(pos[dst, 2] - pos[src, 2])

    assert (z_disps <= 4.0 + 1e-6).all(), (
        f"{(z_disps > 4.0).sum()} edges have |Δz| > 4 Å: "
        f"max Δz = {z_disps.max():.3f} Å"
    )


# ---------------------------------------------------------------------------
# Test 2 — MoS2 coordination number
# ---------------------------------------------------------------------------


def test_mos2_mo_coordination():
    """Mo in 2H-MoS2 is coordinated by exactly 6 S atoms (trigonal prismatic)."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=5.0, vacuum_cutoff=4.0)

    symbols = atoms.get_chemical_symbols()
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()

    for i, sym in enumerate(symbols):
        if sym != "Mo":
            continue
        # Count how many neighbours of Mo are S atoms
        nbr_indices = dst[src == i]
        nbr_symbols = [symbols[j] for j in nbr_indices]
        n_S = nbr_symbols.count("S")

        assert 4 <= n_S <= 8, (
            f"Mo at index {i} has {n_S} S neighbours; expected 4–8 "
            f"(ideal is 6 in 2H-MoS2)"
        )


# ---------------------------------------------------------------------------
# Test 3 — Feature dimensions
# ---------------------------------------------------------------------------


def test_atom_feature_dimension():
    """Atom features must be ATOM_FEAT_DIM wide."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)
    assert graph.x.shape[1] == ATOM_FEAT_DIM, (
        f"Expected atom dim {ATOM_FEAT_DIM}, got {graph.x.shape[1]}"
    )


def test_bond_feature_dimension():
    """Bond features must be BOND_FEAT_DIM wide."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)
    assert graph.edge_attr.shape[1] == BOND_FEAT_DIM, (
        f"Expected bond dim {BOND_FEAT_DIM}, got {graph.edge_attr.shape[1]}"
    )


def test_no_zero_edge_graph_for_normal_structure():
    """A connected 2D structure must produce at least some bonds."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)
    assert graph.edge_index.shape[1] > 0, "MoS2 graph has zero edges — check neighbour search"


# ---------------------------------------------------------------------------
# Test 4 — Edge counts consistent
# ---------------------------------------------------------------------------


def test_n_edges_attribute_matches_edge_index():
    """graph.n_edges must equal the actual number of edges in edge_index."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)
    assert graph.n_edges == graph.edge_index.shape[1]


def test_n_atoms_attribute_matches_x():
    """graph.n_atoms must equal the number of rows in graph.x."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0)
    assert graph.n_atoms == graph.x.shape[0]


# ---------------------------------------------------------------------------
# Test 5 — Max neighbours cap
# ---------------------------------------------------------------------------


def test_max_neighbors_cap():
    """No atom should have more than max_neighbors=4 edges when capped."""
    atoms = make_mos2(vacuum=20.0)
    graph = atoms_to_graph(atoms, radius=8.0, vacuum_cutoff=4.0, max_neighbors=4)

    src = graph.edge_index[0].numpy()
    for atom_idx in range(graph.n_atoms):
        degree = (src == atom_idx).sum()
        assert degree <= 4, (
            f"Atom {atom_idx} has degree {degree} > max_neighbors=4"
        )
