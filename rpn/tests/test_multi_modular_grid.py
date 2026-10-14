"""Unit tests for Multi-Modular Entorhinal Grid Reference Frame & Column."""
import numpy as np
import pytest

from neuron_v2.core.multi_modular_grid import MultiModularGridReferenceFrame, MultiModularCorticalColumn
from neuron_v2.core.item_table import ItemTable
from neuron_v2.substrate.phasor import sim


def test_multi_modular_grid_properties():
    D = 128
    moduli = (3, 5, 7)
    grid = MultiModularGridReferenceFrame(D=D, moduli=moduli, seed=42)

    assert grid.total_physical_vectors == 3 + 5 + 7  # 15 vectors
    assert grid.total_coordinate_capacity == 3 * 5 * 7  # 105 coordinates

    # Test coordinate uniqueness
    c0 = grid.encode_location(0)
    c1 = grid.encode_location(1)
    c104 = grid.encode_location(104)

    assert sim(c0, c1) < 0.25
    assert sim(c0, c104) < 0.25


def test_multi_modular_column_retrieval():
    D = 128
    items = ItemTable(D=D, seed=42)
    col = MultiModularCorticalColumn(D, items, n_shards=4, moduli=(3, 5, 7), seed=42)

    # 12 locations x 4 items = 48 associations
    pairs = [(f"loc_{i}", f"item_{j}", f"obj_{i}_{j}") for i in range(12) for j in range(4)]
    for w, t, o in pairs:
        _ = items[w]; _ = items[t]; _ = items[o]
        col.learn(w, t, o)

    corr = sum(1 for w, t, o in pairs if col.read(w, t)[0] == o)
    assert corr == len(pairs)  # 100% capacity
