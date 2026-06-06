"""Smoke tests for CompositionalColumn (3-RPN role-bound compositional lookup)."""
from neuron_v2.core.column import CompositionalColumn
from neuron_v2.core.item_table import ItemTable

D = 256

PAIRS = [
    ("red", "circle", "apple"),
    ("red", "square", "box"),
    ("blue", "circle", "ball"),
    ("blue", "square", "block"),
]


def _make_column() -> CompositionalColumn:
    items = ItemTable(D, seed=7)
    for color, shape, obj in PAIRS:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]
    return CompositionalColumn(D, items, mode="delta")


def test_compositional_column_smoke_d256():
    col = _make_column()
    for color, shape, obj in PAIRS:
        col.learn(color, shape, obj)

    for color, shape, expected in PAIRS:
        token, prob = col.read(color, shape)
        assert token == expected, f"({color}, {shape}) -> {token!r}, expected {expected!r}"
        assert prob > 0.0
