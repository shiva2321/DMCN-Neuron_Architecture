"""Unit tests for Stress Benchmark 3: Cortical shard overload."""
import pytest

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.laminar_column import LaminarCorticalColumn


def test_capacity_stress_sub_rank_and_saturation():
    D = 128  # Use smaller D for rapid test execution
    items = ItemTable(D=D, seed=42)

    # 1. Sub-rank condition: 4 locations x 8 items = 32 items across 2 shards (16 items/shard < D=128)
    pairs_sub = [(f"loc_{i}", f"item_{j}", f"obj_{i}_{j}") for i in range(4) for j in range(8)]
    for w, t, o in pairs_sub:
        _ = items[w]; _ = items[t]; _ = items[o]

    rcc_sub = LaminarCorticalColumn(D, items, n_shards=2, seed=42)
    for w, t, o in pairs_sub:
        rcc_sub.learn(w, t, o)

    corr_sub = sum(1 for w, t, o in pairs_sub if rcc_sub.read(w, t)[0] == o)
    assert corr_sub / len(pairs_sub) == 1.0  # Perfect sub-rank recall

    # 2. Over-rank condition: 32 locations x 8 items = 256 items in 1 shard (256 items/shard > D=128)
    pairs_over = [(f"loc_{i}", f"item_{j}", f"obj_over_{i}_{j}") for i in range(32) for j in range(8)]
    for w, t, o in pairs_over:
        _ = items[w]; _ = items[t]; _ = items[o]

    rcc_over = LaminarCorticalColumn(D, items, n_shards=1, seed=42)
    for w, t, o in pairs_over:
        rcc_over.learn(w, t, o)

    corr_over = sum(1 for w, t, o in pairs_over if rcc_over.read(w, t)[0] == o)
    acc_over = corr_over / len(pairs_over)
    assert acc_over < 0.80  # Demonstrates real physical saturation threshold
