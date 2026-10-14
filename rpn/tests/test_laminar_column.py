"""Unit tests for the 6-layer Laminar Cortical Column (RCC)."""
import numpy as np
import pytest

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.laminar_column import LaminarCorticalColumn, WhatWhereReferenceFrame
from neuron_v2.substrate.phasor import normalise, random_hv, sim


def test_what_where_reference_frame_properties():
    D = 128
    frame = WhatWhereReferenceFrame(D=D, seed=42)
    where_hv = random_hv(D, seed=1)
    what_hv = random_hv(D, seed=2)

    composite = frame.compose(where_hv, what_hv)
    assert composite.shape == (D,)
    assert np.allclose(np.abs(composite), 1.0, atol=1e-5)

    # Recovery of structural where component
    rec_where = frame.extract_where(composite)
    assert sim(rec_where, where_hv) > 0.40

    # Recovery of sensory what component
    rec_what = frame.extract_what(composite)
    assert sim(rec_what, what_hv) > 0.40


def test_laminar_cortical_column_sharding():
    D = 128
    items = ItemTable(D=D, seed=123)
    column = LaminarCorticalColumn(D=D, items=items, n_shards=4, seed=123)

    s0 = column._get_shard("loc_0")
    s1 = column._get_shard("loc_1")
    s2 = column._get_shard("loc_2")
    s3 = column._get_shard("loc_3")

    # Distinct locations route to distinct shards (up to n_shards)
    shards = {s0, s1, s2, s3}
    assert len(shards) == 4


def test_laminar_cortical_column_capacity():
    D = 256
    items = ItemTable(D=D, seed=99)
    column = LaminarCorticalColumn(D=D, items=items, n_shards=4, seed=99)

    pairs = [
        (f"loc_{loc_i}", f"item_{item_j}", f"obj_{loc_i}_{item_j}")
        for loc_i in range(4)
        for item_j in range(8)
    ]
    for w, t, o in pairs:
        column.learn(w, t, o)

    correct = sum(1 for w, t, o in pairs if column.read(w, t)[0] == o)
    assert correct == len(pairs)  # 100% capacity


def test_laminar_cortical_column_lateral_consensus():
    D = 256
    items = ItemTable(D=D, seed=77)
    column = LaminarCorticalColumn(D=D, items=items, n_shards=2, seed=77)

    column.learn("loc_0", "item_A", "target_object")

    # Read with coherent lateral hypothesis
    target_hv = items["target_object"]
    token, prob = column.read("loc_0", "item_A", lateral_hypotheses=[target_hv])
    assert token == "target_object"
    assert prob > 0.80


def test_laminar_cortical_column_ood_continual_learning():
    D = 256
    items = ItemTable(D=D, seed=55)
    column = LaminarCorticalColumn(D=D, items=items, n_shards=4, seed=55)

    # Train task 1 on loc_0 and loc_1
    task1 = [(f"loc_{loc_i}", f"item_{item_j}", f"obj_{loc_i}_{item_j}") for loc_i in range(2) for item_j in range(4)]
    for w, t, o in task1:
        column.learn(w, t, o)

    # Train task 2 on loc_2 and loc_3
    task2 = [(f"loc_{loc_i}", f"item_{item_j}", f"obj_{loc_i}_{item_j}") for loc_i in (2, 3) for item_j in range(4)]
    for w, t, o in task2:
        column.learn(w, t, o)

    # Verify zero catastrophic forgetting on task 1
    t1_acc = sum(1 for w, t, o in task1 if column.read(w, t)[0] == o) / len(task1)
    t2_acc = sum(1 for w, t, o in task2 if column.read(w, t)[0] == o) / len(task2)

    assert t1_acc == 1.0
    assert t2_acc == 1.0
