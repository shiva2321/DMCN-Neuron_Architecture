"""Tests for T0/T1/T2 tiered memory and consolidation."""
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.tiered_memory import TieredMemory
from neuron_v2.core.tiered_rpn import TieredRPN
from neuron_v2.substrate.phasor import sim


def test_t0_reinstates_working_slot():
    D = 128
    items = ItemTable(D, seed=1)
    mem = TieredMemory(D)
    ctx = items["ctx_a"]
    val = items["tgt_a"]
    mem.set_working(ctx, val)
    pred = mem.predict(ctx, use_t1=False, use_t2=False)
    assert sim(pred, val) > 0.99


def test_consolidation_preserves_old_task_after_t1_reset():
    D = 256
    items = ItemTable(D, seed=2)
    _ = items["c1"]
    _ = items["t1"]
    _ = items["c2"]
    _ = items["t2"]

    rpn = TieredRPN(D, items, consolidate_on_task_end=True)
    rpn.learn("c1", "t1")
    rpn.end_task()

    for _ in range(40):
        rpn.learn("c2", "t2")

    assert rpn.read("c1")[0] == "t1"
    assert rpn.read("c2")[0] == "t2"


def test_episodic_reset_forgets_without_consolidation():
    D = 256
    items = ItemTable(D, seed=3)
    _ = items["c1"]
    _ = items["t1"]
    _ = items["c2"]
    _ = items["t2"]

    rpn = TieredRPN(D, items, consolidate_on_task_end=False)
    rpn.learn("c1", "t1")
    rpn.end_task()
    rpn.learn("c2", "t2")

    assert rpn.read("c2")[0] == "t2"
    assert rpn.read("c1")[0] != "t1"
