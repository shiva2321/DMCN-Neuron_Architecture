"""Smoke tests for B-14 continual stack benchmark."""
from benchmarks.continual_stack import StackConfig, evaluate_verdict, run_benchmark
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.sequence_rpc import SequenceRPC


def test_sequence_rpc_end_task_smoke():
    items = ItemTable(256, seed=4)
    rows = [("red", "circle", "apple", "eat"), ("blue", "square", "block", "push")]
    for c, s, o, a in rows:
        _ = items[c]
        _ = items[s]
        _ = items[o]
        _ = items[a]

    stack = SequenceRPC(256, items, memory="tiered")
    for c, s, o, a in rows:
        stack.learn(c, s, o, a)
    stack.end_task()
    for c, s, o, a in rows:
        assert stack.read_object(c, s)[0] == o
        assert stack.read_action(c, s)[0] == a


def test_continual_stack_smoke():
    cfg = StackConfig(
        D=256,
        n_colors=2,
        shapes_per_task=4,
        n_tasks=2,
        min_task0_object=0.75,
        min_task0_action=0.75,
        max_episodic_object=0.50,
        max_episodic_action=0.50,
        min_object_margin=0.20,
        min_action_margin=0.20,
    )
    results = run_benchmark(cfg)
    by = {r.name: r for r in results}
    assert by["stack_tiered_consolidate"].task0_action > by["stack_tiered_episodic"].task0_action


def test_continual_stack_default_passes():
    cfg = StackConfig()
    assert evaluate_verdict(run_benchmark(cfg), cfg).passed
