"""Smoke tests for B-8 continual compositional benchmark."""
from benchmarks.continual_composition import (
    CompositionConfig,
    evaluate_verdict,
    run_benchmark,
)
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn


def test_tiered_rpc_end_task_smoke():
    items = ItemTable(256, seed=3)
    pairs = [(f"color_{i}", f"shape_{j}", f"obj_{i}_{j}") for i in range(2) for j in range(4)]
    for c, s, o in pairs:
        _ = items[c]
        _ = items[s]
        _ = items[o]

    rpc = ResonantPredictiveColumn(256, items, memory="tiered")
    for c, s, o in pairs:
        rpc.learn(c, s, o)
    rpc.end_task()
    for c, s, o in pairs:
        assert rpc.read(c, s)[0] == o


def test_continual_composition_smoke():
    cfg = CompositionConfig(
        D=256,
        n_colors=2,
        shapes_per_task=4,
        n_tasks=2,
        min_task0_retention=0.75,
        max_episodic_task0=0.50,
        min_retention_margin=0.20,
    )
    results = run_benchmark(cfg)
    by = {r.name: r for r in results}
    assert by["rpc_tiered_consolidate"].task0_final > by["rpc_tiered_episodic_reset"].task0_final


def test_continual_composition_default_passes():
    cfg = CompositionConfig()
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    assert verdict.passed
