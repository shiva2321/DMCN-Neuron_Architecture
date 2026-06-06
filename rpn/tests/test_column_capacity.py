"""Smoke tests for column_capacity benchmark."""
from benchmarks.column_capacity import CapacityConfig, evaluate_verdict, run_benchmark


def test_column_capacity_smoke():
    cfg = CapacityConfig(D=256, n_colors=4, n_shapes=4)
    results = run_benchmark(cfg)
    by = {r.name: r for r in results}
    assert by["color_sharded_column"].accuracy == 1.0
    assert by["color_sharded_column"].n_shards == 4


def test_column_capacity_verdict_default_passes():
    cfg = CapacityConfig()
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    assert verdict.passed
