"""Smoke tests for compositional_bind benchmark machinery."""
from benchmarks.compositional_bind import BindConfig, evaluate_verdict, run_benchmark


def test_compositional_bind_smoke_2x2():
    cfg = BindConfig(D=256, n_colors=2, n_shapes=2)
    results = run_benchmark(cfg)
    by = {r.name: r for r in results}
    assert by["color_sharded_column"].accuracy == 1.0
    assert by["single_rpn"].accuracy == 1.0


def test_compositional_bind_verdict_runs():
    cfg = BindConfig(D=256, n_colors=2, n_shapes=2)
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    assert verdict.lines[-1].startswith("  -")
