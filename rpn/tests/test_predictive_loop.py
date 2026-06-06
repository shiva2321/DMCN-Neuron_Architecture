"""Smoke tests for predictive_loop benchmark."""
from benchmarks.predictive_loop import LoopConfig, evaluate_verdict, run_benchmark


def test_predictive_loop_default_passes():
    cfg = LoopConfig(stream_epochs=6)
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    assert verdict.passed
