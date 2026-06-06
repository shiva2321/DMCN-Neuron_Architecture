"""Smoke tests for continual BWT benchmark."""
from benchmarks.continual_bwt import BWTConfig, evaluate_verdict, run_benchmark


def test_continual_bwt_smoke():
    cfg = BWTConfig(D=256, n_tasks=3, pairs_per_task=8, min_bwt_margin=0.05, min_task0_retention=0.5)
    results = run_benchmark(cfg)
    assert len(results) == 3
    by = {r.name: r for r in results}
    assert by["tiered_t0t2_consolidate"].task0_final_acc > by["tiered_episodic_reset"].task0_final_acc


def test_continual_bwt_default_verdict_passes():
    cfg = BWTConfig(D=512, n_tasks=5, pairs_per_task=32)
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    assert verdict.passed
