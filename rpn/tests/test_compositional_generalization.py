"""Smoke tests for B-6 compositional generalization scorecard."""
from benchmarks.compositional_generalization import (
    ScorecardConfig,
    evaluate_verdict,
    run_scorecard,
)


def test_scorecard_default_passes():
    cfg = ScorecardConfig(
        n_colors=4,
        n_shapes=8,
        n_train_colors=3,
        bind_colors=4,
        bind_shapes=4,
        capacity_single_max=0.75,
        capacity_min_margin=0.25,
        adapt_single_retention_max=0.85,
        adapt_retention_margin=0.10,
        bind_single_max=0.95,
    )
    results = run_scorecard(cfg)
    verdict = evaluate_verdict(results, cfg)
    assert verdict.passed


def test_scorecard_full_config_passes():
    cfg = ScorecardConfig()
    verdict = evaluate_verdict(run_scorecard(cfg), cfg)
    assert verdict.passed
