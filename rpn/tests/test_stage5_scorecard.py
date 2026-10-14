"""Smoke tests for B-15 Stage 5 hardening scorecard."""
from benchmarks.stage5_scorecard import Stage5Config, evaluate_verdict, run_scorecard


def test_stage5_scorecard_smoke():
    cfg = Stage5Config(
        D=256,
        scale_colors=8,
        scale_shapes=16,
        min_scale_rpc=0.90,
        max_scale_single=0.35,
        min_scale_margin=0.50,
        ruler_colors=8,
        ruler_shapes=16,
        ruler_probe=20,
        min_ruler_rpc=0.90,
        max_ruler_single=0.40,
        energy_colors=4,
        energy_shapes=8,
        min_relearn_skip_rate=0.10,
        max_footprint_mb=32.0,
        refusal_colors=8,
        refusal_shapes=16,
        min_refusal_train=0.90,
        max_refusal_holdout_zero=0.15,
    )
    results = run_scorecard(cfg)
    by_name = {r.name: r for r in results}
    assert by_name["scale_capacity"].metrics["rpc"] > by_name["scale_capacity"].metrics["single"]
    assert by_name["refusal_zero_shot"].metrics["holdout_zero"] <= cfg.max_refusal_holdout_zero + 0.05


def test_stage5_scorecard_default_passes():
    cfg = Stage5Config()
    assert evaluate_verdict(run_scorecard(cfg), cfg).passed
