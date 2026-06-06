"""Fast regression on continual shootout (small M, low D)."""
from benchmarks.continual_assoc import ShootoutConfig, evaluate_verdict, run_shootout


def test_continual_shootout_smoke():
    cfg = ShootoutConfig(D=256, M=16, n_column=4, saturation_epochs=4, mlp_hidden=128)
    results = run_shootout(cfg)
    verdict = evaluate_verdict(results)
    assert len(results) == 5
    # Smoke: delta RPN should nail small task; full gates may fail on tiny M vs MLP.
    delta = next(r for r in results if r.name == "rpn_delta")
    assert delta.single_pass_acc >= 0.9
