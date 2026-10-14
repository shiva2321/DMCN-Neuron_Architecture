"""Smoke tests for B-9 compositional OOD scorecard."""
from benchmarks.compositional_ood import OODConfig, evaluate_verdict, run_scorecard
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.reference_frame import ReferenceFrame
from neuron_v2.core.rpc import ResonantPredictiveColumn


def test_agent_action_frame_smoke():
    items = ItemTable(256, seed=2)
    pairs = [(f"agent_{i}", f"action_{j}", f"out_{i}_{j}") for i in range(2) for j in range(4)]
    for a, b, o in pairs:
        _ = items[a]
        _ = items[b]
        _ = items[o]
    rpc = ResonantPredictiveColumn(256, items, frame=ReferenceFrame.agent_action(256))
    for a, b, o in pairs:
        rpc.learn(a, b, o)
    for a, b, o in pairs:
        assert rpc.read(a, b)[0] == o


def test_ood_scorecard_smoke():
    cfg = OODConfig(
        D=256,
        n_colors=4,
        n_shapes=8,
        n_adapt_shapes=2,
        n_agents=2,
        n_actions=4,
        new_what_single_retention_max=0.85,
        new_what_margin=0.10,
        holdout_single_max=0.90,
        holdout_margin=0.05,
    )
    results = run_scorecard(cfg)
    assert len(results) == 3


def test_ood_scorecard_default_passes():
    cfg = OODConfig()
    verdict = evaluate_verdict(run_scorecard(cfg), cfg)
    assert verdict.passed
