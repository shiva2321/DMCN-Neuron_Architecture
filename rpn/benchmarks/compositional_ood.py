"""Compositional OOD scorecard (B-9): new what-fillers, holdout recombine, alt frames.

Scenarios
---------
1. new_what — train known shapes per color; add new shape block; shard retention.
2. holdout_recombine — checkerboard train split; one-shot fill held-out pairs.
3. agent_action — second reference frame (agent×action) reaches column accuracy.

Run: python -m benchmarks.compositional_ood
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.reference_frame import ReferenceFrame
from neuron_v2.core.rpc import ResonantPredictiveColumn
from benchmarks.baselines import SingleRPNComposite


@dataclass
class OODConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 8
    n_shapes: int = 32
    n_adapt_shapes: int = 8
    n_agents: int = 4
    n_actions: int = 8
    new_what_rpc_retention_min: float = 0.95
    new_what_single_retention_max: float = 0.50
    new_what_rpc_new_min: float = 0.95
    new_what_margin: float = 0.50
    holdout_rpc_min: float = 0.95
    holdout_single_max: float = 0.70
    holdout_margin: float = 0.25
    alt_frame_min_acc: float = 0.95


@dataclass
class ScenarioResult:
    name: str
    metrics: dict[str, float]
    passed: bool
    detail: str


def _grid(n_colors: int, n_shapes: int) -> list[tuple[str, str, str]]:
    return [
        (f"color_{ci}", f"shape_{si}", f"obj_{ci}_{si}")
        for ci in range(n_colors)
        for si in range(n_shapes)
    ]


def _register(items: ItemTable, pairs: list[tuple[str, str, str]]) -> None:
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]


def _accuracy(model, pairs: list[tuple[str, str, str]]) -> float:
    ok = sum(1 for c, s, o in pairs if model.read(c, s)[0] == o)
    return ok / len(pairs) if pairs else 0.0


def scenario_new_what(cfg: OODConfig) -> ScenarioResult:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)
    n_train_shapes = cfg.n_shapes - cfg.n_adapt_shapes
    train = [p for p in pairs if int(p[1].split("_")[1]) < n_train_shapes]
    adapt = [p for p in pairs if int(p[1].split("_")[1]) >= n_train_shapes]

    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in train:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    rpc_before = _accuracy(rpc, train)
    single_before = _accuracy(single, train)

    for color, shape, obj in adapt:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    rpc_ret = _accuracy(rpc, train)
    single_ret = _accuracy(single, train)
    rpc_new = _accuracy(rpc, adapt)
    single_new = _accuracy(single, adapt)
    margin = rpc_ret - single_ret

    passed = (
        rpc_ret >= cfg.new_what_rpc_retention_min
        and single_ret <= cfg.new_what_single_retention_max
        and rpc_new >= cfg.new_what_rpc_new_min
        and margin >= cfg.new_what_margin
    )
    return ScenarioResult(
        "new_what_adapt",
        {
            "rpc_retention": rpc_ret,
            "single_retention": single_ret,
            "rpc_new": rpc_new,
            "single_new": single_new,
            "retention_margin": margin,
            "rpc_before": rpc_before,
            "single_before": single_before,
        },
        passed,
        f"retention RPC {rpc_ret:.3f} vs single {single_ret:.3f}; "
        f"new shapes RPC {rpc_new:.3f}",
    )


def scenario_holdout_recombine(cfg: OODConfig) -> ScenarioResult:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)
    train = [p for p in pairs if (int(p[0].split("_")[1]) + int(p[1].split("_")[1])) % 2 == 0]
    holdout = [p for p in pairs if (int(p[0].split("_")[1]) + int(p[1].split("_")[1])) % 2 == 1]

    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in train:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    zero_rpc = _accuracy(rpc, holdout)
    zero_single = _accuracy(single, holdout)

    for color, shape, obj in holdout:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    one_rpc = _accuracy(rpc, holdout)
    one_single = _accuracy(single, holdout)
    train_rpc = _accuracy(rpc, train)
    margin = one_rpc - one_single

    passed = (
        one_rpc >= cfg.holdout_rpc_min
        and one_single <= cfg.holdout_single_max
        and margin >= cfg.holdout_margin
        and train_rpc >= cfg.holdout_rpc_min
    )
    return ScenarioResult(
        "holdout_recombine",
        {
            "zero_rpc": zero_rpc,
            "zero_single": zero_single,
            "one_shot_rpc": one_rpc,
            "one_shot_single": one_single,
            "train_rpc": train_rpc,
            "margin": margin,
        },
        passed,
        f"zero-shot RPC {zero_rpc:.3f}; one-shot RPC {one_rpc:.3f} vs single {one_single:.3f}",
    )


def scenario_agent_action(cfg: OODConfig) -> ScenarioResult:
    pairs = [
        (f"agent_{ai}", f"action_{aj}", f"out_{ai}_{aj}")
        for ai in range(cfg.n_agents)
        for aj in range(cfg.n_actions)
    ]
    items = ItemTable(cfg.D, seed=cfg.seed + 1)
    for agent, action, out in pairs:
        _ = items[agent]
        _ = items[action]
        _ = items[out]

    frame = ReferenceFrame.agent_action(cfg.D)
    rpc = ResonantPredictiveColumn(cfg.D, items, frame=frame)
    for agent, action, out in pairs:
        rpc.learn(agent, action, out)

    acc = _accuracy(rpc, pairs)
    passed = acc >= cfg.alt_frame_min_acc
    return ScenarioResult(
        "agent_action_frame",
        {"rpc": acc},
        passed,
        f"RPC {acc:.3f} on agent×action frame ({cfg.n_agents}×{cfg.n_actions})",
    )


def run_scorecard(cfg: OODConfig) -> list[ScenarioResult]:
    return [
        scenario_new_what(cfg),
        scenario_holdout_recombine(cfg),
        scenario_agent_action(cfg),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ScenarioResult], cfg: OODConfig) -> Verdict:
    lines: list[str] = []
    lines.append(f"Compositional OOD scorecard (D={cfg.D})")
    lines.append("")
    lines.append(f"{'scenario':<22} {'passed':>8}  detail")
    for r in results:
        lines.append(f"{r.name:<22} {'PASS' if r.passed else 'FAIL':>8}  {r.detail}")
    lines.append("")
    lines.append("Metrics:")
    for r in results:
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in r.metrics.items())
        lines.append(f"  {r.name}: {metric_str}")
    lines.append("")
    passed = all(r.passed for r in results)
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - new_what: same where-shard, new what-fillers (shapes) without erasing old.")
    lines.append("  - holdout: checkerboard split; one-shot completes unseen combinations.")
    lines.append("  - agent_action: alternate reference frame shares RPC machinery.")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = OODConfig()
    results = run_scorecard(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
