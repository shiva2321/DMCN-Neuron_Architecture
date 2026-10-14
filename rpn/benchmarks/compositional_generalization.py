"""Compositional generalization scorecard (B-6): full RPC + reference frames.

Scenarios
---------
1. capacity — 8×32 grid; RPC matches sharded column, beats monolithic single RPN.
2. new_where — train 7 colors; add color_8 in isolated where-shard; retention vs interference.
3. bind — 8×8 grid; RPC ≥ column-style accuracy on in-distribution composition.

Run: python -m benchmarks.compositional_generalization
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn
from benchmarks.baselines import SingleRPNComposite


@dataclass
class ScorecardConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 8
    n_shapes: int = 32
    n_train_colors: int = 7
    bind_colors: int = 8
    bind_shapes: int = 8
    capacity_min_acc: float = 0.95
    capacity_single_max: float = 0.30
    capacity_min_margin: float = 0.50
    adapt_rpc_retention_min: float = 0.95
    adapt_single_retention_max: float = 0.50
    adapt_new_where_min: float = 0.95
    adapt_retention_margin: float = 0.50
    bind_min_acc: float = 0.95
    bind_single_max: float = 0.85


@dataclass
class ScenarioResult:
    name: str
    metrics: dict[str, float]
    passed: bool
    detail: str


def _grid(
    n_colors: int,
    n_shapes: int,
) -> list[tuple[str, str, str]]:
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
    return ok / len(pairs)


def scenario_capacity(cfg: ScorecardConfig) -> ScenarioResult:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    rpc_acc = _accuracy(rpc, pairs)
    single_acc = _accuracy(single, pairs)
    margin = rpc_acc - single_acc
    passed = (
        rpc_acc >= cfg.capacity_min_acc
        and single_acc <= cfg.capacity_single_max
        and margin >= cfg.capacity_min_margin
    )
    return ScenarioResult(
        "capacity_8x32",
        {"rpc": rpc_acc, "single": single_acc, "margin": margin},
        passed,
        f"RPC {rpc_acc:.3f} vs single {single_acc:.3f} (margin {margin:.3f})",
    )


def scenario_new_where(cfg: ScorecardConfig) -> ScenarioResult:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    train = [p for p in pairs if int(p[0].split("_")[1]) < cfg.n_train_colors]
    adapt_ci = cfg.n_train_colors
    adapt = [p for p in pairs if int(p[0].split("_")[1]) == adapt_ci]

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
        rpc_ret >= cfg.adapt_rpc_retention_min
        and single_ret <= cfg.adapt_single_retention_max
        and rpc_new >= cfg.adapt_new_where_min
        and margin >= cfg.adapt_retention_margin
    )
    return ScenarioResult(
        "new_where_adapt",
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
        f"new color RPC {rpc_new:.3f}",
    )


def scenario_bind(cfg: ScorecardConfig) -> ScenarioResult:
    pairs = _grid(cfg.bind_colors, cfg.bind_shapes)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    rpc_acc = _accuracy(rpc, pairs)
    single_acc = _accuracy(single, pairs)
    passed = rpc_acc >= cfg.bind_min_acc and single_acc <= cfg.bind_single_max
    return ScenarioResult(
        "bind_8x8",
        {"rpc": rpc_acc, "single": single_acc},
        passed,
        f"RPC {rpc_acc:.3f} vs single {single_acc:.3f}",
    )


def run_scorecard(cfg: ScorecardConfig) -> list[ScenarioResult]:
    return [
        scenario_capacity(cfg),
        scenario_new_where(cfg),
        scenario_bind(cfg),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ScenarioResult], cfg: ScorecardConfig) -> Verdict:
    lines: list[str] = []
    lines.append(
        f"Compositional generalization scorecard (D={cfg.D}, "
        f"frame=color/shape reference)"
    )
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
    lines.append("Pass/fail gates:")
    cap, adapt, bind = results
    checks: list[tuple[str, bool]] = [
        (
            f"capacity RPC >= {cfg.capacity_min_acc:.2f} ({cap.metrics['rpc']:.3f})",
            cap.metrics["rpc"] >= cfg.capacity_min_acc,
        ),
        (
            f"capacity single <= {cfg.capacity_single_max:.2f} "
            f"({cap.metrics['single']:.3f})",
            cap.metrics["single"] <= cfg.capacity_single_max,
        ),
        (
            f"new_where RPC retention >= {cfg.adapt_rpc_retention_min:.2f} "
            f"({adapt.metrics['rpc_retention']:.3f})",
            adapt.metrics["rpc_retention"] >= cfg.adapt_rpc_retention_min,
        ),
        (
            f"new_where single retention <= {cfg.adapt_single_retention_max:.2f} "
            f"({adapt.metrics['single_retention']:.3f})",
            adapt.metrics["single_retention"] <= cfg.adapt_single_retention_max,
        ),
        (
            f"new_where retention margin >= {cfg.adapt_retention_margin:.2f} "
            f"({adapt.metrics['retention_margin']:.3f})",
            adapt.metrics["retention_margin"] >= cfg.adapt_retention_margin,
        ),
        (
            f"bind RPC >= {cfg.bind_min_acc:.2f} ({bind.metrics['rpc']:.3f})",
            bind.metrics["rpc"] >= cfg.bind_min_acc,
        ),
    ]
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(r.passed for r in results)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Reference frames bind where (color) + what (shape) into composite ctx.")
    lines.append("  - Where-sharded RPC adds a new frame without erasing prior where-shards.")
    lines.append("  - Monolithic single RPN interferes when the world grows (new where).")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = ScorecardConfig()
    results = run_scorecard(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
