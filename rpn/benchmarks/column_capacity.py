"""Column capacity at scale: 8-shard color column vs monolithic single RPN.

Teaches M = n_colors × n_shapes role-bound compositional pairs (default 8×32 = 256).
ColorShardedColumn holds n_shapes associations per shard; SingleRPNComposite holds all M.

Run: python -m benchmarks.column_capacity
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.column import ColorShardedColumn
from neuron_v2.core.item_table import ItemTable
from benchmarks.baselines import SingleRPNComposite


@dataclass
class CapacityConfig:
    D: int = 512
    n_colors: int = 8
    n_shapes: int = 32
    seed: int = 42
    column_min_acc: float = 0.95
    single_max_acc: float = 0.75
    min_margin: float = 0.50


@dataclass
class CapacityResult:
    name: str
    accuracy: float
    n_shards: int
    pairs_per_shard: int


def _make_grid(cfg: CapacityConfig) -> list[tuple[str, str, str]]:
    return [
        (f"color_{ci}", f"shape_{si}", f"obj_{ci}_{si}")
        for ci in range(cfg.n_colors)
        for si in range(cfg.n_shapes)
    ]


def _register(items: ItemTable, pairs: list[tuple[str, str, str]]) -> None:
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]


def _accuracy(model, pairs: list[tuple[str, str, str]]) -> float:
    ok = sum(1 for c, s, o in pairs if model.read(c, s)[0] == o)
    return ok / len(pairs)


def run_benchmark(cfg: CapacityConfig) -> list[CapacityResult]:
    pairs = _make_grid(cfg)
    m = len(pairs)

    def _items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        _register(items, pairs)
        return items

    single = SingleRPNComposite(cfg.D, _items(), mode="delta")
    column = ColorShardedColumn(cfg.D, _items(), mode="delta")

    for color, shape, obj in pairs:
        single.learn(color, shape, obj)
        column.learn(color, shape, obj)

    return [
        CapacityResult("single_rpn", _accuracy(single, pairs), 1, m),
        CapacityResult(
            "color_sharded_column",
            _accuracy(column, pairs),
            cfg.n_colors,
            cfg.n_shapes,
        ),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[CapacityResult], cfg: CapacityConfig) -> Verdict:
    by = {r.name: r for r in results}
    col = by["color_sharded_column"]
    single = by["single_rpn"]
    m = cfg.n_colors * cfg.n_shapes

    checks: list[tuple[str, bool]] = [
        (
            f"color_sharded_column acc >= {cfg.column_min_acc:.2f} ({col.accuracy:.3f})",
            col.accuracy >= cfg.column_min_acc,
        ),
        (
            f"single_rpn acc <= {cfg.single_max_acc:.2f} ({single.accuracy:.3f})",
            single.accuracy <= cfg.single_max_acc,
        ),
        (
            f"margin >= {cfg.min_margin:.2f} ({col.accuracy - single.accuracy:.3f})",
            col.accuracy - single.accuracy >= cfg.min_margin,
        ),
        (
            f"column beats single ({col.accuracy:.3f} > {single.accuracy:.3f})",
            col.accuracy > single.accuracy,
        ),
    ]

    lines: list[str] = []
    lines.append(
        f"Column capacity benchmark ({cfg.n_colors} shards x {cfg.n_shapes} pairs/shard "
        f"= M={m}, D={cfg.D})"
    )
    lines.append("")
    lines.append(f"{'model':<24} {'acc':>8} {'shards':>8} {'pairs/shard':>12}")
    for r in results:
        lines.append(
            f"{r.name:<24} {r.accuracy:>8.3f} {r.n_shards:>8} {r.pairs_per_shard:>12}"
        )
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  - Hash-route RPNColumn ties single RPN at M=256 random pairs (see continual_assoc)."
    )
    lines.append(
        "  - Semantic shard (by color) cuts load to n_shapes per neuron — wins at M>=256."
    )
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = CapacityConfig()
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
