"""Sequence composition benchmark (B-11): RPC perception + chain action.

Teach (color, shape) → object → action as two modules; query action via 2-hop stack.
Monolithic single-RPN composite cannot shortcut the object hop from appearance alone.

Run: python -m benchmarks.sequence_composition
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.sequence_rpc import SequenceRPC
from benchmarks.baselines import SingleRPNComposite


@dataclass
class SeqCompConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 8
    n_shapes: int = 16
    min_object_acc: float = 0.98
    min_action_2hop: float = 0.95
    max_color_only_chain: float = 0.10


@dataclass
class SeqCompResult:
    object_acc: float
    action_2hop: float
    direct_action: float
    color_only_chain: float


def _grid(cfg: SeqCompConfig) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for ci in range(cfg.n_colors):
        for si in range(cfg.n_shapes):
            color = f"color_{ci}"
            shape = f"shape_{si}"
            obj = f"obj_{ci}_{si}"
            action = f"act_{ci}_{si}"
            rows.append((color, shape, obj, action))
    return rows


def _register(items: ItemTable, rows: list[tuple[str, str, str, str]]) -> None:
    for color, shape, obj, action in rows:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]
        _ = items[action]


def run_benchmark(cfg: SeqCompConfig) -> SeqCompResult:
    rows = _grid(cfg)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, rows)

    stack = SequenceRPC(cfg.D, items)
    shortcut = SingleRPNComposite(cfg.D, items)
    for color, shape, obj, action in rows:
        stack.learn(color, shape, obj, action)
        shortcut.learn(color, shape, action)

    obj_ok = act_ok = direct_ok = color_ok = 0
    for color, shape, obj, action in rows:
        if stack.read_object(color, shape)[0] == obj:
            obj_ok += 1
        if stack.read_action(color, shape)[0] == action:
            act_ok += 1
        if shortcut.read(color, shape)[0] == action:
            direct_ok += 1
        if stack.chain.read_hops(color, 1)[0] == action:
            color_ok += 1

    n = len(rows)
    return SeqCompResult(
        object_acc=obj_ok / n,
        action_2hop=act_ok / n,
        direct_action=direct_ok / n,
        color_only_chain=color_ok / n,
    )


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(result: SeqCompResult, cfg: SeqCompConfig) -> Verdict:
    checks = [
        (
            f"object (RPC) >= {cfg.min_object_acc:.2f} ({result.object_acc:.3f})",
            result.object_acc >= cfg.min_object_acc,
        ),
        (
            f"action 2-hop >= {cfg.min_action_2hop:.2f} ({result.action_2hop:.3f})",
            result.action_2hop >= cfg.min_action_2hop,
        ),
        (
            f"color-only chain <= {cfg.max_color_only_chain:.2f} ({result.color_only_chain:.3f})",
            result.color_only_chain <= cfg.max_color_only_chain,
        ),
    ]
    lines = [
        f"Sequence composition ({cfg.n_colors}×{cfg.n_shapes}, D={cfg.D})",
        "",
        f"{'metric':<22} {'accuracy':>10}",
        f"{'object (RPC)':<22} {result.object_acc:>10.3f}",
        f"{'action (2-hop)':<22} {result.action_2hop:>10.3f}",
        f"{'direct composite':<22} {result.direct_action:>10.3f}",
        f"{'color-only chain':<22} {result.color_only_chain:>10.3f}",
        "",
        "Pass/fail gates:",
    ]
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = SeqCompConfig()
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
