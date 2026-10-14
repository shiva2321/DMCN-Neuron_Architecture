"""Continual compositional learning (B-8): tiered RPC + task-boundary consolidation.

Protocol
--------
- Shared colors across tasks; each task adds a new shape block in the same where-shards.
- After each task: ``end_task()`` on tiered RPC (T1→T2 replay or episodic reset).
- Measure task-0 retention after the final task.

Run: python -m benchmarks.continual_composition
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn


@dataclass
class CompositionConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 4
    shapes_per_task: int = 8
    n_tasks: int = 3
    min_task0_retention: float = 0.95
    max_episodic_task0: float = 0.35
    min_retention_margin: float = 0.50


@dataclass
class ModelResult:
    name: str
    task_accuracies: list[float]
    task0_final: float
    task0_after_each: list[float]


def _task_pairs(
    task_idx: int,
    cfg: CompositionConfig,
) -> list[tuple[str, str, str]]:
    shape_offset = task_idx * cfg.shapes_per_task
    return [
        (
            f"color_{ci}",
            f"shape_{shape_offset + si}",
            f"obj_{task_idx}_{ci}_{si}",
        )
        for ci in range(cfg.n_colors)
        for si in range(cfg.shapes_per_task)
    ]


def _all_tasks(cfg: CompositionConfig) -> list[list[tuple[str, str, str]]]:
    return [_task_pairs(ti, cfg) for ti in range(cfg.n_tasks)]


def _register(items: ItemTable, tasks: list[list[tuple[str, str, str]]]) -> None:
    for task in tasks:
        for color, shape, obj in task:
            _ = items[color]
            _ = items[shape]
            _ = items[obj]


def _accuracy(model: ResonantPredictiveColumn, pairs: list[tuple[str, str, str]]) -> float:
    ok = sum(1 for c, s, o in pairs if model.read(c, s)[0] == o)
    return ok / len(pairs)


def _run_rpc(
    name: str,
    factory,
    tasks: list[list[tuple[str, str, str]]],
) -> ModelResult:
    items = ItemTable(factory.D, seed=factory.seed)
    _register(items, tasks)
    model = factory.make(items)

    task0_trace: list[float] = []
    final_accs: list[float] = []

    for ti, task in enumerate(tasks):
        for color, shape, obj in task:
            model.learn(color, shape, obj)
        for ej in range(ti + 1):
            acc = _accuracy(model, tasks[ej])
            if ej == 0:
                task0_trace.append(acc)
        model.end_task()
        if ti == len(tasks) - 1:
            for ej in range(ti + 1):
                final_accs.append(_accuracy(model, tasks[ej]))

    return ModelResult(
        name=name,
        task_accuracies=final_accs,
        task0_final=task0_trace[-1] if task0_trace else 0.0,
        task0_after_each=task0_trace,
    )


@dataclass
class _Factory:
    D: int
    seed: int
    memory: str
    consolidate_on_task_end: bool = True

    def make(self, items: ItemTable) -> ResonantPredictiveColumn:
        return ResonantPredictiveColumn(
            self.D,
            items,
            memory=self.memory,  # type: ignore[arg-type]
            consolidate_on_task_end=self.consolidate_on_task_end,
        )


def run_benchmark(cfg: CompositionConfig) -> list[ModelResult]:
    tasks = _all_tasks(cfg)
    base = _Factory(D=cfg.D, seed=cfg.seed, memory="predictive")
    return [
        _run_rpc(
            "rpc_tiered_consolidate",
            _Factory(
                D=cfg.D,
                seed=cfg.seed,
                memory="tiered",
                consolidate_on_task_end=True,
            ),
            tasks,
        ),
        _run_rpc(
            "rpc_tiered_episodic_reset",
            _Factory(
                D=cfg.D,
                seed=cfg.seed,
                memory="tiered",
                consolidate_on_task_end=False,
            ),
            tasks,
        ),
        _run_rpc(
            "rpc_predictive",
            base,
            tasks,
        ),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ModelResult], cfg: CompositionConfig) -> Verdict:
    by = {r.name: r for r in results}
    tiered = by["rpc_tiered_consolidate"]
    episodic = by["rpc_tiered_episodic_reset"]
    predictive = by["rpc_predictive"]
    margin = tiered.task0_final - episodic.task0_final

    checks: list[tuple[str, bool]] = [
        (
            f"tiered consolidate task0 >= {cfg.min_task0_retention:.2f} "
            f"({tiered.task0_final:.3f})",
            tiered.task0_final >= cfg.min_task0_retention,
        ),
        (
            f"tiered episodic-reset task0 <= {cfg.max_episodic_task0:.2f} "
            f"({episodic.task0_final:.3f})",
            episodic.task0_final <= cfg.max_episodic_task0,
        ),
        (
            f"retention margin >= {cfg.min_retention_margin:.2f} ({margin:.3f})",
            margin >= cfg.min_retention_margin,
        ),
        (
            f"predictive task0 >= consolidate - 0.10 ({predictive.task0_final:.3f})",
            predictive.task0_final >= tiered.task0_final - 0.10,
        ),
    ]

    lines: list[str] = []
    lines.append(
        f"Continual composition (colors={cfg.n_colors}, "
        f"shapes/task={cfg.shapes_per_task}, tasks={cfg.n_tasks}, D={cfg.D})"
    )
    lines.append("")
    lines.append(f"{'model':<28} {'task0 final':>12} {'all tasks':>20}")
    for r in results:
        accs = ", ".join(f"{a:.2f}" for a in r.task_accuracies)
        lines.append(f"{r.name:<28} {r.task0_final:>12.3f} [{accs}]")
    lines.append("")
    lines.append("Task-0 trace (after each task):")
    for r in results:
        trace = ", ".join(f"{a:.3f}" for a in r.task0_after_each)
        lines.append(f"  {r.name}: {trace}")
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Same where-shards learn new shape blocks each task.")
    lines.append("  - consolidate: replay into T2 before T1 reset — task-0 retained.")
    lines.append("  - episodic-reset: T1 cleared without T2 — task-0 forgotten.")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = CompositionConfig()
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
