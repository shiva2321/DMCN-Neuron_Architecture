"""Continual learning BWT benchmark (B-4): T1/T2 + consolidation vs plain RPN.

Protocol
--------
- K disjoint tasks; each task has M (ctx -> target) pairs.
- Train tasks 0..K-1 sequentially; after each task, measure accuracy on all
  tasks seen so far.
- BWT = mean_j (R[K-1, j] - R[j, j]) for j < K-1 (higher = less forgetting).

Run: python -m benchmarks.continual_bwt
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN
from neuron_v2.core.tiered_rpn import TieredRPN


@dataclass
class BWTConfig:
    D: int = 512
    n_tasks: int = 5
    pairs_per_task: int = 32
    seed: int = 42
    min_bwt_margin: float = 0.80
    min_task0_retention: float = 0.95
    episodic_max_bwt: float = -0.75


@dataclass
class TaskResult:
    name: str
    accuracy_matrix: np.ndarray
    bwt: float
    final_mean_acc: float
    task0_final_acc: float


def _make_tasks(cfg: BWTConfig) -> list[list[tuple[str, str]]]:
    return [
        [(f"t{ti}_ctx_{pi}", f"t{ti}_tgt_{pi}") for pi in range(cfg.pairs_per_task)]
        for ti in range(cfg.n_tasks)
    ]


def _register(items: ItemTable, tasks: list[list[tuple[str, str]]]) -> None:
    for task in tasks:
        for ctx, tgt in task:
            _ = items[ctx]
            _ = items[tgt]


def _task_accuracy(model, pairs: list[tuple[str, str]]) -> float:
    ok = sum(1 for c, t in pairs if model.read(c)[0] == t)
    return ok / len(pairs)


def _run_model(
    name: str,
    factory,
    tasks: list[list[tuple[str, str]]],
    cfg: BWTConfig,
) -> TaskResult:
    K = cfg.n_tasks
    acc = np.zeros((K, K), dtype=np.float64)

    model = factory()
    for ti in range(K):
        for ctx, tgt in tasks[ti]:
            model.learn(ctx, tgt)
        for ej in range(ti + 1):
            acc[ti, ej] = _task_accuracy(model, tasks[ej])
        if hasattr(model, "end_task"):
            model.end_task()

    bwt_vals = [acc[K - 1, j] - acc[j, j] for j in range(K - 1)]
    bwt = float(np.mean(bwt_vals))
    return TaskResult(
        name=name,
        accuracy_matrix=acc,
        bwt=bwt,
        final_mean_acc=float(np.mean(acc[K - 1, :K])),
        task0_final_acc=float(acc[K - 1, 0]),
    )


def run_benchmark(cfg: BWTConfig) -> list[TaskResult]:
    tasks = _make_tasks(cfg)

    def _items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        _register(items, tasks)
        return items

    factories: list[tuple[str, object]] = [
        (
            "tiered_episodic_reset",
            lambda: TieredRPN(
                cfg.D,
                _items(),
                mode="delta",
                consolidate_on_task_end=False,
            ),
        ),
        (
            "tiered_t0t2_consolidate",
            lambda: TieredRPN(
                cfg.D,
                _items(),
                mode="delta",
                consolidate_on_task_end=True,
            ),
        ),
        ("plain_rpn_monolith", lambda: RPN(cfg.D, _items(), mode="delta")),
    ]

    return [_run_model(name, factory, tasks, cfg) for name, factory in factories]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[TaskResult], cfg: BWTConfig) -> Verdict:
    by = {r.name: r for r in results}
    episodic = by["tiered_episodic_reset"]
    tiered = by["tiered_t0t2_consolidate"]
    plain = by["plain_rpn_monolith"]

    checks: list[tuple[str, bool]] = [
        (
            f"consolidate BWT > episodic-reset ({tiered.bwt:.3f} > {episodic.bwt:.3f})",
            tiered.bwt > episodic.bwt,
        ),
        (
            f"BWT margin >= {cfg.min_bwt_margin:.2f} "
            f"({tiered.bwt - episodic.bwt:.3f})",
            tiered.bwt - episodic.bwt >= cfg.min_bwt_margin,
        ),
        (
            f"episodic-reset BWT <= {cfg.episodic_max_bwt:.2f} ({episodic.bwt:.3f})",
            episodic.bwt <= cfg.episodic_max_bwt,
        ),
        (
            f"task-0 retention >= {cfg.min_task0_retention:.2f} "
            f"({tiered.task0_final_acc:.3f})",
            tiered.task0_final_acc >= cfg.min_task0_retention,
        ),
        (
            f"consolidate >= plain task-0 ({tiered.task0_final_acc:.3f} >= {plain.task0_final_acc:.3f})",
            tiered.task0_final_acc >= plain.task0_final_acc,
        ),
    ]

    lines: list[str] = []
    lines.append(
        f"Continual BWT benchmark (K={cfg.n_tasks}, M={cfg.pairs_per_task}/task, D={cfg.D})"
    )
    lines.append("")
    lines.append(f"{'model':<26} {'BWT':>8} {'task0':>8} {'final avg':>10}")
    for r in results:
        lines.append(
            f"{r.name:<26} {r.bwt:>8.3f} {r.task0_final_acc:>8.3f} {r.final_mean_acc:>10.3f}"
        )
    lines.append("")
    lines.append("Accuracy matrix (rows=after task, cols=eval task) — tiered_t0t2:")
    t = tiered.accuracy_matrix
    header = "       " + "".join(f"t{j:>6}" for j in range(cfg.n_tasks))
    lines.append(header)
    for i in range(cfg.n_tasks):
        row = f"  @{i}  " + "".join(f"{t[i, j]:>6.3f}" for j in range(cfg.n_tasks))
        lines.append(row)
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - episodic-reset: T1 cleared each task without T2 replay — forgetting.")
    lines.append("  - consolidate: replay episodic log into T2 before T1 reset — retention.")
    lines.append("  - plain monolith: single delta matrix (no task boundaries); baseline.")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = BWTConfig()
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
