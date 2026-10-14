"""Cross-module continual composition (B-14): SequenceRPC stack across tasks.

Protocol
--------
- Shared colors; each task adds a new shape block with unique object→action chains.
- ``end_task()`` consolidates tiered RPC memory; ``ChainRPN`` edges persist across tasks.
- Measure task-0 object (RPC) and 2-hop action (RPC + chain) retention after final task.

Run: python -m benchmarks.continual_stack
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.sequence_rpc import SequenceRPC


@dataclass
class StackConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 4
    shapes_per_task: int = 8
    n_tasks: int = 3
    min_task0_object: float = 0.95
    min_task0_action: float = 0.95
    max_episodic_object: float = 0.35
    max_episodic_action: float = 0.35
    min_object_margin: float = 0.50
    min_action_margin: float = 0.50


@dataclass
class StackResult:
    name: str
    task0_object: float
    task0_action: float
    task_object_accs: list[float]
    task_action_accs: list[float]
    task0_object_trace: list[float]
    task0_action_trace: list[float]


def _task_rows(task_idx: int, cfg: StackConfig) -> list[tuple[str, str, str, str]]:
    shape_offset = task_idx * cfg.shapes_per_task
    return [
        (
            f"color_{ci}",
            f"shape_{shape_offset + si}",
            f"obj_{task_idx}_{ci}_{si}",
            f"act_{task_idx}_{ci}_{si}",
        )
        for ci in range(cfg.n_colors)
        for si in range(cfg.shapes_per_task)
    ]


def _all_tasks(cfg: StackConfig) -> list[list[tuple[str, str, str, str]]]:
    return [_task_rows(ti, cfg) for ti in range(cfg.n_tasks)]


def _register(items: ItemTable, tasks: list[list[tuple[str, str, str, str]]]) -> None:
    for task in tasks:
        for color, shape, obj, action in task:
            _ = items[color]
            _ = items[shape]
            _ = items[obj]
            _ = items[action]


def _object_acc(stack: SequenceRPC, rows: list[tuple[str, str, str, str]]) -> float:
    ok = sum(1 for c, s, o, _ in rows if stack.read_object(c, s)[0] == o)
    return ok / len(rows)


def _action_acc(stack: SequenceRPC, rows: list[tuple[str, str, str, str]]) -> float:
    ok = sum(1 for c, s, _, a in rows if stack.read_action(c, s)[0] == a)
    return ok / len(rows)


@dataclass
class _Factory:
    D: int
    seed: int
    memory: str
    consolidate_on_task_end: bool = True
    k: int = 1

    def make(self, items: ItemTable) -> SequenceRPC:
        return SequenceRPC(
            self.D,
            items,
            memory=self.memory,  # type: ignore[arg-type]
            consolidate_on_task_end=self.consolidate_on_task_end,
            k=self.k,
        )


def _run_stack(name: str, factory: _Factory, tasks: list[list[tuple[str, str, str, str]]]) -> StackResult:
    items = ItemTable(factory.D, seed=factory.seed)
    _register(items, tasks)
    stack = factory.make(items)

    obj_trace: list[float] = []
    act_trace: list[float] = []
    final_obj: list[float] = []
    final_act: list[float] = []

    for ti, task in enumerate(tasks):
        for color, shape, obj, action in task:
            stack.learn(color, shape, obj, action)
        obj_trace.append(_object_acc(stack, tasks[0]))
        act_trace.append(_action_acc(stack, tasks[0]))
        stack.end_task()
        if ti == len(tasks) - 1:
            for ej in range(ti + 1):
                final_obj.append(_object_acc(stack, tasks[ej]))
                final_act.append(_action_acc(stack, tasks[ej]))

    return StackResult(
        name=name,
        task0_object=obj_trace[-1] if obj_trace else 0.0,
        task0_action=act_trace[-1] if act_trace else 0.0,
        task_object_accs=final_obj,
        task_action_accs=final_act,
        task0_object_trace=obj_trace,
        task0_action_trace=act_trace,
    )


def run_benchmark(cfg: StackConfig) -> list[StackResult]:
    tasks = _all_tasks(cfg)
    base = dict(D=cfg.D, seed=cfg.seed)
    return [
        _run_stack(
            "stack_tiered_consolidate",
            _Factory(**base, memory="tiered", consolidate_on_task_end=True),
            tasks,
        ),
        _run_stack(
            "stack_tiered_episodic",
            _Factory(**base, memory="tiered", consolidate_on_task_end=False),
            tasks,
        ),
        _run_stack(
            "stack_kuramoto",
            _Factory(**base, memory="kuramoto", consolidate_on_task_end=True, k=4),
            tasks,
        ),
        _run_stack(
            "stack_predictive",
            _Factory(**base, memory="predictive", consolidate_on_task_end=True, k=1),
            tasks,
        ),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[StackResult], cfg: StackConfig) -> Verdict:
    by = {r.name: r for r in results}
    tiered = by["stack_tiered_consolidate"]
    episodic = by["stack_tiered_episodic"]
    kuramoto = by["stack_kuramoto"]
    obj_margin = tiered.task0_object - episodic.task0_object
    act_margin = tiered.task0_action - episodic.task0_action

    checks: list[tuple[str, bool]] = [
        (
            f"consolidate task0 object >= {cfg.min_task0_object:.2f} ({tiered.task0_object:.3f})",
            tiered.task0_object >= cfg.min_task0_object,
        ),
        (
            f"consolidate task0 action >= {cfg.min_task0_action:.2f} ({tiered.task0_action:.3f})",
            tiered.task0_action >= cfg.min_task0_action,
        ),
        (
            f"episodic task0 object <= {cfg.max_episodic_object:.2f} ({episodic.task0_object:.3f})",
            episodic.task0_object <= cfg.max_episodic_object,
        ),
        (
            f"episodic task0 action <= {cfg.max_episodic_action:.2f} ({episodic.task0_action:.3f})",
            episodic.task0_action <= cfg.max_episodic_action,
        ),
        (
            f"object margin >= {cfg.min_object_margin:.2f} ({obj_margin:.3f})",
            obj_margin >= cfg.min_object_margin,
        ),
        (
            f"action margin >= {cfg.min_action_margin:.2f} ({act_margin:.3f})",
            act_margin >= cfg.min_action_margin,
        ),
        (
            f"kuramoto task0 action >= predictive - 0.15 ({kuramoto.task0_action:.3f})",
            kuramoto.task0_action >= by["stack_predictive"].task0_action - 0.15,
        ),
    ]

    lines: list[str] = [
        f"Continual stack (colors={cfg.n_colors}, shapes/task={cfg.shapes_per_task}, "
        f"tasks={cfg.n_tasks}, D={cfg.D})",
        "",
        f"{'model':<26} {'t0 obj':>8} {'t0 act':>8}  all-object / all-action",
    ]
    for r in results:
        objs = ", ".join(f"{a:.2f}" for a in r.task_object_accs)
        acts = ", ".join(f"{a:.2f}" for a in r.task_action_accs)
        lines.append(
            f"{r.name:<26} {r.task0_object:>8.3f} {r.task0_action:>8.3f}  [{objs}] / [{acts}]"
        )
    lines.append("")
    lines.append("Task-0 traces (object / action after each task):")
    for r in results:
        obj_t = ", ".join(f"{a:.3f}" for a in r.task0_object_trace)
        act_t = ", ".join(f"{a:.3f}" for a in r.task0_action_trace)
        lines.append(f"  {r.name}: obj [{obj_t}]  act [{act_t}]")
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - RPC (tiered) holds appearance->object across tasks when consolidated.")
    lines.append("  - Chain edges persist; action retention tracks object retention.")
    lines.append("  - Kuramoto RPC is a non-tiered dynamical baseline (no end_task effect).")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = StackConfig()
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
