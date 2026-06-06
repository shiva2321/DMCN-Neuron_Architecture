"""Continual associative recall shootout: RPN vs RXCA-style vs MLP.

Protocol
--------
1. Build a fixed token pool; sample M unique (context -> target) pairs.
2. Stream-teach each pair once (online, no replay).
3. Measure token-level recall accuracy on all M pairs.
4. Repeat teach for extra epochs (saturation / interference stress).

Pass/fail (ruthless)
--------------------
- Single-pass: best RPN variant must beat RXCA additive AND online MLP on accuracy.
- Saturation: delta RPN (or column) must stay above floor while RXCA/additive degrades.

Run: python -m benchmarks.continual_assoc
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from neuron_v2.core.item_table import ItemTable
from benchmarks.baselines import (
    AdditiveMatrixBrain,
    OnlineMLP,
    RPNColumn,
    RXCAAdditiveBrain,
    SingleRPN,
)


class AssociativeModel(Protocol):
    def learn(self, ctx_token: str, target_token: str) -> None: ...
    def read(self, ctx_token: str, **kwargs) -> tuple[str | None, float]: ...


@dataclass
class ShootoutConfig:
    D: int = 512
    M: int = 48
    n_column: int = 8
    mlp_hidden: int = 256
    mlp_lr: float = 0.08
    seed: int = 42
    saturation_epochs: int = 8


@dataclass
class ShootoutResult:
    name: str
    single_pass_acc: float
    saturation_acc: float


def _make_pairs(M: int, rng: np.random.RandomState) -> list[tuple[str, str]]:
    ctx_pool = [f"ctx_{i}" for i in range(M * 2)]
    tgt_pool = [f"tgt_{i}" for i in range(M * 2)]
    pairs: list[tuple[str, str]] = []
    used_ctx: set[str] = set()
    while len(pairs) < M:
        c = ctx_pool[int(rng.randint(0, len(ctx_pool)))]
        t = tgt_pool[int(rng.randint(0, len(tgt_pool)))]
        if c in used_ctx:
            continue
        used_ctx.add(c)
        pairs.append((c, t))
    return pairs


def _register_tokens(items: ItemTable, pairs: list[tuple[str, str]]) -> None:
    for c, t in pairs:
        _ = items[c]
        _ = items[t]


def _accuracy(model: AssociativeModel, pairs: list[tuple[str, str]]) -> float:
    ok = 0
    for c, t in pairs:
        pred, _ = model.read(c)
        if pred == t:
            ok += 1
    return ok / len(pairs)


def _train_stream(
    model: AssociativeModel,
    pairs: list[tuple[str, str]],
    epochs: int,
) -> None:
    for _ in range(epochs):
        for c, t in pairs:
            model.learn(c, t)


def run_shootout(cfg: ShootoutConfig) -> list[ShootoutResult]:
    rng = np.random.RandomState(cfg.seed)
    pairs = _make_pairs(cfg.M, rng)

    def _fresh_items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        _register_tokens(items, pairs)
        return items

    models: list[tuple[str, Callable[[], AssociativeModel]]] = [
        ("rxca_additive", lambda: RXCAAdditiveBrain(cfg.D, _fresh_items())),
        ("matrix_additive", lambda: AdditiveMatrixBrain(cfg.D, _fresh_items())),
        ("rpn_delta", lambda: SingleRPN(cfg.D, _fresh_items(), mode="delta")),
        (
            f"rpn_column_{cfg.n_column}",
            lambda: RPNColumn(cfg.D, _fresh_items(), cfg.n_column, mode="delta"),
        ),
        (
            "mlp_2layer",
            lambda: OnlineMLP(
                cfg.D,
                _fresh_items(),
                hidden=cfg.mlp_hidden,
                lr=cfg.mlp_lr,
            ),
        ),
    ]

    results: list[ShootoutResult] = []
    for name, factory in models:
        m = factory()
        _train_stream(m, pairs, epochs=1)
        single = _accuracy(m, pairs)

        m2 = factory()
        _train_stream(m2, pairs, epochs=cfg.saturation_epochs)
        sat = _accuracy(m2, pairs)

        results.append(ShootoutResult(name=name, single_pass_acc=single, saturation_acc=sat))
    return results


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ShootoutResult]) -> Verdict:
    by = {r.name: r for r in results}

    def _pick_rpn() -> ShootoutResult:
        col = [r for r in results if r.name.startswith("rpn_column")]
        delta = by["rpn_delta"]
        return max([delta, *col], key=lambda r: r.single_pass_acc)

    rpn_best = _pick_rpn()
    rxca = by["rxca_additive"]
    mlp = by["mlp_2layer"]
    mat_add = by["matrix_additive"]

    lines: list[str] = []
    lines.append("Continual associative shootout (stream-once + saturation)")
    lines.append("")
    lines.append(f"{'model':<22} {'1-pass acc':>10} {'sat acc':>10}")
    for r in results:
        lines.append(f"{r.name:<22} {r.single_pass_acc:>10.3f} {r.saturation_acc:>10.3f}")

    checks: list[tuple[str, bool]] = []

    # Ruthless: RPN must beat both RXCA hologram and online MLP on single-pass.
    c1 = rpn_best.single_pass_acc > rxca.single_pass_acc
    checks.append(
        (
            f"RPN ({rpn_best.name}) single-pass > RXCA additive "
            f"({rpn_best.single_pass_acc:.3f} > {rxca.single_pass_acc:.3f})",
            c1,
        )
    )

    c2 = rpn_best.single_pass_acc > mlp.single_pass_acc
    checks.append(
        (
            f"RPN ({rpn_best.name}) single-pass > online MLP "
            f"({rpn_best.single_pass_acc:.3f} > {mlp.single_pass_acc:.3f})",
            c2,
        )
    )

    # Saturation: delta should not collapse; RXCA or matrix additive should be worse or equal.
    c3 = rpn_best.saturation_acc >= 0.85
    checks.append(
        (f"RPN saturation acc >= 0.85 ({rpn_best.saturation_acc:.3f})", c3)
    )

    c4 = rpn_best.saturation_acc > min(rxca.saturation_acc, mat_add.saturation_acc)
    checks.append(
        (
            f"RPN saturation > worst additive "
            f"({rpn_best.saturation_acc:.3f} > "
            f"{min(rxca.saturation_acc, mat_add.saturation_acc):.3f})",
            c4,
        )
    )

    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation (read this, not just PASS):")
    lines.append(
        "  - RXCA hologram (surface += bind): cheap O(D) memory; collapses as M grows."
    )
    lines.append(
        "  - Matrix additive uses same O(D^2) cost as delta; token accuracy may tie."
    )
    lines.append(
        "  - Delta win on RMS/crosstalk is in capacity_curve, not always token acc."
    )
    lines.append(
        "  - MLP needs many epochs to match RPN; backprop, no built-in refuse."
    )
    lines.append(
        "  - Column helps when sharding raises effective capacity (high M per neuron)."
    )
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = ShootoutConfig()
    results = run_shootout(cfg)
    verdict = evaluate_verdict(results)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
