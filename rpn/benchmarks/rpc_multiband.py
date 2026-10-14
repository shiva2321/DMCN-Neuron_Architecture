"""RPC + multi-band benchmark (B-11): K-band units inside full RPC.

Run: python -m benchmarks.rpc_multiband
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn


@dataclass
class RpcMbConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 8
    n_shapes: int = 32
    K: int = 3
    min_multiband_acc: float = 0.95
    min_predictive_acc: float = 0.95
    max_acc_gap: float = 0.02


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
    return ok / len(pairs)


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def run_benchmark(cfg: RpcMbConfig) -> Verdict:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)

    items_mb = ItemTable(cfg.D, seed=cfg.seed)
    _register(items_mb, pairs)
    mb_rpc = ResonantPredictiveColumn(
        cfg.D, items_mb, memory="multiband", multiband_K=cfg.K
    )

    items_pred = ItemTable(cfg.D, seed=cfg.seed)
    _register(items_pred, pairs)
    pred_rpc = ResonantPredictiveColumn(cfg.D, items_pred, memory="predictive", k=1)

    for color, shape, obj in pairs:
        mb_rpc.learn(color, shape, obj)
        pred_rpc.learn(color, shape, obj)

    mb_acc = _accuracy(mb_rpc, pairs)
    pred_acc = _accuracy(pred_rpc, pairs)
    gap = abs(mb_acc - pred_acc)

    checks = [
        (
            f"multiband RPC >= {cfg.min_multiband_acc:.2f} ({mb_acc:.3f})",
            mb_acc >= cfg.min_multiband_acc,
        ),
        (
            f"predictive RPC >= {cfg.min_predictive_acc:.2f} ({pred_acc:.3f})",
            pred_acc >= cfg.min_predictive_acc,
        ),
        (
            f"parity gap <= {cfg.max_acc_gap:.2f} ({gap:.3f})",
            gap <= cfg.max_acc_gap,
        ),
    ]
    lines = [
        f"RPC multiband (8×32, D={cfg.D}, K={cfg.K})",
        "",
        f"{'model':<20} {'accuracy':>10}",
        f"{'multiband_rpc':<20} {mb_acc:>10.3f}",
        f"{'predictive_rpc':<20} {pred_acc:>10.3f}",
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
    verdict = run_benchmark(RpcMbConfig())
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
