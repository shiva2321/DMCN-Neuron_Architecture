"""RPC + Kuramoto benchmark (B-13): K-band Kuramoto units inside full RPC.

Run: python -m benchmarks.rpc_kuramoto
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn


@dataclass
class RpcKmConfig:
    D: int = 512
    seed: int = 42
    n_colors: int = 8
    n_shapes: int = 32
    K: int = 3
    settle_k: int = 4
    coupling: float = 0.35
    min_kuramoto_acc: float = 0.95
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


def run_benchmark(cfg: RpcKmConfig) -> Verdict:
    pairs = _grid(cfg.n_colors, cfg.n_shapes)

    items_km = ItemTable(cfg.D, seed=cfg.seed)
    _register(items_km, pairs)
    km_rpc = ResonantPredictiveColumn(
        cfg.D,
        items_km,
        memory="kuramoto",
        k=cfg.settle_k,
        kuramoto_K=cfg.K,
        kuramoto_coupling=cfg.coupling,
    )

    items_pred = ItemTable(cfg.D, seed=cfg.seed)
    _register(items_pred, pairs)
    pred_rpc = ResonantPredictiveColumn(
        cfg.D, items_pred, memory="predictive", k=cfg.settle_k
    )

    for color, shape, obj in pairs:
        km_rpc.learn(color, shape, obj)
        pred_rpc.learn(color, shape, obj)

    km_acc = _accuracy(km_rpc, pairs)
    pred_acc = _accuracy(pred_rpc, pairs)
    gap = abs(km_acc - pred_acc)

    checks = [
        (
            f"kuramoto RPC >= {cfg.min_kuramoto_acc:.2f} ({km_acc:.3f})",
            km_acc >= cfg.min_kuramoto_acc,
        ),
        (
            f"predictive_k{cfg.settle_k} RPC >= {cfg.min_predictive_acc:.2f} ({pred_acc:.3f})",
            pred_acc >= cfg.min_predictive_acc,
        ),
        (
            f"parity gap <= {cfg.max_acc_gap:.2f} ({gap:.3f})",
            gap <= cfg.max_acc_gap,
        ),
    ]
    lines = [
        f"RPC Kuramoto (8×32, D={cfg.D}, K={cfg.K}, k={cfg.settle_k})",
        "",
        f"{'model':<22} {'accuracy':>10}",
        f"{'kuramoto_rpc':<22} {km_acc:>10.3f}",
        f"{'predictive_rpc':<22} {pred_acc:>10.3f}",
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
    verdict = run_benchmark(RpcKmConfig())
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
