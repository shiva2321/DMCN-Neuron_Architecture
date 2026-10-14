"""Multi-band dendrite benchmark (B-10): K-band RPN parity + band routing.

Validates Stage-4 multi-band machinery:
  - redundant K=3 matches monolithic single RPN on clean associations
  - sharded routing isolates writes per dendritic band

Run: python -m benchmarks.multiband_denoise
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.band_rpn import MultiBandRPN
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN


@dataclass
class BandConfig:
    D: int = 512
    M: int = 96
    seed: int = 42
    K: int = 3
    min_parity_acc: float = 0.98
    max_parity_gap: float = 0.02
    min_isolation_acc: float = 1.0


@dataclass
class BandResult:
    name: str
    accuracy: float


def _pairs(M: int) -> list[tuple[str, str]]:
    return [(f"ctx_{i}", f"tgt_{i}") for i in range(M)]


def _register(items: ItemTable, pairs: list[tuple[str, str]]) -> None:
    for c, t in pairs:
        _ = items[c]
        _ = items[t]


def _accuracy(model, pairs: list[tuple[str, str]]) -> float:
    ok = sum(1 for c, t in pairs if model.read(c)[0] == t)
    return ok / len(pairs)


def run_parity(cfg: BandConfig) -> tuple[BandResult, BandResult]:
    items = ItemTable(cfg.D, seed=cfg.seed)
    pairs = _pairs(cfg.M)
    _register(items, pairs)

    mb = MultiBandRPN(cfg.D, items, K=cfg.K, redundant=True)
    single = RPN(cfg.D, items)
    for ctx, tgt in pairs:
        mb.learn(ctx, tgt)
        single.learn(ctx, tgt)

    return (
        BandResult("multiband_redundant_k3", _accuracy(mb, pairs)),
        BandResult("single_rpn", _accuracy(single, pairs)),
    )


def run_isolation(cfg: BandConfig) -> float:
    """Each band owns disjoint ctx tokens; reads must stay routed."""
    items = ItemTable(cfg.D, seed=cfg.seed + 1)
    mb = MultiBandRPN(cfg.D, items, K=cfg.K, redundant=False)
    per_band = cfg.M // cfg.K
    pairs: list[tuple[str, str]] = []
    for band in range(cfg.K):
        for j in range(per_band):
            ctx = f"band{band}_ctx_{j}"
            tgt = f"band{band}_tgt_{j}"
            _ = items[ctx]
            _ = items[tgt]
            pairs.append((ctx, tgt))
            mb.learn(ctx, tgt, band=band)

    return _accuracy(mb, pairs)


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(
    parity: tuple[BandResult, BandResult],
    isolation_acc: float,
    cfg: BandConfig,
) -> Verdict:
    mb, single = parity
    gap = abs(mb.accuracy - single.accuracy)
    checks = [
        (
            f"redundant multiband >= {cfg.min_parity_acc:.2f} ({mb.accuracy:.3f})",
            mb.accuracy >= cfg.min_parity_acc,
        ),
        (
            f"single >= {cfg.min_parity_acc:.2f} ({single.accuracy:.3f})",
            single.accuracy >= cfg.min_parity_acc,
        ),
        (
            f"parity gap <= {cfg.max_parity_gap:.2f} ({gap:.3f})",
            gap <= cfg.max_parity_gap,
        ),
        (
            f"sharded isolation >= {cfg.min_isolation_acc:.2f} ({isolation_acc:.3f})",
            isolation_acc >= cfg.min_isolation_acc,
        ),
    ]
    lines = [
        f"Multi-band scorecard (D={cfg.D}, K={cfg.K}, M={cfg.M})",
        "",
        "Parity (redundant K-band vs single):",
        f"{'model':<24} {'accuracy':>10}",
        f"{mb.name:<24} {mb.accuracy:>10.3f}",
        f"{single.name:<24} {single.accuracy:>10.3f}",
        "",
        f"Sharded isolation accuracy: {isolation_acc:.3f}",
        "",
        "Pass/fail gates:",
    ]
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    return Verdict(passed=passed, lines=lines)


def run_benchmark(cfg: BandConfig) -> Verdict:
    parity = run_parity(cfg)
    isolation = run_isolation(cfg)
    return evaluate_verdict(parity, isolation, cfg)


def main() -> int:
    verdict = run_benchmark(BandConfig())
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
