"""Predictive loop benchmark (B-5): k>1 settling + surprise-gated write.

Scenarios
---------
1. Noisy read: train clean, query with phase-noised contexts — k=4 vs k=1.
2. Write efficiency: stream repeated epochs — surprise gate skips redundant writes.

Run: python -m benchmarks.predictive_loop
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.predictive_rpn import PredictiveRPN
from neuron_v2.core.rpn import RPN
from neuron_v2.substrate.phasor import normalise


@dataclass
class LoopConfig:
    D: int = 512
    M: int = 96
    seed: int = 42
    noise_sigma: float = 2.12
    settle_k: int = 4
    surprise_threshold: float = 0.05
    stream_epochs: int = 8
    min_noisy_margin: float = 0.20
    min_clean_acc: float = 0.95
    min_write_reduction: float = 0.75


@dataclass
class LoopResult:
    name: str
    clean_acc: float
    noisy_acc: float
    n_writes: int
    n_skipped: int


def _make_pairs(M: int, _rng: np.random.RandomState) -> list[tuple[str, str]]:
    """Disjoint ctx_i -> tgt_i pairs (prefixes enable ctx-key denoise in settling)."""
    return [(f"ctx_{i}", f"tgt_{i}") for i in range(M)]


def _register(items: ItemTable, pairs: list[tuple[str, str]]) -> None:
    for c, t in pairs:
        _ = items[c]
        _ = items[t]


def _phase_noise(hv: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    phases = np.angle(hv).astype(np.float32) + rng.normal(0.0, sigma, hv.shape[0]).astype(
        np.float32
    )
    return normalise(np.exp(1j * phases).astype(np.complex64))


def _clean_accuracy(model, pairs: list[tuple[str, str]]) -> float:
    ok = sum(1 for c, t in pairs if model.read(c)[0] == t)
    return ok / len(pairs)


def _noisy_accuracy(
    model,
    items: ItemTable,
    pairs: list[tuple[str, str]],
    sigma: float,
    rng: np.random.RandomState,
) -> float:
    ok = 0
    for c, t in pairs:
        noisy = _phase_noise(items[c], sigma, rng)
        if model.read(c, ctx_hv=noisy)[0] == t:
            ok += 1
    return ok / len(pairs)


def _train_stream(model, pairs: list[tuple[str, str]], epochs: int) -> None:
    for _ in range(epochs):
        for c, t in pairs:
            if isinstance(model, PredictiveRPN):
                model.learn(c, t)
            else:
                model.learn(c, t)


def _write_counts(model) -> tuple[int, int]:
    if isinstance(model, PredictiveRPN):
        return model.n_writes_applied, model.n_writes_skipped
    mem = model.mem
    return mem.n_writes, 0


def run_benchmark(cfg: LoopConfig) -> list[LoopResult]:
    rng = np.random.RandomState(cfg.seed)
    pairs = _make_pairs(cfg.M, rng)

    def _items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        _register(items, pairs)
        return items

    results: list[LoopResult] = []

    rpn = RPN(cfg.D, _items(), mode="delta")
    _train_stream(rpn, pairs, epochs=1)
    noise_rng = np.random.RandomState(cfg.seed + 17)
    results.append(
        LoopResult(
            "rpn_k1",
            _clean_accuracy(rpn, pairs),
            _noisy_accuracy(rpn, rpn.items, pairs, cfg.noise_sigma, noise_rng),
            *_write_counts(rpn),
        )
    )

    pred = PredictiveRPN(
        cfg.D,
        _items(),
        mode="delta",
        k=1,
        surprise_threshold=cfg.surprise_threshold,
        gate_writes=False,
    )
    _train_stream(pred, pairs, epochs=1)
    noise_rng = np.random.RandomState(cfg.seed + 17)
    results.append(
        LoopResult(
            "predictive_k1",
            _clean_accuracy(pred, pairs),
            _noisy_accuracy(pred, pred.items, pairs, cfg.noise_sigma, noise_rng),
            *_write_counts(pred),
        )
    )

    pred_k = PredictiveRPN(
        cfg.D,
        _items(),
        mode="delta",
        k=cfg.settle_k,
        surprise_threshold=cfg.surprise_threshold,
        gate_writes=False,
    )
    _train_stream(pred_k, pairs, epochs=1)
    noise_rng = np.random.RandomState(cfg.seed + 17)
    results.append(
        LoopResult(
            f"predictive_k{cfg.settle_k}",
            _clean_accuracy(pred_k, pairs),
            _noisy_accuracy(pred_k, pred_k.items, pairs, cfg.noise_sigma, noise_rng),
            *_write_counts(pred_k),
        )
    )

    stream_total = cfg.stream_epochs * cfg.M
    rpn_stream = RPN(cfg.D, _items(), mode="delta")
    _train_stream(rpn_stream, pairs, epochs=cfg.stream_epochs)
    pred_stream = PredictiveRPN(
        cfg.D,
        _items(),
        mode="delta",
        k=cfg.settle_k,
        surprise_threshold=cfg.surprise_threshold,
        gate_writes=True,
    )
    _train_stream(pred_stream, pairs, epochs=cfg.stream_epochs)

    noise_rng = np.random.RandomState(cfg.seed + 17)
    results.append(
        LoopResult(
            "rpn_stream",
            _clean_accuracy(rpn_stream, pairs),
            _noisy_accuracy(rpn_stream, rpn_stream.items, pairs, cfg.noise_sigma, noise_rng),
            *_write_counts(rpn_stream),
        )
    )
    noise_rng = np.random.RandomState(cfg.seed + 17)
    results.append(
        LoopResult(
            f"predictive_k{cfg.settle_k}_gated",
            _clean_accuracy(pred_stream, pairs),
            _noisy_accuracy(pred_stream, pred_stream.items, pairs, cfg.noise_sigma, noise_rng),
            *_write_counts(pred_stream),
        )
    )

    _ = stream_total  # documented expected upper bound for rpn_stream writes
    return results


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[LoopResult], cfg: LoopConfig) -> Verdict:
    by = {r.name: r for r in results}
    k1 = by["rpn_k1"]
    ksettle = by[f"predictive_k{cfg.settle_k}"]
    gated = by[f"predictive_k{cfg.settle_k}_gated"]
    stream = by["rpn_stream"]

    stream_total = cfg.stream_epochs * cfg.M
    write_reduction = 1.0 - (gated.n_writes / max(stream.n_writes, 1))

    checks: list[tuple[str, bool]] = [
        (
            f"settle noisy margin >= {cfg.min_noisy_margin:.2f} "
            f"({ksettle.noisy_acc - k1.noisy_acc:.3f})",
            ksettle.noisy_acc - k1.noisy_acc >= cfg.min_noisy_margin,
        ),
        (
            f"settle noisy > k1 ({ksettle.noisy_acc:.3f} > {k1.noisy_acc:.3f})",
            ksettle.noisy_acc > k1.noisy_acc,
        ),
        (
            f"clean acc >= {cfg.min_clean_acc:.2f} ({ksettle.clean_acc:.3f})",
            ksettle.clean_acc >= cfg.min_clean_acc,
        ),
        (
            f"write reduction >= {cfg.min_write_reduction:.2f} ({write_reduction:.3f})",
            write_reduction >= cfg.min_write_reduction,
        ),
        (
            f"gated stream acc >= {cfg.min_clean_acc:.2f} ({gated.clean_acc:.3f})",
            gated.clean_acc >= cfg.min_clean_acc,
        ),
        (
            f"gated writes < stream ({gated.n_writes} < {stream.n_writes})",
            gated.n_writes < stream.n_writes,
        ),
    ]

    lines: list[str] = []
    lines.append(
        f"Predictive loop benchmark (M={cfg.M}, D={cfg.D}, k={cfg.settle_k}, "
        f"noise_sigma={cfg.noise_sigma})"
    )
    lines.append("")
    lines.append(f"{'model':<24} {'clean':>8} {'noisy':>8} {'writes':>8} {'skipped':>8}")
    for r in results:
        lines.append(
            f"{r.name:<24} {r.clean_acc:>8.3f} {r.noisy_acc:>8.3f} "
            f"{r.n_writes:>8} {r.n_skipped:>8}"
        )
    lines.append("")
    lines.append(f"Stream upper bound (rpn): {stream_total} writes")
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - k>1 settling: ctx-key denoise + target snap refines noisy probes.")
    lines.append("  - Surprise gate: skips writes when prediction already matches target.")
    lines.append("  - k=1 predictive without settling matches plain RPN on clean reads.")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = LoopConfig()
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
