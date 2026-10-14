"""Kuramoto dynamical mode benchmark (B-12): noisy read vs k=1 RPN.

Train clean ctx→tgt pairs; query with phase-noised contexts. Kuramoto k=4 should
beat plain RPN k=1 under noise (spec Stage 4 robustness).

Run: python -m benchmarks.kuramoto_denoise
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.kuramoto_rpn import KuramotoRPN
from neuron_v2.core.predictive_rpn import PredictiveRPN
from neuron_v2.core.rpn import RPN
from neuron_v2.substrate.phasor import normalise


@dataclass
class KuramotoConfig:
    D: int = 512
    M: int = 96
    seed: int = 42
    K: int = 3
    settle_k: int = 4
    coupling: float = 0.35
    noise_sigma: float = 2.12
    min_clean_acc: float = 0.95
    min_noisy_kuramoto: float = 0.82
    max_noisy_rpn: float = 0.65
    min_noisy_margin: float = 0.20
    min_vs_predictive_margin: float = -0.02


@dataclass
class ModelResult:
    name: str
    clean_acc: float
    noisy_acc: float


def _pairs(M: int) -> list[tuple[str, str]]:
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


def _accuracy(model, pairs: list[tuple[str, str]], *, noisy: bool, cfg: KuramotoConfig) -> float:
    rng = np.random.RandomState(cfg.seed + (1 if noisy else 0))
    ok = 0
    for ctx_t, tgt_t in pairs:
        ctx = model.items[ctx_t]
        if noisy:
            ctx = _phase_noise(ctx, cfg.noise_sigma, rng)
        if model.read(ctx_t, ctx_hv=ctx)[0] == tgt_t:
            ok += 1
    return ok / len(pairs)


def run_benchmark(cfg: KuramotoConfig) -> list[ModelResult]:
    items = ItemTable(cfg.D, seed=cfg.seed)
    pairs = _pairs(cfg.M)
    _register(items, pairs)

    km = KuramotoRPN(
        cfg.D,
        items,
        K=cfg.K,
        k=cfg.settle_k,
        coupling=cfg.coupling,
    )
    rpn = RPN(cfg.D, items)
    pred = PredictiveRPN(
        cfg.D,
        items,
        k=cfg.settle_k,
        gate_writes=False,
        ctx_prefix="ctx_",
        tgt_prefix="tgt_",
    )
    for ctx, tgt in pairs:
        km.learn(ctx, tgt)
        rpn.learn(ctx, tgt)
        pred.learn(ctx, tgt)

    return [
        ModelResult(
            "kuramoto_k4",
            _accuracy(km, pairs, noisy=False, cfg=cfg),
            _accuracy(km, pairs, noisy=True, cfg=cfg),
        ),
        ModelResult(
            "predictive_k4",
            _accuracy(pred, pairs, noisy=False, cfg=cfg),
            _accuracy(pred, pairs, noisy=True, cfg=cfg),
        ),
        ModelResult(
            "rpn_k1",
            _accuracy(rpn, pairs, noisy=False, cfg=cfg),
            _accuracy(rpn, pairs, noisy=True, cfg=cfg),
        ),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ModelResult], cfg: KuramotoConfig) -> Verdict:
    km, pred, rpn = results
    margin = km.noisy_acc - rpn.noisy_acc
    vs_pred = km.noisy_acc - pred.noisy_acc
    checks = [
        (f"kuramoto clean >= {cfg.min_clean_acc:.2f} ({km.clean_acc:.3f})", km.clean_acc >= cfg.min_clean_acc),
        (f"kuramoto noisy >= {cfg.min_noisy_kuramoto:.2f} ({km.noisy_acc:.3f})", km.noisy_acc >= cfg.min_noisy_kuramoto),
        (f"rpn noisy <= {cfg.max_noisy_rpn:.2f} ({rpn.noisy_acc:.3f})", rpn.noisy_acc <= cfg.max_noisy_rpn),
        (f"noisy margin vs rpn >= {cfg.min_noisy_margin:.2f} ({margin:.3f})", margin >= cfg.min_noisy_margin),
        (
            f"kuramoto vs predictive_k4 >= {cfg.min_vs_predictive_margin:.2f} ({vs_pred:.3f})",
            vs_pred >= cfg.min_vs_predictive_margin,
        ),
    ]
    lines = [
        f"Kuramoto denoise (D={cfg.D}, K={cfg.K}, k={cfg.settle_k}, "
        f"sigma={cfg.noise_sigma}, coupling={cfg.coupling})",
        "",
        f"{'model':<14} {'clean':>8} {'noisy':>8}",
        f"{km.name:<14} {km.clean_acc:>8.3f} {km.noisy_acc:>8.3f}",
        f"{pred.name:<14} {pred.clean_acc:>8.3f} {pred.noisy_acc:>8.3f}",
        f"{rpn.name:<14} {rpn.clean_acc:>8.3f} {rpn.noisy_acc:>8.3f}",
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
    verdict = evaluate_verdict(run_benchmark(KuramotoConfig()), KuramotoConfig())
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
