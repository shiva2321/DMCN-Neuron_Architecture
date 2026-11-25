"""Generic null-control gate library (GEMINI.md §5, §6, §7). Standard library only.

Domain-specific controls (what an "untrained", "blind" or "heuristic" system *is* in your project) live in
your project's tests. This module supplies the project-independent pieces: statistics, seed sweeps, and
gate predicates that return a list of human-readable failures (empty list == gate passes).

    from controls import Rate, rate, gate_beats_controls, gate_depends_on_inputs, assert_gate

    system = rate(successes_of_system)                       # list[bool]
    controls = {"untrained": rate(...), "constant": rate(...), "heuristic": rate(...)}
    assert_gate(gate_beats_controls(system, controls, min_margin=0.2))
    assert_gate(gate_depends_on_inputs(system, rate(blind_successes)))

Pair domain tests with ``@pytest.mark.xfail(strict=True, reason="<ledger id> REFUTED")`` while a claim is
refuted: the test then flips to a hard failure (XPASS strict) the moment the claim genuinely survives, which
forces a ledger update instead of a silent change.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

MIN_SEEDS = 20


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.k, self.n)

    def __str__(self) -> str:
        lo, hi = self.ci
        return f"{self.k}/{self.n} ({self.p:.0%}, 95% CI {lo:.0%}-{hi:.0%})"


def rate(outcomes: Iterable[bool]) -> Rate:
    xs = [bool(o) for o in outcomes]
    return Rate(sum(xs), len(xs))


def seed_sweep(fn: Callable[[int], float | bool], seeds: Sequence[int]) -> list:
    """Run ``fn(seed)`` for each seed. ``fn`` must seed all of its own randomness from ``seed``."""
    return [fn(int(s)) for s in seeds]


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def _t_crit(df: int) -> float:
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    for a, b in zip(keys, keys[1:]):
        if a < df < b:
            return _T975[a] + (_T975[b] - _T975[a]) * (df - a) / (b - a)
    return 1.96 if df > keys[-1] else _T975[keys[0]]


def summarize(values: Sequence[float]) -> dict:
    """mean, sd, 95% CI (t-based), min, max, n — report these, never a single run."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = _t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95": (mean - half, mean + half),
            "min": min(values), "max": max(values)}


def permutation_pvalue(
    score_fn: Callable[[Sequence, Sequence], float],
    y_true: Sequence,
    y_pred: Sequence,
    n_perm: int = 1000,
    seed: int = 0,
) -> float:
    """Shuffled-label control: P(score on shuffled labels >= observed). Small p == not chance."""
    rng = random.Random(seed)
    observed = score_fn(y_true, y_pred)
    labels = list(y_true)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(labels)
        if score_fn(labels, y_pred) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------- gate predicates
def gate_min_seeds(n_seeds: int, minimum: int = MIN_SEEDS) -> list[str]:
    return [] if n_seeds >= minimum else [f"only {n_seeds} seeds (< {minimum}); label the claim OBSERVED at most"]


def gate_beats_controls(system: Rate, controls: dict[str, Rate], min_margin: float = 0.2) -> list[str]:
    """System must beat EVERY control by ``min_margin`` (absolute, on success probability)."""
    failures = []
    for name, c in controls.items():
        if c.p + min_margin > system.p:
            failures.append(f"control '{name}' {c} matches/approaches system {system} (needs margin {min_margin:.0%})")
    return failures


def gate_depends_on_inputs(sighted: Rate, blind: Rate, max_ratio: float = 0.5) -> list[str]:
    """Blind (inputs zeroed/constant) success must be well below sighted success."""
    if sighted.p <= 0:
        return [f"sighted system never succeeds ({sighted}); nothing to test"]
    if blind.p > max_ratio * sighted.p:
        return [f"blind success {blind} > {max_ratio:.0%} of sighted {sighted}: system does not need its inputs"]
    return []


def gate_absent_in_null(null_value: float, threshold: float, what: str = "effect") -> list[str]:
    """An 'emergent' effect must be absent (< threshold) in the fresh/untrained/scrambled system."""
    if null_value >= threshold:
        return [f"{what} already present in null system ({null_value:.3g} >= {threshold:.3g})"]
    return []


def gate_metric_improved(initial: float, final: float, higher_is_better: bool, min_rel_change: float = 0.10) -> list[str]:
    """Task metric (not an internal proxy) must improve by at least ``min_rel_change``."""
    change = (final - initial) if higher_is_better else (initial - final)
    base = abs(initial) if initial else 1.0
    if change / base < min_rel_change:
        return [f"task metric improved by {change / base:+.1%} (< {min_rel_change:.0%}): initial={initial:.4g}, final={final:.4g}"]
    return []


def gate_not_worse_than_baseline(system_error: float, baseline_error: float, what: str = "baseline") -> list[str]:
    if system_error > baseline_error:
        return [f"system error {system_error:.4g} is worse than {what} {baseline_error:.4g}"]
    return []


def assert_gate(failures: Sequence[str]) -> None:
    if failures:
        raise AssertionError("integrity gate failed:\n  - " + "\n  - ".join(failures))
