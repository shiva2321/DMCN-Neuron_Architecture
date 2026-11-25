"""TEMPLATE: null-control gate tests for a claim. Copy into your project's tests/integrity/ and adapt.

Pattern (GEMINI.md section 5): for every claim, write tests that run
  1. the system,
  2. the dumbest things that could also produce the number (untrained, majority, constant, heuristic, shuffled),
  3. a blind/shuffled-input version of the system,
and assert the system beats ALL controls and needs its inputs.
While a claim is refuted, mark the test:  @pytest.mark.xfail(strict=True, reason="<ledger id> REFUTED")
so that a silent change (XPASS) breaks the build and forces a ledger update.

This example is a toy classifier so it runs anywhere; replace `system`, `controls` and the data for your domain.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


def _find_integrity() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".integrity" / "controls.py").exists():
            return p / ".integrity"
        if p.name == ".integrity" and (p / "controls.py").exists():
            return p
        if (p / "install.py").exists() and (p / "tools" / "controls.py").exists():  # running inside the kit
            return p / "tools"
    raise RuntimeError(".integrity/controls.py not found")


sys.path.insert(0, str(_find_integrity()))
from controls import (  # noqa: E402
    assert_gate, gate_beats_controls, gate_depends_on_inputs, gate_min_seeds, rate, seed_sweep, summarize,
)

SEEDS = list(range(1000, 1020))  # >= 20 disjoint TEST seeds (tuning used other seeds)


def make_data(seed: int, n: int = 200):
    rng = random.Random(seed)
    xs = [rng.random() for _ in range(n)]
    ys = [int(x > 0.5) if rng.random() > 0.1 else int(x <= 0.5) for x in xs]  # 10% label noise
    return xs, ys


def system(xs, threshold=0.5):
    return [int(x > threshold) for x in xs]


def accuracy_ok(pred, ys) -> bool:
    return sum(p == y for p, y in zip(pred, ys)) / len(ys) > 0.7  # absolute, pre-registered criterion


def test_system_beats_every_null_control():
    sys_hits, maj_hits, rnd_hits = [], [], []
    for seed in SEEDS:
        xs, ys = make_data(seed)
        sys_hits.append(accuracy_ok(system(xs), ys))
        majority = int(sum(ys) * 2 >= len(ys))
        maj_hits.append(accuracy_ok([majority] * len(ys), ys))
        rng = random.Random(seed)
        rnd_hits.append(accuracy_ok([rng.randint(0, 1) for _ in ys], ys))
    assert_gate(gate_min_seeds(len(SEEDS)))
    assert_gate(gate_beats_controls(rate(sys_hits), {"majority": rate(maj_hits), "random": rate(rnd_hits)}))


def test_system_needs_its_inputs():
    """Blind control: shuffle the inputs. A real system collapses; a label-leaking one does not."""
    sighted, blind = [], []
    for seed in SEEDS:
        xs, ys = make_data(seed)
        sighted.append(accuracy_ok(system(xs), ys))
        shuffled = xs[:]
        random.Random(seed + 1).shuffle(shuffled)
        blind.append(accuracy_ok(system(shuffled), ys))
    assert_gate(gate_depends_on_inputs(rate(sighted), rate(blind)))


def test_gate_catches_a_label_leaking_cheater():
    """The cheater reads the label (an oracle). It scores well, but the blind control exposes it."""
    sighted, blind = [], []
    for seed in SEEDS:
        xs, ys = make_data(seed)
        cheat = lambda _xs, _ys=ys: list(_ys)  # noqa: E731  (reads the answer, ignores xs)
        sighted.append(accuracy_ok(cheat(xs), ys))
        shuffled = xs[:]
        random.Random(seed + 1).shuffle(shuffled)
        blind.append(accuracy_ok(cheat(shuffled), ys))
    with pytest.raises(AssertionError):
        assert_gate(gate_depends_on_inputs(rate(sighted), rate(blind)))


def test_report_distribution_not_a_single_run():
    scores = seed_sweep(lambda s: sum(p == y for p, y in zip(system(make_data(s)[0]), make_data(s)[1])) / 200, SEEDS)
    stats = summarize(scores)
    assert stats["n"] >= 20 and stats["ci95"][0] > 0.7
