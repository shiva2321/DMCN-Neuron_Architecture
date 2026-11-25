"""Known-answer tests for the BASELINES used in published comparisons. GEMINI.md section 7:
"Unit-test every baseline against a case with a known answer."
"""
from __future__ import annotations

import numpy as np
import pytest
from controls import assert_gate
from benchmarks.rcpn_autonomous_timing import TappedDelayBaseline
from benchmarks.rcpn_autonomous_xor import TwoLayerMLP, get_xor_patterns


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-16 comparison INVALID (docs/CLAIMS.md): TappedDelayBaseline inserts every sample twice "
                          "when used as step() then learn() (the benchmark's own call pattern)")
def test_tapped_delay_baseline_returns_the_exact_lag_when_used_as_in_the_benchmark():
    tau = 20
    b = TappedDelayBaseline(buffer_size=50, lr=0.0)
    b.w[:] = 0.0
    b.w[tau] = 1.0                       # exact-lag weights: prediction must equal x[t - tau] on a ramp
    errors = []
    for t in range(60):
        x = float(t)
        pred = b.step(x)                 # benchmark call pattern: step(x) then learn(x, target)
        b.learn(x, 0.0)
        if t > tau + 2 and abs(pred - (t - tau)) > 1e-9:
            errors.append((t, pred, t - tau))
    assert_gate([f"{len(errors)} wrong predictions, e.g. t={errors[0][0]}: got {errors[0][1]}, expected {errors[0][2]}"]
                if errors else [])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-16 comparison INVALID (docs/CLAIMS.md): the XOR MLP baseline (4 ReLU hidden) fails to reach "
                          "100% even with a generous budget; a conventional tanh MLP does (see the passing test below)")
def test_xor_mlp_baseline_solves_xor_with_generous_budget():
    accs = []
    for trial in range(10):
        seed = 100 + trial
        data, _ = get_xor_patterns(128, seed)
        m = TwoLayerMLP(seed=seed)
        for _ in range(3000):
            for chs, tgt in data:
                m.learn(chs, tgt)
        accs.append(sum(m.predict(c) == t for c, t in data) / len(data))
    mean = float(np.mean(accs))
    assert_gate([] if mean >= 0.99 else
                [f"baseline MLP mean XOR accuracy {mean:.1%} after 3000 epochs (known solvable: 100%)"])


def test_reference_conventional_mlp_solves_xor():
    """Passing reference: a conventionally configured MLP gets 100% on XOR on every seed (so the baseline above is wrong)."""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    y = np.array([0, 1, 1, 0], float)
    for trial in range(10):
        rng = np.random.RandomState(100 + trial)
        W1, b1, W2, b2 = rng.normal(0, 1, (2, 8)), np.zeros(8), rng.normal(0, 1, 8), 0.0
        for _ in range(3000):
            h = np.tanh(X @ W1 + b1)
            p = 1 / (1 + np.exp(-(h @ W2 + b2)))
            g = p - y
            gh = np.outer(g, W2) * (1 - h ** 2)
            W1 -= 0.3 * (X.T @ gh)
            b1 -= 0.3 * gh.sum(0)
            W2 -= 0.3 * (h.T @ g)
            b2 -= 0.3 * g.sum()
        p = 1 / (1 + np.exp(-(np.tanh(X @ W1 + b1) @ W2 + b2)))
        assert np.all((p > 0.5) == y)


def test_reference_exact_lag_tapped_delay_is_exact():
    """Passing reference: a correctly implemented tapped delay line with the exact tap has zero error."""
    tau, buf = 20, np.zeros(50)
    for t in range(80):
        buf = np.roll(buf, 1)
        buf[0] = float(t)
        if t > tau:
            assert buf[tau] == t - tau
