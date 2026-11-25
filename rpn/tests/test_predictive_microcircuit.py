"""Unit tests for Canonical 3-Layer Cortical Predictive Microcircuit (Track 4).

Verifies:
1. Predictive Silencing: prediction error epsilon(t) drops toward zero on predictable input.
2. Sensory Surprise: sudden frequency shift triggers immediate sharp prediction error spike.
3. Decoupled Feedback: disconnecting top-down prediction leaves unmitigated error epsilon = y.
4. Numerical Stability: internal state space remains bounded and contractive.
"""
from __future__ import annotations

import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.biophysics.predictive_microcircuit import PredictiveCorticalMicrocircuit


def test_predictive_silencing_error_cancellation():
    """Verify that prediction error cancels toward zero on predictable input."""
    col = PredictiveCorticalMicrocircuit(order=8, theta=40.0, dt=0.5, lr=0.05)
    dt = 0.5
    n_steps = 300
    t = np.arange(n_steps) * dt
    y_wave = np.sin(0.2 * t)

    early_errors = []
    late_errors = []
    for step in range(n_steps):
        res = col.step(y_wave[step])
        if step < 50:
            early_errors.append(abs(res["error"]))
        elif step > 200:
            late_errors.append(abs(res["error"]))

    mean_early = float(np.mean(early_errors))
    mean_late = float(np.mean(late_errors))
    # Late error must be at least 5x smaller than early error
    assert mean_late < 0.20 * mean_early
    assert mean_late < 0.05


def test_sensory_surprise_response():
    """Verify that an unexpected sensory deviation triggers an immediate error spike."""
    col = PredictiveCorticalMicrocircuit(order=8, theta=40.0, dt=0.5, lr=0.05)
    dt = 0.5
    n_steps = 300
    t = np.arange(n_steps) * dt

    # Predictable wave up to step 200, then sharp anomalous spike at step 200
    y_wave = np.sin(0.15 * t)
    y_wave[200] += 3.0

    errors = []
    for step in range(250):
        res = col.step(y_wave[step])
        errors.append(abs(res["error"]))

    baseline_error = float(np.mean(errors[150:200]))
    surprise_error = float(errors[200])
    # Anomaly must produce an error spike at least 5x larger than baseline
    assert surprise_error > 5.0 * baseline_error


def test_decoupled_feedback_null_control():
    """Verify that decoupled top-down feedback leaves unmitigated error epsilon = y."""
    col = PredictiveCorticalMicrocircuit(order=8, theta=40.0, dt=0.5, lr=0.05)
    dt = 0.5
    n_steps = 100
    t = np.arange(n_steps) * dt
    y_wave = np.sin(0.2 * t)

    for step in range(n_steps):
        # Override top-down prediction to zero
        res = col.step(y_wave[step], top_down_override=0.0, learn=False)
        assert res["prediction"] == 0.0
        assert res["error"] == pytest.approx(y_wave[step], abs=1e-8)


def test_state_space_continuity():
    """Verify that internal Layer 5 state remains strictly bounded."""
    col = PredictiveCorticalMicrocircuit(order=8, theta=40.0, dt=0.5, lr=0.05)
    n_steps = 500
    rng = np.random.RandomState(42)

    for _ in range(n_steps):
        val = float(rng.normal(0.0, 1.0))
        res = col.step(val)
        m = res["l5_state"]
        assert np.all(np.isfinite(m))
        assert np.all(np.abs(m) < 50.0)
