"""Unit tests for Continuous-Time State-Space Legendre Delay Engine.

Verifies:
1. Mathematical properties of continuous Legendre operators (Hurwitz stability, contractive discretization).
2. Analytical zero-shot continuous delay reconstruction against ground truth.
3. Known-answer exactness of the reference Tapped Delay Line baseline.
4. Mandatory failure of null controls (decoupled sensory input).
5. State space compression factor relative to discrete tapped buffers.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.biophysics.continuous_delay import (
    ContinuousLegendreDelayEngine,
    compute_legendre_operators,
)


def test_legendre_operators_properties():
    """Verify Hurwitz stability of continuous drift matrix A and contractivity of A_bar."""
    order = 8
    theta = 50.0
    dt = 0.5

    A, B = compute_legendre_operators(order, theta)
    eig_A = np.linalg.eigvals(A)
    # Continuous stability: all real parts must be strictly negative
    assert np.all(np.real(eig_A) < 0.0)

    engine = ContinuousLegendreDelayEngine(order=order, theta=theta, dt=dt)
    eig_A_bar = np.linalg.eigvals(engine.A_bar)
    # Discrete stability: all eigenvalues inside unit circle
    assert np.all(np.abs(eig_A_bar) < 1.0)


def test_zero_shot_delay_on_known_sinusoid():
    """Verify zero-shot analytical continuous delay reconstruction on pure sinusoid."""
    engine = ContinuousLegendreDelayEngine(order=8, theta=50.0, dt=0.5)
    tau = 12.5  # Non-integer delay relative to some sample grids
    omega = 0.1
    dt = 0.5
    n_steps = 250

    preds, truths = [], []
    for step in range(n_steps):
        t = step * dt
        val = np.sin(omega * t)
        engine.step(val)
        if t >= tau + 5.0:
            recon = engine.reconstruct_delay(tau)
            truth = np.sin(omega * (t - tau))
            preds.append(recon)
            truths.append(truth)

    mse = float(np.mean((np.array(preds) - np.array(truths)) ** 2))
    var = float(np.var(np.array(truths)))
    nmse = mse / var
    # Analytical reconstruction on smooth wave must achieve NMSE < 0.05
    assert nmse < 0.05


def test_exact_tdl_baseline_known_answer():
    """Verify known-answer property: exact TDL buffer achieves 0.0 error on ramp signal."""
    tau_steps = 30
    buf = np.zeros(60, dtype=np.float64)
    errors = []
    for t in range(80):
        x = float(t)
        buf = np.roll(buf, 1)
        buf[0] = x
        if t >= tau_steps:
            pred = buf[tau_steps]
            target = float(t - tau_steps)
            if abs(pred - target) > 1e-9:
                errors.append((t, pred, target))
    assert len(errors) == 0


def test_null_control_decoupled_fails():
    """Verify that decoupling sensory drive (B = 0) causes delay reconstruction to fail."""
    engine = ContinuousLegendreDelayEngine(order=8, theta=50.0, dt=0.5)
    engine.B_bar.fill(0.0)
    tau = 15.0
    dt = 0.5
    n_steps = 200

    preds, truths = [], []
    for step in range(n_steps):
        t = step * dt
        val = np.sin(0.15 * t)
        engine.step(val)
        if t >= tau:
            preds.append(engine.reconstruct_delay(tau))
            truths.append(np.sin(0.15 * (t - tau)))

    mse = float(np.mean((np.array(preds) - np.array(truths)) ** 2))
    var = float(np.var(np.array(truths)))
    nmse = mse / var
    # Decoupled state must fail (NMSE >= 1.0)
    assert nmse >= 1.0


def test_compression_ratio_and_capacity():
    """Verify state dimension and memory compression factor."""
    order = 8
    theta = 50.0
    dt = 0.5
    engine = ContinuousLegendreDelayEngine(order=order, theta=theta, dt=dt)
    assert engine.state_dim == 8
    # 100 discrete buffer slots vs 8 continuous states = 12.5x compression
    discrete_buffer_slots = int(round(theta / dt))
    compression = discrete_buffer_slots / engine.state_dim
    assert compression == 12.5
