"""Unit tests for Continuous Spatial Wave Delay & Temporal Sequence Tracking.

Verifies:
1. Zero-shot physical delay reconstruction on novel, unseen broadband signals without any trained weights.
2. Space-time isomorphism (tau = x / v) and capacity bounds.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_temporal_tracker import ContinuousWaveDelayDendrite


def test_physical_wave_delay_reconstruction_unseen_broadband():
    """Verify that spatial wave propagation yields exact continuous delayed signals without training."""
    dendrite = ContinuousWaveDelayDendrite(length=10.0, n_nodes=100, velocity=2.0, damping=0.005)
    dt = 0.01

    # Completely unseen broadband multi-frequency signal
    def u_signal(t: float) -> float:
        return float(np.sin(0.8 * t) + 0.6 * np.cos(2.7 * t) + 0.3 * np.sin(5.5 * t))

    test_delays = [0.5, 1.5, 2.5]
    n_steps = 700

    recons = {tau: [] for tau in test_delays}
    truths = {tau: [] for tau in test_delays}

    for step in range(n_steps):
        t = step * dt
        val = u_signal(t)
        dendrite.step(val, dt=dt)

        if t >= 3.5:
            for tau in test_delays:
                recon = dendrite.tap(tau)
                true_val = u_signal(t - tau)
                recons[tau].append(recon)
                truths[tau].append(true_val)

    for tau in test_delays:
        r_arr = np.array(recons[tau])
        t_arr = np.array(truths[tau])
        corr = float(np.corrcoef(r_arr, t_arr)[0, 1])
        mse = float(np.mean((r_arr - t_arr) ** 2))

        # Must achieve >0.95 correlation and <0.05 MSE on completely novel signal with ZERO weights
        assert corr > 0.95
        assert mse < 0.05


def test_dendrite_cable_capacity_bounds():
    """Verify physical maximum delay capacity tau_max = L / v."""
    dendrite = ContinuousWaveDelayDendrite(length=12.0, n_nodes=60, velocity=3.0)
    assert dendrite.max_delay == pytest.approx(4.0, abs=1e-5)
