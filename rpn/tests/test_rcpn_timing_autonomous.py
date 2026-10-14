"""Tests for Autonomous Continuous Delay Discovery in RCPN."""
import numpy as np
import pytest

from neuron_v2.core.rcpn import ContinuousTimeDelay
from benchmarks.rcpn_autonomous_timing import generate_signal


def test_autonomous_delay_learning():
    """Verify that RCPN learns to reconstruct continuous delays with MSE < 0.05."""
    signal = generate_signal(n_steps=200, seed=42)
    delay_engine = ContinuousTimeDelay(order=6, theta=50.0, dt=1.0)
    tau = 15

    preds = []
    targets = []
    for t in range(200):
        x = signal[t]
        delay_engine.step(x)
        if t >= tau:
            tgt = signal[t - tau]
            targets.append(tgt)
            pred = delay_engine.learn_readout(tgt, lr=0.08)
            preds.append(pred)

    # Evaluate over second half after convergence
    half = len(targets) // 2
    mse = np.mean((np.array(targets[half:]) - np.array(preds[half:])) ** 2)
    assert mse < 0.07, f"Autonomous delay learning MSE too high: {mse}"
