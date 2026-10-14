"""Unit tests for Log-Cochlear Multi-Scale Resonator Bank."""
import numpy as np
import pytest

from neuron_v2.core.hpr import LogCochlearResonatorBank


def test_log_cochlear_resonator_properties():
    res = LogCochlearResonatorBank(K=32, omega_min=1e-3, omega_max=10.0, dt=0.5, seed=42)

    assert len(res.omegas) == 32
    assert res.omegas[0] == pytest.approx(1e-3, rel=1e-4)
    assert res.omegas[-1] == pytest.approx(10.0, rel=1e-4)

    # Unitary magnitude strictly <= 1.0 (no explosion)
    assert np.all(np.abs(res.U) <= 1.0)


def test_log_cochlear_resonator_multi_octave_delay():
    dt = 0.5
    res = LogCochlearResonatorBank(K=64, omega_min=1e-3, omega_max=5.0, dt=dt, lr=0.05, seed=42)

    # Test on frequency omega = 0.50 rad/s
    t = np.arange(1000, dtype=np.float64) * dt
    x = np.sin(0.50 * t)
    tau = 10.0

    preds = []
    targets = []
    for step in range(len(x)):
        res.step(x[step])
        pred = res.predict_delay()
        if step >= int(tau / dt):
            target = x[step - int(tau / dt)]
            res.learn_delay(target)
            preds.append(pred)
            targets.append(target)

    mse = float(np.mean((np.array(preds[-200:]) - np.array(targets[-200:])) ** 2))
    assert mse < 0.005  # High-fidelity tracking across octaves
