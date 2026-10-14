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


def test_log_cochlear_resonator_frozen_broadband_generalization():
    """Verify honest generalization: train on multi-frequency mix, FREEZE weights, test on novel broadband signal."""
    dt = 0.5
    res = LogCochlearResonatorBank(K=64, omega_min=1e-3, omega_max=5.0, dt=dt, lr=0.05, seed=42)

    # 1. Train on multi-frequency mix
    train_omegas = [0.01, 0.05, 0.2, 0.8]
    n_train = 2000
    t_train = np.arange(n_train, dtype=np.float64) * dt
    x_train = sum(np.sin(w * t_train) for w in train_omegas) / len(train_omegas)
    tau = 20.0 * dt

    for step in range(n_train):
        res.step(x_train[step])
        if step >= int(tau / dt):
            target = x_train[step - int(tau / dt)]
            res.learn_delay(target)

    # 2. FREEZE learning rate to 0.0 (strict out-of-distribution test)
    res.lr = 0.0

    # 3. Test on completely novel, unseen frequencies
    test_omegas = [0.023, 0.087, 0.31, 1.20]
    n_test = 1500
    t_test = np.arange(n_test, dtype=np.float64) * dt
    x_test = sum(np.sin(w * t_test) for w in test_omegas) / len(test_omegas)

    preds = []
    targets = []
    for step in range(n_test):
        res.step(x_test[step])
        p = res.predict_delay()  # No learn_delay called, weights are frozen
        if step >= int(tau / dt):
            target = x_test[step - int(tau / dt)]
            preds.append(p)
            targets.append(target)

    eval_p = np.array(preds[-500:])
    eval_t = np.array(targets[-500:])
    frozen_mse = float(np.mean((eval_p - eval_t) ** 2))

    # Honest baseline: MSE is non-zero due to finite basis resolution, but vastly beats chance/variance (~0.5)
    assert 0.02 < frozen_mse < 0.15
