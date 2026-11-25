import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.plasticity_engine import ThreeFactorPlasticityNetwork, BackpropBaselineNetwork


def make_dataset(seed: int = 42, n_train: int = 200, n_test: int = 100):
    rng = np.random.RandomState(seed)
    d_in, d_hid, d_out = 8, 16, 4
    w1_true = rng.randn(d_hid, d_in) * 0.5
    w2_true = rng.randn(d_out, d_hid) * 0.5

    def gen(n):
        X = rng.randn(n, d_in)
        H_true = np.tanh(X @ w1_true.T)
        Y = H_true @ w2_true.T + 0.15 * np.sin(2.0 * H_true @ w2_true.T) + rng.normal(0, 0.05, size=(n, d_out))
        return X, Y

    X_tr, Y_tr = gen(n_train)
    X_te, Y_te = gen(n_test)
    return X_tr, Y_tr, X_te, Y_te


def test_forward_dimensions():
    """Verify forward activation shapes and output dimensionality."""
    net = ThreeFactorPlasticityNetwork(d_in=8, d_hid=16, d_out=4)
    x = np.random.randn(8)
    h, y_hat = net.forward(x)
    assert h.shape == (16,)
    assert y_hat.shape == (4,)
    assert np.all(h >= -1.0) and np.all(h <= 1.0)


def test_convergence_without_backprop():
    """Verify network learns non-linear mapping using three-factor plasticity without backprop."""
    X_tr, Y_tr, X_te, Y_te = make_dataset(seed=101)
    net = ThreeFactorPlasticityNetwork(d_in=8, d_hid=16, d_out=4, eta=0.03, rng=np.random.RandomState(42))

    nmse_init = net.evaluate_nmse(X_te, Y_te)

    # Train for 50 epochs
    for epoch in range(50):
        for i in range(len(X_tr)):
            net.step_plasticity(X_tr[i], Y_tr[i], control="none")

    nmse_final = net.evaluate_nmse(X_te, Y_te)
    assert nmse_final < 0.10
    assert nmse_final < nmse_init * 0.15


def test_neuromodulatory_gating_prevents_drift():
    """Verify neuromodulator gating prevents catastrophic drift during uninformative noise."""
    X_tr, Y_tr, X_te, Y_te = make_dataset(seed=102)

    net_intact = ThreeFactorPlasticityNetwork(rng=np.random.RandomState(42))
    net_decoupled = ThreeFactorPlasticityNetwork(rng=np.random.RandomState(42))

    # Phase 1: Train both
    for epoch in range(50):
        for i in range(len(X_tr)):
            net_intact.step_plasticity(X_tr[i], Y_tr[i], control="none")
            net_decoupled.step_plasticity(X_tr[i], Y_tr[i], control="decoupled")

    nmse_pre_intact = net_intact.evaluate_nmse(X_te, Y_te)
    nmse_pre_decoupled = net_decoupled.evaluate_nmse(X_te, Y_te)
    assert nmse_pre_intact < 0.15
    assert nmse_pre_decoupled < 0.15

    # Phase 2: Exposure to uninformative noise drift
    rng_noise = np.random.RandomState(999)
    for _ in range(500):
        x_noise = rng_noise.randn(8) * 0.5
        y_noise = rng_noise.normal(0, 0.2, size=4)
        net_intact.step_plasticity(x_noise, y_noise, control="none", is_noise_drift=True)
        net_decoupled.step_plasticity(x_noise, y_noise, control="decoupled", is_noise_drift=True)

    nmse_post_intact = net_intact.evaluate_nmse(X_te, Y_te)
    nmse_post_decoupled = net_decoupled.evaluate_nmse(X_te, Y_te)

    # Intact network must retain performance (< 0.25)
    assert nmse_post_intact < 0.25
    # Decoupled network must suffer catastrophic drift (> 0.60)
    assert nmse_post_decoupled > 0.60
    assert nmse_post_decoupled > nmse_post_intact * 2.0


def test_shuffled_error_fails():
    """Verify shuffled error feedback destroys multi-layer credit assignment."""
    X_tr, Y_tr, X_te, Y_te = make_dataset(seed=103)
    net_shuffled = ThreeFactorPlasticityNetwork(rng=np.random.RandomState(42))

    for epoch in range(50):
        for i in range(len(X_tr)):
            net_shuffled.step_plasticity(X_tr[i], Y_tr[i], control="shuffled")

    nmse_shuffled = net_shuffled.evaluate_nmse(X_te, Y_te)
    assert nmse_shuffled > 0.50


def test_feedback_alignment_asymmetry():
    """Verify feedback projections B remain strictly asymmetric from W2.T."""
    X_tr, Y_tr, X_te, Y_te = make_dataset(seed=104)
    net = ThreeFactorPlasticityNetwork(rng=np.random.RandomState(42))

    for epoch in range(10):
        for i in range(len(X_tr)):
            net.step_plasticity(X_tr[i], Y_tr[i])

    # B is fixed and asymmetric (B != W2.T)
    assert not np.allclose(net.B, net.W2.T)
