import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.structural_plasticity import StructuralPlasticityNetwork, TimerBasedBaselineNetwork


def make_stationary_stream(seed: int = 42, n_samples: int = 300):
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    W = rng.randn(d_out, d_in) * 0.5
    X = rng.randn(n_samples, d_in)
    Y = X @ W.T + rng.normal(0, 0.05, size=(n_samples, d_out))
    return X, Y


def make_noise_stream(seed: int = 43, n_samples: int = 300):
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    X = rng.randn(n_samples, d_in)
    Y = rng.randn(n_samples, d_out) * 1.0  # Irreducible white noise
    return X, Y


def make_shift_stream(seed: int = 44, n_samples: int = 600):
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    X = np.zeros((n_samples, d_in))
    Y = np.zeros((n_samples, d_out))
    for k in range(3):
        start = k * 200
        end = (k + 1) * 200
        w_k = rng.randn(d_out, d_in) * (0.8 + 0.3 * k)
        X[start:end] = rng.randn(200, d_in) + (k * 2.0)
        Y[start:end] = np.tanh(X[start:end] @ w_k.T) + rng.normal(0, 0.05, size=(200, d_out))
    return X, Y


def test_zero_false_positive_on_stationary_data():
    """Verify strictly zero false-positive neurogenesis on stationary data."""
    X, Y = make_stationary_stream(seed=101)
    net = StructuralPlasticityNetwork(d_in=8, d_initial_hid=4, d_out=4, rng=np.random.RandomState(42))

    res = net.process_stream(X, Y)
    assert res["n_spawn"] == 0
    assert res["final_hid"] == 4


def test_zero_false_positive_on_white_noise():
    """Verify strictly zero false-positive neurogenesis on irreducible white noise."""
    X, Y = make_noise_stream(seed=102)
    net = StructuralPlasticityNetwork(d_in=8, d_initial_hid=4, d_out=4, rng=np.random.RandomState(42))

    res = net.process_stream(X, Y)
    assert res["n_spawn"] == 0
    assert res["final_hid"] == 4


def test_true_positive_growth_under_distribution_shift():
    """Verify adaptive neurogenesis triggers under systematic distribution shifts."""
    X, Y = make_shift_stream(seed=103)
    net = StructuralPlasticityNetwork(d_in=8, d_initial_hid=4, d_out=4, rng=np.random.RandomState(42))

    res = net.process_stream(X, Y)
    assert res["n_spawn"] in [2, 3, 4]
    assert res["final_hid"] == 4 + res["n_spawn"]


def test_metabolic_synaptic_pruning():
    """Verify synaptic connections with magnitude below threshold AND depleted utility are excised."""
    net = StructuralPlasticityNetwork(d_in=8, d_initial_hid=4, d_out=4, prune_threshold=0.01, utility_threshold=0.01)
    net.W1[0, 0] = 0.002  # Below weight threshold
    net.U1[0, 0] = 0.001  # Depleted utility tag
    net.W1[0, 1] = 0.05   # Above weight threshold

    net.prune_synapses()

    # Uninformative, sub-threshold synapse must be zeroed
    assert net.W1[0, 0] == 0.0
    assert net.W1[0, 1] == 0.05


def test_rare_critical_synapse_protected_by_utility_tag():
    """Verify rarely activating but critical synapses are protected from pruning by STC utility tags."""
    net = StructuralPlasticityNetwork(d_in=8, d_initial_hid=4, d_out=4, prune_threshold=0.01, utility_threshold=0.01)
    net.W1[0, 0] = 0.002  # Below conductance threshold
    net.U1[0, 0] = 0.25   # High functional utility tag from past critical surprise event

    net.prune_synapses()

    # Protected synapse must NOT be excised despite small static weight magnitude
    assert net.W1[0, 0] == 0.002


def test_timer_baseline_falsely_spawns():
    """Verify legacy timer-based baseline falsely spawns on stationary and noise data."""
    X_stat, Y_stat = make_stationary_stream(seed=104)
    timer_net = TimerBasedBaselineNetwork(d_in=8, d_initial_hid=4, d_out=4, spawn_period=50)

    res = timer_net.process_stream(X_stat, Y_stat)
    assert res["n_spawn"] == 5  # 50, 100, 150, 200, 250
    assert res["final_hid"] == 9
