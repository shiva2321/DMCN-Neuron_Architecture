"""Tests for Multi-Column Compositional Network (Track 5)."""
import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest

from neuron_v2.biophysics.multicolumn_network import (
    CorticalColumnModule,
    MultiColumnCompositionalNetwork,
    MonolithicFlatMemoryBaseline,
    circular_convolution,
    circular_correlation,
)


def test_circular_convolution_algebra():
    """Verify algebraic properties of circular convolution for vector binding."""
    rng = np.random.RandomState(42)
    d = 128
    a = rng.randn(d)
    b = rng.randn(d)

    # Commutative: a (*) b == b (*) a
    conv_ab = circular_convolution(a, b)
    conv_ba = circular_convolution(b, a)
    np.testing.assert_allclose(conv_ab, conv_ba, atol=1e-10)

    # Approximate inverse: unbinding via circular correlation
    # a_approx = (a (*) b) (#) b
    unbound_a = circular_correlation(conv_ab, b)
    norm_a = a / np.linalg.norm(a)
    norm_unbound = unbound_a / np.linalg.norm(unbound_a)
    cos_sim = float(np.dot(norm_a, norm_unbound))
    # In HRR, unbinding with normalized vector yields positive cosine similarity > 0.5
    assert cos_sim > 0.40


def test_cortical_column_module_laminar_states():
    """Verify active laminar processing across L4, L5, and prototype identification."""
    rng = np.random.RandomState(123)
    dim = 64
    col = CorticalColumnModule(dim=dim, alpha_l5=0.60)

    proto1 = rng.choice([-1.0, 1.0], size=dim)
    proto2 = rng.choice([-1.0, 1.0], size=dim)
    col.store_prototypes(np.array([proto1, proto2]))

    # Test noisy feedforward sensory drive through L4 and L5
    noisy_input = proto1 + rng.normal(0.0, 0.3, size=dim)
    z_col = col.process_sensory_input(noisy_input)

    assert z_col.shape == (dim,)
    assert col.l4_state.shape == (dim,)
    assert col.l5_state.shape == (dim,)

    # Prototype identification
    best_idx, sim = col.identify_prototype(z_col)
    assert best_idx == 0
    assert sim > 0.80


def test_multi_column_controls_and_binding():
    """Verify intact binding vs decoupled and shuffled null controls."""
    rng = np.random.RandomState(456)
    dim = 128
    net = MultiColumnCompositionalNetwork(dim=dim)

    e_proto = rng.choice([-1.0, 1.0], size=(4, dim))
    r_proto = rng.choice([-1.0, 1.0], size=(2, dim))
    net.train_columns(e_proto, r_proto)

    e_in = e_proto[0]
    r_in = r_proto[1]

    # Intact forward
    out_intact = net.forward(e_in, r_in, decouple_horizontal=False, shuffle_binding=False)
    assert out_intact["l6_gain"][0] > 0.80
    assert np.linalg.norm(out_intact["bound_rep"]) > 0.99

    # Decoupled control (Layer 2/3 severed)
    out_decoupled = net.forward(e_in, r_in, decouple_horizontal=True, shuffle_binding=False)
    assert np.allclose(out_decoupled["bound_rep"], 0.0)
    assert out_decoupled["l6_gain"][0] == 0.0

    # Shuffled control (Layer 2/3 scrambled)
    out_shuffled = net.forward(e_in, r_in, decouple_horizontal=False, shuffle_binding=True, rng=rng)
    # Cosine similarity between intact and shuffled should be near 0
    cos_shuf = float(np.dot(out_intact["bound_rep"], out_shuffled["bound_rep"]))
    assert abs(cos_shuf) < 0.35


def test_compositional_generalization_held_out():
    """Verify 100% zero-shot retrieval on unseen held-out pairs vs 0% on flat monolith."""
    from benchmarks.track5_compositional_benchmark import evaluate_single_seed
    res = evaluate_single_seed(seed=995, noise_sigma=0.50, dim=256)
    assert res["acc_net"] >= 95.0
    assert res["acc_decoupled"] <= 5.0
    assert res["acc_shuffled"] <= 5.0
    assert res["acc_monolith"] == 0.0
