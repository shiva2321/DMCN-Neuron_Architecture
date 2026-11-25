import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.universal_column import (
    UniversalCAPNColumn,
    UniversalMotorAttractorField,
    PlasticTransitionModel,
    generate_spherical_basis_vectors,
)


def test_universal_column_arbitrary_dimensions():
    """Verify Universal CAPN column initializes and operates across arbitrary dimensional configurations."""
    # Config A: Low-dimensional robotics (2D planar control, 5 sensory inputs)
    col_a = UniversalCAPNColumn(d_sensory=5, d_state=8, d_action=2, n_action_basis=8)
    x_a = np.random.randn(5)
    z_a = col_a.encode_state(x_a)
    assert z_a.shape == (8,)
    act_a = col_a.select_action(dir_goal=np.ones(2), sensory_hazard_profile=np.zeros(8))
    assert act_a.shape == (2,)

    # Config B: High-dimensional multi-modal sensory stream (24 inputs, 32 state, 4 actions)
    col_b = UniversalCAPNColumn(d_sensory=24, d_state=32, d_action=4, n_action_basis=16)
    x_b = np.random.randn(24)
    z_b = col_b.encode_state(x_b)
    assert z_b.shape == (32,)
    act_b = col_b.select_action(dir_goal=np.ones(4), sensory_hazard_profile=np.zeros(16))
    assert act_b.shape == (4,)


def test_plastic_transition_learning_without_backprop():
    """Verify PlasticTransitionModel learns unknown non-linear dynamics online without backpropagation."""
    rng = np.random.RandomState(42)
    d_state, d_action = 6, 2
    model = PlasticTransitionModel(d_state=d_state, d_action=d_action, d_hid=24, eta=0.08, rng=rng)

    # Unknown non-linear dynamical system: Delta z = 0.3 * tanh(A @ z + B @ u)
    A_true = rng.randn(d_state, d_state) * 0.4
    B_true = rng.randn(d_state, d_action) * 0.4

    errors = []
    z = rng.randn(d_state) * 0.5
    for t in range(250):
        u = rng.randn(d_action) * 0.5
        # True physics transition
        delta_true = 0.3 * np.tanh(A_true @ z + B_true @ u)
        z_next = z + delta_true

        # Learn online from observation
        err = model.learn_step(z, u, z_next)
        errors.append(err)
        z = z_next

    # Early error vs converged error
    early_err = float(np.mean(errors[:30]))
    late_err = float(np.mean(errors[-50:]))
    assert late_err < early_err * 0.4  # Significant online error reduction (>2.5x)


def test_universal_motor_attractor_affordances():
    """Verify Universal Motor Attractor Field operates in arbitrary dimensions and bifurcates under hazard."""
    rng = np.random.RandomState(42)
    d_action = 4
    n_basis = 12
    attractor = UniversalMotorAttractorField(d_action=d_action, n_basis=n_basis, rng=rng)

    dir_goal = np.zeros(d_action)
    dir_goal[0] = 1.0  # Move along first axis
    prev_act = dir_goal.copy()

    # Free space
    affordances_free = attractor.compute_affordances(dir_goal, np.zeros(n_basis), prev_act)
    pop_vec_free = affordances_free[0]
    cos_sim = float(np.dot(pop_vec_free, dir_goal))
    assert cos_sim > 0.90

    # Inhibit first basis vector (hazard directly along movement axis)
    hazard = np.zeros(n_basis)
    hazard[0] = 1.5
    affordances_blocked = attractor.compute_affordances(dir_goal, hazard, prev_act)
    assert len(affordances_blocked) >= 4


def test_chaotic_lorenz_attractor_tracking():
    """Verify universal column tracks and predicts chaotic Lorenz attractor non-linear dynamics."""
    rng = np.random.RandomState(101)
    col = UniversalCAPNColumn(d_sensory=3, d_state=6, d_action=3, n_action_basis=8, eta_dyn=0.06, rng=rng)

    # Generate chaotic Lorenz trajectory: dx = s*(y - x), dy = x*(r - z) - y, dz = x*y - b*z
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt_lorenz = 0.01

    state = np.array([1.0, 1.0, 1.0])
    lorenz_states = []
    for _ in range(500):
        dx = sigma * (state[1] - state[0]) * dt_lorenz
        dy = (state[0] * (rho - state[2]) - state[1]) * dt_lorenz
        dz = (state[0] * state[1] - beta * state[2]) * dt_lorenz
        state = state + np.array([dx, dy, dz])
        lorenz_states.append(state.copy())

    lorenz_states = np.array(lorenz_states)
    # Normalize states to standard variance
    lorenz_norm = lorenz_states / (np.std(lorenz_states, axis=0) + 1e-6)

    tracking_errors = []
    for t in range(len(lorenz_norm)):
        s_t = lorenz_norm[t]
        z_t = col.encode_state(s_t)
        err = col.step_learn_transition(z_t)
        if t > 50:
            tracking_errors.append(err)

    mean_tracking_err = float(np.mean(tracking_errors))
    assert mean_tracking_err < 0.20  # Successful non-backprop tracking of chaotic dynamics
