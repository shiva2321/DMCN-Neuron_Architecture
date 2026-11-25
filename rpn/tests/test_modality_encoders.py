import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.modality_encoders import (
    LegendreTemporalEncoder,
    RetinalReceptiveFieldEncoder,
    PhaseSequenceEncoder,
    MotorCPGController,
)


def test_legendre_temporal_encoder_orthogonality_and_bounds():
    """Verify Legendre temporal encoder maintains bounded orthogonal state."""
    enc = LegendreTemporalEncoder(d_order=4, theta=1.0, dt=0.05)
    states = []
    # Feed chirp signal
    for t in range(100):
        u = float(np.sin(0.1 * t + 0.005 * (t ** 2)))
        m = enc.step(u)
        states.append(m)

    states_arr = np.array(states)
    assert states_arr.shape == (100, 4)
    # Norm remains bounded
    assert np.all(np.abs(states_arr) < 10.0)
    # Trajectory is non-trivial
    assert np.std(states_arr[:, 0]) > 0.01


def test_retinal_receptive_field_center_surround_inhibition():
    """Verify DoG receptive fields exhibit on-center excitation and off-surround inhibition."""
    encoder = RetinalReceptiveFieldEncoder(n_sensors=8, sigma_center=0.5, sigma_surround=1.5)

    # 1. Point obstacle directly at sensor 0 (distance = 0.2), all others far (distance = 5.0)
    dists = np.full(8, 5.0)
    dists[0] = 0.2

    rates = encoder.encode(dists)
    # Sensor 0 must have strongest excitation
    assert rates[0] == np.max(rates)
    # Lateral neighbors (sensors 1 and 7) must have lower activation than sensor 0
    assert rates[1] < rates[0]
    assert rates[7] < rates[0]

    # 2. Uniform sensory field (no spatial contrast)
    uniform_dists = np.full(8, 2.0)
    uniform_rates = encoder.encode(uniform_dists)
    # Due to zero-mean kernel, contrast is zero -> rectified rate is 0.0
    assert np.allclose(uniform_rates, 0.0, atol=1e-5)


def test_phase_sequence_encoder_unitary_and_orthogonality():
    """Verify continuous phase vector encoder is unitary and distinct sequences are separable."""
    seq_enc = PhaseSequenceEncoder(vocab_size=16, d_embed=64, rng=np.random.RandomState(42))

    # Token 1 and 2
    z1 = seq_enc.encode_token(1)
    z2 = seq_enc.encode_token(2)
    assert np.allclose(np.abs(z1), 1.0)
    assert np.allclose(np.abs(z2), 1.0)

    # Circular binding preserves unitary magnitude
    z_bound = seq_enc.bind(z1, z2)
    assert np.allclose(np.abs(z_bound), 1.0)

    # Encode two different sequences
    s1 = seq_enc.encode_sequence([1, 2, 3])
    s2 = seq_enc.encode_sequence([4, 5, 6])

    # Inner product (cosine similarity) between distinct sequences should be low
    sim = np.abs(np.vdot(s1, s2))
    assert sim < 0.40


def test_motor_cpg_controller_smoothness_and_bounds():
    """Verify motor CPG yields smooth, bounded continuous velocities."""
    cpg = MotorCPGController(max_vel=1.0, max_yaw_rate=1.5, dt=0.05)

    # Step response from 0 to full thrust and sharp turn
    vels, yaws = [], []
    for _ in range(30):
        v, yaw = cpg.step(u_thrust=1.0, u_steer=1.0)
        vels.append(v)
        yaws.append(yaw)

    # Velocities within bounds
    assert all(0.0 <= v <= 1.05 for v in vels)
    assert all(-1.55 <= yaw <= 1.55 for yaw in yaws)

    # Smoothness: no instantaneous jump > 0.5 in single dt step
    diff_v = np.diff(vels)
    diff_yaw = np.diff(yaws)
    assert np.all(np.abs(diff_v) < 0.5)
    assert np.all(np.abs(diff_yaw) < 0.5)
