"""Unit tests for Transmission-Line Impedance Matching Local Plasticity (Track 3).

Verifies:
1. Maximum Power Transfer Theorem: conjugate matching Z = Z0* yields Gamma = 0, T = 1.0.
2. Reactance Cancellation: complex conjugate matching cancels membrane capacitive reactance.
3. Autonomous convergence to theoretical conjugate match under coherent drive.
4. Mandatory failure of null controls (decoupled feedback yields Delta_T near zero).
5. Physical passive membrane constraint: Re(Z) >= 0.02.
"""
from __future__ import annotations

import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.biophysics.impedance_plasticity import TransmissionLineImpedanceArray


def test_maximum_power_transfer_theorem():
    """Verify that Z = Z0* gives zero reflection and 100% transmission."""
    Z0 = 1.0 - 0.5j
    array = TransmissionLineImpedanceArray(n_synapses=2, z0_membrane=Z0)
    # Force synapse 0 to theoretical conjugate match
    array.Z[0] = np.conj(Z0)
    # Force synapse 1 to mismatched impedance
    array.Z[1] = 4.0 - 1.5j

    gamma0 = array.reflection_coefficient(0)
    t0 = array.power_transmission(0)
    assert abs(gamma0) == pytest.approx(0.0, abs=1e-8)
    assert t0 == pytest.approx(1.0, abs=1e-8)

    gamma1 = array.reflection_coefficient(1)
    t1 = array.power_transmission(1)
    assert abs(gamma1) > 0.4
    assert t1 < 0.85


def test_reactance_cancellation():
    """Verify that conjugate matching cancels reactive phase shift whereas purely real Z does not."""
    Z0 = 1.0 - 0.8j
    array = TransmissionLineImpedanceArray(n_synapses=2, z0_membrane=Z0)
    # Conjugate match: Z = 1.0 + 0.8j
    array.Z[0] = 1.0 + 0.8j
    # Pure real match: Z = 1.0 + 0.0j (fails to cancel capacitance)
    array.Z[1] = 1.0 + 0.0j

    assert array.power_transmission(0) == pytest.approx(1.0, abs=1e-6)
    # Real-only impedance has reactive reflection
    assert array.power_transmission(1) < 0.90


def test_local_adaptation_convergence_to_theoretical_match():
    """Verify that unsupervised local resonance drives synapses to within 0.05 of theoretical match."""
    array = TransmissionLineImpedanceArray(n_synapses=4, z0_membrane=1.0 - 0.5j, lr=0.08, seed=42)

    dt = 0.5
    n_steps = 250
    t = np.arange(n_steps) * dt
    # Coherent signal driving all synapses across 50 episodes (matching benchmark protocol)
    for ep in range(50):
        omega = 0.2
        wave = np.exp(1j * (omega * t + ep * 0.1))
        u = np.zeros((4, n_steps), dtype=np.complex128)
        for i in range(4):
            u[i] = wave + np.random.normal(0, 0.2, n_steps) + 1j * np.random.normal(0, 0.2, n_steps)
        array.adapt_episode(u)

    for i in range(4):
        dist = array.distance_to_conjugate_match(i)
        assert dist < 0.05, f"Synapse {i} distance {dist} exceeded 0.05"
        assert array.power_transmission(i) > 0.99


def test_null_control_decoupled_fails_to_adapt():
    """Verify that zero somatic feedback prevents selective transmission."""
    array = TransmissionLineImpedanceArray(n_synapses=8, z0_membrane=1.0 - 0.5j, lr=0.08, seed=42)
    dt = 0.5
    n_steps = 200
    t = np.arange(n_steps) * dt

    for ep in range(30):
        u = np.zeros((8, n_steps), dtype=np.complex128)
        wave = np.exp(1j * (0.2 * t))
        for i in range(4):
            u[i] = wave
        for i in range(4, 8):
            u[i] = np.random.normal(0, 1.0, n_steps) + 1j * np.random.normal(0, 1.0, n_steps)
        # Decouple somatic feedback
        zero_soma = np.zeros(n_steps, dtype=np.complex128)
        array.adapt_episode(u, somatic_override=zero_soma)

    final_T = array.get_all_transmissions()
    delta_T = float(np.mean(final_T[:4]) - np.mean(final_T[4:]))
    assert abs(delta_T) <= 0.05


def test_physical_positive_resistance_constraint():
    """Verify that impedance resistance Re(Z) is strictly bounded below by 0.02."""
    array = TransmissionLineImpedanceArray(n_synapses=2, z0_membrane=1.0 - 0.5j)
    array.Z[0] = -5.0 + 1.0j
    # Any update must keep real part >= 0.02
    u = np.zeros((2, 50), dtype=np.complex128)
    array.adapt_episode(u)
    assert array.Z[0].real >= 0.02
