"""Unit tests for the Holomorphic Phase-Resonator (HPR)."""
import numpy as np
import pytest

from neuron_v2.core.hpr import (
    HarmonicSpectralMemory,
    WaveInterferenceNeuron,
    HolomorphicPhaseResonator,
)


def test_wave_interference_analytical_xor():
    """Verify that native wave interference computes exact XOR with anti-phase carriers."""
    neuron = WaveInterferenceNeuron(n_channels=2, threshold=0.50, seed=42)
    # Set carriers to exact anti-phase (0 and pi)
    neuron.phases[0] = 0.0
    neuron.phases[1] = np.pi

    s00, e00 = neuron.forward([0.0, 0.0])
    s10, e10 = neuron.forward([1.0, 0.0])
    s01, e01 = neuron.forward([0.0, 1.0])
    s11, e11 = neuron.forward([1.0, 1.0])

    assert s00 == 0 and e00 == pytest.approx(0.0, abs=1e-6)
    assert s10 == 1 and e10 == pytest.approx(1.0, abs=1e-6)
    assert s01 == 1 and e01 == pytest.approx(1.0, abs=1e-6)
    assert s11 == 0 and e11 == pytest.approx(0.0, abs=1e-6)


def test_wave_interference_autonomous_learning():
    """Verify autonomous phase adaptation converges to 100% XOR accuracy from random initial phases."""
    neuron = WaveInterferenceNeuron(n_channels=2, threshold=0.50, lr=0.15, seed=7)
    patterns = [
        ([0.0, 0.0], 0),
        ([1.0, 0.0], 1),
        ([0.0, 1.0], 1),
        ([1.0, 1.0], 0),
    ]

    for _ in range(120):
        for inp, target in patterns:
            neuron.learn_autonomous_logic(inp, target)

    for inp, target in patterns:
        spike, _ = neuron.forward(inp)
        assert spike == target

    dtheta = neuron.phases[0] - neuron.phases[1]
    assert np.cos(dtheta) < -0.75  # Verified destructive interference regime


def test_harmonic_spectral_memory_unitary_energy_conservation():
    """Verify that unforced evolution preserves wavefield energy."""
    # Zero damping gamma=0.0
    mem = HarmonicSpectralMemory(K=8, gamma=0.0, dt=1.0, seed=42)
    # Initialize with non-zero state
    mem.Psi = np.exp(1j * np.random.RandomState(42).uniform(-np.pi, np.pi, 8))
    initial_energy = float(np.real(np.vdot(mem.Psi, mem.Psi)))

    # Step without input (x = 0)
    for _ in range(50):
        mem.step(0.0)

    final_energy = float(np.real(np.vdot(mem.Psi, mem.Psi)))
    assert initial_energy == pytest.approx(final_energy, rel=1e-5)


def test_harmonic_spectral_memory_delay_reconstruction():
    """Verify spectral phase dispersion delay discovery on continuous signal."""
    mem = HarmonicSpectralMemory(K=16, dt=1.0, lr=0.05, seed=42)
    t = np.arange(1000, dtype=np.float64)
    x = np.sin(0.02 * t) + 0.5 * np.cos(0.05 * t)
    tau = 15.0

    preds = []
    targets = []
    for step in range(len(x)):
        xt = x[step]
        mem.step(xt)
        pred = mem.predict_delay()
        if step >= int(tau):
            target = x[step - int(tau)]
            mem.learn_delay(target)
            preds.append(pred)
            targets.append(target)

    mse = float(np.mean((np.array(preds[-200:]) - np.array(targets[-200:])) ** 2))
    assert mse < 0.005  # High-fidelity continuous reconstruction


def test_holomorphic_phase_resonator_unified():
    """Verify unified HPR container steps both discrete logic and continuous time."""
    hpr = HolomorphicPhaseResonator(n_channels=2, K_spectrum=8, dt=1.0, seed=42)

    logic_out = hpr.step_discrete_logic([1.0, 0.0], target_spike=1)
    assert logic_out["spike"] in (0, 1)

    cont_out = hpr.step_continuous(1.5, target_delay_val=1.2)
    assert len(cont_out["spectral_state"]) == 8
    assert isinstance(cont_out["predicted_delay"], float)
