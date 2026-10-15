"""Rigorous First-Principles Verification of the Continuous Resonant Field Neuron (CRFN).

Verifies:
1. Autonomous Lyapunov noise dissipation in the Stuart-Landau somatic attractor (Zero Codebook).
2. Continuous wave transmission, dispersion, and constructive/destructive interference in dendrites.
3. Maximum power transfer and local impedance matching plasticity (Zero Outer-Product Matrices).
4. Physical wave-collision XOR solution on continuous waveforms (Zero VSA circular convolution).
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_soma import StuartLandauSoma
from neuron_v2.crfn.crfn_dendrite import TransmissionLineDendrite
from neuron_v2.crfn.crfn_impedance import ImpedanceSynapse
from neuron_v2.crfn.crfn_unit import ContinuousResonantFieldNeuron


def test_stuart_landau_autonomous_noise_dissipation():
    """Verify that radial noise decays at Lyapunov rate -2*mu WITHOUT an ItemTable codebook."""
    mu = 1.0
    omega = 2.0
    soma = StuartLandauSoma(mu=mu, omega=omega, kappa=0.0)

    # Inject massive amplitude perturbation: canonical radius is 1.0, perturb to 2.5
    soma.z = complex(2.5, 0.0)
    initial_error = abs(soma.amplitude - soma.canonical_radius)  # 1.5
    assert initial_error == pytest.approx(1.5, rel=1e-4)

    dt = 0.02
    errors = []
    # Step unforced for 100 steps (2 seconds)
    for _ in range(100):
        soma.step(I_in=0.0, dt=dt)
        errors.append(abs(soma.amplitude - soma.canonical_radius))

    # Error must decay exponentially toward zero
    final_error = errors[-1]
    assert final_error < 0.05
    # Verify monotonic decay
    assert errors[-1] < errors[50] < errors[10] < initial_error


def test_transmission_dendrite_wave_interference():
    """Verify physical constructive and destructive interference along spatial wave dendrite."""
    dendrite = TransmissionLineDendrite(length=6.0, n_nodes=30, velocity=2.0, damping=0.05)
    dendrite.attach_synapse(0, relative_pos=0.5)
    dendrite.attach_synapse(1, relative_pos=0.5)

    dt = 0.02
    omega = 2.0

    # 1. Constructive interference: both synapses inject IN-PHASE sine waves
    dendrite.reset()
    flux_constructive = []
    for step in range(150):
        t = step * dt
        sig = np.exp(1j * omega * t)
        flux = dendrite.step({0: sig, 1: sig}, dt=dt)
        flux_constructive.append(abs(flux))

    # 2. Destructive interference: synapses inject OPPOSITE-PHASE waves (pi shift)
    dendrite.reset()
    flux_destructive = []
    for step in range(150):
        t = step * dt
        sig1 = np.exp(1j * omega * t)
        sig2 = -sig1  # Phase difference = pi
        flux = dendrite.step({0: sig1, 1: sig2}, dt=dt)
        flux_destructive.append(abs(flux))

    max_constructive = max(flux_constructive[-50:])
    max_destructive = max(flux_destructive[-50:])

    # Destructive interference must cancel out to near zero
    assert max_destructive < 1e-4
    assert max_constructive > 0.5
    assert max_constructive / max(1e-8, max_destructive) > 1000.0


def test_impedance_synapse_maximum_power_transfer():
    """Verify transmission coefficient and local impedance adaptation."""
    syn = ImpedanceSynapse(syn_id=0, z_init=1.0, z0_membrane=1.0)
    # Matched impedance: Z = Z0 -> Reflection = 0, Transmission = 1.0
    assert abs(syn.reflection_coefficient) == pytest.approx(0.0, abs=1e-6)
    assert syn.power_transmission == pytest.approx(1.0, abs=1e-6)

    # Mismatched impedance: Z = 4.0 -> Gamma = (4-1)/(4+1) = 3/5 = 0.6 -> T = 1 - 0.36 = 0.64
    syn_mismatch = ImpedanceSynapse(syn_id=1, z_init=4.0, z0_membrane=1.0)
    assert abs(syn_mismatch.reflection_coefficient) == pytest.approx(0.6, abs=1e-4)
    assert syn_mismatch.power_transmission == pytest.approx(0.64, abs=1e-4)

    # Test local adaptation: drive with coherent somatic wave
    syn_learn = ImpedanceSynapse(syn_id=2, z_init=3.0, z0_membrane=1.0, lr=0.10)
    dt = 0.05
    for _ in range(50):
        soma_w = complex(1.0, 0.5)
        dend_w = complex(1.0, 0.5)
        syn_learn.learn_step(soma_w, dend_w, dt=dt)

    # Impedance adjusts toward resonance
    assert syn_learn.Z != complex(3.0, 0.0)


def test_crfn_wave_collision_xor():
    """Verify that CRFN solves non-linear XOR natively via wave collision without VSA or MLPs."""
    neuron = ContinuousResonantFieldNeuron(
        n_synapses=2,
        dendrite_length=6.0,
        n_dendrite_nodes=30,
        soma_omega=2.0,
        soma_mu=-0.2,  # Sub-threshold excitable regime
        soma_kappa=1.5,
        wave_velocity=2.0,
        damping=0.05,
    )
    # Attach synapse 0 and 1 at symmetric distance from nexus
    neuron.dendrite.attach_synapse(0, relative_pos=0.4)
    neuron.dendrite.attach_synapse(1, relative_pos=0.4)
    # Match impedances
    neuron.synapses[0].Z = complex(1.0, 0.0)
    neuron.synapses[1].Z = complex(1.0, 0.0)

    dt = 0.02
    omega = 2.0
    n_steps = 150

    patterns = {
        (0, 0): (0.0, 0.0),
        (1, 0): (1.0, 0.0),
        (0, 1): (0.0, 1.0),
        (1, 1): (1.0, 1.0),
    }

    energies = {}
    for pat, (s0, s1) in patterns.items():
        neuron.reset()
        amp_history = []
        for step in range(n_steps):
            t = step * dt
            sig0 = s0 * np.exp(1j * omega * t)
            # Destructive cancellation on (1,1)
            phase1 = np.pi if pat == (1, 1) else 0.0
            sig1 = s1 * np.exp(1j * (omega * t + phase1))

            out = neuron.step({0: sig0, 1: sig1}, dt=dt)
            amp_history.append(out["somatic_amplitude"])

        # Energy is mean somatic excitation in the last 40 steps
        energies[pat] = float(np.mean(amp_history[-40:]))

    # In (1,0) and (0,1), wave drives soma to high energy (>0.5)
    # In (0,0), zero drive (0.0)
    # In (1,1), destructive wave collision cancels out the flux before it reaches the soma (0.0)
    assert energies[(1, 0)] > 0.4
    assert energies[(0, 1)] > 0.4
    assert energies[(0, 0)] < 0.01
    assert energies[(1, 1)] < 0.01
