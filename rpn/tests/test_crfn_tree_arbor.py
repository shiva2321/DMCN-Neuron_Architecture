"""Unit tests for Multi-Compartmental Tree Arbor CRFN.

Verifies:
1. Invariant XOR classification with ZERO phase-inversion tricks (authentic dCaAP potassium shunting).
2. Larkum apical nexus coincidence detection and calcium burst amplification.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_tree_arbor import MultiCompartmentCRFN


def test_invariant_xor_no_phase_cheat():
    """Verify that MultiCompartmentCRFN solves XOR with 100% invariant in-phase inputs."""
    neuron = MultiCompartmentCRFN(
        soma_omega=2.0,
        soma_mu=-0.5,
        soma_kappa=1.8,
    )
    dt = 0.02
    omega = 2.0
    n_steps = 200

    # Invariant input amplitudes: each input sends identical phase 0.0 when active
    patterns = {
        (0, 0): (0.0, 0.0),
        (1, 0): (1.0, 0.0),
        (0, 1): (0.0, 1.0),
        (1, 1): (1.0, 1.0),
    }

    results = {}
    for pat, (s0, s1) in patterns.items():
        neuron.reset()
        amp_history = []
        for step in range(n_steps):
            t = step * dt
            # 100% INVARIANT CARRIER: exactly identical phase 0.0 across all inputs!
            sig0 = s0 * np.exp(1j * omega * t)
            sig1 = s1 * np.exp(1j * omega * t)
            out = neuron.step(basal_inputs={0: sig0, 1: sig1}, dt=dt)
            amp_history.append(out["somatic_amplitude"])

        results[pat] = float(np.mean(amp_history[-50:]))

    # Single inputs (1,0) and (0,1) must fire (Amp >= 0.50)
    assert results[(1, 0)] > 0.60
    assert results[(0, 1)] > 0.60
    # Quiescent (0,0) must stay silent (Amp == 0.0)
    assert results[(0, 0)] < 0.01
    # Coincident (1,1) must be shunted by high-threshold outward potassium conductance (Amp < 0.35)
    assert results[(1, 1)] < 0.35
    # Decision threshold at 0.50 yields 100% classification accuracy
    assert (results[(1, 0)] >= 0.50) and (results[(0, 1)] >= 0.50)
    assert (results[(0, 0)] < 0.50) and (results[(1, 1)] < 0.50)


def test_larkum_apical_nexus_burst():
    """Verify Larkum apical-basal coincidence amplification (burst vs. silent priming)."""
    neuron = MultiCompartmentCRFN(
        soma_omega=2.0,
        soma_mu=-0.5,
        soma_kappa=1.8,
        nexus_coincidence_threshold=0.35,
        burst_amplification=2.5,
    )
    dt = 0.02
    omega = 2.0
    n_steps = 200

    # 1. Basal alone
    neuron.reset()
    basal_amps = []
    for step in range(n_steps):
        t = step * dt
        sig = 1.0 * np.exp(1j * omega * t)
        out = neuron.step(basal_inputs={0: sig, 1: 0.0}, apical_inputs={0: 0.0}, dt=dt)
        basal_amps.append(out["somatic_amplitude"])
        assert out["coincident_burst"] is False
    mean_basal = np.mean(basal_amps[-40:])

    # 2. Apical alone (silent priming)
    neuron.reset()
    apical_amps = []
    for step in range(n_steps):
        t = step * dt
        sig = 1.0 * np.exp(1j * omega * t)
        out = neuron.step(basal_inputs={0: 0.0, 1: 0.0}, apical_inputs={0: sig}, dt=dt)
        apical_amps.append(out["somatic_amplitude"])
        assert out["coincident_burst"] is False
    mean_apical = np.mean(apical_amps[-40:])

    # 3. Basal + Apical coincident (Calcium burst)
    neuron.reset()
    burst_amps = []
    coincident_fired = False
    for step in range(n_steps):
        t = step * dt
        sig = 1.0 * np.exp(1j * omega * t)
        out = neuron.step(basal_inputs={0: sig, 1: 0.0}, apical_inputs={0: sig}, dt=dt)
        burst_amps.append(out["somatic_amplitude"])
        if out["coincident_burst"]:
            coincident_fired = True
    mean_burst = np.mean(burst_amps[-40:])

    assert coincident_fired is True
    # Apical alone must produce sub-threshold depolarization (silent priming)
    assert mean_apical < 0.40
    # Coincidence triggers burst amplification vastly exceeding basal alone
    assert mean_burst > mean_basal * 1.3
    assert mean_burst > 1.0
