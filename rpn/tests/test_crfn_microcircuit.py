"""Unit tests for Coupled CRFN Microcircuit.

Verifies:
1. Spontaneous collective phase synchronization from random initial conditions (Phase noise dissipation).
2. Injection locking and frequency entrainment across Arnold tongues.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_microcircuit import CRFNMicrocircuit


def test_collective_phase_synchronization():
    """Verify that N coupled CRFN units spontaneously phase-lock (R -> 1.0)."""
    micro = CRFNMicrocircuit(
        n_units=8,
        base_omega=2.0,
        omega_dispersion=0.05,
        coupling_strength=2.5,
        soma_mu=1.0,
        seed=42,
    )

    r_initial = micro.order_parameter
    assert r_initial < 0.60  # Initial random dispersion

    dt = 0.02
    r_history = []
    for _ in range(300):
        out = micro.step(external_inputs=0.0, dt=dt)
        r_history.append(out["order_parameter"])

    r_final = r_history[-1]
    # Spontaneous phase locking must achieve near-perfect coherence
    assert r_final > 0.95
    assert r_final > r_initial + 0.35


def test_injection_locking_frequency_entrainment():
    """Verify that external driving frequency locks the microcircuit via Arnold tongue resonance."""
    micro = CRFNMicrocircuit(
        n_units=6,
        base_omega=2.0,
        omega_dispersion=0.02,
        coupling_strength=1.5,
        soma_mu=1.0,
        seed=101,
    )
    dt = 0.02
    omega_ext = 2.12  # Detuned external carrier

    # Drive for 350 steps to achieve stable lock
    for step in range(350):
        t = step * dt
        ext_wave = 0.8 * np.exp(1j * omega_ext * t)
        micro.step(external_inputs=ext_wave, dt=dt)

    # Track unit 0 phase over subsequent 60 steps
    phases = []
    for step in range(60):
        t = (350 + step) * dt
        ext_wave = 0.8 * np.exp(1j * omega_ext * t)
        micro.step(external_inputs=ext_wave, dt=dt)
        phases.append(micro.units[0].phase)

    # Unwrapped phase difference evaluates instantaneous frequency: dphi / dt
    unwrapped = np.unwrap(phases)
    measured_omega = float(np.mean(np.diff(unwrapped)) / dt)

    # Unit must be frequency-locked to the external signal
    assert measured_omega == pytest.approx(omega_ext, abs=0.02)
