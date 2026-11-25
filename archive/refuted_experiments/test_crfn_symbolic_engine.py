"""Unit tests for Weight-Free CRFN Symbolic Engine."""

import numpy as np
import pytest

from neuron_v2.crfn.crfn_symbolic_engine import (
    CRFNSymbolicEngine,
    ImpedanceTransmissionBus,
    StuartLandauSomaticBank,
)


def test_somatic_bank_lyapunov_clean_up() -> None:
    """Verify autonomous somatic limit-cycle amplitude noise dissipation (-2*mu)."""
    rng = np.random.RandomState(42)
    dim = 64
    bank = StuartLandauSomaticBank(dim=dim, mu=1.0)

    # Unit phasor with heavy amplitude noise
    phases = rng.uniform(-np.pi, np.pi, dim)
    noisy_amp = 1.0 + 0.40 * rng.randn(dim)
    noisy_z = noisy_amp * np.exp(1j * phases)

    rad_err_before = float(np.mean(np.abs(np.abs(noisy_z) - 1.0)))
    cleaned_z = bank.clean_up(noisy_z, steps=60, dt=0.02)
    rad_err_after = float(np.mean(np.abs(np.abs(cleaned_z) - 1.0)))

    assert rad_err_after < 0.05
    assert rad_err_after < rad_err_before * 0.20


def test_impedance_transmission_bus() -> None:
    """Verify transmission coefficient calculations at membrane boundaries."""
    bus = ImpedanceTransmissionBus(dim=4, z0=1.0)
    # Perfectly matched
    np.testing.assert_allclose(bus.transmission_coefficients, 1.0, atol=1e-6)

    # Discontinuity: Z = 3.0 -> Gamma = (3-1)/(3+1) = 0.5 -> T = 1 - 0.25 = 0.75
    bus.Z[0] = complex(3.0, 0.0)
    np.testing.assert_allclose(bus.transmission_coefficients[0], 0.75, atol=1e-6)


def test_crfn_analogy_and_null_controls() -> None:
    """Verify relational analogy solving under noise, and verify null controls fail."""
    rng = np.random.RandomState(42)
    dim = 256
    engine = CRFNSymbolicEngine(dim=dim, mu=1.0)

    # Generate source, relation, and target entities
    va = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    vc = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    r = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))

    vb = va * r
    vd = vc * r

    # Add sensory noise
    noise_sigma = 0.20
    noisy_a = va * np.exp(1j * noise_sigma * rng.randn(dim))
    noisy_b = vb * np.exp(1j * noise_sigma * rng.randn(dim))
    noisy_c = vc * np.exp(1j * noise_sigma * rng.randn(dim))

    # 1. Active column
    res_act = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c)
    coherence = float(np.real(np.dot(res_act["output_wave"], np.conj(vd)))) / dim
    assert coherence > 0.85

    # 2. Decoupled control (T = 0)
    res_dec = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, decouple_transmission=True)
    coh_dec = float(np.real(np.dot(res_dec["output_wave"], np.conj(vd)))) / dim
    assert coh_dec == 0.0

    # 3. Blind control
    res_bln = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, blind_input=True)
    coh_bln = float(np.real(np.dot(res_bln["output_wave"], np.conj(vd)))) / dim
    assert coh_bln == 0.0


def test_crfn_transitive_deduction() -> None:
    """Verify multi-hop transitive relation deduction without matrix weights."""
    rng = np.random.RandomState(42)
    dim = 256
    engine = CRFNSymbolicEngine(dim=dim, mu=1.0)

    va = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    rab = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    rbc = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))

    expected_vc = va * rab * rbc

    res = engine.solve_transitive_deduction(va, rab, rbc)
    coherence = float(np.real(np.dot(res["output_wave"], np.conj(expected_vc)))) / dim
    assert coherence > 0.95


def test_crfn_phase_demodulation_arithmetic() -> None:
    """Verify wave arithmetic decoded via direct linear phase demodulation without dictionary lookup."""
    rng = np.random.RandomState(42)
    engine = CRFNSymbolicEngine(n_frequencies=64, omega_min=0.01, omega_max=0.10)

    # 1. Addition
    a, b = 14, 27
    res_add = engine.wave_add(a, b, noise_sigma=0.12, rng=rng)
    assert res_add["predicted_val"] == 41

    # 2. Inversion (subtraction)
    res_inv = engine.wave_invert(41, 14, noise_sigma=0.12, rng=rng)
    assert res_inv["predicted_val"] == 27

    # 3. Decoupled control
    res_dec = engine.wave_add(a, b, decouple_transmission=True)
    assert res_dec["predicted_val"] == -1
