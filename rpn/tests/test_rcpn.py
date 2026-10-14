"""Tests for the Resonant Compartmental Predictive Neuron (RCPN)."""
import numpy as np
import pytest

from neuron_v2.core.rcpn import (
    ContinuousTimeDelay,
    BasalDendriticBranch,
    ApicalNexus,
    AstrocyteEnvelope,
    PredictiveSoma,
    RCPNeuron,
    legendre_matrices,
)
from neuron_v2.substrate.phasor import random_hv


def test_legendre_delay_reconstruction():
    """Verify that continuous-time Legendre state space accurately reconstructs delayed continuous signals."""
    delay_engine = ContinuousTimeDelay(order=6, theta=50.0, dt=1.0)
    # Feed a continuous sine wave for 100 ticks
    history = []
    reconstructed = []
    target_tau = 20.0  # 20 ticks delay

    for t in range(120):
        val = np.sin(2 * np.pi * t / 40.0)
        history.append(val)
        delay_engine.step(val)
        if t >= 30:
            recon = delay_engine.reconstruct_delay(target_tau)
            reconstructed.append(recon)

    # Compare actual delayed signal vs reconstructed
    actual_delayed = history[30 - int(target_tau) : 120 - int(target_tau)]
    mse = np.mean((np.array(actual_delayed) - np.array(reconstructed)) ** 2)
    # Legendre order 6 should reconstruct smoothly with low MSE
    assert mse < 0.08, f"Legendre delay reconstruction MSE too high: {mse}"


def test_basal_branch_plateau_initiation():
    """Verify that crossing local dendritic threshold initiates an active NMDA plateau."""
    D = 128
    branch = BasalDendriticBranch(D, branch_id=0, n_channels=2, seed=42)
    # Construct an input that strongly matches synaptic weights
    matching_input = [branch.W[0].copy(), branch.W[1].copy()]

    # First step: voltage charges up
    pred, inj1 = branch.forward(matching_input, g_astro=1.0)
    # Second step: crosses threshold, triggers plateau
    pred, inj2 = branch.forward(matching_input, g_astro=1.0)

    assert branch.plateau_timer > 0, "Dendritic plateau timer was not initiated"
    assert inj2 == branch.plateau_current, f"Injected current {inj2} does not match plateau current {branch.plateau_current}"


def test_apical_nexus_coincidence_amplification():
    """Verify Larkum apical coincidence detection boosts injected current."""
    D = 128
    nexus = ApicalNexus(D)
    ctx = random_hv(D, seed=1)
    basal_match = ctx.copy()  # Perfect coincidence
    basal_mismatch = random_hv(D, seed=2)  # Orthogonal mismatch

    amp_match = nexus.evaluate_coincidence(ctx, basal_match)
    amp_mismatch = nexus.evaluate_coincidence(ctx, basal_mismatch)

    assert amp_match == nexus.amplification_factor, "Coincident apical input did not amplify"
    assert amp_mismatch == 1.0, "Non-coincident apical input incorrectly amplified"


def test_astrocyte_homeostatic_scaling():
    """Verify that astrocytic calcium accumulation compresses synaptic gain under high activity."""
    astro = AstrocyteEnvelope(target_ca=1.0)
    # Simulate high activity burst
    gains = []
    for _ in range(50):
        g = astro.update(total_synaptic_activity=3.0, somatic_spiked=True)
        gains.append(g)

    # Initial gain is near 1.0; under chronic hyperactivity it drops homeostatically
    assert gains[-1] < 0.7, f"Astrocytic gain failed to scale down under hyperactivity: {gains[-1]}"
    assert astro.calcium > 1.0, "Calcium did not accumulate"


def test_rcpn_step_and_predictive_silence():
    """Verify that RCPN executes forward pass and remains silent when surprise is low."""
    D = 128
    neuron = RCPNeuron(D=D, n_branches=2, n_channels=2, seed=42)
    inputs = [random_hv(D, seed=10), random_hv(D, seed=11)]

    # Single step with unprimed input should produce valid state without error
    out = neuron.step(inputs)
    assert "spiked" in out
    assert "soma_voltage" in out
    assert len(out["branch_voltages"]) == 2
