"""Unit tests for Self-Organizing CRFN Network.

Verifies:
1. Spontaneous neurogenesis (network automatically grows new neurons under unabsorbed sensory tension).
2. Activity-dependent synaptic pruning (anti-resonant/reflective connections are physically severed).
3. Turrigiano homeostatic scaling (prevents epileptic runaway under high continuous excitation).
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_self_organizing import SelfOrganizingCRFNNetwork


def test_resonance_driven_growth():
    """Verify that network automatically buds new neurons when unabsorbed sensory drive accumulates."""
    net = SelfOrganizingCRFNNetwork(
        n_initial_neurons=2,
        base_omega=2.0,
        theta_grow=0.30,  # Sensitive growth threshold for test
        max_neurons=8,
        seed=42,
    )
    assert len(net.somas) == 2

    dt = 0.02
    omega_sensory = 2.8  # Novel frequency mismatched from initial 2.0 rad/s

    # Inject strong persistent novel sensory stream into neuron 0
    for step in range(250):
        t = step * dt
        # High power input
        u = 2.5 * np.exp(1j * omega_sensory * t)
        net.step(sensory_inputs={0: u}, dt=dt)

    # Network must have grown to accommodate the unabsorbed energy
    assert len(net.somas) > 2
    # Verify new neurons are properly connected
    assert len(net.links) >= len(net.somas)


def test_activity_dependent_pruning():
    """Verify that anti-resonant connections with high reflection are automatically pruned."""
    net = SelfOrganizingCRFNNetwork(
        n_initial_neurons=2,
        theta_prune=0.20,
        seed=101,
    )
    # Manually introduce a severely mismatched connection from 0 to 1 with Z = 25.0 (Gamma ~ 0.92, T ~ 0.14)
    net.links[(0, 1)].synapse.Z = complex(25.0, 0.0)
    net.links[(0, 1)].transmission_history = []  # reset history

    dt = 0.02
    # Step for 120 steps with low transmission
    for step in range(120):
        t = step * dt
        u = 1.0 * np.exp(1j * 2.0 * t)
        net.step(sensory_inputs={0: u}, dt=dt)

    # Connection (0, 1) must have been pruned due to low average transmission
    assert (0, 1) not in net.links


def test_homeostatic_stabilization():
    """Verify Turrigiano homeostatic scaling keeps network energy bounded under hyper-excitation."""
    net = SelfOrganizingCRFNNetwork(
        n_initial_neurons=3,
        target_energy=1.0,
        seed=202,
    )
    dt = 0.02
    initial_mus = [s.mu for s in net.somas.values()]

    # Blast with hyper-exciting inputs for 200 steps
    for step in range(200):
        t = step * dt
        u = 4.0 * np.exp(1j * 2.0 * t)
        net.step(sensory_inputs={0: u, 1: u}, dt=dt)

    # Homeostasis must have pushed somatic mu down (inhibitory push) to prevent epileptic runaway
    final_mus = [s.mu for s in net.somas.values()]
    for init_mu, final_mu in zip(initial_mus, final_mus):
        assert final_mu < init_mu
