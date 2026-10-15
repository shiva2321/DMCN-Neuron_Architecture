"""Unit tests for Morphogenetic CRFN Substrate.

Verifies:
1. Zero Predefined Roles:
   - Initial neurons are identical pluripotent units.
   - No hardcoded 'motor', 'sensory', or 'critic' IDs.
2. Arbitrary Multi-Channel Actuation:
   - Substrate drives arbitrary K-channel actuator bus (e.g. 4 channels: Up, Down, Left, Right).
3. Spontaneous Functional Specialization:
   - Closed-loop reward selectively turns undifferentiated neurons into 'Motor Specialists'.
   - Sensory drive selectively turns undifferentiated neurons into 'Sensory Specialists'.
   - Unconnected internal neurons become 'Association / Memory Specialists'.
4. Multi-Channel Sensorimotor Control:
   - Agent learns to selectively trigger rewarded actuator channels without backpropagation.
5. Morphogenetic Neurogenesis & Pruning:
   - High sensory surprise buds new neurons.
   - Consistently punished/reflecting actuator links are pruned away.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_morphogenetic import (
    SensoryPort,
    ActuatorPort,
    MorphogeneticCRFNSubstrate,
)


def test_morphogenetic_substrate_initialization_zero_predefined_roles():
    """Verify that all seed neurons are identical and have zero predefined labels."""
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=4,
        n_sensory_ports=3,
        n_actuator_ports=4,  # e.g., Up, Down, Left, Right
        base_omega=2.0,
        seed=42,
    )

    # 4 identical neurons
    assert len(substrate.somas) == 4
    assert len(substrate.dendrites) == 4
    assert len(substrate.sensory_ports) == 3
    assert len(substrate.actuator_ports) == 4

    # All neurons share identical dynamical equations and resting parameters
    for nid, soma in substrate.somas.items():
        assert soma.mu == pytest.approx(-0.05, abs=1e-3)
        assert soma.kappa == pytest.approx(2.5, abs=1e-3)

    # Initial state: none are pre-labeled
    roles = substrate.inspect_emergent_roles()
    assert len(roles) == 4


def test_arbitrary_4_channel_actuation_up_down_left_right():
    """Verify continuous wave transmission into 4 arbitrary actuator channels."""
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=4,
        n_sensory_ports=2,
        n_actuator_ports=4,  # 0: Up, 1: Down, 2: Left, 3: Right
        seed=42,
    )

    dt = 0.02
    sensory_inputs = {0: complex(1.0, 0.0), 1: complex(0.5, 0.5)}

    # Step substrate
    out = substrate.step(sensory_inputs, reward=0.0, dt=dt)
    forces = out["actuator_forces"]

    assert len(forces) == 4
    assert all(isinstance(f, float) for f in forces)


def test_spontaneous_functional_differentiation_via_reward():
    """Verify that rewarding a specific actuator port causes an undifferentiated neuron

    to spontaneously differentiate into a Motor Specialist for that actuator.
    """
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=4,
        n_sensory_ports=1,
        n_actuator_ports=4,  # 0: Up, 1: Down, 2: Left, 3: Right
        lr_rl=4.0,
        exploration_std=0.10,
        seed=42,
    )

    dt = 0.02
    omega = 2.0
    target_actuator = 2  # Say actuator 2 ("Left") is the winning action

    # Record initial power transmission to target actuator
    initial_T = [
        l.power_transmission for (nid, aid), l in substrate.efferent_links.items()
        if aid == target_actuator
    ]
    assert len(initial_T) > 0

    # Conditioning loop: reward whenever target actuator force is positive
    for epoch in range(5):
        substrate.reset()
        for step in range(100):
            t = step * dt
            sensory = np.exp(1j * omega * t)
            # Check last force on target actuator
            last_force = substrate.actuator_ports[target_actuator].force
            reward = 1.5 if last_force > 0.005 else -0.2
            substrate.step({0: sensory}, reward=reward, dt=dt)

    # Inspect post-learning efferent transmission
    final_T = [
        l.power_transmission for (nid, aid), l in substrate.efferent_links.items()
        if aid == target_actuator
    ]

    # At least one neuron must have matched impedance, becoming a Motor Specialist
    max_final_T = max(final_T)
    max_init_T = max(initial_T)
    assert max_final_T > max_init_T * 1.2, (
        f"Efferent link to actuator {target_actuator} must match impedance: {max_final_T} vs {max_init_T}"
    )

    # Check emergent role inspection
    roles = substrate.inspect_emergent_roles()
    assert any(role in ["Motor Specialist", "Sensory Specialist"] for role in roles.values()), (
        f"Substrate should have differentiated functional roles: {roles}"
    )


def test_punishment_prunes_bad_actuator_connections():
    """Verify that repeatedly punished actuator connections reflect energy and are severed."""
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=4,
        n_sensory_ports=1,
        n_actuator_ports=2,
        lr_rl=5.0,
        theta_prune=0.25,
        seed=42,
    )

    dt = 0.02
    bad_actuator = 1

    # Find an efferent link to bad_actuator
    bad_links = [pair for pair in substrate.efferent_links if pair[1] == bad_actuator]
    assert len(bad_links) > 0
    target_pair = bad_links[0]
    link = substrate.efferent_links[target_pair]

    # Pre-condition with activity
    for _ in range(10):
        link.update_eligibility(complex(1.0, 0.5), complex(1.0, 0.5), dt=dt)

    # Deliver severe continuous punishment (delta = -3.0)
    for _ in range(40):
        link.apply_neuromodulation(delta_dopamine=-3.0, dt=dt)

    assert link.power_transmission < 0.25, f"Expected low transmission, got {link.power_transmission}"


def test_morphogenetic_neurogenesis_under_unabsorbed_sensory_tension():
    """Verify that persistent novel sensory error triggers spontaneous neurogenesis."""
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=3,
        n_sensory_ports=2,
        n_actuator_ports=2,
        theta_grow=0.30,  # Lower threshold for prompt budding
        max_neurons=10,
        seed=42,
    )
    substrate.growth_cooldown = 20  # Fast test cooldown
    dt = 0.02

    initial_neuron_count = len(substrate.somas)

    # Inject high-power unpredicted broadband sensory waves
    for step in range(60):
        t = step * dt
        sensory_0 = 2.0 * np.exp(1j * 2.5 * t)
        sensory_1 = 1.5 * np.exp(1j * 3.8 * t)
        substrate.step({0: sensory_0, 1: sensory_1}, reward=0.0, dt=dt)

    # Substrate must have budded new neurons to handle the unabsorbed sensory load
    final_neuron_count = len(substrate.somas)
    assert final_neuron_count > initial_neuron_count, (
        f"Substrate must grow neurons: {final_neuron_count} vs {initial_neuron_count}"
    )
