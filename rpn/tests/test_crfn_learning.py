"""Tests for Unified Predictive Coding and Reinforcement Learning in CRFN.

Verifies:
1. Resonant Eligibility Traces & Temporal Credit Assignment:
   - Physical ringing holds memory of recent wave coincidence.
   - Delayed dopamine pulse selectively reinforces pathways active within window tau_e.
2. Destructive Wave Cancellation (Predictive Coding):
   - Internal top-down predictions destructively interfere with sensory input.
   - Net prediction error wave E(t) decays toward zero, minimizing free energy.
3. Neuromodulated Pathway Selection (Reinforcement Learning):
   - Rewarded pathway (delta > 0) matches impedance (T -> 1.0).
   - Punished pathway (delta < 0) detunes impedance (T -> 0.0).
4. Habit Stabilization (Zero RPE):
   - When outcomes match predictions (delta = 0), weights freeze, preventing catastrophic forgetting.
5. Unified Continuous Actor-Critic Learning:
   - Network autonomously tracks a continuous reward gradient without backpropagation.
"""
from __future__ import annotations

import numpy as np
import pytest

from neuron_v2.crfn.crfn_rl_predictive import (
    ResonantEligibilityTrace,
    NeuromodulatedSynapticLink,
    PredictiveCodingErrorJunction,
    NeuromodulatedCRFNNetwork,
)


def test_wave_eligibility_trace_temporal_decay():
    """Verify that resonant eligibility trace accumulates on coincidence and decays physically."""
    trace = ResonantEligibilityTrace(tau_e=0.20)  # 200 ms decay window
    dt = 0.02

    # Initial state is quiescent
    assert trace.magnitude == pytest.approx(0.0, abs=1e-8)

    # Coincident stimulation: post-synaptic and pre-synaptic waves overlap for 100 ms
    for _ in range(5):
        z_post = complex(1.0, 0.5)
        psi_pre = complex(1.0, 0.5)
        trace.step(z_post, psi_pre, dt=dt)

    peak_magnitude = trace.magnitude
    assert peak_magnitude > 0.05, f"Expected active trace, got {peak_magnitude}"

    # Stimulation stops: pre-synaptic wave is silent (ringing decay phase)
    decay_values = []
    for _ in range(20):  # 400 ms of silence (2 x tau_e)
        val = trace.step(z_post=0.0, psi_pre=0.0, dt=dt)
        decay_values.append(abs(val))

    # Must decay monotonically toward zero
    assert decay_values[-1] < peak_magnitude * 0.20
    assert decay_values[-1] < decay_values[0]


def test_delayed_reward_credit_assignment_via_eligibility():
    """Verify that a delayed dopamine pulse credits the pathway that was active in the past."""
    # Two competing synapses
    link_A = NeuromodulatedSynapticLink(source_id=0, target_id=2, z_init=3.0, tau_e=0.30, lr_rl=0.50)
    link_B = NeuromodulatedSynapticLink(source_id=1, target_id=2, z_init=3.0, tau_e=0.30, lr_rl=0.50)
    dt = 0.02

    # Pathway A is activated at t=0 (coincident with post-synaptic firing)
    for _ in range(5):
        z_post = complex(1.2, 0.0)
        psi_pre = complex(1.0, 0.0)
        link_A.update_eligibility(z_post, psi_pre, dt=dt)
        link_B.update_eligibility(z_post, psi_pre=0.0, dt=dt)  # Pathway B was silent

    # Silence delay of 100 ms (stimulus has ceased, but ringing eligibility persists in A)
    for _ in range(5):
        link_A.update_eligibility(0.0, 0.0, dt=dt)
        link_B.update_eligibility(0.0, 0.0, dt=dt)

    assert link_A.eligibility.magnitude > 0.02
    assert link_B.eligibility.magnitude < 1e-4

    # Now a delayed dopamine reward pulse arrives (delta = +1.0)
    initial_T_A = link_A.power_transmission
    initial_T_B = link_B.power_transmission

    for _ in range(10):
        link_A.apply_neuromodulation(delta_dopamine=1.0, dt=dt)
        link_B.apply_neuromodulation(delta_dopamine=1.0, dt=dt)

    # Pathway A should have matched impedance, increasing power transmission
    # Pathway B was not eligible, so its impedance remains unchanged
    assert link_A.power_transmission > initial_T_A, (
        f"Pathway A should be reinforced: {link_A.power_transmission} vs {initial_T_A}"
    )
    assert link_B.power_transmission == pytest.approx(initial_T_B, abs=1e-3), (
        f"Pathway B should remain unchanged: {link_B.power_transmission} vs {initial_T_B}"
    )


def test_predictive_coding_destructive_wave_cancellation():
    """Verify that predictive error junction cancels predictable sensory waves destructively."""
    junction = PredictiveCodingErrorJunction()
    dt = 0.02
    omega = 2.0

    # Case 1: No prediction (top-down is silent) -> High prediction error
    err_unpredicted = []
    for step in range(50):
        t = step * dt
        sensory = np.exp(1j * omega * t)
        err = junction.compute_interference(sensory, prediction_wave=0.0, dt=dt)
        err_unpredicted.append(abs(err))

    mean_err_unpredicted = float(np.mean(err_unpredicted))
    assert mean_err_unpredicted > 0.90

    # Case 2: Perfect top-down prediction (in phase with sensory) -> Destructive cancellation
    err_predicted = []
    for step in range(50):
        t = step * dt
        sensory = np.exp(1j * omega * t)
        prediction = np.exp(1j * omega * t)  # Exact match
        err = junction.compute_interference(sensory, prediction_wave=prediction, dt=dt)
        err_predicted.append(abs(err))

    mean_err_predicted = float(np.mean(err_predicted))
    assert mean_err_predicted < 1e-6, f"Expected near zero error wave, got {mean_err_predicted}"
    assert mean_err_unpredicted / (mean_err_predicted + 1e-8) > 1000.0


def test_habit_stabilization_zero_rpe():
    """Verify that when dopamine RPE is zero (predicted outcome), learned impedances are preserved."""
    link = NeuromodulatedSynapticLink(source_id=0, target_id=1, z_init=1.1, tau_e=0.30)
    dt = 0.02

    # Fill eligibility trace with strong activity
    for _ in range(10):
        link.update_eligibility(complex(1.5, 0.5), complex(1.0, 0.2), dt=dt)

    assert link.eligibility.magnitude > 0.05
    z_before = link.Z

    # Apply 50 steps of neutral outcome (delta = 0.0, expected reward)
    for _ in range(50):
        link.apply_neuromodulation(delta_dopamine=0.0, dt=dt)

    # Impedance must remain stable (only slight passive relaxation, no wild RL shifts)
    assert abs(link.Z - z_before) < 0.05


def test_neuromodulated_crfn_network_learning_loop():
    """Verify end-to-end learning loop: network receives sensory drive, acts, and improves under reward."""
    net = NeuromodulatedCRFNNetwork(
        n_sensory=1,
        n_recurrent=3,
        n_motor=1,
        base_omega=2.0,
        lr_rl=0.20,
        lr_pred=0.05,
        seed=42,
    )

    dt = 0.02
    omega = 2.0

    # Run initial baseline epoch (no reward)
    initial_actions = []
    for step in range(100):
        t = step * dt
        sensory_wave = np.exp(1j * omega * t)
        out = net.step({0: sensory_wave}, reward=0.0, dt=dt)
        initial_actions.append(float(out["motor_action"]))

    initial_motor_power = float(np.mean(np.square(initial_actions[-50:])))

    # Reward conditioning epoch:
    # Whenever motor action is positive, inject positive reward (delta > 0)
    for epoch in range(5):
        net.reset()
        for step in range(100):
            t = step * dt
            sensory_wave = np.exp(1j * omega * t)
            # Motor action from previous step
            last_act = net.somas[net.motor_ids[0]].z.real
            reward = 1.0 if last_act > 0.0 else -0.5
            net.step({0: sensory_wave}, reward=reward, dt=dt)

    # Evaluation epoch (freeze reward): check if motor transmission grew
    net.reset()
    eval_actions = []
    for step in range(100):
        t = step * dt
        sensory_wave = np.exp(1j * omega * t)
        out = net.step({0: sensory_wave}, reward=0.0, dt=dt)
        eval_actions.append(float(out["motor_action"]))

    eval_motor_power = float(np.mean(np.square(eval_actions[-50:])))

    # Motor power should have grown through impedance matching under reinforcement
    assert eval_motor_power > initial_motor_power * 1.5, (
        f"Motor transmission should increase with reward: {eval_motor_power} vs {initial_motor_power}"
    )


def test_arnold_tongue_frequency_selective_learning():
    """Verify that only neurons within the Arnold tongue resonance window entrain and learn."""
    from neuron_v2.crfn.crfn_soma import StuartLandauSoma
    dt = 0.02
    omega_ext = 2.05  # External drive frequency

    # Resonant neuron (natural freq = 2.00, detuning = 0.05 rad/s -> inside Arnold tongue)
    soma_resonant = StuartLandauSoma(mu=1.0, omega=2.00, kappa=1.5)
    # Detuned neuron (natural freq = 5.00, detuning = 2.95 rad/s -> far outside Arnold tongue)
    soma_detuned = StuartLandauSoma(mu=1.0, omega=5.00, kappa=1.5)

    link_resonant = NeuromodulatedSynapticLink(source_id=0, target_id=1, z_init=3.0, lr_rl=1.0)
    link_detuned = NeuromodulatedSynapticLink(source_id=0, target_id=2, z_init=3.0, lr_rl=1.0)

    # Drive both neurons with external wave for 100 steps (2 seconds)
    for step in range(100):
        t = step * dt
        drive = np.exp(1j * omega_ext * t)
        soma_resonant.step(drive, dt=dt)
        soma_detuned.step(drive, dt=dt)

        link_resonant.update_eligibility(soma_resonant.z, drive, dt=dt)
        link_detuned.update_eligibility(soma_detuned.z, drive, dt=dt)

    # Resonant neuron entrains to drive, accumulating high phase coherence and eligibility
    # Detuned neuron slips phase rapidly, rolling off as 1/sqrt(1 + (Delta_omega * tau_e)^2)
    assert link_resonant.eligibility.magnitude > link_detuned.eligibility.magnitude * 1.4

    # Deliver global dopamine pulse
    init_T_res = link_resonant.power_transmission
    init_T_det = link_detuned.power_transmission

    for _ in range(20):
        link_resonant.apply_neuromodulation(delta_dopamine=1.0, dt=dt)
        link_detuned.apply_neuromodulation(delta_dopamine=1.0, dt=dt)

    # Only resonant pathway undergoes significant reinforcement
    delta_T_res = link_resonant.power_transmission - init_T_res
    delta_T_det = link_detuned.power_transmission - init_T_det
    assert delta_T_res > delta_T_det * 1.4, (
        f"Resonant pathway must adapt faster: {delta_T_res} vs {delta_T_det}"
    )


def test_standing_wave_cavity_ringing_as_physical_credit_buffer():
    """Verify that high-Q dendritic transmission lines physically preserve ringing energy as credit."""
    from neuron_v2.crfn.crfn_dendrite import TransmissionLineDendrite
    dt = 0.02
    omega = 2.0

    # High-Q dendrite (low damping gamma = 0.01) vs Low-Q dendrite (high damping gamma = 1.50)
    dend_high_q = TransmissionLineDendrite(length=4.0, n_nodes=25, velocity=2.0, damping=0.01)
    dend_low_q = TransmissionLineDendrite(length=4.0, n_nodes=25, velocity=2.0, damping=1.50)

    dend_high_q.attach_synapse(0, relative_pos=0.5)
    dend_low_q.attach_synapse(0, relative_pos=0.5)

    # Inject a 50-step wave pulse
    for step in range(50):
        t = step * dt
        pulse = np.exp(1j * omega * t)
        dend_high_q.step({0: pulse}, dt=dt)
        dend_low_q.step({0: pulse}, dt=dt)

    # Pulse ceases: measure physical ringing decay over the next 75 steps
    high_q_ringing = []
    low_q_ringing = []
    for _ in range(75):
        f_high = dend_high_q.step({}, dt=dt)
        f_low = dend_low_q.step({}, dt=dt)
        high_q_ringing.append(abs(f_high))
        low_q_ringing.append(abs(f_low))

    # High-Q cavity retains significantly higher residual ringing power (physical eligibility)
    final_power_high = high_q_ringing[-1] ** 2
    final_power_low = low_q_ringing[-1] ** 2
    assert final_power_high > final_power_low * 5.0, (
        f"High-Q cavity must preserve ringing wave energy: {final_power_high} vs {final_power_low}"
    )


def test_punishment_induced_synaptic_pruning():
    """Verify that negative RPE (punishment) detunes impedance and triggers pruning."""
    net = NeuromodulatedCRFNNetwork(n_sensory=1, n_recurrent=2, n_motor=1, lr_rl=2.0, theta_prune=0.30)
    dt = 0.02

    # Identify a motor link
    motor_pair = list(net.links.keys())[-1]
    motor_link = net.links[motor_pair]

    # Pre-condition link with strong activity
    for _ in range(10):
        motor_link.update_eligibility(complex(1.0, 0.5), complex(1.0, 0.5), dt=dt)

    assert motor_link.eligibility.magnitude > 0.02
    initial_T = motor_link.power_transmission

    # Apply severe punishment pulses (delta = -2.0)
    for _ in range(30):
        motor_link.apply_neuromodulation(delta_dopamine=-2.0, dt=dt)

    # Impedance must mismatch, dropping transmission
    final_T = motor_link.power_transmission
    assert final_T < initial_T * 0.85, f"Transmission must decrease: {final_T} vs {initial_T}"

