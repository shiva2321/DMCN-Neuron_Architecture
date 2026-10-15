"""Unified Predictive Coding and Reinforcement Learning for Continuous Resonant Field Neurons.

Merges physical wave mechanics with two biological learning principles:
1. Predictive Wave Cancellation (Unsupervised Free-Energy Minimization):
   Top-down prediction waves destructively interfere with bottom-up sensory waves:
       E_wave(t) = Psi_sensory(t) - Psi_hat(t)
   When internal predictions are accurate, destructive interference cancels the wavefield (E -> 0).
   The network consumes minimal metabolic power. When surprises occur, uncancelled error waves
   surge into the soma, driving plastic impedance adaptation to cancel the error.

2. Three-Factor Neuromodulated Plasticity (Reinforcement Learning via Resonant Ringing):
   Solves the temporal credit assignment problem without backpropagation or artificial replay buffers.
   Every physical resonant compartment has a finite Quality Factor Q, causing dendritic waves
   to ring for a characteristic decay window tau_e:
       dC_ij/dt = -C_ij / tau_e + z_post(t) * Psi_pre*(t)
   When a global scalar dopamine pulse delta(t) (Reward Prediction Error) washes over the network:
       dZ_ij/dt = -eta_RL * delta(t) * (C_ij / (|C_ij| + eps)) * (Z_ij - Z_0)
   - Positive RPE (delta > 0) matches impedance (Z -> Z_0), opening transmission T -> 1 for the ringing pathway.
   - Negative RPE (delta < 0) mismatches impedance (Z away from Z_0), reflecting energy away (T -> 0).
   - Zero RPE (delta = 0) locks in habitual policies, preventing catastrophic forgetting.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Dict, Tuple, Optional

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_impedance import ImpedanceSynapse


class ResonantEligibilityTrace:
    """Continuous physical eligibility trace governed by resonant ringing decay.
    
    Tracks the complex cross-correlation between post-synaptic somatic oscillations
    and pre-synaptic dendritic wave arrivals over time window tau_e.
    """

    def __init__(self, tau_e: float = 0.5):
        self.tau_e = float(tau_e)
        self.trace: complex = 0.0 + 0.0j

    def step(self, z_post: complex, psi_pre: complex, dt: float = 0.02) -> complex:
        """Update trace continuously with instantaneous wave coincidence and exponential decay."""
        # Instantaneous wave cross-correlation: z_post * conj(psi_pre)
        instantaneous_coherence = z_post * np.conj(psi_pre)
        decay = np.exp(-dt / self.tau_e)
        self.trace = self.trace * decay + instantaneous_coherence * dt
        return self.trace

    @property
    def magnitude(self) -> float:
        """Coherence strength of recent resonant activity."""
        return float(abs(self.trace))

    @property
    def phase(self) -> float:
        """Phase lag between pre and post wavefields."""
        return float(np.angle(self.trace))

    def reset(self):
        """Quench eligibility trace."""
        self.trace = 0.0 + 0.0j


class NeuromodulatedSynapticLink:
    """Transmission-line synaptic link with continuous eligibility and dopamine gating."""

    def __init__(
        self,
        source_id: int,
        target_id: int,
        dst_syn_id: int = 0,
        z_init: complex | None = None,
        z0_membrane: float = 1.0,
        lr_rl: float = 0.10,
        lr_pred: float = 0.05,
        tau_e: float = 0.40,
        decay: float = 0.005,
    ):
        self.src = int(source_id)
        self.dst = int(target_id)
        self.dst_syn_id = int(dst_syn_id)
        self.synapse = ImpedanceSynapse(
            syn_id=target_id,
            z_init=z_init if z_init is not None else 2.5 + 0.0j,  # Start slightly mismatched
            z0_membrane=z0_membrane,
            lr=lr_pred,
            decay=decay,
        )
        self.eligibility = ResonantEligibilityTrace(tau_e=tau_e)
        self.lr_rl = float(lr_rl)
        self.lr_pred = float(lr_pred)
        self.decay = float(decay)

        # Transmission history for activity-dependent pruning
        self.transmission_history: list[float] = []
        self.history_window = 100

    @property
    def Z(self) -> complex:
        return self.synapse.Z

    @Z.setter
    def Z(self, val: complex):
        self.synapse.Z = complex(val)

    @property
    def power_transmission(self) -> float:
        return self.synapse.power_transmission

    @property
    def mean_transmission(self) -> float:
        if not self.transmission_history:
            return self.power_transmission
        return float(np.mean(self.transmission_history[-self.history_window:]))

    def transmit(self, wave: complex) -> complex:
        """Transmit incoming wave through impedance boundary."""
        tx_wave = self.synapse.transmit(wave)
        T = self.power_transmission
        self.transmission_history.append(T)
        if len(self.transmission_history) > self.history_window * 2:
            self.transmission_history = self.transmission_history[-self.history_window:]
        return tx_wave

    def update_eligibility(self, z_post: complex, psi_pre: complex, dt: float = 0.02) -> complex:
        """Accumulate wave coherence in the resonant ringing trace."""
        return self.eligibility.step(z_post, psi_pre, dt=dt)

    def apply_neuromodulation(self, delta_dopamine: float, dt: float = 0.02):
        """Apply dopamine reward prediction error pulse to eligible pathway.
        
        Physics of impedance matching:
        - When delta > 0 (reward), drive Z towards Z0 (impedance match -> T -> 1.0).
        - When delta < 0 (punishment), drive Z away from Z0 (mismatch -> T -> 0.0).
        - Update is gated by the normalized eligibility trace.
        """
        c_mag = self.eligibility.magnitude
        if c_mag < 1e-6:
            return

        # Normalized direction vector of eligibility
        c_norm = self.eligibility.trace / (c_mag + 1e-4)

        # Distance from perfect impedance match
        z_diff = self.synapse.Z - self.synapse.Z0

        # Reinforcement gradient:
        # If delta > 0, pull Z closer to Z0 by fraction of z_diff
        # If delta < 0, push Z farther away from Z0
        dZ_rl = -self.lr_rl * float(delta_dopamine) * min(1.0, c_mag) * z_diff

        # Passive metabolic relaxation to prevent runaway reactance
        dZ_relax = -self.decay * (self.synapse.Z - 2.0)

        self.synapse.Z += (dZ_rl + dZ_relax) * dt

        # Enforce physical passive resistance constraint: 0.1 <= Re(Z) <= 100.0
        if self.synapse.Z.real < 0.1:
            self.synapse.Z = complex(0.1, self.synapse.Z.imag)
        elif self.synapse.Z.real > 100.0:
            self.synapse.Z = complex(100.0, self.synapse.Z.imag)

    def adapt_predictive(self, somatic_wave: complex, local_dendritic_wave: complex, dt: float = 0.02):
        """Unsupervised predictive impedance adaptation via maximum power transfer."""
        self.synapse.learn_step(somatic_wave, local_dendritic_wave, dt=dt)


class PredictiveCodingErrorJunction:
    """Multi-compartment dendritic junction computing physical wave cancellation.
    
    Computes destructive wave interference between sensory observation and top-down prediction:
        E(t) = Psi_sensory(t) - Psi_predict(t)
    When prediction perfectly matches sensory wave in frequency, amplitude, and phase,
    net error wave E(t) = 0 and somatic tension vanishes.
    """

    def __init__(self, forward_gain: float = 1.0, predict_gain: float = 1.0):
        self.forward_gain = float(forward_gain)
        self.predict_gain = float(predict_gain)
        self.error_wave: complex = 0.0 + 0.0j
        self.error_energy_trace: float = 0.0

    def compute_interference(
        self,
        sensory_wave: complex,
        prediction_wave: complex,
        dt: float = 0.02,
    ) -> complex:
        """Physically superimpose sensory and prediction waves destructively."""
        # Physical destructive interference: sensory - prediction
        self.error_wave = self.forward_gain * sensory_wave - self.predict_gain * prediction_wave
        instantaneous_energy = float(abs(self.error_wave) ** 2)
        self.error_energy_trace = 0.95 * self.error_energy_trace + 0.05 * instantaneous_energy
        return self.error_wave


class NeuromodulatedCRFNNetwork:
    """Unified CRFN Network combining Predictive Coding and Reinforcement Learning.
    
    Features:
    1. Sensory Predictive Coding: Destructive wave cancellation of predictable sensory dynamics.
    2. Recurrent Internal Microcircuit: Resonant memory and state-space dynamics.
    3. Continuous Actor-Critic Output:
       - Critic predicts state value V(t).
       - Actor generates continuous motor wave a(t).
       - Global dopamine RPE delta(t) = r(t) + gamma * V(t) - V(t - dt).
    4. Three-Factor Neuromodulated Synapses with resonant ringing eligibility traces.
    5. Autonomous Self-Organization: Neurogenesis on high prediction error, pruning on low transmission.
    """

    def __init__(
        self,
        n_sensory: int = 1,
        n_recurrent: int = 3,
        n_motor: int = 1,
        base_omega: float = 2.0,
        gamma_discount: float = 0.95,
        lr_rl: float = 1.0,
        lr_pred: float = 0.05,
        theta_grow: float = 0.85,
        theta_prune: float = 0.15,
        exploration_std: float = 0.05,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.base_omega = float(base_omega)
        self.gamma = float(gamma_discount)
        self.lr_rl = float(lr_rl)
        self.lr_pred = float(lr_pred)
        self.theta_grow = float(theta_grow)
        self.theta_prune = float(theta_prune)
        self.exploration_std = float(exploration_std)

        # Allocate neurons
        self.somas: dict[int, StuartLandauSoma] = {}
        self.dendrites: dict[int, TransmissionLineDendrite] = {}
        self.links: dict[tuple[int, int], NeuromodulatedSynapticLink] = {}

        self.sensory_ids = list(range(n_sensory))
        self.recurrent_ids = list(range(n_sensory, n_sensory + n_recurrent))
        self.value_id = n_sensory + n_recurrent  # Critic neuron
        self.motor_ids = [n_sensory + n_recurrent + 1 + i for i in range(n_motor)]

        self.all_ids = self.sensory_ids + self.recurrent_ids + [self.value_id] + self.motor_ids
        self.next_neuron_id = max(self.all_ids) + 1

        # Instantiate physical units: compact columnar scale (L=2.0, v=4.0)
        for nid in self.all_ids:
            # Somas: slightly detuned around base frequency to form rich frequency spectrum
            omega = self.base_omega + self.rng.uniform(-0.10, 0.10)
            # Receptive excitable regime near Hopf bifurcation
            self.somas[nid] = StuartLandauSoma(mu=-0.05, omega=omega, kappa=2.5, z_init=0.0)
            self.dendrites[nid] = TransmissionLineDendrite(
                length=2.0, n_nodes=15, velocity=4.0, damping=0.02
            )
            self.dendrites[nid].attach_synapse(0, relative_pos=0.15)

        # Predictive coding error junctions for sensory neurons
        self.predictive_junctions: dict[int, PredictiveCodingErrorJunction] = {
            s_id: PredictiveCodingErrorJunction() for s_id in self.sensory_ids
        }

        # Initialize synaptic wiring
        # 1. Sensory -> Recurrent (Matched to feed forward sensory power)
        for s in self.sensory_ids:
            for r in self.recurrent_ids:
                self._create_link(s, r, z_init=1.0)

        # 2. Recurrent -> Sensory (Top-Down Predictive Feedback)
        for r in self.recurrent_ids:
            for s in self.sensory_ids:
                self._create_link(r, s, z_init=1.5)

        # 3. Recurrent <-> Recurrent (Lateral Association)
        for r1 in self.recurrent_ids:
            for r2 in self.recurrent_ids:
                if r1 != r2:
                    self._create_link(r1, r2, z_init=1.2)

        # 4. Recurrent -> Value (Critic)
        for r in self.recurrent_ids:
            self._create_link(r, self.value_id, z_init=1.2)

        # 5. Recurrent -> Motor (Actor: initially mismatched so motor is quiet until learned)
        for r in self.recurrent_ids:
            for m in self.motor_ids:
                self._create_link(r, m, z_init=3.5)

        # Value tracking for TD-error
        self.v_previous = 0.0
        self.last_rpe = 0.0
        self.unabsorbed_error_trace = 0.0
        self.steps_since_growth = 0
        self.growth_cooldown = 100

    def _create_link(self, src: int, dst: int, z_init: complex | float = 2.0):
        """Wire a neuromodulated transmission line from src to dst."""
        if (src, dst) not in self.links:
            syn_id = len(self.dendrites[dst]._syn_to_node)
            rel_pos = float(np.clip(0.15 + 0.15 * self.rng.uniform(0, 1), 0.1, 0.35))
            self.dendrites[dst].attach_synapse(syn_id, relative_pos=rel_pos)
            link = NeuromodulatedSynapticLink(
                source_id=src,
                target_id=dst,
                dst_syn_id=syn_id,
                z_init=complex(z_init),
                lr_rl=self.lr_rl,
                lr_pred=self.lr_pred,
            )
            self.links[(src, dst)] = link

    def step(
        self,
        sensory_inputs: dict[int, complex | float],
        reward: float = 0.0,
        dt: float = 0.02,
    ) -> dict[str, float | complex]:
        """Execute one continuous wave step with predictive cancellation and dopamine RPE."""
        soma_states = {nid: s.z for nid, s in self.somas.items()}

        # 1. Predictive Coding Wave Cancellation on Sensory Inputs
        # Sensory dendrites receive top-down predictions from recurrent neurons
        effective_sensory_drive: dict[int, complex] = {}
        total_prediction_error = 0.0

        for s_id in self.sensory_ids:
            actual_input = complex(sensory_inputs.get(s_id, 0.0))

            # Sum incoming top-down prediction waves from recurrent neurons
            top_down_prediction = 0.0 + 0.0j
            for r_id in self.recurrent_ids:
                if (r_id, s_id) in self.links:
                    link = self.links[(r_id, s_id)]
                    top_down_prediction += link.transmit(soma_states[r_id])

            # Destructive wave cancellation at junction
            err_junction = self.predictive_junctions[s_id]
            error_wave = err_junction.compute_interference(actual_input, top_down_prediction, dt=dt)
            effective_sensory_drive[s_id] = error_wave
            total_prediction_error += float(abs(error_wave) ** 2)

        self.unabsorbed_error_trace = (
            0.95 * self.unabsorbed_error_trace + 0.05 * total_prediction_error
        )

        # 2. Inter-neuron Transmission & Dendritic Injections
        transmitted_to_dst: dict[int, dict[int, complex]] = {nid: {} for nid in self.somas}

        # Inject error waves into sensory neurons
        for s_id, err_w in effective_sensory_drive.items():
            transmitted_to_dst[s_id][0] = err_w

        # Transmit waves across all other links
        for (src, dst), link in self.links.items():
            if dst not in self.sensory_ids:  # Sensory top-down handled at junction
                if src in soma_states and dst in self.dendrites:
                    tx = link.transmit(soma_states[src])
                    transmitted_to_dst[dst][link.dst_syn_id] = tx

        # 3. Dendritic Wavefield Propagation & Somatic Limit-Cycles
        for nid in list(self.somas.keys()):
            nexus_flux = self.dendrites[nid].step(transmitted_to_dst[nid], dt=dt)
            # Add stochastic exploratory fluctuation for motor neurons
            if nid in self.motor_ids and self.exploration_std > 0:
                noise = complex(
                    self.rng.normal(0, self.exploration_std),
                    self.rng.normal(0, self.exploration_std),
                )
                nexus_flux += noise
            self.somas[nid].step(nexus_flux, dt=dt)

        new_soma_states = {nid: s.z for nid, s in self.somas.items()}

        # 4. Continuous Value Estimation & Dopamine RPE Computation
        # Critic neuron predicts scalar state value V(t) = |z_critic|^2 * sign(Re(z_critic))
        z_critic = new_soma_states[self.value_id]
        v_current = float(abs(z_critic) ** 2 * np.sign(z_critic.real))

        # Temporal Difference (TD) Reward Prediction Error:
        # delta(t) = r(t) + gamma * V(t) - V(t - dt)
        rpe = float(reward) + self.gamma * v_current - self.v_previous
        self.v_previous = v_current
        self.last_rpe = rpe

        # 5. Continuous Eligibility Traces & Three-Factor Neuromodulation
        for (src, dst), link in self.links.items():
            z_post = new_soma_states[dst]
            psi_pre = link.transmit(soma_states[src])

            # Update molecular/ringing eligibility trace
            link.update_eligibility(z_post, psi_pre, dt=dt)

            # Apply dopamine pulse to motor and value synapses
            if dst in self.motor_ids or dst == self.value_id:
                link.apply_neuromodulation(delta_dopamine=rpe, dt=dt)
            else:
                # Internal synapses adapt to minimize predictive error
                link.adapt_predictive(z_post, self.dendrites[dst].nexus_flux, dt=dt)

        # 6. Autonomous Structural Neurogenesis & Pruning
        self.steps_since_growth += 1
        # Growth: If persistent unabsorbed prediction error > theta_grow and cooldown passed
        if (
            self.steps_since_growth >= self.growth_cooldown
            and self.unabsorbed_error_trace > self.theta_grow
            and len(self.somas) < 15
        ):
            self.steps_since_growth = 0
            new_omega = self.base_omega + self.rng.uniform(-0.4, 0.4)
            new_nid = self.next_neuron_id
            self.next_neuron_id += 1

            self.somas[new_nid] = StuartLandauSoma(mu=-0.05, omega=new_omega, kappa=2.5)
            self.dendrites[new_nid] = TransmissionLineDendrite(
                length=2.0, n_nodes=15, velocity=4.0, damping=0.02
            )
            self.dendrites[new_nid].attach_synapse(0, relative_pos=0.15)
            self.recurrent_ids.append(new_nid)

            # Wire new neuron into circuit
            for s in self.sensory_ids:
                self._create_link(s, new_nid, z_init=1.0)
            for m in self.motor_ids:
                self._create_link(new_nid, m, z_init=3.5)
            self.unabsorbed_error_trace *= 0.5  # Discharge tension

        # Pruning: Sever non-transmitting synapses (T < theta_prune)
        prune_candidates = []
        for pair, link in self.links.items():
            if len(link.transmission_history) >= link.history_window:
                if link.mean_transmission < self.theta_prune:
                    prune_candidates.append(pair)
        for pair in prune_candidates:
            del self.links[pair]

        # Readout motor action
        # Motor action is continuous real flux from motor soma(s)
        motor_actions = [self.somas[m].z.real for m in self.motor_ids]

        return {
            "motor_action": motor_actions[0] if len(motor_actions) == 1 else motor_actions,
            "prediction_error": total_prediction_error,
            "rpe": rpe,
            "value": v_current,
            "n_neurons": len(self.somas),
            "n_links": len(self.links),
        }

    def reset(self):
        """Reset wavefields and traces while preserving learned synaptic impedances."""
        for s in self.somas.values():
            s.reset()
        for d in self.dendrites.values():
            d.reset()
        for link in self.links.values():
            link.eligibility.reset()
        self.v_previous = 0.0
        self.last_rpe = 0.0
        self.unabsorbed_error_trace = 0.0
