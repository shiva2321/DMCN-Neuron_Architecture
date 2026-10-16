"""Morphogenetic Continuous Resonant Field Neuron (CRFN) Substrate.

Implements an authentic, fully undifferentiated neural substrate with ZERO predefined neuron roles:
- No hardcoded "sensory neurons".
- No hardcoded "motor neurons".
- No hardcoded "critic / value neurons".
- No predefined cardinal directions (supports arbitrary M sensory ports and K actuator ports).

All neurons are identical pluripotent physical units:
    dz_j/dt = (mu + i*omega_j - |z_j|^2)*z_j + kappa * I_nexus_j(t) + exploration_noise

Dendritic transmission lines govern physical wave propagation in space-time:
    d^2Psi/dt^2 + gamma*dPsi/dt - v^2*nabla^2 Psi = sum_syn u_syn / dx

Connections (afferent from sensory ports, recurrent between neurons, efferent to actuators)
adapt strictly via local transmission-line impedance matching and 3-factor resonant eligibility:
    dZ/dt = -eta_RL * delta * (C / (|C| + eps)) * (Z - Z_0) + eta_pred * E_local - alpha*|z|^2*(Z - Z_rest)

Functional roles (sensory filters, working memories, motor drivers) emerge spontaneously
through closed-loop sensorimotor resonance, neurogenesis, and synaptic pruning.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Dict, Tuple, List, Optional, Any

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_rl_predictive import (
    ResonantEligibilityTrace,
    NeuromodulatedSynapticLink,
    PredictiveCodingErrorJunction,
)


class SensoryPort:
    """An open physical sensory bus channel carrying continuous input waves."""

    def __init__(self, port_id: int):
        self.id = int(port_id)
        self.error_junction = PredictiveCodingErrorJunction()
        self.last_wave: complex = 0.0 + 0.0j

    def process(self, sensory_wave: complex, top_down_prediction: complex, dt: float = 0.02) -> complex:
        """Destructive wave cancellation: returns unpredicted sensory error wave."""
        err_wave = self.error_junction.compute_interference(sensory_wave, top_down_prediction, dt=dt)
        self.last_wave = err_wave
        return err_wave

    @property
    def unabsorbed_energy(self) -> float:
        return self.error_junction.error_energy_trace


class ActuatorPort:
    """An open physical effector bus channel pooling wave fluxes to drive actuators."""

    def __init__(self, port_id: int):
        self.id = int(port_id)
        self.transmitted_flux: complex = 0.0 + 0.0j

    def reset_flux(self):
        self.transmitted_flux = 0.0 + 0.0j

    def receive_flux(self, wave: complex):
        self.transmitted_flux += wave

    @property
    def force(self) -> float:
        """Physical continuous wave power exerted on this actuator channel."""
        return float(abs(self.transmitted_flux) ** 2)


class MorphogeneticCRFNSubstrate:
    """A completely undifferentiated, self-organizing continuous resonant substrate.
    
    Zero predefined neuron roles. Connects to M arbitrary sensory ports and K arbitrary
    actuator ports. Neurons self-differentiate into sensory, memory, and motor roles
    strictly through closed-loop resonance, eligibility ringing, and dopamine feedback.
    """

    def __init__(
        self,
        n_initial_neurons: int = 4,
        n_sensory_ports: int = 2,
        n_actuator_ports: int = 2,
        base_omega: float = 2.0,
        base_mu: float = -0.05,
        gamma_discount: float = 0.90,
        lr_rl: float = 1.20,
        lr_pred: float = 0.05,
        theta_grow: float = 0.80,
        theta_prune: float = 0.15,
        exploration_std: float = 0.05,
        max_neurons: int = 20,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.base_omega = float(base_omega)
        self.base_mu = float(base_mu)
        self.gamma = float(gamma_discount)
        self.lr_rl = float(lr_rl)
        self.lr_pred = float(lr_pred)
        self.theta_grow = float(theta_grow)
        self.theta_prune = float(theta_prune)
        self.exploration_std = float(exploration_std)
        self.max_neurons = int(max_neurons)

        # 1. Instantiate open physical bus ports (no human semantic labels)
        self.sensory_ports = [SensoryPort(m) for m in range(n_sensory_ports)]
        self.actuator_ports = [ActuatorPort(k) for k in range(n_actuator_ports)]

        # 2. Instantiate undifferentiated pluripotent neurons
        self.next_neuron_id = 0
        self.somas: dict[int, StuartLandauSoma] = {}
        self.dendrites: dict[int, TransmissionLineDendrite] = {}

        # 3. Connection graphs (all use the same physical NeuromodulatedSynapticLink)
        # Afferent: (sensory_port_id, neuron_id) -> link
        self.afferent_links: dict[tuple[int, int], NeuromodulatedSynapticLink] = {}
        # Recurrent: (neuron_src, neuron_dst) -> link
        self.recurrent_links: dict[tuple[int, int], NeuromodulatedSynapticLink] = {}
        # Efferent: (neuron_id, actuator_port_id) -> link
        self.efferent_links: dict[tuple[int, int], NeuromodulatedSynapticLink] = {}

        # Spawn initial seed neurons
        for _ in range(n_initial_neurons):
            self._spawn_neuron(omega=self.base_omega + self.rng.uniform(-0.15, 0.15))

        # Initial stochastic wiring across the open bus
        neuron_ids = list(self.somas.keys())
        for nid in neuron_ids:
            # Connect each sensory port to some neurons
            for sp in self.sensory_ports:
                if self.rng.uniform(0, 1) < 0.8:
                    self._create_afferent_link(sp.id, nid, z_init=1.5)
            # Connect some recurrent links between neurons
            for other_nid in neuron_ids:
                if nid != other_nid and self.rng.uniform(0, 1) < 0.6:
                    self._create_recurrent_link(nid, other_nid, z_init=1.5)
            # Connect neurons to actuator ports (initially high impedance / mismatched)
            for ap in self.actuator_ports:
                if self.rng.uniform(0, 1) < 0.8:
                    self._create_efferent_link(nid, ap.id, z_init=5.0)

        # Continuous physical metabolic expectation baseline (no artificial linear weight vectors)
        self.expected_reward_baseline = 0.0
        self.tau_v = 1.0  # 1-second physical expectation window
        self.last_rpe = 0.0

        # Tension accumulator for resonance-driven neurogenesis
        self.steps_since_growth = 0
        self.growth_cooldown = 80

    def _spawn_neuron(self, omega: float, mu: float | None = None) -> int:
        """Spawn a completely identical pluripotent physical neuron."""
        nid = self.next_neuron_id
        self.next_neuron_id += 1
        effective_mu = self.base_mu if mu is None else float(mu)

        soma = StuartLandauSoma(mu=effective_mu, omega=omega, kappa=2.5, z_init=0.0)
        dendrite = TransmissionLineDendrite(length=2.0, n_nodes=15, velocity=4.0, damping=0.02)
        dendrite.attach_synapse(0, relative_pos=0.15)  # Default primary trunk entry

        self.somas[nid] = soma
        self.dendrites[nid] = dendrite
        return nid

    def _create_afferent_link(self, port_id: int, nid: int, z_init: float = 1.5):
        """Wire an open sensory port into a neuron's dendritic arbor."""
        if (port_id, nid) not in self.afferent_links:
            syn_id = len(self.dendrites[nid]._syn_to_node)
            rel_pos = float(np.clip(0.15 + 0.15 * self.rng.uniform(0, 1), 0.1, 0.35))
            self.dendrites[nid].attach_synapse(syn_id, relative_pos=rel_pos)
            link = NeuromodulatedSynapticLink(
                source_id=-1 - port_id,
                target_id=nid,
                dst_syn_id=syn_id,
                z_init=complex(z_init),
                lr_rl=self.lr_rl,
                lr_pred=self.lr_pred,
            )
            self.afferent_links[(port_id, nid)] = link

    def _create_recurrent_link(self, src: int, dst: int, z_init: float = 1.5):
        """Wire a physical transmission line between two neurons."""
        if (src, dst) not in self.recurrent_links:
            syn_id = len(self.dendrites[dst]._syn_to_node)
            rel_pos = float(np.clip(0.20 + 0.20 * self.rng.uniform(0, 1), 0.1, 0.45))
            self.dendrites[dst].attach_synapse(syn_id, relative_pos=rel_pos)
            link = NeuromodulatedSynapticLink(
                source_id=src,
                target_id=dst,
                dst_syn_id=syn_id,
                z_init=complex(z_init),
                lr_rl=self.lr_rl,
                lr_pred=self.lr_pred,
            )
            self.recurrent_links[(src, dst)] = link

    def _create_efferent_link(self, nid: int, port_id: int, z_init: float = 3.5):
        """Wire a neuron's output flux into an actuator port."""
        if (nid, port_id) not in self.efferent_links:
            link = NeuromodulatedSynapticLink(
                source_id=nid,
                target_id=-1 - port_id,
                dst_syn_id=0,
                z_init=complex(z_init),
                lr_rl=self.lr_rl,
                lr_pred=self.lr_pred,
            )
            self.efferent_links[(nid, port_id)] = link

    def step(
        self,
        sensory_inputs: dict[int, complex | float],
        reward: float = 0.0,
        dt: float = 0.02,
        teacher_actuator_fluxes: dict[int, complex] | None = None,
    ) -> dict[str, Any]:
        """Execute one continuous wave step with emergent role differentiation."""
        soma_states = {nid: s.z for nid, s in self.somas.items()}

        # 1. Open Sensory Bus: Destructive Wave Cancellation on Sensory Ports
        # Compute top-down predictions arriving at sensory ports from the whole substrate
        effective_port_waves: dict[int, complex] = {}
        total_sensory_error = 0.0

        for sp in self.sensory_ports:
            raw_input = complex(sensory_inputs.get(sp.id, 0.0))

            # Internal top-down prediction arriving from any connected neurons
            top_down_pred = 0.0 + 0.0j
            for (pid, nid), link in self.afferent_links.items():
                if pid == sp.id:
                    # Reverse predictive transmission
                    top_down_pred += link.synapse.power_transmission * soma_states[nid]

            err_w = sp.process(raw_input, top_down_pred, dt=dt)
            effective_port_waves[sp.id] = err_w
            total_sensory_error += float(abs(err_w) ** 2)

        # 2. Route Inputs & Inter-Neuron Waves to Dendritic Arbor
        dendrite_inputs: dict[int, dict[int, complex]] = {nid: {} for nid in self.somas}

        # Afferent drive from sensory ports
        for (port_id, nid), link in self.afferent_links.items():
            if port_id in effective_port_waves and nid in self.dendrites:
                tx = link.transmit(effective_port_waves[port_id])
                dendrite_inputs[nid][link.dst_syn_id] = tx

        # Recurrent wave transmission
        for (src, dst), link in self.recurrent_links.items():
            if src in soma_states and dst in self.dendrites:
                tx = link.transmit(soma_states[src])
                dendrite_inputs[dst][link.dst_syn_id] = tx

        # 3. Dendritic Wavefield Propagation & Somatic Limit-Cycles
        for nid in list(self.somas.keys()):
            nexus_flux = self.dendrites[nid].step(dendrite_inputs[nid], dt=dt)
            # Natural thermal/membrane fluctuation for motor exploration
            if self.exploration_std > 0:
                noise = complex(
                    self.rng.normal(0, self.exploration_std),
                    self.rng.normal(0, self.exploration_std),
                )
                nexus_flux += noise
            self.somas[nid].step(nexus_flux, dt=dt)

        new_soma_states = {nid: s.z for nid, s in self.somas.items()}

        # 4. Open Actuator Bus: Pool Transmitted Fluxes to Drive Actuators
        for ap in self.actuator_ports:
            ap.reset_flux()

        for (nid, port_id), link in self.efferent_links.items():
            if nid in new_soma_states:
                tx_flux = link.transmit(new_soma_states[nid])
                self.actuator_ports[port_id].receive_flux(tx_flux)

        if teacher_actuator_fluxes is not None:
            for port_id, flux in teacher_actuator_fluxes.items():
                if 0 <= port_id < len(self.actuator_ports):
                    self.actuator_ports[port_id].receive_flux(complex(flux))

        # 5. Continuous Metabolic Expectation & Dopamine RPE Computation
        # Deviation from running physical baseline: delta(t) = r(t) - R_bar(t)
        rpe = float(reward) - self.expected_reward_baseline
        # Update running metabolic expectation baseline continuously: dR_bar/dt = (reward - R_bar) / tau_v
        self.expected_reward_baseline += (float(reward) - self.expected_reward_baseline) * (dt / self.tau_v)
        self.last_rpe = rpe

        # 6. Three-Factor Neuromodulation & Predictive Plasticity Across All Links
        # A. Afferent links adapt predictively and via global dopamine RPE
        for (port_id, nid), link in self.afferent_links.items():
            z_post = new_soma_states[nid]
            port_w = effective_port_waves.get(port_id, 0.0 + 0.0j)
            link.update_eligibility(z_post, port_w, dt=dt)
            link.adapt_predictive(z_post, self.dendrites[nid].nexus_flux, dt=dt)
            link.apply_neuromodulation(delta_dopamine=rpe, dt=dt)

        # B. Recurrent links adapt via resonance + eligibility + dopamine
        for (src, dst), link in self.recurrent_links.items():
            z_post = new_soma_states[dst]
            psi_pre = soma_states[src]
            link.update_eligibility(z_post, psi_pre, dt=dt)
            link.adapt_predictive(z_post, self.dendrites[dst].nexus_flux, dt=dt)
            link.apply_neuromodulation(delta_dopamine=rpe, dt=dt)

        # C. Efferent links adapt via Dopamine RPE on resonant ringing traces
        for (nid, port_id), link in self.efferent_links.items():
            # Actuator channel post-synaptic activity is the continuous wave flux at that port
            if teacher_actuator_fluxes is not None:
                z_post = teacher_actuator_fluxes.get(port_id, 0.0 + 0.0j)
            else:
                z_post = self.actuator_ports[port_id].transmitted_flux
            psi_pre = new_soma_states[nid]
            link.update_eligibility(z_post, psi_pre, dt=dt)
            link.apply_neuromodulation(delta_dopamine=rpe, dt=dt)

        # 7. Morphogenetic Neurogenesis & Pruning
        self.steps_since_growth += 1
        # Neurogenesis: If sensory ports have high unabsorbed tension and cooldown passed
        if (
            self.steps_since_growth >= self.growth_cooldown
            and total_sensory_error > self.theta_grow
            and len(self.somas) < self.max_neurons
        ):
            self.steps_since_growth = 0
            new_omega = self.base_omega + self.rng.uniform(-0.3, 0.3)
            new_nid = self._spawn_neuron(omega=new_omega)

            # Sprout connections from the highest tension sensory port
            max_sp = max(self.sensory_ports, key=lambda p: p.unabsorbed_energy)
            self._create_afferent_link(max_sp.id, new_nid, z_init=1.0)

            # Sprout recurrent connection to an active peer
            active_peer = max(self.somas.keys(), key=lambda k: self.somas[k].amplitude if k != new_nid else -1.0)
            self._create_recurrent_link(active_peer, new_nid, z_init=1.2)
            self._create_recurrent_link(new_nid, active_peer, z_init=1.2)

            # Sprout exploratory efferent connection to an actuator port
            act_p = self.rng.choice(self.actuator_ports)
            self._create_efferent_link(new_nid, act_p.id, z_init=5.0)

        # Pruning: Sever links that consistently reflect power (T < theta_prune)
        prune_recurrent = [
            pair for pair, l in self.recurrent_links.items()
            if len(l.transmission_history) >= l.history_window and l.mean_transmission < self.theta_prune
        ]
        for pair in prune_recurrent:
            del self.recurrent_links[pair]

        prune_efferent = [
            pair for pair, l in self.efferent_links.items()
            if len(l.transmission_history) >= l.history_window and l.mean_transmission < self.theta_prune
        ]
        for pair in prune_efferent:
            del self.efferent_links[pair]

        # Readout all actuator forces
        actuator_forces = [ap.force for ap in self.actuator_ports]

        return {
            "actuator_forces": actuator_forces,
            "prediction_error": total_sensory_error,
            "rpe": rpe,
            "expected_reward": self.expected_reward_baseline,
            "n_neurons": len(self.somas),
            "n_afferent": len(self.afferent_links),
            "n_recurrent": len(self.recurrent_links),
            "n_efferent": len(self.efferent_links),
        }

    def inspect_emergent_roles(self) -> dict[int, str]:
        """Classify each neuron's emergent role objectively by its transmission dominance.
        
        Roles are NOT predefined; they emerge from which impedances match:
        - Sensory Specialist: High afferent transmission from sensory ports.
        - Motor Specialist: High efferent transmission to actuator ports.
        - Association / Memory Specialist: High recurrent transmission, low direct bus links.
        """
        roles = {}
        for nid in self.somas:
            afferent_T = sum(l.power_transmission for (pid, n), l in self.afferent_links.items() if n == nid)
            efferent_T = sum(l.power_transmission for (n, aid), l in self.efferent_links.items() if n == nid)
            recurrent_T = sum(l.power_transmission for (src, dst), l in self.recurrent_links.items() if src == nid or dst == nid)

            if efferent_T > 0.85 and efferent_T > afferent_T:
                roles[nid] = "Motor Specialist"
            elif afferent_T > 0.85 and afferent_T > efferent_T:
                roles[nid] = "Sensory Specialist"
            elif recurrent_T > 0.5:
                roles[nid] = "Association / Memory Specialist"
            else:
                roles[nid] = "Undifferentiated Pluripotent"
        return roles

    def reset(self):
        """Quench wavefields while preserving learned impedances and network structure."""
        for s in self.somas.values():
            s.reset()
        for d in self.dendrites.values():
            d.reset()
        for l in self.afferent_links.values():
            l.eligibility.reset()
        for l in self.recurrent_links.values():
            l.eligibility.reset()
        for l in self.efferent_links.values():
            l.eligibility.reset()
        for sp in self.sensory_ports:
            sp.last_wave = 0.0 + 0.0j
        for ap in self.actuator_ports:
            ap.reset_flux()
        self.expected_reward_baseline = 0.0
        self.last_rpe = 0.0
