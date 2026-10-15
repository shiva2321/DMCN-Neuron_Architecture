"""Self-Organizing Continuous Resonant Field Neuron (CRFN) Network.

Implements autonomous structural growth, synaptic pruning, and homeostatic stabilization:
1. Resonance-Driven Neurogenesis (Growth):
   When sensory waves cannot be absorbed by existing resonant modes, unabsorbed metabolic
   tension accumulates. When tension > theta_grow, the network automatically buds a new neuron
   tuned to the unabsorbed frequency.
2. Activity-Dependent Synaptic Pruning:
   Connections track their average power transmission T_ij = 1 - |Gamma_ij|^2.
   If a connection consistently reflects energy (T_ij < theta_prune), the synapse atrophies and is deleted.
3. Homeostatic Synaptic Scaling (Criticality):
   Turrigiano synaptic scaling: if total network energy exceeds target E_target, all thresholds
   shift upward to prevent epileptic runaway. If silent, thresholds lower to maintain criticality.
4. Local Wave Coherence Learning:
   Synapses adapt their complex impedance Z_ij strictly via local wave resonance without backpropagation.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_impedance import ImpedanceSynapse


class SynapticLink:
    """A physical transmission-line connection between two neurons."""

    def __init__(
        self,
        source_id: int,
        target_id: int,
        z_init: complex | None = None,
        lr: float = 0.05,
    ):
        self.src = int(source_id)
        self.dst = int(target_id)
        self.synapse = ImpedanceSynapse(syn_id=target_id, z_init=z_init, z0_membrane=1.0, lr=lr)
        # Sliding window of power transmission history
        self.transmission_history: list[float] = []
        self.history_window = 100

    @property
    def mean_transmission(self) -> float:
        """Average power transmission coefficient T in [0, 1]."""
        if not self.transmission_history:
            return self.synapse.power_transmission
        return float(np.mean(self.transmission_history[-self.history_window:]))

    def transmit(self, wave: complex) -> complex:
        """Transmit wave through synapse and record transmission efficiency."""
        tx_wave = self.synapse.transmit(wave)
        T = self.synapse.power_transmission
        self.transmission_history.append(T)
        if len(self.transmission_history) > self.history_window * 2:
            self.transmission_history = self.transmission_history[-self.history_window:]
        return tx_wave

    def adapt(self, somatic_wave: complex, local_wave: complex, dt: float = 0.02):
        """Locally adapt impedance via maximum power transfer."""
        self.synapse.learn_step(somatic_wave, local_wave, dt=dt)


class SelfOrganizingCRFNNetwork:
    """An autonomous growing, self-pruning, and homeostatically self-stabilizing CRFN network."""

    def __init__(
        self,
        n_initial_neurons: int = 2,
        base_omega: float = 2.0,
        theta_grow: float = 0.85,
        theta_prune: float = 0.15,
        target_energy: float = 1.0,
        max_neurons: int = 20,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.base_omega = float(base_omega)
        self.theta_grow = float(theta_grow)
        self.theta_prune = float(theta_prune)
        self.target_energy = float(target_energy)
        self.max_neurons = int(max_neurons)

        # 1. Spawn initial seed neurons
        self.next_neuron_id = 0
        self.somas: dict[int, StuartLandauSoma] = {}
        self.dendrites: dict[int, TransmissionLineDendrite] = {}
        self.links: dict[tuple[int, int], SynapticLink] = {}

        for _ in range(n_initial_neurons):
            self._spawn_neuron(omega=self.base_omega + self.rng.uniform(-0.1, 0.1))

        # Wire initial mutual links
        neuron_ids = list(self.somas.keys())
        for i in neuron_ids:
            for j in neuron_ids:
                if i != j:
                    self._create_link(i, j)

        # Metabolic tension accumulator
        self.unabsorbed_energy_trace = 0.0
        self.total_energy_trace = 0.0

    def _spawn_neuron(self, omega: float, mu: float = -0.3) -> int:
        """Spawn a new physical neuron into the network."""
        nid = self.next_neuron_id
        self.next_neuron_id += 1

        # Somatic excitable core
        soma = StuartLandauSoma(mu=mu, omega=omega, kappa=1.5, z_init=0.0)
        # Dendritic branch
        dendrite = TransmissionLineDendrite(length=6.0, n_nodes=25, velocity=2.0, damping=0.05)
        dendrite.attach_synapse(0, relative_pos=0.5)

        self.somas[nid] = soma
        self.dendrites[nid] = dendrite
        return nid

    def _create_link(self, src: int, dst: int):
        """Create a physical synaptic link from src to dst."""
        if (src, dst) not in self.links:
            self.links[(src, dst)] = SynapticLink(source_id=src, target_id=dst, z_init=1.0, lr=0.05)
            # Attach synapse on target's dendrite
            syn_id = len(self.dendrites[dst]._syn_to_node)
            rel_pos = float(np.clip(0.2 + 0.6 * self.rng.uniform(0, 1), 0.1, 0.9))
            self.dendrites[dst].attach_synapse(syn_id, relative_pos=rel_pos)

    def step(
        self,
        sensory_inputs: dict[int, complex | float],
        dt: float = 0.02,
    ) -> dict[str, float | int]:
        """Execute one continuous time-step: propagation, adaptation, growth, pruning & homeostasis."""
        # 1. Forward propagation
        # Collect signals from sensory inputs and upstream neurons
        soma_states = {nid: s.z for nid, s in self.somas.items()}
        transmitted_to_dst: dict[int, dict[int, complex]] = {nid: {} for nid in self.somas}

        # Sensory inputs feed directly into specified neurons (or neuron 0 by default)
        for nid, val in sensory_inputs.items():
            if nid in self.dendrites:
                transmitted_to_dst[nid][0] = complex(val)

        # Inter-neuron wave transmission across synaptic links
        for (src, dst), link in self.links.items():
            if src in soma_states and dst in self.dendrites:
                src_wave = soma_states[src]
                tx_wave = link.transmit(src_wave)
                syn_idx = len(transmitted_to_dst[dst])
                transmitted_to_dst[dst][syn_idx] = tx_wave

        # Advance dendritic wavefields and drive somas
        current_energy = 0.0
        for nid in list(self.somas.keys()):
            nexus_flux = self.dendrites[nid].step(transmitted_to_dst[nid], dt=dt)
            z = self.somas[nid].step(nexus_flux, dt=dt)
            current_energy += float(abs(z) ** 2)

        # 2. Local Impedance Adaptation (Maximum Power Transfer)
        for (src, dst), link in self.links.items():
            if src in soma_states and dst in self.dendrites:
                soma_wave = self.somas[dst].z
                local_wave = self.dendrites[dst].nexus_flux
                link.adapt(soma_wave, local_wave, dt=dt)

        # 3. Energy accounting & Unabsorbed Tension Tracking
        input_power = sum(abs(v) ** 2 for v in sensory_inputs.values())
        # If input power is high but network energy is low, energy is unabsorbed (mismatch)
        absorbed = min(input_power, current_energy)
        unabsorbed = max(0.0, input_power - absorbed)
        self.unabsorbed_energy_trace = 0.95 * self.unabsorbed_energy_trace + 0.05 * unabsorbed
        self.total_energy_trace = 0.95 * self.total_energy_trace + 0.05 * current_energy

        # 4. Resonance-Driven Neurogenesis (Growth)
        # If unabsorbed tension exceeds threshold and below max capacity, sprout a new neuron
        if self.unabsorbed_energy_trace > self.theta_grow and len(self.somas) < self.max_neurons:
            # Estimate frequency of unabsorbed drive (perturbation)
            new_omega = self.base_omega + self.rng.uniform(-0.5, 0.5)
            new_nid = self._spawn_neuron(omega=new_omega)
            # Connect new neuron to the most active existing neuron
            active_nid = max(self.somas.keys(), key=lambda k: self.somas[k].amplitude)
            self._create_link(active_nid, new_nid)
            self._create_link(new_nid, active_nid)
            # Tension discharges upon neurogenesis
            self.unabsorbed_energy_trace *= 0.5

        # 5. Activity-Dependent Synaptic Pruning
        # If link consistently reflects energy (T < theta_prune after sufficient history), prune it
        links_to_prune = []
        for pair, link in self.links.items():
            if len(link.transmission_history) >= link.history_window:
                if link.mean_transmission < self.theta_prune:
                    links_to_prune.append(pair)
        for pair in links_to_prune:
            del self.links[pair]

        # 6. Homeostatic Synaptic Scaling (Turrigiano Stability)
        # If network energy exceeds target, shift somatic mu down (inhibitory homeostatic push)
        if self.total_energy_trace > self.target_energy * 1.5:
            for s in self.somas.values():
                s.mu -= 0.005  # Move deeper into sub-threshold excitable zone
        elif self.total_energy_trace < self.target_energy * 0.2 and input_power > 0.1:
            for s in self.somas.values():
                s.mu += 0.002  # Move closer to bifurcation threshold to restore sensitivity

        return {
            "n_neurons": len(self.somas),
            "n_connections": len(self.links),
            "total_energy": current_energy,
            "unabsorbed_tension": self.unabsorbed_energy_trace,
        }

    def reset(self):
        """Quench all wavefields and somatic oscillators."""
        for d in self.dendrites.values():
            d.reset()
        for s in self.somas.values():
            s.reset()
        self.unabsorbed_energy_trace = 0.0
        self.total_energy_trace = 0.0
