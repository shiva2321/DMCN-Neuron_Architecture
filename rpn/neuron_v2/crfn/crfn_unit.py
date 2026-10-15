"""Continuous Resonant Field Neuron (CRFN) Unified Cellular Unit.

Unifies:
1. Spatial wave transmission dendrites with continuous partial differential equation wave dynamics.
2. Stuart-Landau limit-cycle somatic oscillator with autonomous Lyapunov noise dissipation.
3. Local membrane impedance plasticity via principle of maximum power transfer.

Completely free of:
- External codebooks / symbol tables (ItemTable).
- Matrix outer-product memories (DeltaMemory).
- Symbolic vector bindings (a * b circular convolution).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_impedance import ImpedanceSynapse


class ContinuousResonantFieldNeuron:
    """The Continuous Resonant Field Neuron (CRFN)."""

    def __init__(
        self,
        n_synapses: int = 4,
        dendrite_length: float = 10.0,
        n_dendrite_nodes: int = 50,
        soma_omega: float = 1.0,
        soma_mu: float = 1.0,
        soma_kappa: float = 0.8,
        wave_velocity: float = 1.0,
        damping: float = 0.10,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.n_synapses = int(n_synapses)
        self.rng = np.random.RandomState(seed)

        # 1. Somatic limit-cycle oscillator
        self.soma = StuartLandauSoma(
            mu=soma_mu,
            omega=soma_omega,
            kappa=soma_kappa,
        )

        # 2. Spatially extended wave transmission dendrite
        self.dendrite = TransmissionLineDendrite(
            length=dendrite_length,
            n_nodes=n_dendrite_nodes,
            velocity=wave_velocity,
            damping=damping,
            absorbing_boundary=True,
        )

        # 3. Local synapses with independent membrane impedances
        self.synapses: dict[int, ImpedanceSynapse] = {}
        # Distribute synapses evenly along the dendritic branch
        for i in range(self.n_synapses):
            rel_pos = float(0.2 + 0.7 * (i / max(1, self.n_synapses - 1)))
            self.dendrite.attach_synapse(i, relative_pos=rel_pos)
            self.synapses[i] = ImpedanceSynapse(
                syn_id=i,
                z0_membrane=1.0,
                lr=lr,
                decay=0.02,
            )

    @property
    def somatic_amplitude(self) -> float:
        """Current amplitude of the somatic limit cycle."""
        return self.soma.amplitude

    @property
    def somatic_phase(self) -> float:
        """Current phase of the somatic limit cycle in [-pi, pi]."""
        return self.soma.phase

    def step(
        self,
        inputs: dict[int, complex | float] | Sequence[complex | float],
        dt: float = 0.05,
    ) -> dict[str, complex | float]:
        """Execute one continuous time-step dt of the cellular wavefield."""
        # Standardize input format
        if isinstance(inputs, (list, tuple, np.ndarray)):
            input_dict = {i: inputs[i] for i in range(min(len(inputs), self.n_synapses))}
        else:
            input_dict = inputs

        # 1. Synaptic filtering via characteristic membrane impedance matching
        transmitted_inputs: dict[int, complex] = {}
        for syn_id, raw_signal in input_dict.items():
            if syn_id in self.synapses:
                transmitted_inputs[syn_id] = self.synapses[syn_id].transmit(raw_signal)

        # 2. Wave propagation and interference along transmission dendrite
        nexus_flux = self.dendrite.step(transmitted_inputs, dt=dt)

        # 3. Drive somatic limit-cycle oscillator with continuous nexus flux
        z_soma = self.soma.step(nexus_flux, dt=dt)

        return {
            "z_soma": z_soma,
            "somatic_amplitude": self.soma.amplitude,
            "somatic_phase": self.soma.phase,
            "nexus_flux": nexus_flux,
        }

    def learn(
        self,
        target_somatic_resonance: complex | float,
        dt: float = 0.05,
    ):
        """Autonomously adapt synaptic impedances using physical wave coincidence."""
        soma_wave = complex(target_somatic_resonance)
        for syn_id, syn in self.synapses.items():
            node_idx = self.dendrite._syn_to_node.get(syn_id, 0)
            local_wave = self.dendrite.psi[node_idx]
            syn.learn_step(soma_wave, local_wave, dt=dt)

    def reset(self):
        """Quench wavefields back to baseline limit cycle."""
        self.dendrite.reset()
        self.soma.reset()
