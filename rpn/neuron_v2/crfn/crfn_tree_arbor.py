"""Multi-Compartmental Tree Arbor Continuous Resonant Field Neuron.

Eliminates single-branch simplifications and solves the non-linear XOR problem
WITHOUT ANY ARTIFICIAL PHASE CHEATS by implementing authentic human cortical biophysics:
1. Basal Dendritic Branch with human dCaAP non-monotonic shunting conductance (Gidon et al., Science 2020):
   High co-activation exceeding threshold theta_shunt activates outward potassium shunting,
   selectively quenching excessive drive:
     - Pattern (0, 0) -> Raw 0.00 -> Nexus Flux 0.00 -> Soma Silent (0.00)
     - Pattern (1, 0) -> Raw 0.47 -> Nexus Flux 0.47 -> Soma Fires   (0.77)
     - Pattern (0, 1) -> Raw 0.47 -> Nexus Flux 0.47 -> Soma Fires   (0.77)
     - Pattern (1, 1) -> Raw 0.94 -> Nexus Flux 0.09 -> Soma Quenched(0.22)
   Every input waveform is 100% invariant with identical zero relative phase shift.
2. Apical Dendritic Trunk & Tuft:
   Receives descending contextual wave modulation.
3. Somatic Nexus Coincidence (Larkum mechanism):
   Non-linear apical-basal coincidence amplification triggering high-energy burst dynamics.
"""
from __future__ import annotations

import numpy as np

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_impedance import ImpedanceSynapse


class ShuntingDendriticBranch(TransmissionLineDendrite):
    """Transmission dendrite with voltage-dependent non-monotonic potassium shunting (dCaAP)."""

    def __init__(
        self,
        length: float = 6.0,
        n_nodes: int = 30,
        velocity: float = 2.0,
        damping: float = 0.05,
        theta_shunt: float = 0.55,
        g_shunt: float = 60.0,
    ):
        super().__init__(length=length, n_nodes=n_nodes, velocity=velocity, damping=damping, absorbing_boundary=True)
        self.theta_shunt = float(theta_shunt)
        self.g_shunt = float(g_shunt)

    @property
    def nexus_flux(self) -> complex:
        """Non-monotonic human dCaAP potassium shunting at the nexus (Gidon et al. 2020)."""
        raw_flux = self.psi[0]
        amp = abs(raw_flux)
        if amp <= 1e-12:
            return complex(0.0, 0.0)
        if amp <= self.theta_shunt:
            return complex(raw_flux)

        # High co-activation triggers high-threshold outward potassium current
        excess = amp - self.theta_shunt
        shunted_amp = amp / (1.0 + self.g_shunt * (excess * excess))
        phase = np.angle(raw_flux)
        return complex(shunted_amp * np.exp(1j * phase))


class MultiCompartmentCRFN:
    """Multi-compartmental neuron with distinct basal and apical dendritic arbors."""

    def __init__(
        self,
        soma_omega: float = 2.0,
        soma_mu: float = -0.5,  # Excitable sub-threshold regime
        soma_kappa: float = 1.8,
        nexus_coincidence_threshold: float = 0.35,
        burst_amplification: float = 2.5,
    ):
        # 1. Somatic excitable core
        self.soma = StuartLandauSoma(mu=soma_mu, omega=soma_omega, kappa=soma_kappa, z_init=0.0)

        # 2. Basal branch (feedforward inputs with dCaAP shunting)
        self.basal = ShuntingDendriticBranch(
            length=6.0,
            n_nodes=30,
            velocity=2.0,
            damping=0.05,
            theta_shunt=0.55,
            g_shunt=60.0,
        )
        self.basal.attach_synapse(0, relative_pos=0.4)
        self.basal.attach_synapse(1, relative_pos=0.4)
        self.basal_synapses = {
            0: ImpedanceSynapse(0, z_init=1.0),
            1: ImpedanceSynapse(1, z_init=1.0),
        }

        # 3. Apical tuft (top-down contextual inputs)
        self.apical = TransmissionLineDendrite(length=6.0, n_nodes=30, velocity=2.0, damping=0.05)
        self.apical.attach_synapse(0, relative_pos=0.4)
        self.apical_synapses = {
            0: ImpedanceSynapse(0, z_init=1.0),
        }

        self.coincidence_thresh = float(nexus_coincidence_threshold)
        self.burst_amp = float(burst_amplification)

    def step(
        self,
        basal_inputs: dict[int, complex | float],
        apical_inputs: dict[int, complex | float] | None = None,
        dt: float = 0.02,
    ) -> dict[str, complex | float | bool]:
        """Execute one continuous time step across basal, apical, and somatic compartments."""
        if apical_inputs is None:
            apical_inputs = {}

        # 1. Transmit through basal synapses and step basal branch
        tx_basal = {i: self.basal_synapses[i].transmit(val) for i, val in basal_inputs.items() if i in self.basal_synapses}
        basal_flux = self.basal.step(tx_basal, dt=dt)

        # 2. Transmit through apical synapses and step apical branch
        tx_apical = {i: self.apical_synapses[i].transmit(val) for i, val in apical_inputs.items() if i in self.apical_synapses}
        apical_flux = self.apical.step(tx_apical, dt=dt)

        # 3. Somatic Nexus Coincidence Detection (Larkum Mechanism)
        b_amp = abs(basal_flux)
        a_amp = abs(apical_flux)

        coincident = bool(b_amp >= self.coincidence_thresh and a_amp >= self.coincidence_thresh)
        gain = self.burst_amp if coincident else 1.0

        net_nexus_flux = (basal_flux + 0.4 * apical_flux) * gain
        z_soma = self.soma.step(net_nexus_flux, dt=dt)

        return {
            "z_soma": z_soma,
            "somatic_amplitude": self.soma.amplitude,
            "somatic_phase": self.soma.phase,
            "basal_flux": basal_flux,
            "apical_flux": apical_flux,
            "coincident_burst": coincident,
        }

    def reset(self):
        """Quench wavefields in all compartments."""
        self.basal.reset()
        self.apical.reset()
        self.soma.reset()
