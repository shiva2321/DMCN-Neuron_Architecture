"""Coupled Continuous Resonant Field Neuron (CRFN) Microcircuit.

Models an authentic cortical microcolumn of N coupled non-linear field neurons:
1. Mutual Phase Entrainment (Phase Noise Dissipation):
   Solves the marginal phase stability problem of isolated oscillators.
   When coupling strength K exceeds the Kuramoto critical threshold K_c, all N neurons
   spontaneously phase-lock into a coherent collective wave:
       R(t) = |(1/N) * sum_j exp(i * phi_j)| -> 1.0
   Phase jitter is actively dissipated by the collective mean-field without a central dictionary.
2. Injection Locking (Arnold Tongue Frequency Resonance):
   When driven by an external sensory wave u(t) = exp(i * Omega * t), the microcircuit
   entrains its collective frequency directly to Omega, locking input to internal representation.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

from .crfn_unit import ContinuousResonantFieldNeuron
from .crfn_soma import StuartLandauSoma


class CRFNMicrocircuit:
    """A cortical microcircuit of N coupled Continuous Resonant Field Neurons."""

    def __init__(
        self,
        n_units: int = 8,
        base_omega: float = 2.0,
        omega_dispersion: float = 0.05,
        coupling_strength: float = 2.0,
        soma_mu: float = 1.0,
        seed: int = 42,
    ):
        self.N = int(n_units)
        self.K = float(coupling_strength)
        self.rng = np.random.RandomState(seed)

        # Distribute intrinsic frequencies around base_omega
        self.omegas = base_omega + self.rng.uniform(-omega_dispersion, omega_dispersion, self.N)

        # Initialize N somatic oscillators with random phases
        self.units: list[StuartLandauSoma] = []
        for i in range(self.N):
            phase = self.rng.uniform(-np.pi, np.pi)
            soma = StuartLandauSoma(mu=soma_mu, omega=float(self.omegas[i]), kappa=1.0)
            soma.z = complex(np.sqrt(max(0.01, soma_mu)) * np.exp(1j * phase))
            self.units.append(soma)

    @property
    def state_vector(self) -> np.ndarray:
        """Vector of complex somatic states z in C^N."""
        return np.array([u.z for u in self.units], dtype=np.complex128)

    @property
    def order_parameter(self) -> float:
        """Kuramoto order parameter R in [0, 1] measuring phase coherence."""
        phases = np.array([u.phase for u in self.units])
        mean_phasor = np.mean(np.exp(1j * phases))
        return float(abs(mean_phasor))

    @property
    def collective_phase(self) -> float:
        """Mean phase of the collective wavefield."""
        phases = np.array([u.phase for u in self.units])
        mean_phasor = np.mean(np.exp(1j * phases))
        return float(np.angle(mean_phasor))

    def step(
        self,
        external_inputs: Sequence[complex | float] | complex | float = 0.0,
        dt: float = 0.02,
    ) -> dict[str, float | np.ndarray]:
        """Execute one continuous time step with lateral diffusive wave coupling."""
        # Standardize external inputs
        if isinstance(external_inputs, (int, float, complex)):
            ext = np.full(self.N, complex(external_inputs), dtype=np.complex128)
        else:
            ext = np.array([complex(x) for x in external_inputs], dtype=np.complex128)

        current_Z = self.state_vector
        mean_Z = np.mean(current_Z)

        # Lateral coupling: K * (mean_Z - Z_i)
        lateral_drives = self.K * (mean_Z - current_Z)

        # Advance each unit with lateral coupling + external driving wave
        next_Z = []
        for i, unit in enumerate(self.units):
            total_drive = lateral_drives[i] + ext[i]
            z_next = unit.step(total_drive, dt=dt)
            next_Z.append(z_next)

        return {
            "order_parameter": self.order_parameter,
            "collective_phase": self.collective_phase,
            "states": np.array(next_Z, dtype=np.complex128),
        }

    def reset(self):
        """Re-randomize phases to test spontaneous resynchronization."""
        for unit in self.units:
            phase = self.rng.uniform(-np.pi, np.pi)
            unit.reset(phase=phase)
