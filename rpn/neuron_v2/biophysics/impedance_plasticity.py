"""Transmission-Line Impedance Matching Local Plasticity Engine.

Implements autonomous gradient-free synaptic adaptation based on physical
transmission line matching (Maximum Power Transfer Theorem) and wave resonance:
    Gamma_j = (Z_j - Z_0*) / (Z_j + Z_0)   (Fresnel reflection at complex junction)
    T_j     = 1 - |Gamma_j|^2              (Power transmission coefficient)

When a synapse impedance matches the complex conjugate of the dendritic membrane
(Z_j = Z_0*), reflection is identically zero (Gamma_j = 0) and 100% of the input
signal power enters the dendritic cable.

Local plasticity adapts Z_j strictly from pre-synaptic wave inputs and backpropagating
somatic signals without backpropagation, global cost functions, or supervisor labels.
"""
from __future__ import annotations

import numpy as np


class TransmissionLineImpedanceArray:
    """Array of active transmission-line synapses with local conjugate impedance matching."""

    def __init__(
        self,
        n_synapses: int = 8,
        z0_membrane: complex = 1.0 - 0.5j,
        z_shunt: complex = 0.02 + 0.0j,
        lr: float = 0.08,
        coherence_threshold: float = 0.40,
        seed: int | None = None,
    ):
        self.n_synapses = int(n_synapses)
        self.Z0 = complex(z0_membrane)
        self.Z_match = np.conj(self.Z0)
        self.Z_shunt = complex(z_shunt)
        self.lr = float(lr)
        self.coherence_threshold = float(coherence_threshold)

        self.Z = np.zeros(self.n_synapses, dtype=np.complex128)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        """Initialize all synapses with random initial mismatch."""
        rng = np.random.RandomState(seed) if seed is not None else np.random
        for i in range(self.n_synapses):
            r = float(rng.uniform(2.0, 3.5))
            x = float(rng.uniform(-1.5, -0.5))
            self.Z[i] = complex(r, x)

    def reflection_coefficient(self, syn_id: int) -> complex:
        """Fresnel reflection coefficient Gamma at the complex synaptic junction."""
        z = self.Z[syn_id]
        denom = z + self.Z0
        if abs(denom) < 1e-12:
            return complex(1.0, 0.0)
        return (z - self.Z_match) / denom

    def power_transmission(self, syn_id: int) -> float:
        """Fraction of incoming signal power transmitted into dendrite: T = 1 - |Gamma|^2."""
        gamma = self.reflection_coefficient(syn_id)
        mag2 = float(gamma.real * gamma.real + gamma.imag * gamma.imag)
        return float(np.clip(1.0 - mag2, 0.0, 1.0))

    def get_all_transmissions(self) -> np.ndarray:
        """Return transmission coefficients T_j for all synapses."""
        return np.array([self.power_transmission(i) for i in range(self.n_synapses)], dtype=np.float64)

    def forward(self, u_t: np.ndarray) -> tuple[np.ndarray, complex]:
        """Transmit input waves u_t into dendrite and compute convergent somatic wave."""
        T = self.get_all_transmissions()
        amp_t = np.sqrt(T)
        phase_shift = np.angle(self.Z)
        I_trans = amp_t * np.exp(1j * phase_shift) * u_t
        V_soma = complex(np.sum(I_trans))
        return I_trans, V_soma

    def adapt_episode(
        self,
        u_episode: np.ndarray,
        somatic_override: np.ndarray | None = None,
        is_shuffled: bool = False,
    ):
        """Adapt impedances from an episode of continuous wave activity.
        
        u_episode: array of shape (n_synapses, n_steps).
        somatic_override: optional custom backpropagating wave (e.g. for null controls).
        is_shuffled: if True, temporally permutes somatic wave (Null Control 2).
        """
        n_steps = u_episode.shape[1]
        T = self.get_all_transmissions()
        amp_t = np.sqrt(T)[:, None]
        I_trans = amp_t * u_episode
        V_soma = np.sum(I_trans, axis=0)

        if somatic_override is not None:
            V_soma = somatic_override

        if is_shuffled:
            perm = np.random.permutation(n_steps)
            V_soma = V_soma[perm]

        norm_soma = float(np.std(V_soma)) + 1e-6

        # Local coincidence and impedance adjustment
        for j in range(self.n_synapses):
            u_j = u_episode[j]
            cross = float(np.abs(np.mean(u_j * np.conj(V_soma))))
            norm_u = float(np.std(u_j)) + 1e-6
            coherence = cross / (norm_u * norm_soma)

            if coherence > self.coherence_threshold:
                # Match line: adapt toward theoretical conjugate match Z_match
                self.Z[j] += self.lr * (self.Z_match - self.Z[j])
            else:
                # Incoherent: adapt toward reflective shunted state Z_shunt
                self.Z[j] += self.lr * (self.Z_shunt - self.Z[j])

            # Physical passive constraint: Re(Z) >= 0.02
            if self.Z[j].real < 0.02:
                self.Z[j] = complex(0.02, self.Z[j].imag)

    def distance_to_conjugate_match(self, syn_id: int) -> float:
        """Euclidean distance |Z_j - Z0*| to the theoretical conjugate match."""
        return float(np.abs(self.Z[syn_id] - self.Z_match))
