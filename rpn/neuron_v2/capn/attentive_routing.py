"""Dynamic Attentional Gating and Task Routing Engine (CAPN Pillar 3).

Integrates:
- Communication-Through-Coherence (CTC): Inter-areal gamma-band oscillatory coupling
  where top-down disinhibition modulates coupling strength K_i = kappa * c_i, driving
  phase synchronization and opening the transmission window without central oracles.
- Top-Down Apical Dendritic Shunting: Contextual inhibition selectively shunts
  task-irrelevant pathways to suppress cross-talk without soft-max attention matrices.
"""

from typing import Any, Dict, Optional
import numpy as np


class AttentiveRoutingEngine:
    """Cortical attentional routing engine using stochastic phase coherence and apical shunting."""

    def __init__(
        self,
        dim: int = 64,
        omega_gamma: float = 0.25,      # rad/ms (~40 Hz base)
        coupling_kappa: float = 0.50,   # Inter-areal coupling strength
        tau_context: float = 5.0,       # Context filter time constant (ms)
        sigma_noise: float = 0.02,      # Physiological membrane phase noise
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        """Initialize routing engine.

        Args:
            dim: Dimension D of stream vectors.
            omega_gamma: Intrinsic base gamma frequency in rad/ms.
            coupling_kappa: Synaptic phase coupling strength.
            tau_context: Context filter time constant in ms.
            sigma_noise: Standard deviation of membrane phase noise.
            rng: RandomState for stochastic membrane fluctuations.
        """
        self.omega_recv = omega_gamma
        # Natural 32-50 Hz biological gamma band spread across distinct cortical areas
        self.omega_1 = omega_gamma * 1.25
        self.omega_2 = omega_gamma * 0.80
        self.coupling_kappa = coupling_kappa
        self.tau_context = tau_context
        self.sigma_noise = sigma_noise
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Dynamic phase registers initialized randomly on circle [0, 2pi)
        self.phi_recv: float = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phi_1: float = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phi_2: float = float(self.rng.uniform(0.0, 2.0 * np.pi))

        # Filtered context vector
        self.c_filtered: np.ndarray = np.array([1.0, 0.0], dtype=np.float64)

    def reset_state(self, seed: Optional[int] = None) -> None:
        """Reset internal phase registers randomly on circle."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.phi_recv = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phi_1 = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phi_2 = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.c_filtered = np.array([1.0, 0.0], dtype=np.float64)

    def step(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        c_task: np.ndarray,
        dt: float = 0.5,
        decouple_phase: bool = False,
        ablate_shunting: bool = False,
    ) -> Dict[str, Any]:
        """Process one time-step of attentive routing across dual input streams.

        Args:
            x1: Input stream 1 vector (dim,).
            x2: Input stream 2 vector (dim,).
            c_task: 2D task context command [c1, c2].
            dt: Time-step in ms.
            decouple_phase: Null control; disables phase synchronization (uncoupled).
            ablate_shunting: Null control; disables apical dendritic shunting.

        Returns:
            Dictionary containing routed output and diagnostic transmission coefficients.
        """
        # 1. Update filtered context vector (smooth biophysical transition)
        dc = (-self.c_filtered + c_task) / self.tau_context
        self.c_filtered += dc * dt
        self.c_filtered = np.clip(self.c_filtered, 0.0, 1.0)

        # 2. Stochastic Kuramoto phase oscillator dynamics
        noise_recv = float(self.rng.normal(0.0, self.sigma_noise))
        noise_1 = float(self.rng.normal(0.0, self.sigma_noise))
        noise_2 = float(self.rng.normal(0.0, self.sigma_noise))

        # Receiver area oscillates at base gamma frequency
        dphi_recv = self.omega_recv + noise_recv

        if decouple_phase:
            # Control: Uncoupled autonomous oscillators with natural frequency dispersion
            dphi_1 = self.omega_1 + noise_1
            dphi_2 = self.omega_2 + noise_2
        else:
            # Context-modulated inter-areal coupling: disinhibited stream entrains to receiver
            k1 = self.coupling_kappa * self.c_filtered[0]
            k2 = self.coupling_kappa * self.c_filtered[1]

            dphi_1 = self.omega_1 + k1 * np.sin(self.phi_recv - self.phi_1) + noise_1
            dphi_2 = self.omega_2 + k2 * np.sin(self.phi_recv - self.phi_2) + noise_2

        self.phi_recv = (self.phi_recv + dphi_recv * dt) % (2.0 * np.pi)
        self.phi_1 = (self.phi_1 + dphi_1 * dt) % (2.0 * np.pi)
        self.phi_2 = (self.phi_2 + dphi_2 * dt) % (2.0 * np.pi)

        # Phase coherence transmission factor: peak when phase difference is 0
        d_phi1 = (self.phi_1 - self.phi_recv + np.pi) % (2.0 * np.pi) - np.pi
        d_phi2 = (self.phi_2 - self.phi_recv + np.pi) % (2.0 * np.pi) - np.pi

        t1_ctc = float(np.cos(d_phi1 / 2.0) ** 2)
        t2_ctc = float(np.cos(d_phi2 / 2.0) ** 2)

        # 3. Top-down apical dendritic shunting
        if ablate_shunting:
            g1_dend = 1.0
            g2_dend = 1.0
        else:
            g1_dend = float(self.c_filtered[0])
            g2_dend = float(self.c_filtered[1])

        # Net effective transmission factor for each stream
        T1 = g1_dend * t1_ctc
        T2 = g2_dend * t2_ctc

        # Output projection
        y_out = T1 * x1 + T2 * x2

        return {
            "y_out": y_out,
            "T1": T1,
            "T2": T2,
            "c_filtered": self.c_filtered.copy(),
        }
