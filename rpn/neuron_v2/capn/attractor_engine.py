"""Continuous Attractor and Dual-Store Memory Engine (CAPN Pillar 1 & 2).

Integrates:
- Pillar 1: Non-linear continuous attractor dynamics with saturating dendritic transfer functions.
- Pillar 2: Dual-store memory architecture combining fast dynamic local memory
  (Tsodyks-Markram Short-Term Plasticity) with slow structural synaptic matrices.
"""

from typing import Tuple
import numpy as np


class ContinuousAttractorMemoryUnit:
    """Continuous-time non-linear attractor unit with short-term synaptic plasticity."""

    def __init__(
        self,
        dim: int = 128,
        tau_m: float = 20.0,
        u0: float = 0.20,
        tau_f: float = 1500.0,
        tau_d: float = 50.0,
        g_rec: float = 3.5,
        gain_act: float = 1.8,
    ) -> None:
        """Initialize continuous attractor unit.

        Args:
            dim: Dimension D of somatic/membrane state vector.
            tau_m: Membrane integration time constant in ms.
            u0: Baseline synaptic release probability.
            tau_f: Synaptic facilitation time constant in ms.
            tau_d: Synaptic depression recovery time constant in ms.
            g_rec: Global recurrent synaptic gain.
            gain_act: Non-linear dendritic activation scaling.
        """
        self.dim = dim
        self.tau_m = tau_m
        self.u0 = u0
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.g_rec = g_rec
        self.gain_act = gain_act

        # Structural synaptic matrix (long-term consolidated memory)
        self.W_rec: np.ndarray = np.zeros((dim, dim), dtype=np.float64)

        # Dynamic state registers
        self.z: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.u: np.ndarray = np.full(dim, u0, dtype=np.float64)
        self.x_res: np.ndarray = np.ones(dim, dtype=np.float64)

    def store_attractor_patterns(self, patterns: np.ndarray) -> None:
        """Store canonical attractor patterns into slow structural synaptic matrix.

        Args:
            patterns: Array of shape (K, D) with canonical attractor representations.
        """
        # Outer-product attractor matrix
        self.W_rec = (self.g_rec / self.dim) * (patterns.T @ patterns)

    def reset_state(self) -> None:
        """Reset fast dynamic states to baseline."""
        self.z = np.zeros(self.dim, dtype=np.float64)
        self.u = np.full(self.dim, self.u0, dtype=np.float64)
        self.x_res = np.ones(self.dim, dtype=np.float64)

    def step(
        self,
        I_ext: np.ndarray,
        dt: float = 0.5,
        linear: bool = False,
        ablate_stp: bool = False,
    ) -> np.ndarray:
        """Integrate continuous dynamics by one time-step dt.

        Args:
            I_ext: External sensory driving current (dim,).
            dt: Integration time-step in ms.
            linear: Null control; if True, evaluates linear activation without saturation.
            ablate_stp: Null control; if True, freezes facilitation and depression.

        Returns:
            Updated membrane state vector z(t).
        """
        # Presynaptic somatic activation
        if linear:
            r = self.z.copy()
        else:
            r = np.tanh(self.z)

        # Dynamic synaptic transmission efficacy
        if ablate_stp:
            s = self.u0 * r
        else:
            s = self.u * self.x_res * r

        # Recurrent input integration
        I_tot = self.W_rec @ s + I_ext

        # Membrane integration
        if linear:
            dz = (-self.z + I_tot) / self.tau_m
        else:
            dz = (-self.z + np.tanh(I_tot * self.gain_act)) / self.tau_m
        self.z += dz * dt

        # Fast local STP state integration
        if not ablate_stp and not linear:
            abs_r = np.abs(r)
            du = (self.u0 - self.u) / self.tau_f + 0.3 * (1.0 - self.u) * abs_r
            dx = (1.0 - self.x_res) / self.tau_d - 0.1 * self.u * self.x_res * abs_r
            self.u = np.clip(self.u + du * dt, 0.0, 1.0)
            self.x_res = np.clip(self.x_res + dx * dt, 0.0, 1.0)

        return self.z.copy()

    def decode_pattern(self, candidate_patterns: np.ndarray) -> Tuple[int, float, float]:
        """Decode the active attractor state against candidate patterns.

        Args:
            candidate_patterns: Array of shape (K, D).

        Returns:
            Tuple of (predicted_index, target_similarity, contrast_margin_db).
        """
        sims = (candidate_patterns @ self.z) / self.dim
        best_k = int(np.argmax(sims))
        best_sim = float(sims[best_k])

        other_sims = np.delete(sims, best_k)
        max_other = float(np.max(np.abs(other_sims))) if len(other_sims) > 0 else 1e-12
        margin_db = float(20.0 * np.log10(np.maximum(1e-12, best_sim) / (max_other + 1e-12)))

        return best_k, best_sim, margin_db
