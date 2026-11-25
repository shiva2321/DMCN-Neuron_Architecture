"""Three-Factor Hybrid Plasticity Engine (CAPN Pillar 5).

Implements multi-layer gradient-free credit assignment using:
1. Three-Factor Synaptic Plasticity:
   Delta W_ij = eta * Pre_j * Local_Error_i * M(t)
2. Feedback Alignment:
   Fixed asymmetric feedback projections B decouple forward and backward paths,
   eliminating the biological implausibility of weight transport (W^T) and backpropagation error tapes.
3. Neuromodulatory Gating M(t):
   Dopamine/surprise gating prevents catastrophic synaptic overwriting during periods
   of uninformative noise drift.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


class ThreeFactorPlasticityNetwork:
    """Multi-layer network learning via three-factor local predictive plasticity."""

    def __init__(
        self,
        d_in: int = 8,
        d_hid: int = 16,
        d_out: int = 4,
        eta: float = 0.03,
        sigma_mod: float = 0.5,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out
        self.eta = eta
        self.sigma_mod = sigma_mod
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Forward weights
        self.W1 = self.rng.randn(d_hid, d_in) * 0.1
        self.b1 = np.zeros(d_hid, dtype=np.float64)
        self.W2 = self.rng.randn(d_out, d_hid) * 0.1
        self.b2 = np.zeros(d_out, dtype=np.float64)

        # Fixed asymmetric feedback projection (Feedback Alignment - Lillicrap et al. 2016)
        # Note: B != W2.T, solving the weight transport problem
        self.B = self.rng.randn(d_hid, d_out) * 0.2

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward activation through hierarchical layers.

        Args:
            x: Input vector (d_in,).

        Returns:
            Tuple of (hidden activation h, output prediction y_hat).
        """
        h = np.tanh(self.W1 @ x + self.b1)
        y_hat = self.W2 @ h + self.b2
        return h, y_hat

    def step_plasticity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        control: str = "none",
        is_noise_drift: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Execute single online three-factor plasticity update.

        Args:
            x: Presynaptic input (d_in,).
            y: Target supervisory signal (d_out,).
            control: One of 'none', 'decoupled', 'shuffled'.
            is_noise_drift: If True, indicates uninformative noise exposure phase.

        Returns:
            Tuple of (output prediction y_hat, scalar prediction error norm).
        """
        h, y_hat = self.forward(x)

        # Output prediction error (Factor 2 for Layer 2)
        eps_y = y - y_hat

        # Hidden dendritic error via fixed feedback alignment (Factor 2 for Layer 1)
        if control == "shuffled":
            # Control: Shuffled error feedback destroys local credit assignment
            eps_y_eff = self.rng.permutation(eps_y)
        else:
            eps_y_eff = eps_y

        dh = 1.0 - h ** 2  # Local postsynaptic derivative
        eps_h = dh * (self.B @ eps_y_eff)

        # Neuromodulator gate M(t) (Factor 3)
        if control == "decoupled":
            # Control: M(t) = 1.0 constant, unselective weight updates
            M_t = 1.0
        elif is_noise_drift:
            # Low neuromodulator tone during uninformative / task-free periods
            M_t = 0.05
        else:
            # Surprise / reward prediction error gate
            err_norm_sq = float(np.sum(eps_y ** 2))
            M_t = float(np.tanh(err_norm_sq / self.sigma_mod))

        # Three-factor synaptic weight updates: Delta W = eta * Pre * Local_Error * M(t)
        self.W2 += self.eta * np.outer(eps_y_eff, h) * M_t
        self.b2 += self.eta * eps_y_eff * M_t
        self.W1 += self.eta * np.outer(eps_h, x) * M_t
        self.b1 += self.eta * eps_h * M_t

        return y_hat, float(np.linalg.norm(eps_y))

    def evaluate_nmse(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Normalized Mean-Squared Error on dataset.

        Args:
            X: Input matrix (N, d_in).
            Y: Target matrix (N, d_out).

        Returns:
            NMSE scalar = MSE / Var(Y).
        """
        H = np.tanh(X @ self.W1.T + self.b1)
        Y_pred = H @ self.W2.T + self.b2
        mse = float(np.mean((Y - Y_pred) ** 2))
        var_y = float(np.var(Y)) + 1e-12
        return mse / var_y


class BackpropBaselineNetwork:
    """Exact backpropagation baseline with identical multi-layer architecture."""

    def __init__(
        self,
        d_in: int = 8,
        d_hid: int = 16,
        d_out: int = 4,
        eta: float = 0.03,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out
        self.eta = eta
        self.rng = rng if rng is not None else np.random.RandomState(42)

        self.W1 = self.rng.randn(d_hid, d_in) * 0.1
        self.b1 = np.zeros(d_hid, dtype=np.float64)
        self.W2 = self.rng.randn(d_out, d_hid) * 0.1
        self.b2 = np.zeros(d_out, dtype=np.float64)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.W1 @ x + self.b1)
        y_hat = self.W2 @ h + self.b2
        return h, y_hat

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        h, y_hat = self.forward(x)
        eps_y = y - y_hat  # Gradient of 0.5 * ||y - y_hat||^2 is -eps_y

        # Exact backpropagation with weight transpose W2.T
        dh = 1.0 - h ** 2
        grad_h = (self.W2.T @ eps_y) * dh

        self.W2 += self.eta * np.outer(eps_y, h)
        self.b2 += self.eta * eps_y
        self.W1 += self.eta * np.outer(grad_h, x)
        self.b1 += self.eta * grad_h

    def evaluate_nmse(self, X: np.ndarray, Y: np.ndarray) -> float:
        H = np.tanh(X @ self.W1.T + self.b1)
        Y_pred = H @ self.W2.T + self.b2
        mse = float(np.mean((Y - Y_pred) ** 2))
        var_y = float(np.var(Y)) + 1e-12
        return mse / var_y
