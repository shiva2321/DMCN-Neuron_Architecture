"""Continuous-Time State-Space Legendre Delay Engine.

Mathematically projects continuous input history u(t - s) for s in [0, theta]
onto an orthogonal shifted Legendre polynomial basis (Voelker & Eliasmith, 2018).

Provides:
1. Continuous-time state-space integration via matrix exponential discretization.
2. Zero-shot analytical delay reconstruction for arbitrary continuous delay tau in [0, theta].
3. Compression of continuous temporal memory: O(order) state variables vs O(theta / dt)
   for discrete tapped delay buffers (12.5x memory compression at dt=0.5 ms, theta=50 ms).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


def compute_legendre_operators(order: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytical continuous-time state-space drift and input matrices.
    
    A_ij = -(2i+1)/theta if i < j else ((2i+1)/theta) * (-1)**(i - j + 1)
    B_i  = ((2i+1)/theta) * (-1)**i
    """
    A = np.zeros((order, order), dtype=np.float64)
    B = np.zeros((order, 1), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            if i < j:
                A[i, j] = -(2.0 * i + 1.0) / theta
            else:
                A[i, j] = ((2.0 * i + 1.0) / theta) * ((-1.0) ** (i - j + 1))
        B[i, 0] = ((2.0 * i + 1.0) / theta) * ((-1.0) ** i)
    return A, B


class ContinuousLegendreDelayEngine:
    """Continuous analog state-space memory engine for temporal delay tracking."""

    def __init__(
        self,
        order: int = 8,
        theta: float = 50.0,
        dt: float = 0.5,
    ):
        self.order = int(order)
        self.theta = float(theta)
        self.dt = float(dt)

        self.A, self.B = compute_legendre_operators(self.order, self.theta)
        # Discretize continuous LTI system dx/dt = A x + B u with zero-order hold
        self.A_bar = scipy.linalg.expm(self.A * self.dt)
        self.B_bar = np.linalg.solve(self.A, (self.A_bar - np.eye(self.order, dtype=np.float64)) @ self.B)

        self.state = np.zeros((self.order, 1), dtype=np.float64)

    @property
    def state_dim(self) -> int:
        return self.order

    def step(self, u_t: float) -> np.ndarray:
        """Advance continuous state space by dt: m_{t+1} = A_bar * m_t + B_bar * u_t."""
        self.state = self.A_bar @ self.state + self.B_bar * float(u_t)
        return self.state.flatten()

    def get_delay_kernel(self, tau: float) -> np.ndarray:
        """Compute analytical shifted Legendre reconstruction vector for continuous delay tau."""
        tau_clamped = float(np.clip(tau, 0.0, self.theta))
        z = 1.0 - (2.0 * tau_clamped / self.theta)

        P = np.zeros(self.order, dtype=np.float64)
        if self.order > 0:
            P[0] = 1.0
        if self.order > 1:
            P[1] = z
        for i in range(1, self.order - 1):
            P[i + 1] = ((2.0 * i + 1.0) * z * P[i] - i * P[i - 1]) / (i + 1)

        c = np.array([((-1.0) ** i) * P[i] for i in range(self.order)], dtype=np.float64)
        return c

    def reconstruct_delay(self, tau: float) -> float:
        """Analytically reconstruct u(t - tau) from the continuous Legendre state."""
        c = self.get_delay_kernel(tau)
        return float(np.dot(c, self.state.flatten()))

    def reset(self):
        """Reset internal memory states to zero."""
        self.state.fill(0.0)
