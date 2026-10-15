"""Continuous Spatial Wave Temporal Sequence & Delay Tracker.

Replaces artificial tapped delay lines, discrete registers, and polynomial filters (LMU)
with the physical law of continuous wave propagation along an active dendritic cable:
    dPsi/dt + v * dPsi/dx + gamma * Psi = 0

Because waves propagate at physical velocity v, the spatial coordinate x along the dendritic shaft
is identically isomorphic to continuous time delay:
    tau(x) = x / v

Tapping the continuous spatial wavefield at physical position x = v * tau yields the delayed signal
u(t - tau) natively with zero parameter training, zero curve-fitting, and zero discrete clocks.
"""
from __future__ import annotations

import numpy as np


class ContinuousWaveDelayDendrite:
    """Continuous 1-way wave transmission dendritic cable for physical delay tracking."""

    def __init__(
        self,
        length: float = 10.0,
        n_nodes: int = 100,
        velocity: float = 2.0,
        damping: float = 0.005,
    ):
        self.L = float(length)
        self.Nx = int(n_nodes)
        self.dx = self.L / max(1, self.Nx - 1)
        self.v = float(velocity)
        self.gamma = float(damping)

        # Spatial wavefield along the cable
        self.psi = np.zeros(self.Nx, dtype=np.float64)

    @property
    def max_delay(self) -> float:
        """Maximum continuous delay capacity of the physical cable: tau_max = L / v."""
        return self.L / self.v

    def step(self, u_t: float, dt: float = 0.01) -> np.ndarray:
        """Advance continuous wavefield using 1st-order upwind wave advection."""
        # Spatial advection: dPsi/dt = -v * dPsi/dx - gamma * Psi
        dpsi_dx = (self.psi[1:] - self.psi[:-1]) / self.dx
        dpsi_dt = -self.v * dpsi_dx - self.gamma * self.psi[1:]

        self.psi[1:] += dpsi_dt * dt
        # Boundary injection at x = 0
        self.psi[0] = float(u_t)

        return self.psi

    def tap(self, tau: float) -> float:
        """Reconstruct continuous delayed signal u(t - tau) via physical spatial lookup.
        
        Requires ZERO learned parameters: delay is a direct consequence of space-time wave motion.
        """
        tau_clamped = np.clip(tau, 0.0, self.max_delay)
        target_x = self.v * tau_clamped
        # Continuous linear interpolation between spatial lattice points
        node_float = target_x / self.dx
        idx_low = int(np.floor(node_float))
        idx_high = min(self.Nx - 1, idx_low + 1)
        alpha = node_float - idx_low

        recon = (1.0 - alpha) * self.psi[idx_low] + alpha * self.psi[idx_high]
        # Compensate for known exponential membrane damping: exp(+gamma * tau)
        recon_calibrated = recon * np.exp(self.gamma * tau_clamped)
        return float(recon_calibrated)

    def reset(self):
        """Quench wavefield back to zero."""
        self.psi.fill(0.0)
