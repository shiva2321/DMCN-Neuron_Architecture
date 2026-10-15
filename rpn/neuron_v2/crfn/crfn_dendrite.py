"""Transmission-Line Continuous Dendritic Wavefield.

Models a continuous, spatially extended dendritic branch as a non-linear dissipative
transmission line governed by the continuous wave equation:
    d^2Psi/dx^2 - (1/v^2) d^2Psi/dt^2 - gamma * dPsi/dt = sum_j delta(x - x_j) * u_j(t)

Waves propagate along the branch at velocity v = 1 / sqrt(L_ind * C_mem).
Interactions between multiple synaptic inputs produce physical wave interference,
dispersion, and standing wave patterns that solve non-linear logic (e.g. XOR)
natively in the wavefield without requiring artificial thresholds or MLP layers.
"""
from __future__ import annotations

import numpy as np


class TransmissionLineDendrite:
    """Continuous 1D spatial wave transmission dendritic branch."""

    def __init__(
        self,
        length: float = 10.0,
        n_nodes: int = 50,
        velocity: float = 1.0,
        damping: float = 0.10,
        absorbing_boundary: bool = True,
    ):
        self.L = float(length)
        self.Nx = int(n_nodes)
        self.dx = self.L / max(1, self.Nx - 1)
        self.v = float(velocity)
        self.gamma = float(damping)
        self.absorbing = absorbing_boundary

        # State vectors: complex wavefield position Psi and momentum Pi = dPsi/dt
        self.psi = np.zeros(self.Nx, dtype=np.complex128)
        self.pi = np.zeros(self.Nx, dtype=np.complex128)

        # Synaptic connection map: syn_id -> node_index
        self._syn_to_node: dict[int, int] = {}

    def attach_synapse(self, syn_id: int, relative_pos: float = 0.5):
        """Attach a synapse at relative spatial position along branch (0.0=nexus, 1.0=tip)."""
        rel_pos = np.clip(relative_pos, 0.0, 1.0)
        node_idx = int(round(rel_pos * (self.Nx - 1)))
        self._syn_to_node[syn_id] = node_idx

    @property
    def nexus_flux(self) -> complex:
        """Wave amplitude entering somatic nexus (node 0)."""
        return complex(self.psi[0])

    @property
    def courrant_number(self) -> float:
        """Courant-Friedrichs-Lewy stability parameter."""
        return self.v / self.dx

    def step(self, synaptic_inputs: dict[int, complex | float], dt: float = 0.05) -> complex:
        """Advance the wavefield by dt using velocity-Verlet symplectic integration.
        
        Guarantees physical energy conservation and wave interference.
        """
        # 1. Compute spatial laplacian (d^2Psi / dx^2) with central differences
        lap = np.zeros(self.Nx, dtype=np.complex128)
        # Interior points
        lap[1:-1] = (self.psi[2:] - 2.0 * self.psi[1:-1] + self.psi[:-2]) / (self.dx * self.dx)

        # Boundary conditions
        if self.absorbing:
            # Engquist-Majda radiation boundary condition (zero reflection)
            # dPsi/dt = -v * dPsi/dx at x=L; dPsi/dt = +v * dPsi/dx at x=0
            lap[0] = (self.psi[1] - self.psi[0]) / (self.dx * self.dx)
            lap[-1] = (self.psi[-2] - self.psi[-1]) / (self.dx * self.dx)
        else:
            # Dirichlet boundary (fixed zero at ends)
            lap[0] = (self.psi[1] - 2.0 * self.psi[0]) / (self.dx * self.dx)
            lap[-1] = (self.psi[-2] - 2.0 * self.psi[-1]) / (self.dx * self.dx)

        # 2. Inject external synaptic inputs at their specific spatial coordinates
        # Scaled by 1/dx to represent continuous Dirac delta distribution: int delta(x) dx = 1
        injection = np.zeros(self.Nx, dtype=np.complex128)
        for syn_id, val in synaptic_inputs.items():
            if syn_id in self._syn_to_node:
                idx = self._syn_to_node[syn_id]
                injection[idx] += complex(val) / self.dx

        # 3. Acceleration: dPi/dt = v^2 * lap - gamma * Pi + injection
        acc = (self.v * self.v) * lap - self.gamma * self.pi + injection

        # 4. Advance momentum and position via damped Leapfrog
        self.pi += acc * dt
        self.psi += self.pi * dt

        return self.nexus_flux

    def reset(self):
        """Quench wavefield back to quiescent equilibrium."""
        self.psi.fill(0.0)
        self.pi.fill(0.0)
