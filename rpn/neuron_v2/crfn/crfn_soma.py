"""Stuart-Landau Limit-Cycle Somatic Oscillator Core.

Implements autonomous noise clean-up via supercritical Hopf bifurcation dynamics:
    dz/dt = (mu + i*omega - |z|^2) * z + kappa * I_in(t)

For mu > 0, the unforced system possesses an intrinsic circular limit-cycle attractor
of radius R_0 = sqrt(mu). Perturbations in amplitude decay exponentially at the Lyapunov
rate lambda = -2*mu without requiring any external dictionary or codebook lookup.
"""
from __future__ import annotations

import numpy as np


class StuartLandauSoma:
    """Non-linear limit-cycle somatic oscillator core."""

    def __init__(
        self,
        mu: float = 1.0,
        omega: float = 1.0,
        kappa: float = 0.5,
        z_init: complex | None = None,
    ):
        self.mu = float(mu)
        self.omega = float(omega)
        self.kappa = float(kappa)
        if z_init is not None:
            self.z = complex(z_init)
        elif self.mu > 0:
            # Initialize on canonical limit cycle with random phase
            phase = np.random.uniform(-np.pi, np.pi)
            self.z = complex(np.sqrt(self.mu) * np.exp(1j * phase))
        else:
            # Excitable resting state at origin
            self.z = complex(0.0, 0.0)

    @property
    def canonical_radius(self) -> float:
        """Intrinsic limit cycle radius R_0 = sqrt(mu)."""
        return float(np.sqrt(max(0.0, self.mu)))

    @property
    def lyapunov_damping_rate(self) -> float:
        """Autonomous exponential rate of radial perturbation decay (-2*mu)."""
        return -2.0 * self.mu

    @property
    def amplitude(self) -> float:
        """Current instantaneous amplitude |z(t)|."""
        return float(abs(self.z))

    @property
    def phase(self) -> float:
        """Current instantaneous phase angle arg(z(t)) in [-pi, pi]."""
        return float(np.angle(self.z))

    def _drift(self, z: complex, I_in: complex) -> complex:
        """Evaluate continuous vector field f(z, I)."""
        r2 = z.real * z.real + z.imag * z.imag
        linear_part = complex(self.mu, self.omega) * z
        cubic_damping = r2 * z
        return linear_part - cubic_damping + self.kappa * I_in

    def step(self, I_in: complex | float, dt: float = 0.05) -> complex:
        """Advance the somatic state by dt using 4th-order Runge-Kutta (RK4) integration."""
        I_c = complex(I_in)
        z = self.z

        k1 = self._drift(z, I_c)
        k2 = self._drift(z + 0.5 * dt * k1, I_c)
        k3 = self._drift(z + 0.5 * dt * k2, I_c)
        k4 = self._drift(z + dt * k3, I_c)

        self.z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.z

    def reset(self, phase: float | None = None):
        """Reset state onto the canonical limit cycle (or origin if excitable)."""
        if self.mu > 0:
            if phase is None:
                phase = np.random.uniform(-np.pi, np.pi)
            self.z = complex(self.canonical_radius * np.exp(1j * phase))
        else:
            self.z = complex(0.0, 0.0)
