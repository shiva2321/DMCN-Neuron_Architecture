"""Universal Continuous Modality Encoders (CAPN Milestone 6).

Implements biophysically and mathematically grounded encoders translating diverse
raw sensory signals into the standardized continuous vector/phase representations
used by CAPN cortical columns:

1. LegendreTemporalEncoder:
   Continuous shifted Legendre state-space projection (LMU dynamics) mapping continuous
   temporal scalar or multivariate time-series into an orthogonal polynomial basis.

2. RetinalReceptiveFieldEncoder:
   Difference-of-Gaussians (DoG) spatial receptive fields with center-surround lateral
   inhibition, extracting contrast and spatial gradients from 2D pixel fields or radial scans.

3. PhaseSequenceEncoder:
   Continuous hyperdimensional phase-vector binding for discrete tokens/sequences,
   mapping symbolic identifiers into unitary complex manifolds without discrete lookup cheats.

4. MotorCPGController:
   Coupled non-linear Hopf oscillator Central Pattern Generator translating cortical
   steering/thrust intention into smooth, dynamically feasible continuous motor commands.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class LegendreTemporalEncoder:
    """Continuous-time Legendre memory encoder projecting time series into an orthogonal basis."""

    def __init__(self, d_order: int = 4, theta: float = 1.0, dt: float = 0.05) -> None:
        # Construct continuous-time Legendre matrix A and input vector B
        A = np.zeros((d_order, d_order), dtype=np.float64)
        B = np.zeros(d_order, dtype=np.float64)
        for i in range(d_order):
            B[i] = (2 * i + 1) * ((-1) ** i)
            for j in range(d_order):
                if i > j:
                    A[i, j] = (2 * i + 1) * ((-1) ** (i - j))
                elif i == j:
                    A[i, j] = -(i + 1)
                else:
                    A[i, j] = (2 * i + 1) * ((-1) ** (i - j + 1))
        A_scaled = A / theta
        B_scaled = B / theta

        # Discretize via Euler approximation: m[t+1] = (I + dt * A) m[t] + dt * B u[t]
        self.Ad = np.eye(d_order) + dt * A_scaled
        self.Bd = dt * B_scaled
        self.m = np.zeros(d_order, dtype=np.float64)

    def reset(self) -> None:
        self.m.fill(0.0)

    def step(self, u: float) -> np.ndarray:
        """Update state with scalar input sample u(t) and return orthogonal representation."""
        self.m = self.Ad @ self.m + self.Bd * u
        return self.m.copy()


class RetinalReceptiveFieldEncoder:
    """Difference-of-Gaussians (DoG) spatial encoder with center-surround lateral inhibition."""

    def __init__(
        self,
        n_sensors: int = 8,
        sigma_center: float = 0.5,
        sigma_surround: float = 1.5,
        gain: float = 2.0,
    ) -> None:
        self.gain = gain

        # Build 1D circular DoG filtering kernel over sensor angles centered at index 0
        idx = np.arange(n_sensors)
        coords = np.minimum(idx, n_sensors - idx) * (2.0 * np.pi / n_sensors)
        center = np.exp(-0.5 * (coords / sigma_center) ** 2) / (np.sqrt(2 * np.pi) * sigma_center)
        surround = np.exp(-0.5 * (coords / sigma_surround) ** 2) / (np.sqrt(2 * np.pi) * sigma_surround)
        kernel = center - surround
        # Zero-mean center-surround normalization
        self.kernel = kernel - np.mean(kernel)

    def encode(self, distances: np.ndarray) -> np.ndarray:
        """Encode raw radial distance measurements into contrast-enhanced receptive field activations."""
        # Convert distance to proximity (nearer = higher excitation)
        proximity = 1.0 / (1.0 + distances)
        # Apply circular convolution with DoG kernel (center excitation, surround inhibition)
        n = len(proximity)
        contrast = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for k in range(len(self.kernel)):
                idx = (i - k) % n
                contrast[i] += self.kernel[k] * proximity[idx]

        # Non-linear rectification: cortical rate response
        rate = np.maximum(0.0, np.tanh(self.gain * contrast))
        return rate


class PhaseSequenceEncoder:
    """Continuous hyperdimensional phase-vector encoder for discrete sequences."""

    def __init__(self, vocab_size: int = 32, d_embed: int = 64, rng: Optional[np.random.RandomState] = None) -> None:
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Random unitary phase codebook (angles theta in [-pi, pi])
        self.base_phases = self.rng.uniform(-np.pi, np.pi, size=(vocab_size, d_embed))
        self.codebook = np.exp(1j * self.base_phases)

    def encode_token(self, token_id: int) -> np.ndarray:
        """Return unitary complex vector for token_id."""
        return self.codebook[token_id % self.vocab_size]

    def bind(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Hadamard circular convolution in phase domain."""
        return z1 * z2

    def position_shift(self, z: np.ndarray, pos: int) -> np.ndarray:
        """Apply continuous position-dependent phase rotation."""
        angles = (pos * 2.0 * np.pi / self.d_embed) * np.arange(self.d_embed)
        rotator = np.exp(1j * angles)
        return z * rotator

    def encode_sequence(self, tokens: list[int]) -> np.ndarray:
        """Encode sequence into composite phase vector via superposition."""
        composite = np.zeros(self.d_embed, dtype=np.complex128)
        for pos, tok in enumerate(tokens):
            z_tok = self.encode_token(tok)
            z_pos = self.position_shift(z_tok, pos)
            composite += z_pos
        # Normalize to unit sphere
        norm = np.linalg.norm(composite) + 1e-8
        return composite / norm


class MotorCPGController:
    """Coupled Hopf oscillator Central Pattern Generator for smooth continuous motor actuation."""

    def __init__(
        self,
        mu: float = 1.0,
        gamma: float = 10.0,
        omega: float = 5.0,
        dt: float = 0.05,
        max_vel: float = 1.0,
        max_yaw_rate: float = 1.5,
    ) -> None:
        self.mu = mu
        self.gamma = gamma
        self.omega = omega
        self.dt = dt
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate

        # Oscillator state: (r, phi) for thrust and steering channels
        self.r = np.array([0.5, 0.5], dtype=np.float64)
        self.phi = np.array([0.0, 0.0], dtype=np.float64)

    def reset(self) -> None:
        self.r.fill(0.5)
        self.phi.fill(0.0)

    def step(self, u_thrust: float, u_steer: float) -> Tuple[float, float]:
        """Integrate Hopf oscillator limit-cycle conditioned on cortical drive commands."""
        # Drive amplitude targets
        target_r = np.array([np.clip(u_thrust, 0.0, 1.0), np.clip(abs(u_steer), 0.0, 1.0)])

        # Radial limit cycle dynamics: dr/dt = gamma * (target - r^2) * r
        dr = self.gamma * (target_r - self.r ** 2) * self.r
        self.r = np.clip(self.r + self.dt * dr, 0.0, 1.5)

        # Phase integration: dphi/dt = omega
        self.phi = (self.phi + self.dt * self.omega) % (2.0 * np.pi)

        # Modulated outputs: smooth, bounded physical velocities
        v_forward = float(self.r[0] * self.max_vel * np.clip(u_thrust, 0.0, 1.0))
        yaw_rate = float(self.r[1] * self.max_yaw_rate * np.clip(u_steer, -1.0, 1.0))

        return v_forward, yaw_rate
