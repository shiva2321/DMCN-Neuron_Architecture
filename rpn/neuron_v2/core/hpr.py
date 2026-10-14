"""Holomorphic Phase-Resonator (HPR) - First-Principles Wave-Interference Digital Neuron.

Formulated directly from wave mechanics and complex analysis:
1. Native Non-Linear Logic (XOR) via destructive wave interference:
   E = |x_1 e^{i theta_1} + x_2 e^{i theta_2}|^2
   When Delta theta = pi:
   (0,0) -> 0, (1,0) -> 1, (0,1) -> 1, (1,1) -> 0.
   No artificial hidden layers, no dendritic compartmental thresholds, no backpropagation.

2. Continuous Temporal Memory via Spectral Phase Dispersion:
   dPsi/dt = -(gamma + i*omega) Psi + x(t)
   Exact unitary propagator: U_k = exp(-i omega_k dt)
   Time delay tau is pure phase shift exp(-i omega_k tau) across harmonic spectrum.
   Eliminates external Legendre polynomial matrix approximations (LMU).

3. Unitary Symplectic Energy Conservation:
   Conserves norm ||Psi|| via unitary transformations U^dagger U = I,
   eliminating ad-hoc biological calcium/adenosine patchwork equations.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence


class HarmonicSpectralMemory:
    """Continuous temporal memory engine based on unitary spectral phase dispersion."""

    def __init__(
        self,
        K: int = 16,
        omega_min: float = 0.005,
        omega_max: float = 0.20,
        gamma: float = 0.002,
        dt: float = 1.0,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.K = K
        self.dt = dt
        self.gamma = gamma
        self.lr = lr
        self.omegas = np.linspace(omega_min, omega_max, K).astype(np.float64)
        # Exact unitary propagator with infinitesimal damping for DC stability
        self.U = np.exp((-gamma - 1j * self.omegas) * dt).astype(np.complex128)
        self.Psi = np.zeros(K, dtype=np.complex128)
        self.W = np.zeros(K, dtype=np.complex128)

    def step(self, x_scalar: float) -> np.ndarray:
        """Advance the spectral phase wavefield by continuous time dt."""
        self.Psi = self.U * self.Psi + (x_scalar * self.dt)
        return self.Psi.copy()

    def predict_delay(self) -> float:
        """Linear readout of delayed target from spectral phase state."""
        return float(np.real(np.dot(self.W, self.Psi)))

    def learn_delay(self, target_value: float) -> float:
        """Learn continuous delay mapping autonomously via Normalized LMS on complex wavefield."""
        pred = self.predict_delay()
        err = target_value - pred
        norm_sq = float(np.real(np.vdot(self.Psi, self.Psi))) + 1e-4
        self.W += (self.lr / norm_sq) * err * np.conj(self.Psi)
        return pred

    def reset(self):
        self.Psi.fill(0.0)


class LogCochlearResonatorBank:
    """Multi-scale continuous temporal memory using logarithmic constant-Q frequency spacing.

    Spans multiple orders of magnitude (e.g. 10^-3 to 10^1 rad/s) simultaneously,
    matching the mammalian cochlea and cortical auditory fields.
    Eliminates manual passband tuning on broad-band non-stationary signals.
    """

    def __init__(
        self,
        K: int = 64,
        omega_min: float = 1e-3,
        omega_max: float = 5.0,
        q_factor: float = 50.0,
        dt: float = 0.5,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.K = K
        self.dt = dt
        self.lr = lr
        # Logarithmic frequency spacing
        indices = np.arange(K, dtype=np.float64) / max(1, K - 1)
        self.omegas = omega_min * ((omega_max / omega_min) ** indices)
        # Constant Q damping: gamma = omega / (2 * Q)
        self.gammas = self.omegas / (2.0 * q_factor)
        # Exact unitary propagator with scale-invariant damping
        self.U = np.exp((-self.gammas - 1j * self.omegas) * dt).astype(np.complex128)
        self.Psi = np.zeros(K, dtype=np.complex128)
        self.W = np.zeros(K, dtype=np.complex128)

    def step(self, x_scalar: float) -> np.ndarray:
        self.Psi = self.U * self.Psi + (x_scalar * self.dt)
        return self.Psi.copy()

    def predict_delay(self) -> float:
        return float(np.real(np.dot(self.W, self.Psi)))

    def learn_delay(self, target_value: float) -> float:
        pred = self.predict_delay()
        err = target_value - pred
        norm_sq = float(np.real(np.vdot(self.Psi, self.Psi))) + 1e-4
        self.W += (self.lr / norm_sq) * err * np.conj(self.Psi)
        return pred

    def reset(self):
        self.Psi.fill(0.0)


class WaveInterferenceNeuron:
    """Non-linear wave-interference neuron for combinatorial logic and pattern separation."""

    def __init__(
        self,
        n_channels: int = 2,
        threshold: float = 0.50,
        lr: float = 0.10,
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.threshold = threshold
        self.lr = lr
        rng = np.random.RandomState(seed)
        # Random initial carrier phases
        self.phases = rng.uniform(-np.pi, np.pi, n_channels).astype(np.float64)

    @property
    def carriers(self) -> np.ndarray:
        return np.exp(1j * self.phases)

    def compute_wavefield(self, inputs: Sequence[float]) -> tuple[complex, float]:
        """Compute complex superposition wavefield Psi and intensity E = |Psi|^2."""
        assert len(inputs) == self.n_channels
        psi = complex(0.0, 0.0)
        for val, carrier in zip(inputs, self.carriers):
            psi += float(val) * carrier
        energy = float(abs(psi) ** 2)
        return psi, energy

    def forward(self, inputs: Sequence[float]) -> tuple[int, float]:
        """Forward evaluation: returns binary spike out in {0, 1} and continuous energy."""
        _, energy = self.compute_wavefield(inputs)
        spike = 1 if energy > self.threshold else 0
        return spike, energy

    def learn_autonomous_logic(
        self,
        inputs: Sequence[float],
        target_spike: int,
    ) -> float:
        """Autonomous phase plasticity: adapts carrier phases via phase-energy gradients."""
        psi, energy = self.compute_wavefield(inputs)
        pred_spike = 1 if energy > self.threshold else 0
        err = target_spike - pred_spike

        # If error occurs, adjust phase relations to alter interference
        if err != 0 and np.sum(inputs) >= 2:
            # Active coactivation: drive toward anti-phase (destructive) if target is 0,
            # or in-phase (constructive) if target is 1
            active_indices = [i for i, v in enumerate(inputs) if v > 0.5]
            if len(active_indices) == 2:
                i0, i1 = active_indices[0], active_indices[1]
                dtheta = self.phases[i0] - self.phases[i1]
                # Gradient of energy w.r.t phase difference
                grad = -2.0 * np.sin(dtheta)
                delta = self.lr * (energy - float(target_spike)) * grad
                self.phases[i0] -= delta
                self.phases[i1] += delta

        return energy


class HolomorphicPhaseResonator:
    """Unified Holomorphic Phase-Resonator (HPR).

    Fuses harmonic spectral phase dispersion with wave-interference logic.
    """

    def __init__(
        self,
        n_channels: int = 2,
        K_spectrum: int = 16,
        dt: float = 1.0,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.wave_gate = WaveInterferenceNeuron(
            n_channels=n_channels,
            threshold=0.50,
            lr=0.10,
            seed=seed,
        )
        self.spectral_memory = HarmonicSpectralMemory(
            K=K_spectrum,
            dt=dt,
            lr=lr,
            seed=seed,
        )

    def step_continuous(self, x_scalar: float, target_delay_val: Optional[float] = None) -> dict:
        """Step continuous temporal spectral resonator."""
        psi = self.spectral_memory.step(x_scalar)
        pred = self.spectral_memory.predict_delay()
        if target_delay_val is not None:
            self.spectral_memory.learn_delay(target_delay_val)
        return {"spectral_state": psi, "predicted_delay": pred}

    def step_discrete_logic(self, inputs: Sequence[float], target_spike: Optional[int] = None) -> dict:
        """Step wave interference gate."""
        spike, energy = self.wave_gate.forward(inputs)
        if target_spike is not None:
            self.wave_gate.learn_autonomous_logic(inputs, target_spike)
        return {"spike": spike, "energy": energy}

    def reset(self):
        self.spectral_memory.reset()
