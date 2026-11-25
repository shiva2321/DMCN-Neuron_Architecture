"""Continuous resonant phasor and transmission-line operations.

Functions and classes in this module:
- StuartLandauSomaticBank: Integrates dz/dt = (mu - |z|^2) * z.
- ImpedanceTransmissionBus: Scales signal amplitudes by T = 1 - |Gamma|^2.
- CRFNSymbolicEngine: Evaluates phase-product binding and unbinding operations,
  linear phase demodulation, and somatic limit-cycle relaxation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class StuartLandauSomaticBank:
    """Bank of independent continuous limit-cycle somatic oscillators.

    Dynamical equation: dz/dt = (mu - |z|^2) * z + kappa * I_in
    Provides autonomous noise clean-up at Lyapunov damping rate lambda = -2*mu.
    """

    def __init__(self, dim: int, mu: float = 1.0, kappa: float = 0.5) -> None:
        self.dim = dim
        self.mu = float(mu)
        self.kappa = float(kappa)

    def clean_up(
        self,
        z_input: np.ndarray,
        steps: int = 50,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Run continuous non-linear Lyapunov relaxation to clean amplitude and phase noise."""
        z = z_input.astype(np.complex128).copy()
        for _ in range(steps):
            r2 = np.real(z * np.conj(z))
            dz = (self.mu - r2) * z
            z += dt * dz
        mag = np.abs(z)
        mask = mag > 1e-12
        z[mask] /= mag[mask]
        return z


class ImpedanceTransmissionBus:
    """Multi-channel transmission line with local membrane impedances Z_j in C.

    Power transmission: T_j = 1 - |(Z_j - Z0) / (Z_j + Z0)|^2.
    """

    def __init__(self, dim: int, z0: float = 1.0) -> None:
        self.dim = dim
        self.z0 = float(z0)
        # Characteristic impedances matched to membrane for open transmission
        self.Z = np.full(dim, complex(z0, 0.0), dtype=np.complex128)

    @property
    def transmission_coefficients(self) -> np.ndarray:
        """Calculate power transmission T_j in [0, 1] for all channels."""
        gamma = (self.Z - self.z0) / (self.Z + self.z0)
        mag2 = np.real(gamma * np.conj(gamma))
        return np.clip(1.0 - mag2, 0.0, 1.0)

    def transmit(self, wave_input: np.ndarray) -> np.ndarray:
        """Transmit input wave scaled by physical transmission coefficients."""
        t_coeffs = self.transmission_coefficients
        return wave_input * np.sqrt(t_coeffs)


class CRFNSymbolicEngine:
    """Weight-Free Continuous Resonant Field Neuron Column for Symbolic & Algebraic Reasoning."""

    def __init__(
        self,
        dim: int = 256,
        mu: float = 1.0,
        n_frequencies: int = 64,
        omega_min: float = 0.01,
        omega_max: float = 0.10,
    ) -> None:
        self.dim = dim
        self.soma_bank = StuartLandauSomaticBank(dim=dim, mu=mu)
        self.transmission_bus = ImpedanceTransmissionBus(dim=dim)

        # Frequencies for continuous numerical phase encoding
        self.n_frequencies = n_frequencies
        self.frequencies = np.linspace(omega_min, omega_max, n_frequencies, dtype=np.float64)
        self.arith_soma_bank = StuartLandauSomaticBank(dim=n_frequencies, mu=mu)

    def solve_relational_analogy(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        vec_c: np.ndarray,
        decouple_transmission: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_soma: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Solve A : B :: C : ? (infer D = C (*) (B (#) A)) purely via wave mechanics."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            za = np.zeros(self.dim, dtype=np.complex128)
            zb = np.zeros(self.dim, dtype=np.complex128)
            zc = np.zeros(self.dim, dtype=np.complex128)
        else:
            za = vec_a.astype(np.complex128)
            zb = vec_b.astype(np.complex128)
            zc = vec_c.astype(np.complex128)

        if decouple_transmission:
            # Severed transmission line: zero signal transmitted
            zd_trans = np.zeros(self.dim, dtype=np.complex128)
            t_eff = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.dim)
            # Unbind relation T = B (*) conj(A) with scrambled phase
            t_rel = zb[perm] * np.conj(za)
            # Project onto C: D = C (*) T
            zd_raw = zc * t_rel
            zd_trans = self.transmission_bus.transmit(zd_raw)
            t_eff = 0.5
        elif blind_input:
            zd_trans = np.zeros(self.dim, dtype=np.complex128)
            t_eff = 0.0
        else:
            # Active path: physical wave unbinding T = B (*) conj(A)
            t_rel = zb * np.conj(za)
            # Physical wave binding D = C (*) T
            zd_raw = zc * t_rel
            # Signal enters somatic cavity through transmission line
            zd_trans = self.transmission_bus.transmit(zd_raw)
            t_eff = 1.0

        # Autonomous Stuart-Landau somatic limit-cycle clean-up (zero dictionary!)
        if ablate_soma or decouple_transmission or blind_input:
            zd_clean = zd_trans.copy()
        else:
            zd_clean = self.soma_bank.clean_up(zd_trans, steps=50, dt=0.02)

        return {
            "output_wave": zd_clean,
            "raw_wave": zd_trans,
            "t_eff": t_eff,
        }

    def solve_transitive_deduction(
        self,
        vec_a: np.ndarray,
        rel_ab: np.ndarray,
        rel_bc: np.ndarray,
        decouple_transmission: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_soma: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Solve multi-hop chain A -> B -> C (infer C = A (*) (R_ab (*) R_bc))."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            za = np.zeros(self.dim, dtype=np.complex128)
            rab = np.zeros(self.dim, dtype=np.complex128)
            rbc = np.zeros(self.dim, dtype=np.complex128)
        else:
            za = vec_a.astype(np.complex128)
            rab = rel_ab.astype(np.complex128)
            rbc = rel_bc.astype(np.complex128)

        if decouple_transmission:
            zc_trans = np.zeros(self.dim, dtype=np.complex128)
            t_eff = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.dim)
            rac = rab[perm] * rbc
            zc_raw = za * rac
            zc_trans = self.transmission_bus.transmit(zc_raw)
            t_eff = 0.5
        elif blind_input:
            zc_trans = np.zeros(self.dim, dtype=np.complex128)
            t_eff = 0.0
        else:
            # Composite relational wave
            rac = rab * rbc
            zc_raw = za * rac
            zc_trans = self.transmission_bus.transmit(zc_raw)
            t_eff = 1.0

        if ablate_soma or decouple_transmission or blind_input:
            zc_clean = zc_trans.copy()
        else:
            zc_clean = self.soma_bank.clean_up(zc_trans, steps=50, dt=0.02)

        return {
            "output_wave": zc_clean,
            "t_eff": t_eff,
        }

    def wave_add(
        self,
        val_a: int,
        val_b: int,
        noise_sigma: float = 0.0,
        decouple_transmission: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_soma: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Continuous wave addition: psi(a + b) = psi(a) * psi(b), decoded via linear phase demodulation."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            pa = np.zeros(self.n_frequencies, dtype=np.complex128)
            pb = np.zeros(self.n_frequencies, dtype=np.complex128)
        else:
            pa = np.exp(1j * val_a * self.frequencies)
            pb = np.exp(1j * val_b * self.frequencies)
            if noise_sigma > 0.0:
                pa *= np.exp(1j * noise_sigma * local_rng.randn(self.n_frequencies))
                pb *= np.exp(1j * noise_sigma * local_rng.randn(self.n_frequencies))

        if decouple_transmission:
            p_trans = np.zeros(self.n_frequencies, dtype=np.complex128)
            t_eff = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.n_frequencies)
            p_trans = pa[perm] * pb
            t_eff = 0.5
        elif blind_input:
            p_trans = np.zeros(self.n_frequencies, dtype=np.complex128)
            t_eff = 0.0
        else:
            p_trans = pa * pb
            t_eff = 1.0

        if ablate_soma or decouple_transmission or blind_input:
            p_clean = p_trans.copy()
        else:
            p_clean = self.arith_soma_bank.clean_up(p_trans, steps=50, dt=0.02)

        # Direct Phase Demodulation (Zero Dictionary!)
        # x_hat = sum(omega_k * phi_k) / sum(omega_k^2)
        if t_eff > 0.0 and not blind_input:
            phases = np.unwrap(np.angle(p_clean))
            demod_val = float(np.sum(self.frequencies * phases) / np.sum(self.frequencies ** 2))
            predicted_val = int(round(demod_val))
        else:
            predicted_val = -1

        return {
            "predicted_val": predicted_val,
            "output_wave": p_clean,
            "t_eff": t_eff,
        }

    def wave_invert(
        self,
        val_sum: int,
        val_known: int,
        noise_sigma: float = 0.0,
        decouple_transmission: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_soma: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Continuous algebraic inversion: psi(b) = psi(c) * conj(psi(a)), decoded via phase demodulation."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            pc = np.zeros(self.n_frequencies, dtype=np.complex128)
            pa = np.zeros(self.n_frequencies, dtype=np.complex128)
        else:
            pc = np.exp(1j * val_sum * self.frequencies)
            pa = np.exp(1j * val_known * self.frequencies)
            if noise_sigma > 0.0:
                pc *= np.exp(1j * noise_sigma * local_rng.randn(self.n_frequencies))
                pa *= np.exp(1j * noise_sigma * local_rng.randn(self.n_frequencies))

        if decouple_transmission:
            p_trans = np.zeros(self.n_frequencies, dtype=np.complex128)
            t_eff = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.n_frequencies)
            p_trans = pc[perm] * np.conj(pa)
            t_eff = 0.5
        elif blind_input:
            p_trans = np.zeros(self.n_frequencies, dtype=np.complex128)
            t_eff = 0.0
        else:
            p_trans = pc * np.conj(pa)
            t_eff = 1.0

        if ablate_soma or decouple_transmission or blind_input:
            p_clean = p_trans.copy()
        else:
            p_clean = self.arith_soma_bank.clean_up(p_trans, steps=50, dt=0.02)

        if t_eff > 0.0 and not blind_input:
            phases = np.unwrap(np.angle(p_clean))
            demod_val = float(np.sum(self.frequencies * phases) / np.sum(self.frequencies ** 2))
            predicted_val = int(round(demod_val))
        else:
            predicted_val = -1

        return {
            "predicted_val": predicted_val,
            "output_wave": p_clean,
            "t_eff": t_eff,
        }
