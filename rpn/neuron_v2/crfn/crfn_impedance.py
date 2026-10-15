"""Local Membrane Impedance Adaptation (Principle of Maximum Power Transfer).

Replaces external weight matrices (W in C^{D x D}) with local complex membrane impedances Z_j.
Signal transmission through each synapse is governed by transmission-line impedance matching:
    Gamma = (Z_syn - Z_0) / (Z_syn + Z_0)   (Reflection Coefficient)
    T = 1 - |Gamma|^2                       (Power Transmission Coefficient)

When the synapse impedance matches the membrane (Z_syn = Z_0), reflection is zero (Gamma = 0)
and 100% of the signal energy enters the dendritic wavefield.
Plasticity updates Z_j locally from somatic-dendritic wave resonance without backpropagation:
    dZ_j/dt = eta * [ z(t) * Psi*(x_j, t) - alpha * |z(t)|^2 * Z_j ]
"""
from __future__ import annotations

import numpy as np


class ImpedanceSynapse:
    """Active transmission-line synapse with continuous impedance matching."""

    def __init__(
        self,
        syn_id: int,
        z_init: complex | None = None,
        z0_membrane: float = 1.0,
        lr: float = 0.05,
        decay: float = 0.01,
    ):
        self.id = int(syn_id)
        self.Z0 = float(z0_membrane)
        self.lr = float(lr)
        self.alpha = float(decay)

        # Characteristic complex impedance Z = R + i*X (Resistance + Reactance)
        if z_init is not None:
            self.Z = complex(z_init)
        else:
            # Initialize with small random mismatch
            r = np.random.uniform(0.5, 1.5)
            x = np.random.uniform(-0.5, 0.5)
            self.Z = complex(r, x)

    @property
    def reflection_coefficient(self) -> complex:
        """Fresnel reflection coefficient Gamma at the synaptic membrane boundary."""
        denom = self.Z + self.Z0
        if abs(denom) < 1e-12:
            return complex(1.0, 0.0)
        return (self.Z - self.Z0) / denom

    @property
    def power_transmission(self) -> float:
        """Fraction of incoming wave power transmitted into dendrite: T = 1 - |Gamma|^2."""
        gamma = self.reflection_coefficient
        mag2 = gamma.real * gamma.real + gamma.imag * gamma.imag
        return float(np.clip(1.0 - mag2, 0.0, 1.0))

    def transmit(self, input_signal: complex | float) -> complex:
        """Transmit input wave into the dendrite scaled by impedance transmission."""
        T = self.power_transmission
        # Amplitude transmission is sqrt(T), with phase shift from synaptic reactance
        amp_t = np.sqrt(T)
        phase_shift = np.angle(self.Z)
        return complex(amp_t * np.exp(1j * phase_shift) * input_signal)

    def learn_step(
        self,
        somatic_wave: complex,
        local_dendritic_wave: complex,
        dt: float = 0.05,
    ) -> complex:
        """Local impedance adaptation via maximum power transfer / least action.
        
        Learns strictly from local wave coincidence without global gradients.
        """
        # Cross-correlation between arriving somatic bAP and local forward wave
        drive = somatic_wave * np.conj(local_dendritic_wave)
        soma_energy = somatic_wave.real * somatic_wave.real + somatic_wave.imag * somatic_wave.imag

        # Update equation: dZ/dt = lr * (drive - alpha * soma_energy * Z)
        dZ = self.lr * (drive - self.alpha * soma_energy * self.Z)
        self.Z += dZ * dt

        # Enforce positive resistance (physical passive membrane constraint R > 0.05)
        if self.Z.real < 0.05:
            self.Z = complex(0.05, self.Z.imag)

        return self.Z

    def reset(self, z_init: complex | None = None):
        """Reset impedance state."""
        if z_init is not None:
            self.Z = complex(z_init)
        else:
            self.Z = complex(1.0, 0.0)
