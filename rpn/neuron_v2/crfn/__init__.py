"""Continuous Resonant Field Neuron (CRFN) Package.

A genuine first-principles physical alternative to Point Perceptrons and Vector Symbolic Architectures.
Features:
- Continuous wavefield dynamics on spatial transmission dendrites.
- Stuart-Landau limit-cycle oscillator soma with autonomous Lyapunov noise dissipation.
- Local membrane impedance plasticity via principle of maximum power transfer.
- Zero ItemTable codebooks, zero outer-product matrices, zero symbolic vector bindings.
"""
from __future__ import annotations

from .crfn_soma import StuartLandauSoma
from .crfn_dendrite import TransmissionLineDendrite
from .crfn_impedance import ImpedanceSynapse
from .crfn_unit import ContinuousResonantFieldNeuron

__all__ = [
    "StuartLandauSoma",
    "TransmissionLineDendrite",
    "ImpedanceSynapse",
    "ContinuousResonantFieldNeuron",
]
