"""Phasor VSA substrate (ported subset of Resonance_X/resonance_x/substrate/phasor.py).

Every symbol is a vector of unit complex numbers x in C^D, x_j = e^{i*theta_j}.
Guarantees: |bind|=1 ; unbind(bind(a,b),a)=b ; sim(v,v)=1 ; E[sim(x,y)]=0.
"""
from __future__ import annotations
import numpy as np


def random_hv(D: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed) if seed is not None else np.random
    phases = rng.uniform(-np.pi, np.pi, D).astype(np.float32)
    return np.exp(1j * phases).astype(np.complex64)


def normalise(v: np.ndarray) -> np.ndarray:
    mag = np.abs(v)
    out = v.copy()
    mask = mag > 1e-12
    out[mask] /= mag[mask]
    return out.astype(np.complex64)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).astype(np.complex64)


def unbind(ab: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (ab * np.conj(a)).astype(np.complex64)


def bundle(*vectors: np.ndarray, normalise_output: bool = True) -> np.ndarray:
    if not vectors:
        raise ValueError("bundle() requires at least one vector")
    s = vectors[0].astype(np.complex64).copy()
    for v in vectors[1:]:
        s += v.astype(np.complex64)
    return normalise(s) if normalise_output else s.astype(np.complex64)


def sim(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.real(np.dot(x, np.conj(y)))) / len(x)


def sim_batch(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    D = matrix.shape[1]
    mat_real = matrix.view(np.float32).reshape(len(matrix), -1)
    qry_real = query.view(np.float32)
    return (mat_real @ qry_real / D).astype(np.float32)
