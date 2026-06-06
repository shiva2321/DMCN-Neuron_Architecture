"""DeltaMemory - fast-weight associative memory (key -> value).

State is a (D, D) complex weight matrix W (a 'fast weight' / linear-attention state).
  read(key):       v_est = W @ k_hat                      (k_hat = key / ||key||)
  additive write:  W += outer(value, conj(k_hat))         (Hebbian outer product; saturates)
  delta write:     W += lr * outer(value - W @ k_hat, conj(k_hat))   (delta / LMS rule)

The delta rule subtracts the current readout before writing the residual, so:
  - re-storing a known (key, value) adds ~0 (residual -> 0): no runaway saturation
  - it error-corrects crosstalk (online least-squares), beating the additive outer-product sum.
This is the VSA/phasor analogue of the delta rule used by Gated DeltaNet / Titans.

Why a matrix (not a single holographic vector): for unit phasors,
bind(ctx, unbind(surface, ctx)) == surface, so subtracting a rebound readout from a single
vector surface erases the whole memory. A (D x D) outer-product state localises the update to
the key subspace, which is what makes the delta rule work.
"""
from __future__ import annotations
import numpy as np

from ..substrate.phasor import normalise


class DeltaMemory:
    def __init__(self, D: int, mode: str = "delta", lr: float = 1.0):
        if mode not in ("delta", "additive"):
            raise ValueError("mode must be 'delta' or 'additive'")
        self.D = D
        self.mode = mode
        self.lr = float(lr)
        self.W = np.zeros((D, D), dtype=np.complex64)
        self.n_writes = 0

    def _khat(self, key: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(key))
        if n < 1e-12:
            return key.astype(np.complex64)
        return (key / n).astype(np.complex64)

    def predict(self, key: np.ndarray) -> np.ndarray:
        """Raw value estimate W @ k_hat (not normalised)."""
        return (self.W @ self._khat(key)).astype(np.complex64)

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        k = self._khat(key)
        if self.mode == "additive":
            self.W = (self.W + np.outer(value, np.conj(k))).astype(np.complex64)
        else:  # delta / LMS
            pred = self.W @ k
            self.W = (self.W + self.lr * np.outer(value - pred, np.conj(k))).astype(np.complex64)
        self.n_writes += 1

    def recall(self, key: np.ndarray) -> np.ndarray:
        """Normalised (unit-phasor) value estimate, for similarity comparison."""
        return normalise(self.predict(key))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.abs(self.W) ** 2)))
