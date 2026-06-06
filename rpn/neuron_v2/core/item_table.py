"""ItemTable - maps tokens to stable phasor HVs and builds a candidate matrix."""
from __future__ import annotations
import hashlib
import numpy as np

from ..substrate.phasor import random_hv


def _token_seed(token: str, base_seed: int) -> int:
    """Stable cross-process seed (Python hash() is salted per interpreter)."""
    digest = hashlib.md5(f"{base_seed}:{token}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


class ItemTable:
    def __init__(self, D: int, seed: int = 42):
        self.D = D
        self.seed = seed
        self._t: dict[str, np.ndarray] = {}

    def __getitem__(self, token: str) -> np.ndarray:
        if token not in self._t:
            self._t[token] = random_hv(self.D, seed=_token_seed(token, self.seed))
        return self._t[token]

    def matrix(self) -> tuple[list[str], np.ndarray]:
        tokens = list(self._t.keys())
        mat = np.stack([self._t[t] for t in tokens]).astype(np.complex64)
        return tokens, mat
