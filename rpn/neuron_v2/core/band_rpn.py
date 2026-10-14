"""Multi-band RPN (B-10) — K frequency-split dendritic bands per unit."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..substrate.phasor import bundle, normalise, sim_batch
from .born_gate import rectified_born
from .delta_memory import DeltaMemory
from .item_table import ItemTable


def band_slice(D: int, band: int, K: int) -> slice:
    if K < 1 or band < 0 or band >= K:
        raise ValueError("invalid band index")
    size = D // K
    start = band * size
    end = D if band == K - 1 else start + size
    return slice(start, end)


def project_band(v: np.ndarray, band: int, K: int) -> np.ndarray:
    """Keep one contiguous frequency band; zero and renormalize."""
    out = np.zeros_like(v)
    out[band_slice(len(v), band, K)] = v[band_slice(len(v), band, K)]
    mag = np.linalg.norm(out)
    if mag < 1e-12:
        return out.astype(np.complex64)
    return (out / mag).astype(np.complex64)


def corrupt_bands(
    v: np.ndarray,
    K: int,
    corrupt: set[int],
    sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Add phase noise to selected bands (frequency-division stress test)."""
    out = v.astype(np.complex64).copy()
    for band in corrupt:
        sl = band_slice(len(out), band, K)
        chunk = out[sl]
        phases = np.angle(chunk).astype(np.float32) + rng.normal(
            0.0, sigma, chunk.shape[0]
        ).astype(np.float32)
        out[sl] = np.exp(1j * phases).astype(np.complex64)
    return normalise(out)


def scramble_bands(
    v: np.ndarray,
    K: int,
    scramble: set[int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Replace selected bands with random phasors (destroys band-local binding)."""
    from ..substrate.phasor import random_hv

    out = v.astype(np.complex64).copy()
    for band in scramble:
        sl = band_slice(len(out), band, K)
        n = sl.stop - sl.start
        out[sl] = random_hv(n, seed=int(rng.randint(0, 2**31 - 1)))
    return normalise(out)


def random_band_scramble(
    v: np.ndarray,
    K: int,
    p_scramble: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, set[int]]:
    """Independently scramble each band with probability *p_scramble*.

    Returns corrupted context and the set of bands left intact.
    """
    out = v.astype(np.complex64).copy()
    clean: set[int] = set()
    for band in range(K):
        if rng.rand() < p_scramble:
            sl = band_slice(len(out), band, K)
            n = sl.stop - sl.start
            from ..substrate.phasor import random_hv

            out[sl] = random_hv(n, seed=int(rng.randint(0, 2**31 - 1)))
        else:
            clean.add(band)
    return normalise(out), clean


def band_route(token: str, K: int) -> int:
    """Stable hash route: spread associations across dendritic bands."""
    return sum(ord(ch) for ch in token) % K


class MultiBandRPN:
    """K band-isolated delta memories; predict bundles band-wise reads."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        mode: str = "delta",
        *,
        K: int = 3,
        redundant: bool = True,
    ):
        if K < 1:
            raise ValueError("K must be >= 1")
        self.D = D
        self.items = items
        self.K = K
        self.redundant = redundant
        self.bands: list[DeltaMemory] = [DeltaMemory(D, mode=mode) for _ in range(K)]
        self.shard_of: dict[str, int] = {}
        self.mode = mode

    @property
    def mem(self) -> DeltaMemory:
        """RPC/column compatibility: first band's delta memory."""
        return self.bands[0]

    def learn_hv(self, ctx: np.ndarray, target: np.ndarray) -> None:
        """HV-level write (RPC integrator/slot paths). Requires redundant mode."""
        if not self.redundant:
            raise ValueError("learn_hv requires redundant=True")
        for k in range(self.K):
            self.bands[k].write(project_band(ctx, k, self.K), target)

    def route_band(self, ctx_token: str) -> int:
        return band_route(ctx_token, self.K)

    def learn(
        self,
        ctx_token: str,
        target_token: str,
        *,
        band: int | None = None,
    ) -> None:
        ctx = self.items[ctx_token]
        target = self.items[target_token]
        if self.redundant:
            for k in range(self.K):
                self.bands[k].write(project_band(ctx, k, self.K), target)
        else:
            k = band if band is not None else self.route_band(ctx_token)
            if k < 0 or k >= self.K:
                raise ValueError("band index out of range")
            self.shard_of[ctx_token] = k
            self.bands[k].write(project_band(ctx, k, self.K), target)

    def predict(self, ctx: np.ndarray, *, active_bands: set[int] | None = None) -> np.ndarray:
        bands = range(self.K) if active_bands is None else sorted(active_bands)
        if not bands:
            return np.zeros(self.D, dtype=np.complex64)
        parts = [
            self.bands[k].predict(project_band(ctx, k, self.K)) for k in bands
        ]
        return bundle(*parts, normalise_output=False).astype(np.complex64)

    def _born_read(
        self,
        pred: np.ndarray,
        *,
        gate_threshold: float,
        confidence_threshold: float,
        min_resonance: float,
    ) -> tuple[Optional[str], float]:
        confidence = float(np.linalg.norm(pred)) / np.sqrt(self.D)
        if confidence < confidence_threshold:
            return None, 0.0

        echo = normalise(pred)
        tokens, mat = self.items.matrix()
        res = sim_batch(mat, echo)
        out = rectified_born(
            res,
            gate_threshold=gate_threshold,
            min_resonance=min_resonance,
        )
        if not out.gate_passed:
            return None, out.best_prob
        return tokens[out.best_idx], out.best_prob

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
        *,
        ctx_hv: np.ndarray | None = None,
        active_bands: set[int] | None = None,
    ) -> tuple[Optional[str], float]:
        ctx = ctx_hv if ctx_hv is not None else self.items[ctx_token]

        if not self.redundant and active_bands is None:
            k = self.shard_of.get(ctx_token, self.route_band(ctx_token))
            pred = self.bands[k].predict(project_band(ctx, k, self.K))
            return self._born_read(
                pred,
                gate_threshold=gate_threshold,
                confidence_threshold=confidence_threshold,
                min_resonance=min_resonance,
            )

        # Partial-band reads: pick the strongest intact dendrite (avoid bundling noise).
        if active_bands is not None and 0 < len(active_bands) < self.K:
            best_token: Optional[str] = None
            best_prob = -1.0
            for k in sorted(active_bands):
                pred = self.bands[k].predict(project_band(ctx, k, self.K))
                token, prob = self._born_read(
                    pred,
                    gate_threshold=gate_threshold,
                    confidence_threshold=confidence_threshold,
                    min_resonance=min_resonance,
                )
                if token is not None and prob > best_prob:
                    best_token, best_prob = token, prob
            if best_token is None:
                return None, 0.0
            return best_token, best_prob

        return self._born_read(
            self.predict(ctx, active_bands=active_bands),
            gate_threshold=gate_threshold,
            confidence_threshold=confidence_threshold,
            min_resonance=min_resonance,
        )
