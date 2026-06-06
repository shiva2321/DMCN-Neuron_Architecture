"""PredictiveRPN — k>1 settling read + surprise-gated delta write (Stage 3 / B-5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..substrate.phasor import normalise, sim, sim_batch
from .born_gate import rectified_born
from .delta_memory import DeltaMemory
from .item_table import ItemTable
from .settling import settle_predict


@dataclass
class LearnStats:
    surprise: float
    wrote: bool


class PredictiveRPN:
    """Resonant Predictive Neuron with predictive loop (k>1) and surprise gating."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        mode: str = "delta",
        *,
        k: int = 4,
        surprise_threshold: float = 0.05,
        settle_beta: float = 0.55,
        ctx_prefix: str | None = "ctx_",
        tgt_prefix: str | None = "tgt_",
        gate_writes: bool = True,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.D = D
        self.items = items
        self.k = k
        self.surprise_threshold = float(surprise_threshold)
        self.settle_beta = float(settle_beta)
        self.ctx_prefix = ctx_prefix
        self.tgt_prefix = tgt_prefix
        self.gate_writes = gate_writes
        self.mem = DeltaMemory(D, mode=mode)
        self.n_writes_applied = 0
        self.n_writes_skipped = 0

    def predict(
        self,
        ctx: np.ndarray,
        *,
        k: int | None = None,
    ) -> np.ndarray:
        return settle_predict(
            self.mem,
            ctx,
            self.items,
            self.k if k is None else k,
            beta=self.settle_beta,
            ctx_prefix=self.ctx_prefix,
            tgt_prefix=self.tgt_prefix,
        )

    def surprise(self, ctx: np.ndarray, target: np.ndarray) -> float:
        pred = self.predict(ctx)
        return 1.0 - sim(normalise(pred), normalise(target))

    def learn(
        self,
        ctx_token: str,
        target_token: str,
        *,
        gate_surprise: bool | None = None,
    ) -> LearnStats:
        ctx = self.items[ctx_token]
        target = self.items[target_token]
        s = self.surprise(ctx, target)
        gate = self.gate_writes if gate_surprise is None else gate_surprise
        if gate and s < self.surprise_threshold:
            self.n_writes_skipped += 1
            return LearnStats(surprise=s, wrote=False)
        self.mem.write(ctx, target)
        self.n_writes_applied += 1
        return LearnStats(surprise=s, wrote=True)

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
        *,
        ctx_hv: np.ndarray | None = None,
        k: int | None = None,
    ) -> tuple[Optional[str], float]:
        ctx = ctx_hv if ctx_hv is not None else self.items[ctx_token]
        pred = self.predict(ctx, k=k)
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
