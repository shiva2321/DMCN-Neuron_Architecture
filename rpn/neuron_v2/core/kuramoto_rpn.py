"""KuramotoRPN — multi-band Kuramoto settling read + delta write (B-12)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..substrate.phasor import normalise, sim_batch
from .born_gate import rectified_born
from .delta_memory import DeltaMemory
from .item_table import ItemTable
from .kuramoto import kuramoto_predict


class KuramotoRPN:
    """RPN with K-band Kuramoto relaxation before delta-memory read."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        mode: str = "delta",
        *,
        K: int = 3,
        k: int = 4,
        coupling: float = 0.35,
        settle_dt: float = 0.2,
        settle_beta: float = 0.55,
        ctx_prefix: str | None = "ctx_",
        tgt_prefix: str | None = "tgt_",
    ):
        if K < 1 or k < 1:
            raise ValueError("K and k must be >= 1")
        self.D = D
        self.items = items
        self.K = K
        self.k = k
        self.coupling = float(coupling)
        self.settle_dt = float(settle_dt)
        self.settle_beta = float(settle_beta)
        self.ctx_prefix = ctx_prefix
        self.tgt_prefix = tgt_prefix
        self.mem = DeltaMemory(D, mode=mode)

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.mem.write(self.items[ctx_token], self.items[target_token])

    def learn_hv(self, ctx: np.ndarray, target: np.ndarray) -> None:
        self.mem.write(ctx, target)

    def predict(self, ctx: np.ndarray, *, k: int | None = None) -> np.ndarray:
        return kuramoto_predict(
            self.mem,
            ctx,
            self.items,
            K=self.K,
            k=self.k if k is None else k,
            coupling=self.coupling,
            dt=self.settle_dt,
            beta=self.settle_beta,
            ctx_prefix=self.ctx_prefix,
            tgt_prefix=self.tgt_prefix,
        )

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
