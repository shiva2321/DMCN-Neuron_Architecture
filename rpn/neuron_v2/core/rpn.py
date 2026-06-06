"""RPN - Resonant Predictive Neuron (k=1 one-shot core).

learn(ctx, target): write the association into local DeltaMemory.
read(ctx): recall echo -> score candidates -> Born gate -> emit token or refuse (None).
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from ..substrate.phasor import normalise, sim_batch
from .delta_memory import DeltaMemory
from .born_gate import rectified_born
from .item_table import ItemTable


class RPN:
    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.D = D
        self.items = items
        self.mem = DeltaMemory(D, mode=mode)

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.mem.write(self.items[ctx_token], self.items[target_token])

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
        *,
        ctx_hv: np.ndarray | None = None,
    ) -> tuple[Optional[str], float]:
        ctx = ctx_hv if ctx_hv is not None else self.items[ctx_token]
        pred = self.mem.predict(ctx)
        # Unit phasor HVs have L2 norm sqrt(D); a confident recall is O(sqrt(D)).
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
