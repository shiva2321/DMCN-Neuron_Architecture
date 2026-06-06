"""TieredRPN — RPN with T0/T1/T2 memory and task-boundary consolidation."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..substrate.phasor import normalise, sim_batch
from .born_gate import rectified_born
from .item_table import ItemTable
from .tiered_memory import TieredMemory, WorkingSlot


class TieredRPN:
    def __init__(
        self,
        D: int,
        items: ItemTable,
        mode: str = "delta",
        *,
        consolidate_on_task_end: bool = True,
        t1_decay: float = 1.0,
        surprise_threshold: float = 0.0,
    ):
        self.D = D
        self.items = items
        self.consolidate_on_task_end = consolidate_on_task_end
        self.mem = TieredMemory(
            D,
            mode=mode,
            t1_decay=t1_decay,
            surprise_threshold=surprise_threshold,
        )

    def learn(self, ctx_token: str, target_token: str) -> float:
        ctx = self.items[ctx_token]
        target = self.items[target_token]
        self.mem.set_working(ctx, target)
        return self.mem.write_episodic(ctx, target)

    def end_task(self) -> None:
        if self.consolidate_on_task_end:
            self.mem.consolidate(reset_t1=True)
        else:
            # Bounded episodic buffer: clear T1 without T2 promotion (forgetting).
            self.mem.t1.W.fill(0)
            self.mem.t1.n_writes = 0
            self.mem.t0 = WorkingSlot()

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
        *,
        use_t0: bool = True,
        use_t1: bool = True,
        use_t2: bool = True,
    ) -> tuple[Optional[str], float]:
        ctx = self.items[ctx_token]
        pred = self.mem.predict(
            ctx,
            use_t0=use_t0,
            use_t1=use_t1,
            use_t2=use_t2,
        )
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
