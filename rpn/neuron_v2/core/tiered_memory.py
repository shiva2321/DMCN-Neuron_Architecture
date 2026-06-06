"""T0/T1/T2 memory hierarchy + consolidation replay (Stage 2 / B-4).

T0 — working slot (transient, exact ctx reinstatement)
T1 — delta-rule fast weights (episodic, sec–min)
T2 — delta-rule semantic store (consolidated via replay)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..substrate.phasor import normalise, sim
from .delta_memory import DeltaMemory


@dataclass
class WorkingSlot:
    ctx: np.ndarray | None = None
    value: np.ndarray | None = None


@dataclass
class ConsolidationStats:
    n_replayed: int
    t2_writes_before: int
    t2_writes_after: int


class TieredMemory:
    """Three-tier memory with episodic log and T1→T2 consolidation."""

    def __init__(
        self,
        D: int,
        mode: str = "delta",
        *,
        lr: float = 1.0,
        t1_decay: float = 1.0,
        surprise_threshold: float = 0.0,
        t0_match_threshold: float = 0.99,
    ):
        self.D = D
        self.mode = mode
        self.t1_decay = float(t1_decay)
        self.surprise_threshold = float(surprise_threshold)
        self.t0_match_threshold = float(t0_match_threshold)
        self.t0 = WorkingSlot()
        self.t1 = DeltaMemory(D, mode=mode, lr=lr)
        self.t2 = DeltaMemory(D, mode=mode, lr=lr)
        self._episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self._consolidated_upto: int = 0

    def set_working(self, ctx: np.ndarray, value: np.ndarray) -> None:
        self.t0 = WorkingSlot(
            ctx.astype(np.complex64),
            value.astype(np.complex64),
        )

    def surprise(self, ctx: np.ndarray, value: np.ndarray) -> float:
        pred = self.predict(ctx, use_t0=False)
        return 1.0 - sim(normalise(pred), normalise(value))

    def write_episodic(
        self,
        ctx: np.ndarray,
        value: np.ndarray,
        *,
        gate_surprise: bool = True,
    ) -> float:
        """Surprise-gated delta write into T1; append to episodic log."""
        s = self.surprise(ctx, value)
        if gate_surprise and s < self.surprise_threshold:
            return s
        self.t1.write(ctx, value)
        self._episodes.append((ctx.copy(), value.copy()))
        if self.t1_decay < 1.0:
            self.t1.W = (self.t1.W * self.t1_decay).astype(np.complex64)
        return s

    def consolidate(
        self,
        *,
        replay_all: bool = False,
        reset_t1: bool = True,
    ) -> ConsolidationStats:
        """Replay episodic pairs into T2 (ReSC+ / hippocampal→neocortical)."""
        start = 0 if replay_all else self._consolidated_upto
        batch = self._episodes[start:]
        before = self.t2.n_writes
        for ctx, value in batch:
            self.t2.write(ctx, value)
        self._consolidated_upto = len(self._episodes)
        if reset_t1:
            self.t1.W.fill(0)
            self.t1.n_writes = 0
        return ConsolidationStats(
            n_replayed=len(batch),
            t2_writes_before=before,
            t2_writes_after=self.t2.n_writes,
        )

    def predict(
        self,
        ctx: np.ndarray,
        *,
        use_t0: bool = True,
        use_t1: bool = True,
        use_t2: bool = True,
    ) -> np.ndarray:
        if use_t0 and self.t0.ctx is not None and self.t0.value is not None:
            if sim(self.t0.ctx, ctx) >= self.t0_match_threshold:
                return self.t0.value.copy()

        out = np.zeros(self.D, dtype=np.complex64)
        if use_t2:
            out += self.t2.predict(ctx)
        if use_t1:
            out += self.t1.predict(ctx)
        return out.astype(np.complex64)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)
