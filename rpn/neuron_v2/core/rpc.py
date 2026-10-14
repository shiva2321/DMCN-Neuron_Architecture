"""Resonant Predictive Column (RPC) — reference-frame composition + CMP vote."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from .band_rpn import MultiBandRPN
from .column import _read_unit_ctx
from .item_table import ItemTable
from .kuramoto_rpn import KuramotoRPN
from .predictive_rpn import PredictiveRPN
from .reference_frame import ReferenceFrame
from .rpn import RPN
from .tiered_rpn import TieredRPN

MemoryMode = Literal["predictive", "plain", "tiered", "multiband", "kuramoto"]
Unit = Union[RPN, PredictiveRPN, TieredRPN, MultiBandRPN, KuramotoRPN]


@dataclass
class CorticalMessage:
    """One neuron read proposal (Cortical-Messaging-Protocol payload)."""

    source: str
    token: Optional[str]
    prob: float


def _cmp_resolve(
    messages: list[CorticalMessage],
    *,
    integrator_source: str = "integrator",
) -> tuple[Optional[str], float]:
    """Integrator wins when it Born-passes; else best prob among passing messages."""
    by_source = {m.source: m for m in messages}
    integrator = by_source.get(integrator_source)
    if integrator is not None and integrator.token is not None:
        return integrator.token, integrator.prob

    passing = [m for m in messages if m.token is not None]
    if not passing:
        return None, 0.0
    best = max(passing, key=lambda m: m.prob)
    return best.token, best.prob


def _unit_mode(unit: Unit) -> str:
    return unit.mem.mode if hasattr(unit, "mem") else "delta"


def _make_unit(
    D: int,
    items: ItemTable,
    mode: str,
    *,
    memory: MemoryMode,
    k: int,
    surprise_threshold: float,
    consolidate_on_task_end: bool,
    multiband_K: int = 3,
    kuramoto_K: int = 3,
    kuramoto_coupling: float = 0.35,
) -> Unit:
    if memory == "kuramoto":
        return KuramotoRPN(
            D,
            items,
            mode=mode,
            K=kuramoto_K,
            k=k,
            coupling=kuramoto_coupling,
            ctx_prefix=None,
            tgt_prefix=None,
        )
    if memory == "multiband":
        return MultiBandRPN(D, items, mode=mode, K=multiband_K, redundant=True)
    if memory == "tiered":
        return TieredRPN(
            D,
            items,
            mode=mode,
            consolidate_on_task_end=consolidate_on_task_end,
            surprise_threshold=surprise_threshold,
        )
    if memory == "predictive":
        return PredictiveRPN(
            D,
            items,
            mode=mode,
            k=k,
            surprise_threshold=surprise_threshold,
        )
    return RPN(D, items, mode=mode)


class ResonantPredictiveColumn:
    """Full RPC: reference-frame bind, where-sharded specialists, integrator + CMP vote."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        frame: ReferenceFrame | None = None,
        mode: str = "delta",
        *,
        shard_slot: str = ReferenceFrame.SLOT_WHERE,
        what_slot: str = ReferenceFrame.SLOT_WHAT,
        memory: MemoryMode = "predictive",
        k: int = 1,
        surprise_threshold: float = 0.05,
        consolidate_on_task_end: bool = True,
        multiband_K: int = 3,
        kuramoto_K: int = 3,
        kuramoto_coupling: float = 0.35,
    ):
        self.D = D
        self.items = items
        self.frame = frame or ReferenceFrame.color_shape(D)
        self.shard_slot = shard_slot
        self.what_slot = what_slot
        self.memory = memory
        self.k = k
        self.surprise_threshold = surprise_threshold
        self.consolidate_on_task_end = consolidate_on_task_end
        self.multiband_K = multiband_K
        self.kuramoto_K = kuramoto_K
        self.kuramoto_coupling = kuramoto_coupling

        def factory() -> Unit:
            return _make_unit(
                D,
                items,
                mode,
                memory=memory,
                k=k,
                surprise_threshold=surprise_threshold,
                consolidate_on_task_end=consolidate_on_task_end,
                multiband_K=multiband_K,
                kuramoto_K=kuramoto_K,
                kuramoto_coupling=kuramoto_coupling,
            )

        self.integrator = factory()
        self.slot_units: dict[str, Unit] = {
            spec.name: factory() for spec in self.frame.slots
        }
        self._where_shards: dict[str, Unit] = {}

    def _fillers(self, color: str, shape: str) -> dict[str, str]:
        return {self.shard_slot: color, self.what_slot: shape}

    def _what_ctx(self, shape: str) -> np.ndarray:
        return self.frame.bind_slot(self.items, self.what_slot, shape)

    def _shard(self, where_token: str) -> Unit:
        if where_token not in self._where_shards:
            self._where_shards[where_token] = _make_unit(
                self.D,
                self.items,
                _unit_mode(self.integrator),
                memory=self.memory,
                k=self.k,
                surprise_threshold=self.surprise_threshold,
                consolidate_on_task_end=self.consolidate_on_task_end,
                multiband_K=self.multiband_K,
                kuramoto_K=self.kuramoto_K,
                kuramoto_coupling=self.kuramoto_coupling,
            )
        return self._where_shards[where_token]

    def _write(self, unit: Unit, ctx: np.ndarray, target: np.ndarray) -> None:
        if isinstance(unit, PredictiveRPN):
            unit.learn_hv(ctx, target)
        elif isinstance(unit, TieredRPN):
            unit.learn_hv(ctx, target)
        elif isinstance(unit, MultiBandRPN):
            unit.learn_hv(ctx, target)
        elif isinstance(unit, KuramotoRPN):
            unit.learn_hv(ctx, target)
        else:
            unit.mem.write(ctx, target)

    def learn(self, color: str, shape: str, object_token: str) -> None:
        fillers = self._fillers(color, shape)
        target = self.items[object_token]
        composite = self.frame.compose(self.items, fillers)
        self._write(self.integrator, composite, target)
        for slot, token in fillers.items():
            ctx = self.frame.bind_slot(self.items, slot, token)
            self._write(self.slot_units[slot], ctx, target)
        self._write(self._shard(fillers[self.shard_slot]), self._what_ctx(shape), target)

    def read(
        self,
        color: str,
        shape: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        fillers = self._fillers(color, shape)
        gate = dict(
            gate_threshold=gate_threshold,
            confidence_threshold=confidence_threshold,
            min_resonance=min_resonance,
        )

        if color in self._where_shards:
            return _read_unit_ctx(
                self._where_shards[color],
                self.D,
                self.items,
                self._what_ctx(shape),
                **gate,
            )

        composite = self.frame.compose(self.items, fillers)
        messages: list[CorticalMessage] = []
        token, prob = _read_unit_ctx(self.integrator, self.D, self.items, composite, **gate)
        messages.append(CorticalMessage("integrator", token, prob))

        for slot, filler in fillers.items():
            ctx = self.frame.bind_slot(self.items, slot, filler)
            token, prob = _read_unit_ctx(self.slot_units[slot], self.D, self.items, ctx, **gate)
            messages.append(CorticalMessage(f"slot:{slot}", token, prob))

        return _cmp_resolve(messages)

    def end_task(self) -> None:
        """Consolidate tiered units at a task boundary (no-op for plain/predictive memory)."""
        units: list[Unit] = [
            self.integrator,
            *self.slot_units.values(),
            *self._where_shards.values(),
        ]
        for unit in units:
            if isinstance(unit, TieredRPN):
                unit.end_task()

    def n_where_shards(self) -> int:
        return len(self._where_shards)

    def write_stats(self) -> tuple[int, int]:
        """Applied and skipped surprise-gated writes across predictive units."""
        applied = skipped = 0
        units: list[Unit] = [self.integrator, *self.slot_units.values(), *self._where_shards.values()]
        for unit in units:
            if isinstance(unit, PredictiveRPN):
                applied += unit.n_writes_applied
                skipped += unit.n_writes_skipped
        return applied, skipped
