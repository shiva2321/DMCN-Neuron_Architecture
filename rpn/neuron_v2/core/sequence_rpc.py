"""Sequence RPC (B-11) — compositional perception + associative action chain."""
from __future__ import annotations

from typing import Optional

from .item_table import ItemTable
from .rpc import ResonantPredictiveColumn
from .sequence_rpn import ChainRPN


class SequenceRPC:
    """Two-stage stack: RPC (color, shape) → object, then ChainRPN object → action."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        *,
        rpc: ResonantPredictiveColumn | None = None,
        chain: ChainRPN | None = None,
        mode: str = "delta",
        **rpc_kwargs,
    ):
        self.D = D
        self.items = items
        self.rpc = rpc or ResonantPredictiveColumn(D, items, mode=mode, **rpc_kwargs)
        self.chain = chain or ChainRPN(D, items, mode=mode)

    def learn(self, color: str, shape: str, object_token: str, action_token: str) -> None:
        self.rpc.learn(color, shape, object_token)
        self.chain.learn_edge(object_token, action_token)

    def read_object(
        self,
        color: str,
        shape: str,
        **gate,
    ) -> tuple[Optional[str], float]:
        return self.rpc.read(color, shape, **gate)

    def read_action(
        self,
        color: str,
        shape: str,
        *,
        action_hops: int = 1,
        **gate,
    ) -> tuple[Optional[str], float]:
        obj, obj_prob = self.read_object(color, shape, **gate)
        if obj is None:
            return None, 0.0
        action, act_prob = self.chain.read_hops(obj, action_hops, **gate)
        if action is None:
            return None, obj_prob
        return action, min(obj_prob, act_prob)

    def end_task(self) -> None:
        """Task boundary on the perception RPC (tiered consolidation); chain edges persist."""
        self.rpc.end_task()
