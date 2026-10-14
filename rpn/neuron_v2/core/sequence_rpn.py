"""Sequence / multi-hop RPN (B-10) — chained associative reads."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .item_table import ItemTable
from .rpn import RPN


class ChainRPN:
    """Associative edges with explicit multi-hop walk (A→B→C)."""

    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.D = D
        self.items = items
        self.rpn = RPN(D, items, mode=mode)

    def learn_edge(self, src_token: str, dst_token: str) -> None:
        self.rpn.learn(src_token, dst_token)

    def read_hops(
        self,
        start_token: str,
        hops: int,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        if hops < 1:
            raise ValueError("hops must be >= 1")
        token: Optional[str] = start_token
        prob = 0.0
        for _ in range(hops):
            if token is None:
                return None, 0.0
            token, prob = self.rpn.read(
                token,
                gate_threshold=gate_threshold,
                confidence_threshold=confidence_threshold,
                min_resonance=min_resonance,
            )
        return token, prob


def build_chain_edges(chain_id: int, length: int) -> list[tuple[str, str]]:
    """Linear chain: chain{c}_s{i} -> chain{c}_s{i+1}."""
    edges: list[tuple[str, str]] = []
    for i in range(length - 1):
        edges.append(
            (
                f"chain{chain_id}_s{i}",
                f"chain{chain_id}_s{i + 1}",
            )
        )
    return edges


def chain_endpoints(chain_id: int, length: int, hops: int) -> tuple[str, str]:
    """Start token and expected token after *hops* edges."""
    if hops < 1 or hops >= length:
        raise ValueError("invalid hops for chain length")
    return f"chain{chain_id}_s0", f"chain{chain_id}_s{hops}"
