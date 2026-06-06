"""Bounded settling dynamics for predictive read (Stage 3 / B-5).

k micro-steps: context denoise (optional) + memory pull + target codebook snap.
"""
from __future__ import annotations

import numpy as np

from ..substrate.phasor import normalise, sim_batch
from .delta_memory import DeltaMemory
from .item_table import ItemTable


def _prefix_matrix(items: ItemTable, prefix: str) -> tuple[list[str], np.ndarray | None]:
    tokens = [t for t in items._t if t.startswith(prefix)]
    if not tokens:
        return [], None
    mat = np.stack([items._t[t] for t in tokens]).astype(np.complex64)
    return tokens, mat


def _denoise_key(ctx: np.ndarray, ctx_mat: np.ndarray) -> np.ndarray:
    obs = normalise(ctx)
    idx = int(np.argmax(sim_batch(ctx_mat, obs)))
    return ctx_mat[idx]


def settle_predict(
    mem: DeltaMemory,
    ctx: np.ndarray,
    items: ItemTable,
    k: int,
    *,
    beta: float = 0.55,
    ctx_prefix: str | None = None,
    tgt_prefix: str | None = None,
) -> np.ndarray:
    """Run *k* settling steps; return raw (unnormalised) value estimate."""
    if k < 1:
        raise ValueError("k must be >= 1")

    if k == 1:
        return mem.predict(ctx)

    _, ctx_mat = _prefix_matrix(items, ctx_prefix) if ctx_prefix else ([], None)
    _, tgt_mat = _prefix_matrix(items, tgt_prefix) if tgt_prefix else ([], None)

    key = _denoise_key(ctx, ctx_mat) if ctx_mat is not None else ctx
    state = mem.predict(key)

    if tgt_mat is None:
        tokens, mat = items.matrix()
        if not tokens:
            return state
        tgt_mat = mat

    steps = k - 1
    for step in range(steps):
        mag = float(np.linalg.norm(state))
        if mag < 1e-8:
            break
        echo = normalise(state)
        res = sim_batch(tgt_mat, echo)
        hint = tgt_mat[int(np.argmax(res))]
        anneal = 1.0 - 0.5 * (step / max(steps, 1))
        b = beta * anneal
        fresh = mem.predict(key)
        state = ((1.0 - b) * fresh + b * hint * mag).astype(np.complex64)
    return state
