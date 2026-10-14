"""Kuramoto band relaxation (B-12 / spec Stage 4).

Local phase coupling across K dendritic bands before memory read. Adjacent-band
coupling only — avoids global sync collapse (spec §16).
"""
from __future__ import annotations

import numpy as np

from ..substrate.phasor import bundle, normalise, sim_batch
from .band_rpn import project_band
from .delta_memory import DeltaMemory
from .item_table import ItemTable
from .settling import _denoise_key, _prefix_matrix


def band_order_parameter(v: np.ndarray) -> float:
    """Circular mean phase of a band vector."""
    phases = np.angle(v)
    return float(np.angle(np.mean(np.exp(1j * phases))))


def kuramoto_relax(
    band_states: list[np.ndarray],
    k: int,
    *,
    coupling: float = 0.35,
    dt: float = 0.2,
) -> list[np.ndarray]:
    """Run *k* local Kuramoto micro-steps on K band phasors."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not band_states:
        return []
    states = [s.astype(np.complex64).copy() for s in band_states]
    if k == 1:
        return states

    n_bands = len(states)
    for _ in range(k - 1):
        thetas = [band_order_parameter(s) for s in states]
        next_states: list[np.ndarray] = []
        for i in range(n_bands):
            dtheta = 0.0
            for j in (i - 1, i + 1):
                if 0 <= j < n_bands:
                    dtheta += coupling * np.sin(thetas[j] - thetas[i])
            rot = np.exp(1j * (dtheta * dt)).astype(np.complex64)
            next_states.append(normalise(states[i] * rot))
        states = next_states
    return states


def kuramoto_predict(
    mem: DeltaMemory,
    ctx: np.ndarray,
    items: ItemTable,
    *,
    K: int = 3,
    k: int = 4,
    coupling: float = 0.35,
    dt: float = 0.2,
    beta: float = 0.55,
    ctx_prefix: str | None = "ctx_",
    tgt_prefix: str | None = "tgt_",
) -> np.ndarray:
    """Kuramoto band sync + ctx denoise + memory pull with tgt hint feedback."""
    if k < 1:
        raise ValueError("k must be >= 1")

    _, ctx_mat = _prefix_matrix(items, ctx_prefix) if ctx_prefix else ([], None)
    base = _denoise_key(ctx, ctx_mat) if ctx_mat is not None else ctx

    bands = [project_band(base, b, K) for b in range(K)]
    relaxed = kuramoto_relax(bands, k, coupling=coupling, dt=dt)
    key = bundle(*relaxed, normalise_output=True)

    if k == 1:
        return mem.predict(key)

    _, tgt_mat = _prefix_matrix(items, tgt_prefix) if tgt_prefix else ([], None)
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
