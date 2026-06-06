"""Rectified Born Rule emit gate (anti-hallucination).

raw_i = max(0, r_i)^2 ; P_i = raw_i / sum(raw) ; emit iff best_P >= threshold.
Negative resonances -> zero probability (eliminated, not flipped). If no positive
mass exists, the gate refuses (best_idx = -1).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BornResult:
    probs: np.ndarray
    best_idx: int
    best_prob: float
    gate_passed: bool


def rectified_born(
    resonances: np.ndarray,
    gate_threshold: float = 0.01,
    min_resonance: float = 0.0,
) -> BornResult:
    sims = np.asarray(resonances, dtype=np.float32)
    raw = np.maximum(sims, 0.0) ** 2
    total = float(np.sum(raw))
    if total <= 1e-14:
        n = len(sims)
        return BornResult(np.zeros(n, dtype=np.float32), -1, 0.0, False)
    probs = (raw / total).astype(np.float32)
    best_idx = int(np.argmax(raw))
    best_prob = float(probs[best_idx])
    best_sim = float(sims[best_idx])
    gate_passed = best_prob >= gate_threshold and best_sim >= min_resonance
    return BornResult(probs, best_idx, best_prob, gate_passed)
