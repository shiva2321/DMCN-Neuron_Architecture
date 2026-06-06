"""CompositionalColumn — 3-RPN mini column for role-bound compositional lookup."""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from ..substrate.phasor import bind, bundle, normalise, permute, random_hv, sim_batch
from .born_gate import rectified_born
from .item_table import ItemTable
from .rpn import RPN


def _role_seed(name: str) -> int:
    digest = hashlib.md5(f"compositional-role:{name}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


class CompositionalColumn:
    """3-RPN column: integrator + role-bound color/shape specialists."""

    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.D = D
        self.items = items
        self.integrator = RPN(D, items, mode=mode)
        self.color_rpn = RPN(D, items, mode=mode)
        self.shape_rpn = RPN(D, items, mode=mode)
        self.role_color = random_hv(D, seed=_role_seed("role_color"))
        self.role_shape = random_hv(D, seed=_role_seed("role_shape"))

    def _color_ctx(self, color: str) -> np.ndarray:
        return bind(permute(self.role_color, 1), self.items[color])

    def _shape_ctx(self, shape: str) -> np.ndarray:
        return bind(permute(self.role_shape, 2), self.items[shape])

    def _composite_ctx(self, color: str, shape: str) -> np.ndarray:
        return bundle(self._color_ctx(color), self._shape_ctx(shape))

    def learn(self, color: str, shape: str, object_token: str) -> None:
        target = self.items[object_token]
        self.integrator.mem.write(self._composite_ctx(color, shape), target)
        self.color_rpn.mem.write(self._color_ctx(color), target)
        self.shape_rpn.mem.write(self._shape_ctx(shape), target)

    def _read_ctx(
        self,
        rpn: RPN,
        ctx: np.ndarray,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        pred = rpn.mem.predict(ctx)
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

    def read(
        self,
        color: str,
        shape: str,
        **gate_kwargs,
    ) -> tuple[Optional[str], float]:
        integrator_out = self._read_ctx(
            self.integrator, self._composite_ctx(color, shape), **gate_kwargs
        )
        color_out = self._read_ctx(self.color_rpn, self._color_ctx(color), **gate_kwargs)
        shape_out = self._read_ctx(self.shape_rpn, self._shape_ctx(shape), **gate_kwargs)

        passed = [out for out in (integrator_out, color_out, shape_out) if out[0] is not None]
        if not passed:
            return None, 0.0

        # Integrator holds the unique composite binding; role-only specialists
        # collide when a color/shape maps to multiple objects.
        if integrator_out[0] is not None:
            return integrator_out
        return max(passed, key=lambda pair: pair[1])
