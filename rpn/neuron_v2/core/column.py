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


def vote_read_ctx(
    rpns: list[RPN],
    D: int,
    items: ItemTable,
    ctx: np.ndarray,
    gate_threshold: float = 0.05,
    confidence_threshold: float = 0.1,
    min_resonance: float = 0.1,
) -> tuple[Optional[str], float]:
    """Query every neuron; return token with highest Born prob among gate-passing reads."""
    best_token: Optional[str] = None
    best_prob = 0.0
    for rpn in rpns:
        token, prob = _read_rpn_ctx(
            rpn,
            D,
            items,
            ctx,
            gate_threshold=gate_threshold,
            confidence_threshold=confidence_threshold,
            min_resonance=min_resonance,
        )
        if token is not None and prob > best_prob:
            best_token, best_prob = token, prob
    return best_token, best_prob


def _read_rpn_ctx(
    rpn: RPN,
    D: int,
    items: ItemTable,
    ctx: np.ndarray,
    gate_threshold: float = 0.05,
    confidence_threshold: float = 0.1,
    min_resonance: float = 0.1,
) -> tuple[Optional[str], float]:
    pred = rpn.mem.predict(ctx)
    confidence = float(np.linalg.norm(pred)) / np.sqrt(D)
    if confidence < confidence_threshold:
        return None, 0.0

    echo = normalise(pred)
    tokens, mat = items.matrix()
    res = sim_batch(mat, echo)
    out = rectified_born(
        res,
        gate_threshold=gate_threshold,
        min_resonance=min_resonance,
    )
    if not out.gate_passed:
        return None, out.best_prob
    return tokens[out.best_idx], out.best_prob


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

    def read(
        self,
        color: str,
        shape: str,
        **gate_kwargs,
    ) -> tuple[Optional[str], float]:
        """Read object for (color, shape).

        Integrator-first when it Born-passes (composite ctx is unique); else max-prob
        among passing specialists. See stage2b plan for why strict global vote fails.
        """
        integrator_out = _read_rpn_ctx(
            self.integrator,
            self.D,
            self.items,
            self._composite_ctx(color, shape),
            **gate_kwargs,
        )
        color_out = _read_rpn_ctx(
            self.color_rpn, self.D, self.items, self._color_ctx(color), **gate_kwargs
        )
        shape_out = _read_rpn_ctx(
            self.shape_rpn, self.D, self.items, self._shape_ctx(shape), **gate_kwargs
        )

        passed = [out for out in (integrator_out, color_out, shape_out) if out[0] is not None]
        if not passed:
            return None, 0.0

        # Integrator holds the unique composite binding; role-only specialists
        # collide when a color/shape maps to multiple objects.
        if integrator_out[0] is not None:
            return integrator_out
        return max(passed, key=lambda pair: pair[1])


class ColorShardedColumn:
    """One RPN per color — shape->object inside each shard (Path B capacity win).

    Single monolithic memory must store all color×shape pairs and suffers
    interference; each shard holds only |shapes| associations.

    Set ``vote_on_read=True`` to query every shard and take best Born prob
    (B-3); default routes by color on read.
    """

    def __init__(
        self,
        D: int,
        items: ItemTable,
        mode: str = "delta",
        *,
        vote_on_read: bool = False,
    ):
        self.D = D
        self.items = items
        self.mode = mode
        self.vote_on_read = vote_on_read
        self.role_shape = random_hv(D, seed=_role_seed("role_shape"))
        self._shards: dict[str, RPN] = {}

    def _shape_ctx(self, shape: str) -> np.ndarray:
        return bind(permute(self.role_shape, 2), self.items[shape])

    def _shard(self, color: str) -> RPN:
        if color not in self._shards:
            self._shards[color] = RPN(self.D, self.items, mode=self.mode)
        return self._shards[color]

    def learn(self, color: str, shape: str, object_token: str) -> None:
        target = self.items[object_token]
        self._shard(color).mem.write(self._shape_ctx(shape), target)

    def read(
        self,
        color: str,
        shape: str,
        **gate_kwargs,
    ) -> tuple[Optional[str], float]:
        ctx = self._shape_ctx(shape)
        if self.vote_on_read:
            if not self._shards:
                return None, 0.0
            return vote_read_ctx(
                list(self._shards.values()),
                self.D,
                self.items,
                ctx,
                **gate_kwargs,
            )
        if color not in self._shards:
            return None, 0.0
        return _read_rpn_ctx(
            self._shards[color],
            self.D,
            self.items,
            ctx,
            **gate_kwargs,
        )


class HashShardedCompositeColumn:
    """Composite ctx sharded by hash; read routes to one neuron or votes all (B-3)."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        n_shards: int,
        mode: str = "delta",
        *,
        vote_on_read: bool = False,
    ):
        if n_shards < 1:
            raise ValueError("n_shards must be >= 1")
        self.D = D
        self.items = items
        self.n_shards = n_shards
        self.vote_on_read = vote_on_read
        self.role_color = random_hv(D, seed=_role_seed("role_color"))
        self.role_shape = random_hv(D, seed=_role_seed("role_shape"))
        self.neurons = [RPN(D, items, mode=mode) for _ in range(n_shards)]

    def _color_ctx(self, color: str) -> np.ndarray:
        return bind(permute(self.role_color, 1), self.items[color])

    def _shape_ctx(self, shape: str) -> np.ndarray:
        return bind(permute(self.role_shape, 2), self.items[shape])

    def _composite_ctx(self, color: str, shape: str) -> np.ndarray:
        return bundle(self._color_ctx(color), self._shape_ctx(shape))

    def _route(self, color: str, shape: str) -> int:
        digest = hashlib.md5(f"composite-shard:{color}:{shape}".encode()).hexdigest()
        return int(digest[:8], 16) % self.n_shards

    def learn(self, color: str, shape: str, object_token: str) -> None:
        ctx = self._composite_ctx(color, shape)
        target = self.items[object_token]
        self.neurons[self._route(color, shape)].mem.write(ctx, target)

    def read(
        self,
        color: str,
        shape: str,
        **gate_kwargs,
    ) -> tuple[Optional[str], float]:
        ctx = self._composite_ctx(color, shape)
        if self.vote_on_read:
            return vote_read_ctx(
                self.neurons,
                self.D,
                self.items,
                ctx,
                **gate_kwargs,
            )
        return _read_rpn_ctx(
            self.neurons[self._route(color, shape)],
            self.D,
            self.items,
            ctx,
            **gate_kwargs,
        )
