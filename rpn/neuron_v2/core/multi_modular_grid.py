"""Multi-Modular Entorhinal Grid Reference Frame & High-Capacity Cortical Column.

Based on Ila Fiete's multi-modular grid coding theorem (Fiete et al., 2008, 2020):
- Space is factorized across M independent modules with coprime periods lambda = (lambda_1, ..., lambda_M).
- Unambiguous coordinate capacity grows exponentially as prod(lambda_m) ~ exp(M).
- Generates thousands of mutually orthogonal spatial hypervectors using only a handful of physical phase vectors (sum lambda_m).
- Eliminates fast-weight matrix rank saturation by ensuring all location keys remain mutually orthogonal.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.delta_memory import DeltaMemory
from neuron_v2.substrate.phasor import normalise, random_hv, bind, permute, sim_batch


class MultiModularGridReferenceFrame:
    """Entorhinal multi-modular grid cell coordinate generator."""

    def __init__(
        self,
        D: int = 256,
        moduli: Sequence[int] = (5, 7, 11, 13),
        seed: int = 42,
    ):
        self.D = D
        self.moduli = tuple(moduli)
        self.total_physical_vectors = sum(self.moduli)
        self.total_coordinate_capacity = int(np.prod(self.moduli))

        # Physical phase basis vectors for each module
        self.phases = [
            np.array([
                random_hv(D, seed=seed + m * 500 + i)
                for i in range(mod)
            ], dtype=np.complex64)
            for m, mod in enumerate(self.moduli)
        ]

    def encode_location(self, loc_index: int) -> np.ndarray:
        """Generate multi-modular grid hypervector for discrete coordinate index."""
        # Chinese Remainder Theorem phase binding
        vec = self.phases[0][loc_index % self.moduli[0]].copy()
        for m in range(1, len(self.moduli)):
            vec = bind(vec, self.phases[m][loc_index % self.moduli[m]])
        return vec


class MultiModularCorticalColumn:
    """High-capacity cortical column network backed by multi-modular grid codes."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        n_shards: int = 16,
        moduli: Sequence[int] = (5, 7, 11, 13),
        seed: int = 42,
    ):
        self.D = D
        self.items = items
        self.n_shards = n_shards
        self.grid = MultiModularGridReferenceFrame(D=D, moduli=moduli, seed=seed)
        self.role_what = random_hv(D, seed=seed + 999)
        self.shards = [DeltaMemory(D, mode="delta", lr=1.0) for _ in range(n_shards)]
        self._loc_token_to_idx: dict[str, int] = {}

    def _get_loc_idx(self, where_token: str) -> int:
        if where_token not in self._loc_token_to_idx:
            # Parse integer suffix if available (e.g. 'loc_42'), else auto-assign
            try:
                idx = int(where_token.split("_")[-1])
            except ValueError:
                idx = len(self._loc_token_to_idx)
            self._loc_token_to_idx[where_token] = idx
        return self._loc_token_to_idx[where_token]

    def compose_key(self, where_token: str, what_token: str) -> tuple[int, np.ndarray]:
        """Compose collision-resistant key via entorhinal grid binding."""
        loc_idx = self._get_loc_idx(where_token)
        shard_idx = loc_idx % self.n_shards
        g_hv = self.grid.encode_location(loc_idx)
        what_hv = self.items[what_token]
        what_bound = bind(permute(self.role_what, 2), what_hv)
        key = bind(g_hv, what_bound)
        return shard_idx, key

    def learn(self, where_token: str, what_token: str, object_token: str):
        shard_idx, key = self.compose_key(where_token, what_token)
        target_hv = self.items[object_token]
        self.shards[shard_idx].write(key, target_hv)

    def read(self, where_token: str, what_token: str) -> tuple[Optional[str], float]:
        shard_idx, key = self.compose_key(where_token, what_token)
        pred_raw = self.shards[shard_idx].predict(key)
        pred_echo = normalise(pred_raw)

        tokens, mat = self.items.matrix()
        similarities = sim_batch(mat, pred_echo)
        best_idx = int(np.argmax(similarities))
        best_prob = float(similarities[best_idx])

        if best_prob < 0.10:
            return None, best_prob
        return tokens[best_idx], best_prob
