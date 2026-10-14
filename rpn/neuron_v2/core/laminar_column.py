"""Laminar Resonant Cortical Column (RCC).

Implements the 6-layer canonical neocortical microcircuit:
- Layer 1: Apical tuft reception & top-down feedback routing.
- Layer 2/3: Superficial prediction error units (gamma band) & horizontal consensus voting.
- Layer 4: Thalamic granular input gate & what-where reference frame factorizer.
- Layer 5: Deep generative predictive units (alpha/beta band) & motor/action output.
- Layer 6: Corticothalamic precision weighting & gain control.

Unifies Tolman-Eichenbaum Machine (TEM) factorization with Communication-Through-Coherence (CTC).
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.delta_memory import DeltaMemory
from neuron_v2.core.rcpn import RCPNeuron
from neuron_v2.substrate.phasor import normalise, bind, unbind, bundle, permute, random_hv, sim, sim_batch


class WhatWhereReferenceFrame:
    """Tolman-Eichenbaum Machine (TEM) factorized reference frame generator."""

    def __init__(self, D: int, seed: int = 42):
        self.D = D
        self.role_where = random_hv(D, seed=seed + 101)
        self.role_what = random_hv(D, seed=seed + 202)

    def compose(self, where_hv: np.ndarray, what_hv: np.ndarray) -> np.ndarray:
        """Compose factorized composite context: bundle(where_bound, what_bound)."""
        where_bound = bind(permute(self.role_where, 1), where_hv)
        what_bound = bind(permute(self.role_what, 2), what_hv)
        return bundle(where_bound, what_bound)

    def extract_where(self, composite_hv: np.ndarray) -> np.ndarray:
        """Unbind and recover structural where component."""
        return unbind(composite_hv, permute(self.role_where, 1))

    def extract_what(self, composite_hv: np.ndarray) -> np.ndarray:
        """Unbind and recover sensory what component."""
        return unbind(composite_hv, permute(self.role_what, 2))


class LaminarCorticalColumn:
    """The 6-Layer Canonical Resonant Cortical Column (RCC)."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        n_shards: int = 8,
        lr: float = 0.20,
        seed: int = 42,
    ):
        self.D = D
        self.items = items
        self.n_shards = n_shards
        self.lr = lr
        self.frame = WhatWhereReferenceFrame(D, seed=seed)

        # Layer 5: Generative state units & fast-weight associative memories (sharded by 'where' metric reference frames)
        self.l5_units = [
            RCPNeuron(D=D, n_branches=2, n_channels=2, lr=lr, seed=seed + i * 37)
            for i in range(n_shards)
        ]
        self.l5_mems = [
            DeltaMemory(D=D, mode="delta", lr=1.0)
            for _ in range(n_shards)
        ]

        # Layer 2/3: Prediction error and lateral voting units
        self.l23_error_state = np.zeros(D, dtype=np.complex64)
        self.l23_consensus_weight = 0.25

        # Layer 6: Corticothalamic precision / gain modulator
        self.l6_precision = 1.0

        # Memory store mapping where_token -> shard index
        self._where_to_shard: dict[str, int] = {}

    def _get_shard(self, where_token: str) -> int:
        if where_token not in self._where_to_shard:
            self._where_to_shard[where_token] = len(self._where_to_shard) % self.n_shards
        return self._where_to_shard[where_token]

    def learn(
        self,
        where_token: str,
        what_token: str,
        object_token: str,
        top_down_context: Optional[str] = None,
    ):
        """Execute one complete laminar learning cycle (L4 -> L5 -> L2/3 -> L6)."""
        where_hv = self.items[where_token]
        what_hv = self.items[what_token]
        obj_hv = self.items[object_token]
        ctx_hv = self.items[top_down_context] if top_down_context else None

        # 1. Layer 4: Compose factorized reference frame
        where_bound = bind(permute(self.frame.role_where, 1), where_hv)
        what_bound = bind(permute(self.frame.role_what, 2), what_hv)
        composite_input = bundle(where_bound, what_bound)

        # 2. Layer 5: Route to specialized infragranular predictive unit & memory
        shard_idx = self._get_shard(where_token)
        unit = self.l5_units[shard_idx]
        mem = self.l5_mems[shard_idx]

        # Forward pass through L5 unit (integrating basal and apical feedback)
        inputs = [what_bound, where_hv]
        out = unit.step(inputs, top_down_context=ctx_hv)

        # Predict target object via fast-weight memory
        pred = mem.predict(what_bound)
        residual = obj_hv - pred
        surprise = float(np.linalg.norm(residual)) / np.sqrt(self.D)

        # 3. Layer 2/3: Compute prediction error & write fast-weights
        mem.write(what_bound, obj_hv)
        unit.learn_autonomous_xor(inputs, target_spike=True, spiked=out["spiked"])

        # 4. Layer 6: Update corticothalamic precision
        self.l6_precision = 1.0 / (1.0 + surprise)
        unit.reset()

    def read(
        self,
        where_token: str,
        what_token: str,
        top_down_context: Optional[str] = None,
        lateral_hypotheses: Optional[list[np.ndarray]] = None,
    ) -> tuple[Optional[str], float]:
        """Readout from column using canonical microcircuit and lateral CTC consensus."""
        where_hv = self.items[where_token]
        what_hv = self.items[what_token]
        ctx_hv = self.items[top_down_context] if top_down_context else None

        # Layer 4: Factorize
        what_bound = bind(permute(self.frame.role_what, 2), what_hv)

        # Layer 5: Retrieve from corresponding where-shard fast-weight memory
        shard_idx = self._get_shard(where_token)
        mem = self.l5_mems[shard_idx]
        pred_raw = mem.predict(what_bound)

        # Layer 2/3: Lateral consensus voting via phase coherence
        if lateral_hypotheses:
            # Constructive interference with coherent lateral inputs
            consensus_bundle = bundle(pred_raw, *lateral_hypotheses)
            pred_echo = normalise(pred_raw + self.l23_consensus_weight * consensus_bundle)
        else:
            pred_echo = normalise(pred_raw)

        # Layer 1: Top-down context modulation if present
        if ctx_hv is not None:
            pred_echo = normalise(pred_echo + 0.15 * ctx_hv)

        # Clean-up memory lookup using item table
        tokens, mat = self.items.matrix()
        similarities = sim_batch(mat, pred_echo)

        best_idx = int(np.argmax(similarities))
        best_prob = float(similarities[best_idx])

        # Confidence threshold
        if best_prob < 0.10:
            return None, best_prob
        return tokens[best_idx], best_prob
