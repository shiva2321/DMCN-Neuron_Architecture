"""Hierarchical Tree Clean-Up Memory for Sub-Linear Concept Retrieval.

Eliminates the O(V * D) linear scan bottleneck in Vector Symbolic Architectures
by organizing concept vocabularies into a multi-level hierarchical proximity tree:
- Level 1: Super-category cluster centroids (K1 prototypes).
- Level 2: Sub-category leaves (K2 prototypes per centroid).
Search complexity drops from O(V) = K1 * K2 to O(sqrt(V)) = K1 + K2.
For V = 10,000: 200 dot products instead of 10,000 (50x speedup).
For V = 1,000,000 (3 levels): 300 dot products instead of 1,000,000 (3,333x speedup).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

from neuron_v2.substrate.phasor import normalise, random_hv, sim_batch


class HierarchicalTreeCleanUpMemory:
    """Sub-linear hierarchical associative clean-up memory."""

    def __init__(
        self,
        D: int = 256,
        n_clusters: int = 100,
        items_per_cluster: int = 100,
        seed: int = 42,
    ):
        self.D = D
        self.n_clusters = n_clusters
        self.items_per_cluster = items_per_cluster
        self.total_capacity = n_clusters * items_per_cluster
        self.rng = np.random.RandomState(seed)

        # 1. Level 1: Centroids matrix of shape (n_clusters, D)
        self.centroids = np.array([
            random_hv(D, seed=seed + 1000 + i) for i in range(n_clusters)
        ], dtype=np.complex64)

        # 2. Level 2: Leaves nested per cluster
        # Shape: (n_clusters, items_per_cluster, D)
        self.leaves = np.zeros((n_clusters, items_per_cluster, D), dtype=np.complex64)
        self.tokens: list[list[str]] = [[] for _ in range(n_clusters)]
        self._token_to_pos: dict[str, tuple[int, int]] = {}

        # Pre-populate structured tree
        for c in range(n_clusters):
            c_vec = self.centroids[c]
            for j in range(items_per_cluster):
                token = f"concept_{c}_{j}"
                noise = random_hv(D, seed=seed + c * 1000 + j)
                leaf_vec = normalise(c_vec + 0.35 * noise)
                self.leaves[c, j] = leaf_vec
                self.tokens[c].append(token)
                self._token_to_pos[token] = (c, j)

    def search(self, query_hv: np.ndarray) -> tuple[str, float]:
        """Sub-linear 2-level hierarchical search."""
        # Step 1: Search centroids (n_clusters dot products)
        sims_c = sim_batch(self.centroids, query_hv)
        best_c = int(np.argmax(sims_c))

        # Step 2: Search leaves within best cluster (items_per_cluster dot products)
        sims_leaf = sim_batch(self.leaves[best_c], query_hv)
        best_item = int(np.argmax(sims_leaf))
        best_sim = float(sims_leaf[best_item])

        return self.tokens[best_c][best_item], best_sim

    def brute_force_search(self, query_hv: np.ndarray) -> tuple[str, float]:
        """Exhaustive linear scan baseline for verification and benchmarking."""
        flat_leaves = self.leaves.reshape(-1, self.D)
        flat_sims = sim_batch(flat_leaves, query_hv)
        best_idx = int(np.argmax(flat_sims))
        c_idx = best_idx // self.items_per_cluster
        j_idx = best_idx % self.items_per_cluster
        return self.tokens[c_idx][j_idx], float(flat_sims[best_idx])
