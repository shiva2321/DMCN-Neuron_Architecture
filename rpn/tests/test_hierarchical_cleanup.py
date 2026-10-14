"""Unit tests for Hierarchical Tree Clean-Up Memory."""
import numpy as np
import pytest

from neuron_v2.core.hierarchical_cleanup import HierarchicalTreeCleanUpMemory
from neuron_v2.substrate.phasor import normalise, random_hv


def test_hierarchical_cleanup_exact_and_noisy():
    D = 128
    tree = HierarchicalTreeCleanUpMemory(D=D, n_clusters=10, items_per_cluster=10, seed=42)

    # 1. Exact retrieval
    target = tree.leaves[3, 7]
    token, sim_val = tree.search(target)
    assert token == "concept_3_7"
    assert sim_val > 0.99

    # 2. Noisy query retrieval (20% noise)
    noise = random_hv(D, seed=999)
    noisy = normalise(target + 0.20 * noise)
    token_noisy, sim_noisy = tree.search(noisy)
    assert token_noisy == "concept_3_7"
    assert sim_noisy > 0.70
