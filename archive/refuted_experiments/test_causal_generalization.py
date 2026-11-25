"""Unit tests for CRFN Causal Generalization and Counterfactual Rollouts."""

import numpy as np
import pytest

from neuron_v2.crfn.crfn_causal_engine import (
    CausalInversionTransmissionLine,
    CounterfactualTrapSimulator,
    DisjointCompositionalEvaluator,
)


def test_disjoint_compositional_split_is_strictly_disjoint() -> None:
    """Verify that train and held-out pairs are strictly disjoint."""
    rng = np.random.RandomState(42)
    evaluator = DisjointCompositionalEvaluator(n_entities=16, n_relations=8)
    train_pairs, held_out_pairs = evaluator.generate_disjoint_split(rng, train_fraction=0.70)

    assert len(train_pairs) == 89 or len(train_pairs) == 90
    assert len(held_out_pairs) == 39 or len(held_out_pairs) == 38
    set_train = set(train_pairs)
    set_held = set(held_out_pairs)

    intersection = set_train.intersection(set_held)
    assert len(intersection) == 0, f"Train and held-out must be disjoint, found {intersection}"


def test_counterfactual_trap_avoidance_vs_greedy() -> None:
    """Verify counterfactual latent rollout avoids trap while greedy selection falls into it."""
    sim = CounterfactualTrapSimulator(horizon=3)

    # 1. Greedy statistical selection falls into the trap (Action 0)
    res_greedy = sim.simulate_choice(use_latent_rollout=False)
    assert res_greedy["chosen_action"] == 0
    assert res_greedy["avoided_trap"] is False

    # 2. Active latent rollout prunes the collision and picks safe bypass (Action 1)
    res_rollout = sim.simulate_choice(use_latent_rollout=True)
    assert res_rollout["chosen_action"] == 1
    assert res_rollout["avoided_trap"] is True


def test_causal_inversion_impedance_adaptation() -> None:
    """Verify transmission line adapts locally under sudden environment impedance inversion."""
    line = CausalInversionTransmissionLine(z0=1.0, eta=0.15)

    # 1. Active adaptation
    res_active = line.run_inversion_adaptation(total_steps=100, inversion_step=40)
    assert res_active["recovery_latency"] < 15, f"Latency {res_active['recovery_latency']} must be < 15"
    assert res_active["final_transmission"] > 0.85

    # 2. Decoupled plasticity control fails to adapt
    res_decoupled = line.run_inversion_adaptation(total_steps=100, inversion_step=40, decouple_plasticity=True)
    assert res_decoupled["recovery_latency"] == 60  # max timeout
    assert res_decoupled["final_transmission"] < 0.60
