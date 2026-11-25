"""Unit tests for Continuous Symbolic Reasoning Column & Resonant Wavefield Engine."""

import numpy as np
import pytest

from neuron_v2.capn.continuous_symbolic_column import (
    ContinuousAttractorCleanUp,
    ContinuousSymbolicReasoningColumn,
    ResonantWavefieldArithmeticEngine,
    MonolithicFlatMemoryAnalogyBaseline,
    encode_phasor_wavefield,
    generate_log_cochlear_frequencies,
    normalise_phasor,
    phasor_bind,
    phasor_similarity,
    phasor_unbind,
    wave_interference_add,
    wave_interference_invert,
)


def test_phasor_algebra_properties() -> None:
    """Verify unitary phasor binding and unbinding algebra."""
    rng = np.random.RandomState(42)
    dim = 256
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))

    # 1. Commutativity of binding
    ab = phasor_bind(a, b)
    ba = phasor_bind(b, a)
    np.testing.assert_allclose(ab, ba, atol=1e-10)

    # 2. Exact unbinding
    recovered_a = phasor_unbind(ab, b)
    sim = phasor_similarity(recovered_a, a)
    assert sim > 0.9999, f"Exact phasor unbinding similarity {sim} must be > 0.9999"


def test_resonant_phasor_wavefield_properties() -> None:
    """Verify that complex Hadamard product implements exact continuous wave addition and inversion."""
    frequencies = generate_log_cochlear_frequencies(128, omega_min=0.01, omega_max=0.80)
    val_a = 14
    val_b = 27
    expected_sum = 41

    p_a, _ = encode_phasor_wavefield(val_a, frequencies)
    p_b, _ = encode_phasor_wavefield(val_b, frequencies)
    p_sum_expected, _ = encode_phasor_wavefield(expected_sum, frequencies)

    # 1. Direct wave interference addition
    p_sum_actual, _ = wave_interference_add(p_a, p_b)
    np.testing.assert_allclose(p_sum_actual, p_sum_expected, atol=1e-10)

    # 2. Exact algebraic inversion (subtraction)
    p_diff_actual, _ = wave_interference_invert(p_sum_actual, p_a)
    np.testing.assert_allclose(p_diff_actual, p_b, atol=1e-10)


def test_continuous_attractor_lyapunov_relaxation() -> None:
    """Verify that Layer 5 CANN dynamics minimizes energy and restores corrupted prototypes."""
    rng = np.random.RandomState(42)
    dim = 256
    n_prototypes = 10
    phases = [rng.uniform(-np.pi, np.pi, dim) for _ in range(n_prototypes)]
    prototypes = [np.exp(1j * p) for p in phases]

    attractor = ContinuousAttractorCleanUp(dim=dim, beta=40.0)
    attractor.set_prototypes(np.array(prototypes))

    target_idx = 3
    true_target = prototypes[target_idx]
    # Add heavy phase noise (sigma = 0.35 rad)
    noisy_input = true_target * np.exp(1j * 0.35 * rng.randn(dim))

    settled, energy = attractor.settle(noisy_input, t_steps=10)
    best_idx, sim = attractor.decode_nearest(settled)

    assert best_idx == target_idx
    assert sim > 0.95
    assert energy < 0.0, f"Lyapunov energy {energy} must be negative at attractor minimum"


def test_relational_analogy_and_null_controls() -> None:
    """Verify relational analogy solving under noise, and verify null controls fail."""
    rng = np.random.RandomState(42)
    dim = 512
    n_entities = 16

    entity_phases = [rng.uniform(-np.pi, np.pi, dim) for _ in range(n_entities)]
    entities = [np.exp(1j * p) for p in entity_phases]

    rel_phase = rng.uniform(-np.pi, np.pi, dim)
    rel_op = np.exp(1j * rel_phase)

    # Define analogy pairs: B = A (*) R, D = C (*) R
    idx_a = 0
    idx_c = 2
    vec_a = entities[idx_a]
    vec_c = entities[idx_c]

    vec_b = phasor_bind(vec_a, rel_op)
    vec_d = phasor_bind(vec_c, rel_op)
    idx_d = 5
    entities[1] = vec_b
    entities[idx_d] = vec_d  # Target entity

    col = ContinuousSymbolicReasoningColumn(dim=dim)
    col.set_knowledge_base(np.array(entities), np.array([rel_op]))

    # Test under sensory noise
    noise_sigma = 0.25
    noisy_a = vec_a * np.exp(1j * noise_sigma * rng.randn(dim))
    noisy_b = vec_b * np.exp(1j * noise_sigma * rng.randn(dim))
    noisy_c = vec_c * np.exp(1j * noise_sigma * rng.randn(dim))

    # 1. Active network: should correctly infer D
    res_active = col.solve_relational_analogy(noisy_a, noisy_b, noisy_c)
    assert res_active["predicted_idx"] == idx_d
    assert res_active["similarity"] > 0.85

    # 2. Decoupled control: L2/3 severed -> similarity 0.0
    res_decoupled = col.solve_relational_analogy(noisy_a, noisy_b, noisy_c, decouple_horizontal=True)
    assert res_decoupled["similarity"] == 0.0

    # 3. Blind control: inputs zeroed -> similarity 0.0
    res_blind = col.solve_relational_analogy(noisy_a, noisy_b, noisy_c, blind_input=True)
    assert res_blind["similarity"] == 0.0


def test_transitive_deduction_multi_hop() -> None:
    """Verify multi-hop transitive relation deduction: A -> B -> C."""
    rng = np.random.RandomState(42)
    dim = 512
    n_entities = 8

    entity_phases = [rng.uniform(-np.pi, np.pi, dim) for _ in range(n_entities)]
    entities = [np.exp(1j * p) for p in entity_phases]

    rel_ab = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    rel_bc = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))

    rel_ac = phasor_bind(rel_ab, rel_bc)
    target_c = phasor_bind(entities[0], rel_ac)
    idx_c = 4
    entities[idx_c] = target_c

    col = ContinuousSymbolicReasoningColumn(dim=dim)
    col.set_knowledge_base(np.array(entities), np.array([rel_ab, rel_bc, rel_ac]))

    res = col.solve_transitive_deduction(entities[0], rel_ab, rel_bc)
    assert res["predicted_idx"] == idx_c
    assert res["similarity"] > 0.90


def test_wavefield_reversible_arithmetic() -> None:
    """Verify continuous wave arithmetic and algebraic inversion across held-out values."""
    rng = np.random.RandomState(42)
    engine = ResonantWavefieldArithmeticEngine(n_frequencies=256, max_val=50)

    # 1. Forward continuous addition under noise
    a = 15
    b = 23
    expected_sum = 38
    res_add = engine.add(a, b, noise_sigma=0.15, rng=rng)
    assert res_add["predicted_val"] == expected_sum
    assert res_add["similarity"] > 0.85

    # 2. Algebraic inversion under noise
    res_inv = engine.invert(expected_sum, a, noise_sigma=0.15, rng=rng)
    assert res_inv["predicted_val"] == b
    assert res_inv["similarity"] > 0.85

    # 3. Decoupled control fails
    res_decoupled = engine.add(a, b, decouple_wave=True)
    assert res_decoupled["similarity"] == 0.0

    # 4. Blind control fails
    res_blind = engine.add(a, b, blind_input=True)
    assert res_blind["similarity"] == 0.0
