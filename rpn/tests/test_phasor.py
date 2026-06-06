import numpy as np
from neuron_v2.substrate.phasor import (
    random_hv, normalise, bind, unbind, bundle, sim, sim_batch, permute,
)

D = 4096


def test_g1_bind_is_unit_magnitude():
    a = random_hv(D, seed=1)
    b = random_hv(D, seed=2)
    ab = bind(a, b)
    assert np.allclose(np.abs(ab), 1.0, atol=1e-5)


def test_g2_unbind_recovers_b_exactly():
    a = random_hv(D, seed=1)
    b = random_hv(D, seed=2)
    rec = unbind(bind(a, b), a)
    assert sim(rec, b) > 0.999


def test_g3_self_similarity_is_one():
    v = random_hv(D, seed=3)
    assert abs(sim(v, v) - 1.0) < 1e-4


def test_g4_random_pairs_near_orthogonal():
    a = random_hv(D, seed=4)
    b = random_hv(D, seed=5)
    assert abs(sim(a, b)) < 0.1


def test_bundle_is_similar_to_each_member():
    a = random_hv(D, seed=6)
    b = random_hv(D, seed=7)
    c = bundle(a, b)
    assert sim(c, a) > 0.3 and sim(c, b) > 0.3


def test_sim_batch_matches_sim():
    M = np.stack([random_hv(D, seed=s) for s in range(8)])
    q = random_hv(D, seed=3)
    batch = sim_batch(M, q)
    for i in range(8):
        assert abs(batch[i] - sim(M[i], q)) < 1e-4


def test_permute_unpermute_recovers_vector():
    v = random_hv(D, seed=10)
    for k in (1, 7, 42, D - 3):
        assert sim(permute(permute(v, k), -k), v) > 0.9999


def test_permute_role_binding_unbind_recovers_filler():
    a = random_hv(D, seed=11)
    b = random_hv(D, seed=12)
    k = 17
    role = permute(a, k)
    rec = unbind(bind(role, b), role)
    assert sim(rec, b) > 0.999


def test_permute_distributes_over_bind():
    a = random_hv(D, seed=13)
    b = random_hv(D, seed=14)
    k = 23
    assert sim(permute(bind(a, b), k), bind(permute(a, k), permute(b, k))) > 0.9999


def test_normalise_unit_magnitude_and_zero_safe():
    v = (np.random.RandomState(9).randn(D) + 1j * np.random.RandomState(8).randn(D)).astype(np.complex64)
    n = normalise(v)
    assert np.allclose(np.abs(n), 1.0, atol=1e-5)
    z = np.zeros(D, dtype=np.complex64)
    nz = normalise(z)
    assert np.all(np.isfinite(nz.view(np.float32))) and not np.any(np.abs(nz) > 0)
