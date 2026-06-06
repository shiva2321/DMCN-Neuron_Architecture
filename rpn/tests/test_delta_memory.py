import numpy as np
from neuron_v2.substrate.phasor import random_hv, sim
from neuron_v2.core.delta_memory import DeltaMemory

D = 512
M = 128     # distinct associations
E = 8       # training epochs (repeated writes)


def _make_pairs(n, seed=0):
    rng = np.random.RandomState(seed)
    ctxs = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(n)]
    tgts = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(n)]
    return ctxs, tgts


def _train_and_score(mode):
    mem = DeltaMemory(D, mode=mode)
    ctxs, tgts = _make_pairs(M, seed=42)
    for _ in range(E):
        for c, t in zip(ctxs, tgts):
            mem.write(c, t)
    recalls = [sim(mem.recall(c), t) for c, t in zip(ctxs, tgts)]
    return float(np.mean(recalls)), mem.rms()


def test_delta_beats_additive_recall_under_repeats():
    delta_recall, _ = _train_and_score("delta")
    add_recall, _ = _train_and_score("additive")
    assert delta_recall > 0.5
    assert delta_recall > add_recall + 0.05


def test_additive_surface_saturates_delta_does_not():
    _, delta_rms = _train_and_score("delta")
    _, add_rms = _train_and_score("additive")
    assert add_rms > 2.0 * delta_rms


def test_delta_repeated_write_is_near_noop_after_learning():
    mem = DeltaMemory(D, mode="delta")
    c = random_hv(D, seed=11)
    t = random_hv(D, seed=12)
    mem.write(c, t)
    rms_after_first = mem.rms()
    for _ in range(20):
        mem.write(c, t)
    assert mem.rms() < 1.5 * rms_after_first
    assert sim(mem.recall(c), t) > 0.9
