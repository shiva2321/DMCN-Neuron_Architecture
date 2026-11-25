"""Tests for Dynamic Attentional Gating and Task Routing Engine (CAPN M2)."""
import numpy as np
import pytest

from neuron_v2.capn.attentive_routing import AttentiveRoutingEngine


def test_phase_entrainment_under_context():
    """Verify that disinhibited pathway phase-locks while unattended pathway drifts."""
    rng = np.random.RandomState(42)
    engine = AttentiveRoutingEngine(dim=32, rng=rng)

    # Context focused on Stream 1
    c_task = np.array([1.0, 0.0])
    x1 = np.ones(32)
    x2 = np.ones(32)

    # Step for 100 ms (200 steps)
    for _ in range(200):
        out = engine.step(x1, x2, c_task, dt=0.5)

    # Stream 1 should have high transmission; Stream 2 should have near zero
    assert out["T1"] > 0.90
    assert out["T2"] < 0.05


def test_dynamic_task_switching():
    """Verify that switching context shifts routing smoothly in < 20 ms."""
    rng = np.random.RandomState(123)
    engine = AttentiveRoutingEngine(dim=32, rng=rng)

    x1 = np.ones(32)
    x2 = np.ones(32)

    # Run on Stream 1 for 100 ms
    for _ in range(200):
        engine.step(x1, x2, np.array([1.0, 0.0]), dt=0.5)

    # Switch to Stream 2 and measure switching time
    switch_steps = 0
    for step in range(100):  # 50 ms
        out = engine.step(x1, x2, np.array([0.0, 1.0]), dt=0.5)
        if out["T2"] > out["T1"]:
            switch_steps = step
            break

    # Switch must occur within 30 steps (15 ms)
    assert switch_steps > 0
    assert switch_steps * 0.5 <= 15.0


def test_decoupled_phase_control():
    """Verify that decoupling phase reduces transmission coherence."""
    rng = np.random.RandomState(456)
    engine = AttentiveRoutingEngine(dim=32, rng=rng)

    c_task = np.array([1.0, 0.0])
    x1 = np.ones(32)
    x2 = np.ones(32)

    # With decoupled phase
    t1_samples = []
    for _ in range(200):
        out = engine.step(x1, x2, c_task, dt=0.5, decouple_phase=True)
        t1_samples.append(out["T1"])

    # Without phase entrainment, transmission fluctuates around mean ~ 0.5
    assert np.mean(t1_samples) < 0.85


def test_apical_shunt_ablation_causes_crosstalk():
    """Verify that ablating apical shunting allows unwanted cross-talk."""
    rng = np.random.RandomState(789)
    engine = AttentiveRoutingEngine(dim=32, rng=rng)

    c_task = np.array([1.0, 0.0])
    x1 = np.ones(32)
    x2 = np.ones(32)

    for _ in range(200):
        out = engine.step(x1, x2, c_task, dt=0.5, ablate_shunting=True)

    # Without shunting, stream 2 transmission is not suppressed to zero
    assert out["T2"] > 0.01
