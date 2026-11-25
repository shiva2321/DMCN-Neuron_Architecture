"""Tests for Continuous Attractor Memory Unit with Short-Term Plasticity (CAPN M1)."""
import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest

from neuron_v2.capn.attractor_engine import ContinuousAttractorMemoryUnit


def test_continuous_attractor_energy_relaxation():
    """Verify that a noisy pattern settles into the correct canonical attractor basin."""
    rng = np.random.RandomState(42)
    dim = 64
    k_patterns = 4
    patterns = rng.choice([-1.0, 1.0], size=(k_patterns, dim))

    unit = ContinuousAttractorMemoryUnit(dim=dim, g_rec=3.5)
    unit.store_attractor_patterns(patterns)

    # Cue pattern 0 for 50 ms (100 steps with dt=0.5)
    for _ in range(100):
        unit.step(I_ext=2.0 * patterns[0], dt=0.5)

    # Allow to relax without external input for 100 ms (200 steps)
    for _ in range(200):
        unit.step(I_ext=np.zeros(dim), dt=0.5)

    best_k, best_sim, margin_db = unit.decode_pattern(patterns)
    assert best_k == 0
    assert best_sim > 0.50
    assert margin_db > 10.0


def test_dual_store_stp_dynamics():
    """Verify that short-term facilitation u increases with activity and recovers."""
    dim = 32
    unit = ContinuousAttractorMemoryUnit(dim=dim, u0=0.20, tau_f=500.0, tau_d=50.0)

    # Initial state
    assert np.allclose(unit.u, 0.20)
    assert np.allclose(unit.x_res, 1.0)

    # Inject strong continuous drive
    for _ in range(50):
        unit.step(I_ext=2.0 * np.ones(dim), dt=0.5)

    # Peak facilitation
    u_peak = unit.u.copy()
    assert np.all(u_peak > 0.40)
    assert np.all(unit.x_res < 0.95)

    # Remove drive and allow recovery (2000 steps = 1000 ms)
    for _ in range(2000):
        unit.step(I_ext=np.zeros(dim), dt=0.5)

    # Fast variables recover toward baseline (decaying below peak)
    assert np.all(unit.u < 0.32)
    assert np.all(unit.u < u_peak)
    assert np.all(unit.x_res > 0.90)


def test_linear_control_fails_saturation():
    """Verify that linear activation without dendritic saturation causes instability and poor SNR."""
    from benchmarks.capn_m1_attractor_benchmark import evaluate_attractor_seed
    res = evaluate_attractor_seed(seed=994, dim=64, k_patterns=4, g_rec=3.5, noise_sigma=0.40)
    # Intact CAPU must achieve 100% and high SNR
    assert res["acc_capu"] == 100.0
    # Linear control has significantly worse SNR and/or retrieval errors
    assert res["snr_capu"] > res["snr_linear"] + 5.0


def test_stp_ablation_degrades_snr():
    """Verify that ablating facilitation degrades contrast margin under noise."""
    from benchmarks.capn_m1_attractor_benchmark import evaluate_attractor_seed
    res = evaluate_attractor_seed(seed=994, dim=128, k_patterns=8, g_rec=3.5, noise_sigma=0.40)
    assert res["acc_capu"] >= 95.0
    assert res["snr_capu"] > res["snr_unfac"]
    assert res["snr_capu"] > res["snr_linear"]
