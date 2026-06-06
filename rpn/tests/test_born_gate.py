import numpy as np
from neuron_v2.core.born_gate import rectified_born


def test_rectify_eliminates_negative_and_picks_best():
    res = np.array([-0.8, 0.1, 0.9, 0.2], dtype=np.float32)
    out = rectified_born(res, gate_threshold=0.01)
    assert out.best_idx == 2
    assert out.gate_passed is True
    assert out.best_prob > 0.5


def test_no_signal_refuses():
    res = np.array([-0.5, -0.2, -0.9], dtype=np.float32)
    out = rectified_born(res, gate_threshold=0.01)
    assert out.best_idx == -1
    assert out.gate_passed is False


def test_tiny_positive_flake_refuses_with_min_resonance():
    res = np.array([-0.5, -0.2, 0.008], dtype=np.float32)
    out = rectified_born(res, gate_threshold=0.05, min_resonance=0.1)
    assert out.best_prob == 1.0
    assert out.gate_passed is False
