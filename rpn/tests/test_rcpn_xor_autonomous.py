"""Tests for Autonomous XOR Symmetry-Breaking in RCPN."""
import pytest
from benchmarks.rcpn_autonomous_xor import evaluate_rcpn_xor, PointLIFNeuron, LinearPerceptron


def test_rcpn_autonomous_xor_convergence():
    """Verify that RCPN converges to >= 90% accuracy without pre-wired channel routing."""
    # Test on multiple distinct seeds
    scores = [evaluate_rcpn_xor(D=128, n_epochs=50, seed=s) for s in [42, 101, 202]]
    mean_acc = sum(scores) / len(scores)
    assert mean_acc >= 0.90, f"Autonomous XOR failed to achieve >= 90%: {scores}"


def test_point_lif_fails_xor():
    """Verify that Point-LIF cannot solve non-linearly separable XOR (mathematical baseline)."""
    dataset = [([1, 3], False), ([0, 3], True), ([1, 2], True), ([0, 2], False)]
    lif = PointLIFNeuron(seed=42)
    for _ in range(50):
        for chs, tgt in dataset:
            lif.learn(chs, tgt)
    acc = sum(1 for chs, tgt in dataset if lif.predict(chs) == tgt) / len(dataset)
    assert acc <= 0.75, "Point-LIF unexpectedly solved non-separable gate without compartments"
