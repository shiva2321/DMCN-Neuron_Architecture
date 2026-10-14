"""Autonomous XOR Symmetry-Breaking Benchmark (Zero-Cheat).

Verifies that the Resonant Compartmental Predictive Neuron (RCPN) learns the non-linearly
separable XOR gate from scratch without pre-wired channel routing, while standard
Point-LIF and Perceptron baselines fail (locked at 50% chance limit).

Patterns:
  (0, 0) -> [NOT_A, NOT_B] -> Target: Silent (0)
  (1, 0) -> [    A, NOT_B] -> Target: Spike  (1)
  (0, 1) -> [NOT_A,     B] -> Target: Spike  (1)
  (1, 1) -> [    A,     B] -> Target: Silent (0)
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, asdict

from neuron_v2.core.rcpn import RCPNeuron
from neuron_v2.substrate.phasor import random_hv


class PointLIFNeuron:
    """Standard point-neuron baseline (single compartment, no dendritic branches)."""

    def __init__(self, n_channels: int = 4, lr: float = 0.05, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = rng.uniform(-0.1, 0.1, n_channels).astype(np.float32)
        self.bias = 0.0
        self.lr = lr

    def predict(self, active_channels: list[int]) -> bool:
        net = sum(self.weights[c] for c in active_channels) + self.bias
        return bool(net > 0.0)

    def learn(self, active_channels: list[int], target: bool):
        pred = self.predict(active_channels)
        err = (1.0 if target else 0.0) - (1.0 if pred else 0.0)
        for c in active_channels:
            self.weights[c] += self.lr * err
        self.bias += self.lr * err * 0.5


class LinearPerceptron:
    """Standard linear classification perceptron."""

    def __init__(self, n_channels: int = 4, lr: float = 0.05, seed: int = 42):
        rng = np.random.RandomState(seed + 1)
        self.weights = rng.uniform(-0.1, 0.1, n_channels).astype(np.float32)
        self.bias = 0.0
        self.lr = lr

    def predict(self, active_channels: list[int]) -> bool:
        net = sum(self.weights[c] for c in active_channels) + self.bias
        return bool(net > 0.0)

    def learn(self, active_channels: list[int], target: bool):
        pred = self.predict(active_channels)
        err = (1.0 if target else 0.0) - (1.0 if pred else 0.0)
        for c in active_channels:
            self.weights[c] += self.lr * err
        self.bias += self.lr * err


class TwoLayerMLP:
    """2-Layer Multi-Layer Perceptron (4 inputs -> 4 hidden -> 1 output)."""

    def __init__(self, hidden: int = 4, lr: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed + 2)
        self.W1 = rng.uniform(-0.5, 0.5, (4, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.uniform(-0.5, 0.5, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)[0]))  # Sigmoid
        return h, float(out)

    def predict(self, active_channels: list[int]) -> bool:
        x = np.zeros(4, dtype=np.float32)
        for c in active_channels:
            x[c] = 1.0
        _, prob = self.forward(x)
        return bool(prob >= 0.5)

    def learn(self, active_channels: list[int], target: bool):
        x = np.zeros(4, dtype=np.float32)
        for c in active_channels:
            x[c] = 1.0
        y = 1.0 if target else 0.0
        h, prob = self.forward(x)
        err = y - prob
        d_out = err * prob * (1.0 - prob)
        d_W2 = np.outer(h, d_out)
        d_b2 = d_out
        d_h = (d_out * self.W2.T)[0]
        d_h[h <= 0] = 0.0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h
        self.W2 += self.lr * d_W2
        self.b2 += self.lr * d_b2
        self.W1 += self.lr * d_W1
        self.b1 += self.lr * d_b1


def get_xor_patterns(D: int, seed: int) -> tuple[list[tuple[list[int], bool]], dict[int, np.ndarray]]:
    """Return truth table and orthogonal channel hypervectors."""
    # Channel 0: A, 1: NOT_A, 2: B, 3: NOT_B
    channels = {i: random_hv(D, seed=seed * 10 + i) for i in range(4)}
    dataset = [
        ([1, 3], False),  # NOT_A, NOT_B -> 0
        ([0, 3], True),   # A, NOT_B     -> 1
        ([1, 2], True),   # NOT_A, B     -> 1
        ([0, 2], False),  # A, B         -> 0
    ]
    return dataset, channels


def evaluate_rcpn_xor(D: int = 128, n_epochs: int = 60, seed: int = 42) -> float:
    """Train a single RCPN with 2 basal branches on XOR from random weights."""
    neuron = RCPNeuron(D=D, n_branches=2, n_channels=4, lr=0.18, seed=seed)
    dataset, channels = get_xor_patterns(D, seed)
    rng = np.random.RandomState(seed)

    for epoch in range(n_epochs):
        # Shuffle patterns
        indices = rng.permutation(len(dataset))
        for idx in indices:
            active_ch, target = dataset[idx]
            # Construct input list (active channels carry their HV, inactive are zeros)
            inputs = [channels[c] if c in active_ch else np.zeros(D, dtype=np.complex64) for c in range(4)]

            # Run 3 forward steps to allow dendritic summation
            spiked = False
            for _ in range(3):
                out = neuron.step(inputs)
                if out["spiked"]:
                    spiked = True

            # Autonomous competitive plasticity update
            neuron.learn_autonomous_xor(inputs, target_spike=target, spiked=spiked)
            neuron.reset()

    # Final evaluation pass
    correct = 0
    for active_ch, target in dataset:
        inputs = [channels[c] if c in active_ch else np.zeros(D, dtype=np.complex64) for c in range(4)]
        spiked = False
        for _ in range(3):
            out = neuron.step(inputs)
            if out["spiked"]:
                spiked = True
        neuron.reset()
        if spiked == target:
            correct += 1
    return correct / len(dataset)


def run_benchmark(n_trials: int = 20, D: int = 128) -> dict:
    """Execute multi-seed evaluation of RCPN vs PointLIF, Perceptron, and MLP."""
    print(f"Running Autonomous XOR Benchmark across {n_trials} independent seeds (D={D})...")

    rcpn_scores = []
    lif_scores = []
    perceptron_scores = []
    mlp_scores = []

    for trial in range(n_trials):
        seed = trial + 100
        dataset, _ = get_xor_patterns(D, seed)

        # 1. RCPN
        acc_rcpn = evaluate_rcpn_xor(D=D, n_epochs=50, seed=seed)
        rcpn_scores.append(acc_rcpn)

        # 2. Point-LIF
        lif = PointLIFNeuron(seed=seed)
        for _ in range(50):
            for chs, tgt in dataset:
                lif.learn(chs, tgt)
        correct_lif = sum(1 for chs, tgt in dataset if lif.predict(chs) == tgt)
        lif_scores.append(correct_lif / len(dataset))

        # 3. Perceptron
        perc = LinearPerceptron(seed=seed)
        for _ in range(50):
            for chs, tgt in dataset:
                perc.learn(chs, tgt)
        correct_perc = sum(1 for chs, tgt in dataset if perc.predict(chs) == tgt)
        perceptron_scores.append(correct_perc / len(dataset))

        # 4. 2-Layer MLP
        mlp = TwoLayerMLP(seed=seed)
        for _ in range(150):
            for chs, tgt in dataset:
                mlp.learn(chs, tgt)
        correct_mlp = sum(1 for chs, tgt in dataset if mlp.predict(chs) == tgt)
        mlp_scores.append(correct_mlp / len(dataset))

    results = {
        "RCPN": {
            "mean": float(np.mean(rcpn_scores)),
            "std": float(np.std(rcpn_scores)),
            "scores": rcpn_scores,
        },
        "TwoLayerMLP": {
            "mean": float(np.mean(mlp_scores)),
            "std": float(np.std(mlp_scores)),
            "scores": mlp_scores,
        },
        "PointLIF": {
            "mean": float(np.mean(lif_scores)),
            "std": float(np.std(lif_scores)),
            "scores": lif_scores,
        },
        "LinearPerceptron": {
            "mean": float(np.mean(perceptron_scores)),
            "std": float(np.std(perceptron_scores)),
            "scores": perceptron_scores,
        },
    }

    # Print summary scorecard
    print("\n" + "=" * 65)
    print("       AUTONOMOUS XOR BENCHMARK SCORECARD (N=20 Trials)")
    print("=" * 65)
    print(f"{'Model':<20} {'Mean Accuracy':<18} {'Std Dev':<12} {'Verdict'}")
    print("-" * 65)
    for model_name, res in results.items():
        verdict = "PASS" if res["mean"] >= 0.90 or (model_name in ["PointLIF", "LinearPerceptron"] and res["mean"] <= 0.55) else "FAIL"
        print(f"{model_name:<20} {res['mean'] * 100:>6.1f}%          ±{res['std'] * 100:>4.1f}%       {verdict}")
    print("=" * 65)

    # Serialize to disk (ZIMO discipline)
    with open("rcpn_xor_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results successfully saved to rcpn_xor_results.json")
    return results


if __name__ == "__main__":
    run_benchmark(n_trials=20)
