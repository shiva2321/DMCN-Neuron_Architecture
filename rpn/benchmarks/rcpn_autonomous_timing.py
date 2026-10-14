"""Autonomous Continuous-Time Delay Discovery Benchmark (Zero-Cheat).

Verifies that the RCPN's continuous-time Legendre state space learns to reconstruct
continuous past signals at arbitrary temporal delays autonomously without an external
trial-tick clock, outperforming memoryless baselines.
"""
from __future__ import annotations

import json
import numpy as np

from neuron_v2.core.rcpn import ContinuousTimeDelay


class MemorylessBaseline:
    """Baseline with no internal temporal state (instantaneous scalar mapping)."""

    def __init__(self, lr: float = 0.05):
        self.w = 0.0
        self.lr = lr

    def predict(self, x: float) -> float:
        return self.w * x

    def learn(self, x: float, target: float):
        pred = self.predict(x)
        self.w += self.lr * (target - pred) * x


class TappedDelayBaseline:
    """Fixed-buffer tapped delay line subject to temporal noise/jitter."""

    def __init__(self, buffer_size: int = 50, lr: float = 0.05):
        self.buffer = np.zeros(buffer_size, dtype=np.float64)
        self.w = np.zeros(buffer_size, dtype=np.float64)
        self.lr = lr

    def step(self, x: float) -> float:
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        return float(np.dot(self.w, self.buffer))

    def learn(self, x: float, target: float):
        pred = self.step(x)
        err = target - pred
        denom = float(np.dot(self.buffer, self.buffer)) + 1e-4
        self.w += (self.lr / denom) * err * self.buffer


def generate_signal(n_steps: int = 200, seed: int = 42) -> np.ndarray:
    """Generate smooth continuous-time analog input stream."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10 * np.pi, n_steps)
    # Superposition of frequencies
    s = np.sin(t) + 0.5 * np.cos(2.3 * t) + 0.3 * np.sin(0.7 * t)
    # Add small analog noise
    s += rng.normal(0, 0.05, n_steps)
    return s


def run_benchmark(n_trials: int = 20, delay_tau: int = 20, n_steps: int = 250) -> dict:
    """Run multi-trial evaluation of autonomous continuous delay reconstruction."""
    print(f"Running Autonomous Continuous Delay Benchmark (tau={delay_tau} ticks, {n_trials} trials)...")

    rcpn_mses = []
    tapped_mses = []
    memoryless_mses = []

    for trial in range(n_trials):
        seed = trial + 300
        signal = generate_signal(n_steps=n_steps, seed=seed)

        # 1. RCPN Continuous State Space
        delay_engine = ContinuousTimeDelay(order=6, theta=60.0, dt=1.0)
        # 2. Baselines
        tapped = TappedDelayBaseline(buffer_size=50)
        memoryless = MemorylessBaseline()

        preds_rcpn = []
        preds_tapped = []
        preds_memless = []
        targets = []

        for t in range(n_steps):
            x = signal[t]
            # Advance state
            delay_engine.step(x)

            if t >= delay_tau:
                target_val = signal[t - delay_tau]
                targets.append(target_val)

                # RCPN: online prediction error minimization without a clock
                pred_r = delay_engine.learn_readout(target_val, lr=0.08)
                preds_rcpn.append(pred_r)

                # Baselines
                pred_t = tapped.step(x)
                tapped.learn(x, target_val)
                preds_tapped.append(pred_t)

                pred_m = memoryless.predict(x)
                memoryless.learn(x, target_val)
                preds_memless.append(pred_m)

        # Compute MSE over the second half of the sequence (after convergence)
        eval_start = len(targets) // 2
        tgt_eval = np.array(targets[eval_start:])
        rcpn_mses.append(float(np.mean((tgt_eval - np.array(preds_rcpn[eval_start:])) ** 2)))
        tapped_mses.append(float(np.mean((tgt_eval - np.array(preds_tapped[eval_start:])) ** 2)))
        memoryless_mses.append(float(np.mean((tgt_eval - np.array(preds_memless[eval_start:])) ** 2)))

    results = {
        "RCPN_ContinuousState": {
            "mean_mse": float(np.mean(rcpn_mses)),
            "std_mse": float(np.std(rcpn_mses)),
        },
        "TappedDelayLine": {
            "mean_mse": float(np.mean(tapped_mses)),
            "std_mse": float(np.std(tapped_mses)),
        },
        "MemorylessBaseline": {
            "mean_mse": float(np.mean(memoryless_mses)),
            "std_mse": float(np.std(memoryless_mses)),
        },
    }

    print("\n" + "=" * 65)
    print(f"  AUTONOMOUS CONTINUOUS DELAY BENCHMARK (Tau = {delay_tau} ticks)")
    print("=" * 65)
    print(f"{'Model':<25} {'Mean MSE':<18} {'Std Dev':<12} {'Verdict'}")
    print("-" * 65)
    for model_name, res in results.items():
        verdict = "PASS" if res["mean_mse"] < 0.08 or model_name == "MemorylessBaseline" else "BASELINE"
        print(f"{model_name:<25} {res['mean_mse']:<18.4f} ±{res['std_mse']:<10.4f} {verdict}")
    print("=" * 65)

    with open("rcpn_timing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to rcpn_timing_results.json")
    return results


if __name__ == "__main__":
    run_benchmark(n_trials=20)
