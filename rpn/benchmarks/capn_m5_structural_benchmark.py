"""Milestone 5: Information-Theoretic Structural Plasticity Benchmark.

Evaluates adaptive neurogenesis and metabolic synaptic pruning across 20 held-out independent test seeds [680..699].

Quantifies:
1. Primary Metric: False-Positive Neurogenesis on Stationary & Irreducible Noise Streams (target: 0.0).
2. Secondary Metric: Shift Adaptation Growth Rate (new units spawned under systematic distribution shifts).

Evaluates mandatory null controls and baselines:
- Null Control 1: Stationary Stream Control (intact network under stationary Gaussian mixture).
- Null Control 2: White Noise Control (intact network under irreducible white noise).
- Baseline: Periodic Timer-Based Neurogenesis (spawns unit every K steps regardless of data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.capn.structural_plasticity import StructuralPlasticityNetwork, TimerBasedBaselineNetwork


def make_stationary_stream(seed: int, n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    W = rng.randn(d_out, d_in) * 0.5
    X = rng.randn(n_samples, d_in)
    Y = X @ W.T + rng.normal(0, 0.05, size=(n_samples, d_out))
    return X, Y


def make_noise_stream(seed: int, n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    X = rng.randn(n_samples, d_in)
    Y = rng.randn(n_samples, d_out) * 1.0  # Irreducible white noise
    return X, Y


def make_shift_stream(seed: int, n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4
    X = np.zeros((n_samples, d_in))
    Y = np.zeros((n_samples, d_out))
    for k in range(3):
        start = k * 200
        end = (k + 1) * 200
        w_k = rng.randn(d_out, d_in) * (0.8 + 0.3 * k)
        X[start:end] = rng.randn(200, d_in) + (k * 2.0)
        Y[start:end] = np.tanh(X[start:end] @ w_k.T) + rng.normal(0, 0.05, size=(200, d_out))
    return X, Y


def evaluate_single_seed(seed: int) -> Dict[str, Any]:
    """Evaluate structural plasticity on stationary, noise, and shift streams for one seed."""
    X_stat, Y_stat = make_stationary_stream(seed)
    X_noise, Y_noise = make_noise_stream(seed)
    X_shift, Y_shift = make_shift_stream(seed)

    # 1. Intact model on stationary stream
    net_stat = StructuralPlasticityNetwork(
        d_in=8, d_initial_hid=4, d_out=4,
        theta_growth=0.30, theta_rho=0.28, sustained_steps=7, refractory_steps=40,
        rng=np.random.RandomState(seed),
    )
    res_stat = net_stat.process_stream(X_stat, Y_stat)

    # 2. Intact model on white noise stream (Null Control 2)
    net_noise = StructuralPlasticityNetwork(
        d_in=8, d_initial_hid=4, d_out=4,
        theta_growth=0.30, theta_rho=0.28, sustained_steps=7, refractory_steps=40,
        rng=np.random.RandomState(seed),
    )
    res_noise = net_noise.process_stream(X_noise, Y_noise)

    # 3. Intact model on systematic distribution shift stream
    net_shift = StructuralPlasticityNetwork(
        d_in=8, d_initial_hid=4, d_out=4,
        theta_growth=0.30, theta_rho=0.28, sustained_steps=7, refractory_steps=40,
        rng=np.random.RandomState(seed),
    )
    res_shift = net_shift.process_stream(X_shift, Y_shift)

    # 4. Timer-based comparative baseline
    timer_net = TimerBasedBaselineNetwork(
        d_in=8, d_initial_hid=4, d_out=4, spawn_period=50,
        rng=np.random.RandomState(seed),
    )
    res_timer_stat = timer_net.process_stream(X_stat, Y_stat)

    return {
        "seed": seed,
        "n_spawn_stat": res_stat["n_spawn"],
        "n_spawn_noise": res_noise["n_spawn"],
        "n_spawn_shift": res_shift["n_spawn"],
        "final_hid_shift": res_shift["final_hid"],
        "mean_error_shift": res_shift["mean_error"],
        "n_spawn_timer_stat": res_timer_stat["n_spawn"],
    }


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * sd / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95": ci95,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_frozen_benchmark() -> None:
    """Run frozen benchmark across 20 pre-registered test seeds [680..699]."""
    test_seeds = list(range(680, 700))
    n_seeds = len(test_seeds)

    per_seed_results: List[Dict[str, Any]] = []
    for seed in test_seeds:
        res = evaluate_single_seed(seed)
        per_seed_results.append(res)

    spawns_stat = [r["n_spawn_stat"] for r in per_seed_results]
    spawns_noise = [r["n_spawn_noise"] for r in per_seed_results]
    spawns_shift = [r["n_spawn_shift"] for r in per_seed_results]
    final_hids = [r["final_hid_shift"] for r in per_seed_results]
    spawns_timer = [r["n_spawn_timer_stat"] for r in per_seed_results]

    stat_summary = compute_stats(spawns_stat)
    noise_summary = compute_stats(spawns_noise)
    shift_summary = compute_stats(spawns_shift)
    hid_summary = compute_stats(final_hids)
    timer_summary = compute_stats(spawns_timer)

    # Compute false-positive rate
    fp_stat_count = sum(1 for s in spawns_stat if s > 0)
    fp_noise_count = sum(1 for s in spawns_noise if s > 0)

    # Meta
    meta = results_meta.build_meta(
        exp_id="capn_m5_structural_plasticity",
        seeds=test_seeds,
        config={
            "n_seeds": n_seeds,
            "d_in": 8,
            "d_initial_hid": 4,
            "d_out": 4,
            "theta_growth": 0.30,
            "theta_rho": 0.28,
            "sustained_steps": 7,
            "refractory_steps": 40,
            "prune_threshold": 0.005,
        },
        command="python rpn/benchmarks/capn_m5_structural_benchmark.py",
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / "capn_m5_structural_plasticity" / "results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "false_positive_stat_count": fp_stat_count,
            "false_positive_noise_count": fp_noise_count,
            "n_spawn_stat": stat_summary,
            "n_spawn_noise": noise_summary,
            "n_spawn_shift": shift_summary,
            "final_hid_shift": hid_summary,
            "n_spawn_timer_baseline": timer_summary,
            "exit_criterion_met": bool(fp_stat_count == 0 and fp_noise_count == 0 and shift_summary["mean"] >= 1.0),
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("==================================================")
    print("CAPN MILESTONE 5 STRUCTURAL PLASTICITY BENCHMARK")
    print("==================================================")
    print(f"Test Seeds: {n_seeds} ({test_seeds[0]}..{test_seeds[-1]})")
    print(f"FP Spawns on Stationary: {fp_stat_count}/{n_seeds} (Mean: {stat_summary['mean']:.4f})")
    print(f"FP Spawns on White Noise: {fp_noise_count}/{n_seeds} (Mean: {noise_summary['mean']:.4f})")
    print(f"Adaptive Neurogenesis Spawns on Shift: Mean {shift_summary['mean']:.2f} +/- {shift_summary['ci95']:.2f} (Min: {shift_summary['min']:.0f}, Max: {shift_summary['max']:.0f})")
    print(f"Final Hidden Dim on Shift: Mean {hid_summary['mean']:.2f} +/- {hid_summary['ci95']:.2f}")
    print(f"Timer Baseline Spurious Spawns: Mean {timer_summary['mean']:.2f} (Spawns unconditionally regardless of stationary data)")
    print(f"Exit Criterion Met: {payload['summary']['exit_criterion_met']}")
    print(f"Results written to: {out_file}")


if __name__ == "__main__":
    run_frozen_benchmark()
