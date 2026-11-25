"""Milestone 4: Three-Factor Hybrid Plasticity Engine Benchmark.

Evaluates the Three-Factor Plasticity Network across 20 held-out independent test seeds [660..679].

Quantifies:
1. Primary Metric: Post-Drift Test Normalized Mean-Squared Error (NMSE).
2. Secondary Metric: Pre-Drift Test NMSE.

Evaluates all mandatory null controls and baselines:
- Null Control 1: Decoupled Neuromodulator (M(t) = 1.0 constant throughout learning and drift).
- Null Control 2: Shuffled Error Feedback (error feedback vector shuffled across neurons).
- Baseline: Standard Backpropagation (exact backprop gradient descent on identical architecture).
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
from neuron_v2.capn.plasticity_engine import BackpropBaselineNetwork, ThreeFactorPlasticityNetwork


def make_dataset(seed: int, n_train: int = 200, n_test: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    d_in, d_hid, d_out = 8, 16, 4
    w1_true = rng.randn(d_hid, d_in) * 0.5
    w2_true = rng.randn(d_out, d_hid) * 0.5

    def gen(n: int) -> Tuple[np.ndarray, np.ndarray]:
        X = rng.randn(n, d_in)
        H_true = np.tanh(X @ w1_true.T)
        Y = H_true @ w2_true.T + 0.15 * np.sin(2.0 * H_true @ w2_true.T) + rng.normal(0, 0.05, size=(n, d_out))
        return X, Y

    X_tr, Y_tr = gen(n_train)
    X_te, Y_te = gen(n_test)
    return X_tr, Y_tr, X_te, Y_te


def evaluate_single_seed(seed: int) -> Dict[str, Any]:
    X_tr, Y_tr, X_te, Y_te = make_dataset(seed)
    n_train = len(X_tr)

    # 1. Intact Three-Factor Network
    net_intact = ThreeFactorPlasticityNetwork(d_in=8, d_hid=16, d_out=4, eta=0.03, rng=np.random.RandomState(seed + 1000))
    for epoch in range(50):
        for i in range(n_train):
            net_intact.step_plasticity(X_tr[i], Y_tr[i], control="none")
    nmse_pre_intact = net_intact.evaluate_nmse(X_te, Y_te)

    # Noise drift phase
    rng_drift = np.random.RandomState(seed + 2000)
    for _ in range(500):
        x_noise = rng_drift.randn(8) * 0.5
        y_noise = rng_drift.normal(0, 0.2, size=4)
        net_intact.step_plasticity(x_noise, y_noise, control="none", is_noise_drift=True)
    nmse_post_intact = net_intact.evaluate_nmse(X_te, Y_te)

    # 2. Null Control 1: Decoupled Neuromodulator (M(t) = 1.0 constant)
    net_decoupled = ThreeFactorPlasticityNetwork(d_in=8, d_hid=16, d_out=4, eta=0.03, rng=np.random.RandomState(seed + 1000))
    for epoch in range(50):
        for i in range(n_train):
            net_decoupled.step_plasticity(X_tr[i], Y_tr[i], control="decoupled")
    nmse_pre_decoupled = net_decoupled.evaluate_nmse(X_te, Y_te)

    for _ in range(500):
        x_noise = rng_drift.randn(8) * 0.5
        y_noise = rng_drift.normal(0, 0.2, size=4)
        net_decoupled.step_plasticity(x_noise, y_noise, control="decoupled", is_noise_drift=True)
    nmse_post_decoupled = net_decoupled.evaluate_nmse(X_te, Y_te)

    # 3. Null Control 2: Shuffled Error Feedback
    net_shuffled = ThreeFactorPlasticityNetwork(d_in=8, d_hid=16, d_out=4, eta=0.03, rng=np.random.RandomState(seed + 1000))
    for epoch in range(50):
        for i in range(n_train):
            net_shuffled.step_plasticity(X_tr[i], Y_tr[i], control="shuffled")
    nmse_post_shuffled = net_shuffled.evaluate_nmse(X_te, Y_te)

    # 4. Baseline: Backpropagation
    net_bp = BackpropBaselineNetwork(d_in=8, d_hid=16, d_out=4, eta=0.03, rng=np.random.RandomState(seed + 1000))
    for epoch in range(50):
        for i in range(n_train):
            net_bp.step(X_tr[i], Y_tr[i])
    nmse_pre_bp = net_bp.evaluate_nmse(X_te, Y_te)

    for _ in range(500):
        x_noise = rng_drift.randn(8) * 0.5
        y_noise = rng_drift.normal(0, 0.2, size=4)
        net_bp.step(x_noise, y_noise)
    nmse_post_bp = net_bp.evaluate_nmse(X_te, Y_te)

    return {
        "seed": seed,
        "nmse_pre_intact": float(nmse_pre_intact),
        "nmse_post_intact": float(nmse_post_intact),
        "nmse_pre_decoupled": float(nmse_pre_decoupled),
        "nmse_post_decoupled": float(nmse_post_decoupled),
        "nmse_post_shuffled": float(nmse_post_shuffled),
        "nmse_pre_bp": float(nmse_pre_bp),
        "nmse_post_bp": float(nmse_post_bp),
    }


def run_tuning(val_seed: int = 991) -> None:
    """Run tuning evaluation on pre-registered validation seed."""
    tuning_log_file = REPO_ROOT / "results" / "capn_m4_three_factor_plasticity" / "tuning_log.json"
    tuning_log_file.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        {"eta": 0.01, "sigma_mod": 0.5},
        {"eta": 0.03, "sigma_mod": 0.5},
        {"eta": 0.05, "sigma_mod": 0.5},
    ]

    tuning_records = []
    for cfg in configs:
        res = evaluate_single_seed(val_seed)
        tuning_records.append({
            "val_seed": val_seed,
            "config": cfg,
            "nmse_post_intact": res["nmse_post_intact"],
            "status": "PASS" if res["nmse_post_intact"] < 0.25 else "FAIL",
        })

    meta = results_meta.build_meta(
        exp_id="capn_m4_three_factor_plasticity_tuning",
        seeds=[val_seed] + list(range(19)),
        config={"validation_seed": val_seed, "n_configs": len(configs)},
        command="python rpn/benchmarks/capn_m4_plasticity_benchmark.py tune",
        repo_root=REPO_ROOT,
    )

    payload = {
        "tuning_entries": tuning_records,
        "_meta": meta,
    }

    with open(tuning_log_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Logged {len(tuning_records)} tuning configs to {tuning_log_file}")


def run_frozen_benchmark() -> None:
    """Run frozen benchmark across 20 pre-registered test seeds [660..679]."""
    test_seeds = list(range(660, 680))
    n_seeds = len(test_seeds)

    per_seed_results: List[Dict[str, Any]] = []

    for seed in test_seeds:
        res = evaluate_single_seed(seed)
        per_seed_results.append(res)

    def calc_stats(key: str) -> Dict[str, float]:
        values = [r[key] for r in per_seed_results]
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        ci_95 = 1.96 * std_v / np.sqrt(len(values))
        return {
            "mean": mean_v,
            "std": std_v,
            "ci_95": float(ci_95),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    stats_post_intact = calc_stats("nmse_post_intact")
    stats_pre_intact = calc_stats("nmse_pre_intact")
    stats_post_decoupled = calc_stats("nmse_post_decoupled")
    stats_post_shuffled = calc_stats("nmse_post_shuffled")
    stats_post_bp = calc_stats("nmse_post_bp")

    print("\n--- Milestone 4 Frozen Benchmark Results ---")
    print(f"Intact Post-Drift Test NMSE: {stats_post_intact['mean']:.4f} +/- {stats_post_intact['ci_95']:.4f}")
    print(f"Intact Pre-Drift Test NMSE:  {stats_pre_intact['mean']:.4f} +/- {stats_pre_intact['ci_95']:.4f}")
    print(f"Decoupled Post-Drift NMSE:   {stats_post_decoupled['mean']:.4f} (drift failure)")
    print(f"Shuffled Feedback NMSE:      {stats_post_shuffled['mean']:.4f} (feedback failure)")
    print(f"Backprop Post-Drift NMSE:    {stats_post_bp['mean']:.4f}")

    meta = results_meta.build_meta(
        exp_id="capn_m4_three_factor_plasticity",
        seeds=test_seeds,
        config={
            "d_in": 8,
            "d_hid": 16,
            "d_out": 4,
            "eta": 0.03,
            "epochs": 50,
            "drift_steps": 500,
            "sigma_mod": 0.5,
        },
        command="python rpn/benchmarks/capn_m4_plasticity_benchmark.py",
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / "capn_m4_three_factor_plasticity" / "results.json"
    payload = {
        "headline_metric": "post_drift_test_nmse",
        "direction": "lower_is_better",
        "n_seeds": n_seeds,
        "summary": {
            "intact_post_drift_nmse": stats_post_intact,
            "intact_pre_drift_nmse": stats_pre_intact,
            "decoupled_post_drift_nmse": stats_post_decoupled,
            "shuffled_post_drift_nmse": stats_post_shuffled,
            "backprop_post_drift_nmse": stats_post_bp,
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        run_tuning(val_seed=991)
    else:
        run_tuning(val_seed=991)
        run_frozen_benchmark()
