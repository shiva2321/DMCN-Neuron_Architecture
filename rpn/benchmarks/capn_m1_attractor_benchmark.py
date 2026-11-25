"""Milestone 1: Continuous Attractor and Dual-Store Memory Benchmark.

Evaluates the Continuous Attractor Memory Unit (CAPU) across 20 held-out independent
test seeds [600..619] under continuous sensory noise.

Quantifies:
1. Primary Metric: Working Memory Retrieval Accuracy after stimulus offset (%) in [0, 100%].
2. Secondary Metric: Post-Relaxation Contrast Margin SNR (dB).

Evaluates all mandatory null controls and baselines:
- Null Control 1: Linear Recurrent Network (linear phi(x) = x, tests dendritic saturation).
- Null Control 2: Unfacilitated Fixed-Synapse Network (STP ablated, u=U0 frozen).
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
from neuron_v2.capn.attractor_engine import ContinuousAttractorMemoryUnit


def evaluate_attractor_seed(
    seed: int,
    dim: int = 128,
    k_patterns: int = 8,
    g_rec: float = 3.5,
    noise_sigma: float = 0.40,
) -> Dict[str, Any]:
    """Evaluate working memory retention on a single seed."""
    rng = np.random.RandomState(seed)
    patterns = rng.choice([-1.0, 1.0], size=(k_patterns, dim))

    dt = 0.5
    n_steps = 600  # 300 ms total

    # Initialize unit
    unit = ContinuousAttractorMemoryUnit(dim=dim, g_rec=g_rec)
    unit.store_attractor_patterns(patterns)

    correct_capu = 0
    snr_capu_list: List[float] = []

    correct_lin = 0
    snr_lin_list: List[float] = []

    correct_unfac = 0
    snr_unfac_list: List[float] = []

    for k in range(k_patterns):
        target_p = patterns[k]
        noise_stream = [rng.normal(0.0, noise_sigma, size=dim) for _ in range(n_steps)]

        # 1. Intact CAPU
        unit.reset_state()
        for step in range(n_steps):
            t = step * dt
            I_ext = 2.0 * target_p if t <= 50.0 else noise_stream[step]
            unit.step(I_ext, dt=dt, linear=False, ablate_stp=False)
        pred_k, _, m_db = unit.decode_pattern(patterns)
        if pred_k == k:
            correct_capu += 1
        snr_capu_list.append(m_db)

        # 2. Linear Control
        unit.reset_state()
        for step in range(n_steps):
            t = step * dt
            I_ext = 2.0 * target_p if t <= 50.0 else noise_stream[step]
            unit.step(I_ext, dt=dt, linear=True, ablate_stp=False)
        pred_lin, _, m_db_lin = unit.decode_pattern(patterns)
        if pred_lin == k:
            correct_lin += 1
        snr_lin_list.append(m_db_lin)

        # 3. Unfacilitated Control (STP Ablated)
        unit.reset_state()
        for step in range(n_steps):
            t = step * dt
            I_ext = 2.0 * target_p if t <= 50.0 else noise_stream[step]
            unit.step(I_ext, dt=dt, linear=False, ablate_stp=True)
        pred_unfac, _, m_db_unfac = unit.decode_pattern(patterns)
        if pred_unfac == k:
            correct_unfac += 1
        snr_unfac_list.append(m_db_unfac)

    return {
        "acc_capu": 100.0 * correct_capu / k_patterns,
        "snr_capu": float(np.mean(snr_capu_list)),
        "acc_linear": 100.0 * correct_lin / k_patterns,
        "snr_linear": float(np.mean(snr_lin_list)),
        "acc_unfac": 100.0 * correct_unfac / k_patterns,
        "snr_unfac": float(np.mean(snr_unfac_list)),
    }


def run_tuning(val_seed: int = 994) -> None:
    """Run exploratory tuning on validation seed only and save tuning_log.json."""
    tuning_entries = []

    configs = [
        {"config_id": 1, "g_rec": 2.5, "noise": 0.40, "desc": "Sub-critical recurrent gain (g=2.5)"},
        {"config_id": 2, "g_rec": 3.5, "noise": 0.40, "desc": "Calibrated supercritical gain (g=3.5)"},
        {"config_id": 3, "g_rec": 4.5, "noise": 0.40, "desc": "Strong recurrent gain (g=4.5)"},
        {"config_id": 4, "g_rec": 3.5, "noise": 0.50, "desc": "Stress noise test (sigma=0.50)"},
    ]

    for cfg in configs:
        res = evaluate_attractor_seed(val_seed, g_rec=cfg["g_rec"], noise_sigma=cfg["noise"])
        tuning_entries.append({
            "config_id": cfg["config_id"],
            "seed": val_seed,
            "g_rec": cfg["g_rec"],
            "noise_sigma": cfg["noise"],
            "description": cfg["desc"],
            "val_acc_capu": res["acc_capu"],
            "val_snr_capu": res["snr_capu"],
            "val_acc_linear": res["acc_linear"],
            "val_snr_linear": res["snr_linear"],
            "val_acc_unfac": res["acc_unfac"],
            "val_snr_unfac": res["snr_unfac"],
            "status": "SELECTED_FOR_FROZEN_EVALUATION" if cfg["config_id"] == 2 else "EXPLORATION",
        })

    out_file = REPO_ROOT / "results" / "capn_m1_attractor_memory" / "tuning_log.json"
    meta = results_meta.build_meta(
        exp_id="capn_m1_attractor_memory_tuning",
        seeds=list(range(20)),
        config={"purpose": "validation_tuning"},
        command="tuning",
        repo_root=REPO_ROOT,
    )
    payload = {
        "tuning_entries": tuning_entries,
        "_meta": meta,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Validation tuning log saved to {out_file}")


def run_frozen_benchmark() -> None:
    """Run frozen protocol on held-out test seeds [600..619]."""
    test_seeds = list(range(600, 620))
    n_seeds = len(test_seeds)

    acc_capu_list: List[float] = []
    snr_capu_list: List[float] = []
    acc_lin_list: List[float] = []
    snr_lin_list: List[float] = []
    acc_unfac_list: List[float] = []
    snr_unfac_list: List[float] = []

    per_seed_results = []

    print(f"Running Milestone 1 frozen benchmark across {n_seeds} test seeds...")
    for s in test_seeds:
        res = evaluate_attractor_seed(s, g_rec=3.5, noise_sigma=0.40)
        acc_capu_list.append(res["acc_capu"])
        snr_capu_list.append(res["snr_capu"])
        acc_lin_list.append(res["acc_linear"])
        snr_lin_list.append(res["snr_linear"])
        acc_unfac_list.append(res["acc_unfac"])
        snr_unfac_list.append(res["snr_unfac"])

        per_seed_results.append({
            "seed": s,
            **res,
        })
        print(f"Seed {s}: CAPU Acc={res['acc_capu']:.1f}%, SNR={res['snr_capu']:.2f}dB | Lin Acc={res['acc_linear']:.1f}% | Unfac Acc={res['acc_unfac']:.1f}%")

    def calc_stats(values: List[float]) -> Dict[str, float]:
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci_95 = 1.96 * std_v / np.sqrt(len(values))
        return {
            "mean": mean_v,
            "std": std_v,
            "ci_95": float(ci_95),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    stats_acc_capu = calc_stats(acc_capu_list)
    stats_snr_capu = calc_stats(snr_capu_list)
    stats_acc_lin = calc_stats(acc_lin_list)
    stats_snr_lin = calc_stats(snr_lin_list)
    stats_acc_unfac = calc_stats(acc_unfac_list)
    stats_snr_unfac = calc_stats(snr_unfac_list)

    print("\n--- Milestone 1 Frozen Benchmark Results ---")
    print(f"Intact CAPU Accuracy: {stats_acc_capu['mean']:.2f}% +/- {stats_acc_capu['ci_95']:.2f}%")
    print(f"Intact CAPU SNR: {stats_snr_capu['mean']:.2f} dB +/- {stats_snr_capu['ci_95']:.2f} dB")
    print(f"Linear Control Accuracy: {stats_acc_lin['mean']:.2f}% (SNR: {stats_snr_lin['mean']:.2f} dB)")
    print(f"STP-Ablated Control Accuracy: {stats_acc_unfac['mean']:.2f}% (SNR: {stats_snr_unfac['mean']:.2f} dB)")

    meta = results_meta.build_meta(
        exp_id="capn_m1_attractor_memory",
        seeds=test_seeds,
        config={
            "dim": 128,
            "k_patterns": 8,
            "g_rec": 3.5,
            "noise_sigma": 0.40,
            "tau_m": 20.0,
            "tau_f": 1500.0,
            "tau_d": 50.0,
        },
        command="python rpn/benchmarks/capn_m1_attractor_benchmark.py",
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / "capn_m1_attractor_memory" / "results.json"
    payload = {
        "headline_metric": "working_memory_retrieval_accuracy",
        "direction": "higher_is_better",
        "n_seeds": n_seeds,
        "summary": {
            "intact_capu_accuracy": stats_acc_capu,
            "intact_capu_snr_db": stats_snr_capu,
            "linear_control_accuracy": stats_acc_lin,
            "linear_control_snr_db": stats_snr_lin,
            "stp_ablated_control_accuracy": stats_acc_unfac,
            "stp_ablated_control_snr_db": stats_snr_unfac,
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        run_tuning(val_seed=994)
    else:
        run_tuning(val_seed=994)
        run_frozen_benchmark()
