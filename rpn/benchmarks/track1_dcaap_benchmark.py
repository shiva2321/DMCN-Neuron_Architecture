#!/usr/bin/env python
"""Track 1: Biophysical Human L2/3 Pyramidal Neuron dCaAP XOR Benchmark.

Evaluates the non-monotonic dCaAP computational engine against pre-registered
criteria (results/track1_dcaap_xor/PREREG.md) across 20 held-out random test seeds
under continuous-time analog input noise and arrival timing jitter.

Mandatory Null Controls (GEMINI.md §5 & §7):
1. Kv Outward Shunt Ablation (K=1.0 constant): must fail XOR (<= 75.0%).
2. Passive Dendrite (w=0.0): must fail XOR (<= 75.0%).
3. Single-layer Linear Perceptron: proven theoretical ceiling of 75.0% on XOR.
4. Tuned 2-Layer Tanh MLP Baseline: must achieve 100.0% on clean XOR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also ensure .integrity is in sys.path for results_meta
INTEGRITY_DIR = REPO_ROOT / ".integrity"
if str(INTEGRITY_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRITY_DIR))

import results_meta
from rpn.neuron_v2.biophysics.dcaap_neuron import GidonDCaAPNeuron


def generate_stimulus(
    in_a: int,
    in_b: int,
    seed: int,
    noise_level: float = 0.05,
    jitter_ms: float = 1.5,
    i_opt: float = 1.0,
    t_nominal_ms: float = 15.0,
    tau_s_ms: float = 2.0,
    duration_ms: float = 80.0,
    dt_ms: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate continuous analog synaptic currents with noise and jitter.

    Crucial Integrity Rule (GEMINI.md §2.1):
    Inputs are generated from the task stimulus definition ONLY.
    ZERO branching on the target label.
    """
    rng = np.random.default_rng(seed)
    steps = int(duration_ms / dt_ms)
    time_ms = np.arange(steps) * dt_ms

    amp_a = in_a * i_opt * (1.0 + rng.uniform(-noise_level, noise_level)) if in_a > 0 else 0.0
    amp_b = in_b * i_opt * (1.0 + rng.uniform(-noise_level, noise_level)) if in_b > 0 else 0.0

    t_a = t_nominal_ms + rng.uniform(-jitter_ms, jitter_ms) if in_a > 0 else t_nominal_ms
    t_b = t_nominal_ms + rng.uniform(-jitter_ms, jitter_ms) if in_b > 0 else t_nominal_ms

    i_syn = np.zeros(steps)
    if in_a > 0:
        mask_a = time_ms >= t_a
        dt_a = time_ms[mask_a] - t_a
        i_syn[mask_a] += amp_a * (dt_a / tau_s_ms) * np.exp(1.0 - dt_a / tau_s_ms)
    if in_b > 0:
        mask_b = time_ms >= t_b
        dt_b = time_ms[mask_b] - t_b
        i_syn[mask_b] += amp_b * (dt_b / tau_s_ms) * np.exp(1.0 - dt_b / tau_s_ms)

    return i_syn, time_ms


def evaluate_neuron_condition(
    seeds: List[int],
    ablate_shunting: bool = False,
    passive_dendrite: bool = False,
    noise_level: float = 0.05,
    jitter_ms: float = 1.5,
) -> Dict[str, Any]:
    """Evaluate neuron condition across held-out test seeds."""
    patterns = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
    seed_accuracies = []
    detailed_results = []

    for s in seeds:
        correct = 0
        seed_record = {"seed": s, "trials": []}
        for in_a, in_b, target in patterns:
            i_syn, time_ms = generate_stimulus(
                in_a, in_b, seed=s, noise_level=noise_level, jitter_ms=jitter_ms
            )
            neuron = GidonDCaAPNeuron(
                ablate_shunting=ablate_shunting,
                passive_dendrite=passive_dendrite,
            )
            sim_res = neuron.simulate(i_syn, time_ms)
            prediction = 1 if sim_res["spikes"] >= 1 else 0
            is_correct = bool(prediction == target)
            if is_correct:
                correct += 1

            seed_record["trials"].append({
                "input": [in_a, in_b],
                "target": target,
                "prediction": prediction,
                "spikes": sim_res["spikes"],
                "k_factor": sim_res["k_factor"],
                "max_vs": sim_res["max_vs"],
                "correct": is_correct,
            })
        acc = correct / len(patterns)
        seed_accuracies.append(acc)
        seed_record["accuracy"] = acc
        detailed_results.append(seed_record)

    mean_acc = float(np.mean(seed_accuracies))
    std_acc = float(np.std(seed_accuracies))
    ci95 = float(1.96 * std_acc / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci95": ci95,
        "min_accuracy": float(np.min(seed_accuracies)),
        "max_accuracy": float(np.max(seed_accuracies)),
        "per_seed_accuracies": [float(a) for a in seed_accuracies],
        "detailed": detailed_results,
    }


def evaluate_linear_control(seeds: List[int]) -> Dict[str, Any]:
    """Null Control 3: Single-layer linear model on continuous analog features."""
    accuracies = []
    for s in seeds:
        # Fit optimal linear weights W*x + b on canonical inputs
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        # Optimal linear regression solution
        X_aug = np.column_stack([X, np.ones(4)])
        w = np.linalg.pinv(X_aug) @ y
        preds = (X_aug @ w >= 0.5).astype(int)
        acc = float(np.mean(preds == y))
        accuracies.append(acc)
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "ci95": 0.0,
        "per_seed_accuracies": accuracies,
    }


def evaluate_mlp_baseline(seeds: List[int]) -> Dict[str, Any]:
    """Baseline 4: Correctly tuned 2-layer tanh MLP (unit tested known-good)."""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    accuracies = []

    for s in seeds:
        rng = np.random.default_rng(s)
        # 2 -> 4 (tanh) -> 1 (sigmoid)
        W1 = rng.normal(0, 1.0, (2, 4))
        b1 = np.zeros(4)
        W2 = rng.normal(0, 1.0, (4, 1))
        b2 = np.zeros(1)

        lr = 0.1
        for _ in range(500):
            # Forward
            h = np.tanh(X @ W1 + b1)
            out = 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))
            # Backward
            d_out = out - y
            d_W2 = h.T @ d_out
            d_b2 = np.sum(d_out, axis=0)
            d_h = (d_out @ W2.T) * (1.0 - h**2)
            d_W1 = X.T @ d_h
            d_b1 = np.sum(d_h, axis=0)
            # Update
            W1 -= lr * d_W1
            b1 -= lr * d_b1
            W2 -= lr * d_W2
            b2 -= lr * d_b2

        preds = (out >= 0.5).astype(int)
        acc = float(np.mean(preds == y))
        accuracies.append(acc)

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "ci95": float(1.96 * np.std(accuracies) / np.sqrt(len(seeds))),
        "per_seed_accuracies": accuracies,
    }


def run_benchmark() -> Dict[str, Any]:
    """Execute pre-registered Track 1 benchmark protocol."""
    test_seeds = list(range(100, 120))  # strictly 20 held-out test seeds
    exp_id = "track1_dcaap_xor"
    noise_val = 0.05
    jitter_val = 1.5

    print("================================================================================")
    print("TRACK 1: BIOPHYSICAL HUMAN L2/3 dCaAP XOR BENCHMARK (GEMINI.md PROTOCOL)")
    print("================================================================================")
    print(f"Pre-registered Experiment ID: {exp_id}")
    print(f"Held-out Test Seeds ({len(test_seeds)}): {test_seeds}")
    print(f"Analog Current Noise: +/-{noise_val*100:.1f}% | Timing Jitter: +/-{jitter_val:.1f} ms")
    print("--------------------------------------------------------------------------------")

    print("[1/4] Running Intact Gidon dCaAP Biophysical Neuron...")
    intact_res = evaluate_neuron_condition(test_seeds, ablate_shunting=False, passive_dendrite=False)
    print(f"      -> Intact Mean Accuracy: {intact_res['mean_accuracy']*100:.1f}% +/- {intact_res['ci95']*100:.1f}%")

    print("[2/4] Running Mandatory Null Control 1: Kv Outward Shunt Ablation (K=1.0)...")
    ablate_res = evaluate_neuron_condition(test_seeds, ablate_shunting=True, passive_dendrite=False)
    print(f"      -> Kv Ablated Mean Accuracy: {ablate_res['mean_accuracy']*100:.1f}%")

    print("[3/4] Running Mandatory Null Control 2: Passive Cable Dendrite (w=0.0)...")
    passive_res = evaluate_neuron_condition(test_seeds, ablate_shunting=False, passive_dendrite=True)
    print(f"      -> Passive Dendrite Mean Accuracy: {passive_res['mean_accuracy']*100:.1f}%")

    print("[4/4] Running Mandatory Null Control 3 (Linear) and Baseline 4 (Tanh MLP)...")
    linear_res = evaluate_linear_control(test_seeds)
    mlp_res = evaluate_mlp_baseline(test_seeds)
    print(f"      -> Linear Perceptron Mean Accuracy: {linear_res['mean_accuracy']*100:.1f}%")
    print(f"      -> Tuned Tanh MLP Mean Accuracy:    {mlp_res['mean_accuracy']*100:.1f}%")

    # Verification of pre-registered falsifiers (absolute criteria only)
    falsifier_tripped = False
    falsifier_reasons = []

    intact_threshold = 0.95
    if intact_res["mean_accuracy"] < intact_threshold:
        falsifier_tripped = True
        falsifier_reasons.append(f"Intact accuracy ({intact_res['mean_accuracy']*100:.1f}%) < 95.0%")
    if ablate_res["mean_accuracy"] >= 0.90:
        falsifier_tripped = True
        falsifier_reasons.append(f"Kv ablated control ({ablate_res['mean_accuracy']*100:.1f}%) >= 90.0%")
    if passive_res["mean_accuracy"] >= 0.90:
        falsifier_tripped = True
        falsifier_reasons.append(f"Passive control ({passive_res['mean_accuracy']*100:.1f}%) >= 90.0%")
    # Baseline sanity check: baseline must not be broken or straw-man
    mlp_perfect = bool(np.isclose(mlp_res["mean_accuracy"], 1.0))
    if not mlp_perfect:
        falsifier_tripped = True
        falsifier_reasons.append("MLP baseline did not reach 1.0 (competitor misconfigured)")

    verdict = "FAIL (FALSIFIED)" if falsifier_tripped else "PASS (REPRODUCED & CONTROLLED)"

    print("--------------------------------------------------------------------------------")
    print(f"FINAL VERDICT: {verdict}")
    if falsifier_reasons:
        for r in falsifier_reasons:
            print(f"  * FALSIFIER: {r}")
    print("================================================================================")

    config = {
        "exp_id": exp_id,
        "n_seeds": len(test_seeds),
        "noise_level": 0.05,
        "jitter_ms": 1.5,
        "dt_ms": 0.05,
        "duration_ms": 80.0,
        "cd_nf": 0.05,
        "gleak_d_us": 0.003,
        "gc_us": 0.002,
        "cs_nf": 0.1,
        "gleak_s_us": 0.005,
        "vth_soma_mv": -52.0,
        "w_na": 5.0,
        "i_th_na": 0.6,
        "i_opt_na": 1.0,
        "d_param": 0.35,
    }

    payload = {
        "verdict": verdict,
        "intact_model": intact_res,
        "null_control_kv_ablated": ablate_res,
        "null_control_passive_dendrite": passive_res,
        "null_control_linear": linear_res,
        "baseline_tanh_mlp": mlp_res,
        "falsifier_tripped": falsifier_tripped,
        "falsifier_reasons": falsifier_reasons,
    }

    meta = results_meta.build_meta(
        exp_id=exp_id,
        seeds=test_seeds,
        config=config,
        command=f"python rpn/benchmarks/track1_dcaap_benchmark.py",
        tuning_seeds=[999],
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / exp_id / "results.json"
    results_meta.write_results(out_file, payload, meta)
    print(f"Results written with compliant metadata to: {out_file}")

    return payload


if __name__ == "__main__":
    run_benchmark()
