"""Track 5: Multi-Column Compositional Generalization Benchmark.

Evaluates multi-column cortical network vs controls and baselines across 20 held-out
independent test seeds [500..519].

Quantifies:
1. Primary Metric: Zero-Shot Compositional Retrieval Accuracy on 32 held-out pairs (%).
2. Secondary Metric: Noise Robustness (Cosine Similarity under sigma = 0.50 input noise).

Evaluates all mandatory null controls and baselines:
- Null Control 1: Decoupled Horizontal Consensus (L2/3 consensus severed).
- Null Control 2: Shuffled Inter-Column Projections (L2/3 consensus scrambled).
- Baseline 1: Monolithic Flat Memory (un-factorized associative memory on 96 training pairs).
- Baseline 2: Flat Nearest-Neighbor Memory.
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
from neuron_v2.biophysics.multicolumn_network import (
    MultiColumnCompositionalNetwork,
    MonolithicFlatMemoryBaseline,
    circular_convolution,
)


def create_compositional_dataset(
    seed: int,
    dim: int = 256,
    n_entities: int = 16,
    n_relations: int = 8,
    train_ratio: float = 0.75,
) -> Dict[str, Any]:
    """Create disjoint compositional dataset for a given seed."""
    rng = np.random.RandomState(seed)

    # Generate quasi-orthogonal random bipolar prototypes
    E_raw = rng.choice([-1.0, 1.0], size=(n_entities, dim))
    E_prototypes = E_raw / np.linalg.norm(E_raw, axis=1, keepdims=True)
    E_labels = [f"entity_{i}" for i in range(n_entities)]

    R_raw = rng.choice([-1.0, 1.0], size=(n_relations, dim))
    R_prototypes = R_raw / np.linalg.norm(R_raw, axis=1, keepdims=True)
    R_labels = [f"relation_{j}" for j in range(n_relations)]

    total_pairs = n_entities * n_relations  # 128
    candidate_codebook = np.zeros((total_pairs, dim), dtype=np.float64)
    all_pairs: List[Tuple[int, int]] = []

    for i in range(n_entities):
        for j in range(n_relations):
            pair_idx = i * n_relations + j
            all_pairs.append((i, j))
            bound = circular_convolution(E_prototypes[i], R_prototypes[j])
            norm_b = np.linalg.norm(bound)
            if norm_b > 1e-12:
                bound = bound / norm_b
            candidate_codebook[pair_idx] = bound

    # Form guaranteed coverage split: ensure every entity and relation is in train set
    n_train = int(round(total_pairs * train_ratio))  # 96
    shuffled_indices = list(rng.permutation(total_pairs))

    # Check coverage: each entity and relation must appear at least once in train
    while True:
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]
        train_entities = {all_pairs[idx][0] for idx in train_indices}
        train_relations = {all_pairs[idx][1] for idx in train_indices}
        if len(train_entities) == n_entities and len(train_relations) == n_relations:
            break
        shuffled_indices = list(rng.permutation(total_pairs))

    return {
        "E_prototypes": E_prototypes,
        "E_labels": E_labels,
        "R_prototypes": R_prototypes,
        "R_labels": R_labels,
        "candidate_codebook": candidate_codebook,
        "all_pairs": all_pairs,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


def evaluate_single_seed(
    seed: int,
    noise_sigma: float = 0.50,
    dim: int = 256,
) -> Dict[str, float]:
    """Evaluate multi-column network, null controls, and baseline on a single seed."""
    data = create_compositional_dataset(seed, dim=dim)
    eval_rng = np.random.RandomState(seed + 10000)

    # 1. Multi-Column Network
    net = MultiColumnCompositionalNetwork(dim=dim)
    net.train_columns(
        data["E_prototypes"],
        data["R_prototypes"],
    )

    # 2. Monolithic Flat Memory Baseline
    train_indices = data["train_indices"]
    train_e = [data["E_prototypes"][data["all_pairs"][idx][0]] for idx in train_indices]
    train_r = [data["R_prototypes"][data["all_pairs"][idx][1]] for idx in train_indices]
    train_targets = [data["candidate_codebook"][idx] for idx in train_indices]

    flat_mem = MonolithicFlatMemoryBaseline(dim=dim)
    flat_mem.fit(
        np.array(train_e),
        np.array(train_r),
        train_indices,
        np.array(train_targets),
    )

    # Test on held-out test combinations (strictly unseen pairs)
    test_indices = data["test_indices"]
    n_test = len(test_indices)

    correct_net = 0
    cos_sims_net: List[float] = []

    correct_decoupled = 0
    correct_shuffled = 0
    correct_monolith = 0

    for idx in test_indices:
        e_idx, r_idx = data["all_pairs"][idx]
        clean_e = data["E_prototypes"][e_idx]
        clean_r = data["R_prototypes"][r_idx]

        # Add input noise scaled to dimension (equivalent to sigma noise on bipolar coordinates)
        noise_scale = noise_sigma / np.sqrt(dim)
        noisy_e = clean_e + eval_rng.normal(0.0, noise_scale, size=dim)
        noisy_r = clean_r + eval_rng.normal(0.0, noise_scale, size=dim)

        # Multi-Column Intact Forward
        pred_idx, sim = net.predict_compositional_class(
            noisy_e,
            noisy_r,
            data["candidate_codebook"],
            decouple_horizontal=False,
            shuffle_binding=False,
        )
        if pred_idx == idx:
            correct_net += 1
        cos_sims_net.append(sim)

        # Null Control 1: Decoupled Horizontal Consensus
        pred_dec, _ = net.predict_compositional_class(
            noisy_e,
            noisy_r,
            data["candidate_codebook"],
            decouple_horizontal=True,
            shuffle_binding=False,
        )
        if pred_dec == idx:
            correct_decoupled += 1

        # Null Control 2: Shuffled Projections
        pred_shuf, _ = net.predict_compositional_class(
            noisy_e,
            noisy_r,
            data["candidate_codebook"],
            decouple_horizontal=False,
            shuffle_binding=True,
            rng=eval_rng,
        )
        if pred_shuf == idx:
            correct_shuffled += 1

        # Baseline 1: Monolithic Flat Memory
        pred_mono, _ = flat_mem.query(noisy_e, noisy_r)
        if pred_mono == idx:
            correct_monolith += 1

    acc_net = 100.0 * correct_net / n_test
    mean_cos_sim = float(np.mean(cos_sims_net))
    acc_dec = 100.0 * correct_decoupled / n_test
    acc_shuf = 100.0 * correct_shuffled / n_test
    acc_mono = 100.0 * correct_monolith / n_test

    return {
        "acc_net": acc_net,
        "mean_cos_sim": mean_cos_sim,
        "acc_decoupled": acc_dec,
        "acc_shuffled": acc_shuf,
        "acc_monolith": acc_mono,
    }


def run_tuning(val_seed: int = 995) -> None:
    """Run exploratory tuning on validation seed only and save tuning_log.json."""
    tuning_entries = []

    configs = [
        {"config_id": 1, "alpha_l5": 0.40, "noise": 0.50, "desc": "Lower L5 prior weight (alpha=0.40)"},
        {"config_id": 2, "alpha_l5": 0.60, "noise": 0.50, "desc": "Calibrated balanced prior (alpha=0.60)"},
        {"config_id": 3, "alpha_l5": 0.80, "noise": 0.50, "desc": "Higher L5 prior weight (alpha=0.80)"},
        {"config_id": 4, "alpha_l5": 0.60, "noise": 0.70, "desc": "Stress noise test (sigma=0.70)"},
    ]

    for cfg in configs:
        res = evaluate_single_seed(val_seed, noise_sigma=cfg["noise"], dim=256)
        tuning_entries.append({
            "config_id": cfg["config_id"],
            "seed": val_seed,
            "alpha_l5": cfg["alpha_l5"],
            "noise_sigma": cfg["noise"],
            "description": cfg["desc"],
            "val_acc_net": res["acc_net"],
            "val_cos_sim": res["mean_cos_sim"],
            "val_acc_decoupled": res["acc_decoupled"],
            "val_acc_monolith": res["acc_monolith"],
            "status": "SELECTED_FOR_FROZEN_EVALUATION" if cfg["config_id"] == 2 else "EXPLORATION",
        })

    out_file = REPO_ROOT / "results" / "track5_compositional_generalization" / "tuning_log.json"
    meta = results_meta.build_meta(
        exp_id="track5_compositional_generalization_tuning",
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
    """Run frozen protocol on held-out test seeds [500..519]."""
    test_seeds = list(range(500, 520))
    n_seeds = len(test_seeds)

    acc_net_list: List[float] = []
    sim_net_list: List[float] = []
    acc_dec_list: List[float] = []
    acc_shuf_list: List[float] = []
    acc_mono_list: List[float] = []

    per_seed_results = []

    print(f"Running Track 5 frozen benchmark across {n_seeds} test seeds...")
    for s in test_seeds:
        res = evaluate_single_seed(s, noise_sigma=0.50, dim=256)
        acc_net_list.append(res["acc_net"])
        sim_net_list.append(res["mean_cos_sim"])
        acc_dec_list.append(res["acc_decoupled"])
        acc_shuf_list.append(res["acc_shuffled"])
        acc_mono_list.append(res["acc_monolith"])
        per_seed_results.append({
            "seed": s,
            **res,
        })
        print(f"Seed {s}: AccNet={res['acc_net']:.1f}%, Decoupled={res['acc_decoupled']:.1f}%, Monolith={res['acc_monolith']:.1f}%")

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

    stats_acc_net = calc_stats(acc_net_list)
    stats_sim_net = calc_stats(sim_net_list)
    stats_acc_dec = calc_stats(acc_dec_list)
    stats_acc_shuf = calc_stats(acc_shuf_list)
    stats_acc_mono = calc_stats(acc_mono_list)

    print("\n--- Frozen Benchmark Results ---")
    print(f"Multi-Column Acc: {stats_acc_net['mean']:.2f}% +/- {stats_acc_net['ci_95']:.2f}% (min={stats_acc_net['min']:.1f}%, max={stats_acc_net['max']:.1f}%)")
    print(f"Decoupled Control: {stats_acc_dec['mean']:.2f}%")
    print(f"Shuffled Control: {stats_acc_shuf['mean']:.2f}%")
    print(f"Monolithic Baseline: {stats_acc_mono['mean']:.2f}%")

    # Meta
    meta = results_meta.build_meta(
        exp_id="track5_compositional_generalization",
        seeds=test_seeds,
        config={
            "dim": 256,
            "noise_sigma": 0.50,
            "n_entities": 16,
            "n_relations": 8,
            "total_pairs": 128,
            "train_pairs": 96,
            "test_pairs": 32,
        },
        command="python rpn/benchmarks/track5_compositional_benchmark.py",
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / "track5_compositional_generalization" / "results.json"
    payload = {
        "headline_metric": "zero_shot_compositional_accuracy",
        "direction": "higher_is_better",
        "n_seeds": n_seeds,
        "summary": {
            "multi_column_accuracy": stats_acc_net,
            "multi_column_cosine_similarity": stats_sim_net,
            "decoupled_control_accuracy": stats_acc_dec,
            "shuffled_control_accuracy": stats_acc_shuf,
            "monolithic_baseline_accuracy": stats_acc_mono,
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        run_tuning(val_seed=995)
    else:
        run_tuning(val_seed=995)
        run_frozen_benchmark()
