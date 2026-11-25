"""Continuous Resonant Field Neuron Symbolic Reasoning Benchmark (Card B-47).

Evaluates CRFNSymbolicEngine across 20 independent test seeds [920..939]:
- Task 1: Relational Analogy Generalization (held-out A:B::C:? under phase noise sigma=0.25).
- Task 2: Multi-Hop Transitive Deduction (held-out chains under phase noise sigma=0.25).
- Task 3: Continuous Wavefield Arithmetic & Inversion via linear phase demodulation (sigma=0.15).

Includes null controls (decoupled, shuffled phase, blind, ablated soma)
and baselines (monolithic flat memory, uniform random).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.crfn.crfn_symbolic_engine import CRFNSymbolicEngine


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, standard deviation, 95% confidence interval, min, and max."""
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


class MonolithicFlatMemoryAnalogyBaseline:
    """Non-factorized associative baseline that memorizes training triplets as flat concatenated vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.stored_keys: List[np.ndarray] = []
        self.stored_target_indices: List[int] = []

    def fit(self, training_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> None:
        """Store training analogies as flat concatenated real keys [Re(A); Im(A); Re(B); Im(B); ...] -> D."""
        self.stored_keys = []
        self.stored_target_indices = []
        for va, vb, vc, target_idx in training_triplets:
            flat_real = np.concatenate([
                np.real(va), np.imag(va),
                np.real(vb), np.imag(vb),
                np.real(vc), np.imag(vc),
            ])
            norm_k = np.linalg.norm(flat_real)
            if norm_k > 1e-12:
                flat_real = flat_real / norm_k
            self.stored_keys.append(flat_real)
            self.stored_target_indices.append(target_idx)

    def query(self, va: np.ndarray, vb: np.ndarray, vc: np.ndarray) -> Tuple[int, float]:
        """Query flat memory with test instance."""
        if not self.stored_keys:
            return -1, 0.0
        q = np.concatenate([
            np.real(va), np.imag(va),
            np.real(vb), np.imag(vb),
            np.real(vc), np.imag(vc),
        ])
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-12:
            q = q / norm_q
        sims = [float(np.dot(q, k)) for k in self.stored_keys]
        best_idx = int(np.argmax(sims))
        return self.stored_target_indices[best_idx], sims[best_idx]


def evaluate_crfn_seed(seed: int) -> Dict[str, Any]:
    """Run full benchmark for a single seed across all tasks, null controls, and baselines."""
    rng = np.random.RandomState(seed)
    dim = 256
    n_entities = 16
    n_relations = 8

    entity_phases = [rng.uniform(-np.pi, np.pi, dim) for _ in range(n_entities)]
    entities = [np.exp(1j * p) for p in entity_phases]

    rel_phases = [rng.uniform(-np.pi, np.pi, dim) for _ in range(n_relations)]
    relations = [np.exp(1j * p) for p in rel_phases]

    engine = CRFNSymbolicEngine(
        dim=dim,
        mu=1.0,
        n_frequencies=64,
        omega_min=0.01,
        omega_max=0.10,
    )

    # -------------------------------------------------------------
    # Task 1: Relational Analogy Generalization (A : B :: C : D)
    # -------------------------------------------------------------
    train_analogies = []
    for _ in range(64):
        ea_idx = rng.randint(0, n_entities)
        ec_idx = rng.randint(0, n_entities)
        r_idx = rng.randint(0, n_relations)
        va = entities[ea_idx]
        vc = entities[ec_idx]
        r = relations[r_idx]
        vb = va * r
        train_analogies.append((va, vb, vc, ec_idx))

    flat_baseline = MonolithicFlatMemoryAnalogyBaseline(dim=dim)
    flat_baseline.fit(train_analogies)

    n_test_analogies = 32
    correct_active_analogy = 0
    correct_decoupled_analogy = 0
    correct_shuffled_analogy = 0
    correct_blind_analogy = 0
    correct_unsettled_analogy = 0
    correct_flat_analogy = 0
    correct_random_analogy = 0

    noise_sigma = 0.25

    for _ in range(n_test_analogies):
        idx_a = rng.randint(0, n_entities)
        idx_c = rng.randint(0, n_entities)
        idx_d = rng.randint(0, n_entities)
        vec_a = entities[idx_a]
        vec_c = entities[idx_c]
        vec_d = entities[idx_d]

        rel_cd = vec_d * np.conj(vec_c)
        vec_b = vec_a * rel_cd

        noisy_a = vec_a * np.exp(1j * noise_sigma * rng.randn(dim))
        noisy_b = vec_b * np.exp(1j * noise_sigma * rng.randn(dim))
        noisy_c = vec_c * np.exp(1j * noise_sigma * rng.randn(dim))

        # 1. Active CRFN column
        res_act = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, rng=rng)
        # Classify by phase coherence against candidate entities
        sims_act = [float(np.real(np.dot(res_act["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_act)) == idx_d and float(np.max(sims_act)) > 0.50:
            correct_active_analogy += 1

        # 2. Decoupled control (T = 0)
        res_dec = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, decouple_transmission=True, rng=rng)
        sims_dec = [float(np.real(np.dot(res_dec["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_dec)) == idx_d and float(np.max(sims_dec)) > 0.50:
            correct_decoupled_analogy += 1

        # 3. Shuffled phase control
        res_shf = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, shuffle_phase=True, rng=rng)
        sims_shf = [float(np.real(np.dot(res_shf["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_shf)) == idx_d:
            correct_shuffled_analogy += 1

        # 4. Blind control (zeros)
        res_bln = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, blind_input=True, rng=rng)
        sims_bln = [float(np.real(np.dot(res_bln["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_bln)) == idx_d and float(np.max(sims_bln)) > 0.50:
            correct_blind_analogy += 1

        # 5. Ablated soma control
        res_uns = engine.solve_relational_analogy(noisy_a, noisy_b, noisy_c, ablate_soma=True, rng=rng)
        sims_uns = [float(np.real(np.dot(res_uns["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_uns)) == idx_d:
            correct_unsettled_analogy += 1

        # 6. Monolithic flat memory baseline
        pred_flat, _ = flat_baseline.query(noisy_a, noisy_b, noisy_c)
        if pred_flat == idx_d:
            correct_flat_analogy += 1

        # 7. Random uniform baseline
        if rng.randint(0, n_entities) == idx_d:
            correct_random_analogy += 1

    # -------------------------------------------------------------
    # Task 2: Multi-Hop Transitive Deduction (A -> B -> C)
    # -------------------------------------------------------------
    n_test_transitive = 20
    correct_active_trans = 0
    correct_decoupled_trans = 0
    correct_shuffled_trans = 0
    correct_blind_trans = 0
    correct_unsettled_trans = 0
    correct_random_trans = 0

    for _ in range(n_test_transitive):
        idx_a = rng.randint(0, n_entities)
        idx_c = rng.randint(0, n_entities)
        vec_a = entities[idx_a]
        vec_c = entities[idx_c]

        r_ab = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
        r_bc = (vec_c * np.conj(vec_a)) * np.conj(r_ab)

        noisy_a = vec_a * np.exp(1j * noise_sigma * rng.randn(dim))
        noisy_ab = r_ab * np.exp(1j * noise_sigma * rng.randn(dim))
        noisy_bc = r_bc * np.exp(1j * noise_sigma * rng.randn(dim))

        res_act = engine.solve_transitive_deduction(noisy_a, noisy_ab, noisy_bc, rng=rng)
        sims_act = [float(np.real(np.dot(res_act["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_act)) == idx_c and float(np.max(sims_act)) > 0.50:
            correct_active_trans += 1

        res_dec = engine.solve_transitive_deduction(noisy_a, noisy_ab, noisy_bc, decouple_transmission=True, rng=rng)
        sims_dec = [float(np.real(np.dot(res_dec["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_dec)) == idx_c and float(np.max(sims_dec)) > 0.50:
            correct_decoupled_trans += 1

        res_shf = engine.solve_transitive_deduction(noisy_a, noisy_ab, noisy_bc, shuffle_phase=True, rng=rng)
        sims_shf = [float(np.real(np.dot(res_shf["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_shf)) == idx_c:
            correct_shuffled_trans += 1

        res_bln = engine.solve_transitive_deduction(noisy_a, noisy_ab, noisy_bc, blind_input=True, rng=rng)
        sims_bln = [float(np.real(np.dot(res_bln["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_bln)) == idx_c and float(np.max(sims_bln)) > 0.50:
            correct_blind_trans += 1

        res_uns = engine.solve_transitive_deduction(noisy_a, noisy_ab, noisy_bc, ablate_soma=True, rng=rng)
        sims_uns = [float(np.real(np.dot(res_uns["output_wave"], np.conj(e)))) / dim for e in entities]
        if int(np.argmax(sims_uns)) == idx_c:
            correct_unsettled_trans += 1

        if rng.randint(0, n_entities) == idx_c:
            correct_random_trans += 1

    # -------------------------------------------------------------
    # Task 3: Continuous Wavefield Arithmetic & Inversion
    # -------------------------------------------------------------
    n_arithmetic_ops = 40  # 20 additions, 20 inversions
    correct_active_arith = 0
    correct_decoupled_arith = 0
    correct_shuffled_arith = 0
    correct_blind_arith = 0
    correct_unsettled_arith = 0
    correct_random_arith = 0

    arith_noise = 0.15

    for _ in range(20):
        a = rng.randint(0, 25)
        b = rng.randint(0, 25)
        expected_c = a + b

        res_act = engine.wave_add(a, b, noise_sigma=arith_noise, rng=rng)
        if res_act["predicted_val"] == expected_c:
            correct_active_arith += 1

        res_dec = engine.wave_add(a, b, decouple_transmission=True, rng=rng)
        if res_dec["predicted_val"] == expected_c:
            correct_decoupled_arith += 1

        res_shf = engine.wave_add(a, b, shuffle_phase=True, rng=rng)
        if res_shf["predicted_val"] == expected_c:
            correct_shuffled_arith += 1

        res_bln = engine.wave_add(a, b, blind_input=True, rng=rng)
        if res_bln["predicted_val"] == expected_c:
            correct_blind_arith += 1

        res_uns = engine.wave_add(a, b, ablate_soma=True, rng=rng)
        if res_uns["predicted_val"] == expected_c:
            correct_unsettled_arith += 1

        if rng.randint(0, 51) == expected_c:
            correct_random_arith += 1

    for _ in range(20):
        a = rng.randint(0, 25)
        b = rng.randint(0, 25)
        c = a + b

        res_act = engine.wave_invert(c, a, noise_sigma=arith_noise, rng=rng)
        if res_act["predicted_val"] == b:
            correct_active_arith += 1

        res_dec = engine.wave_invert(c, a, decouple_transmission=True, rng=rng)
        if res_dec["predicted_val"] == b:
            correct_decoupled_arith += 1

        res_shf = engine.wave_invert(c, a, shuffle_phase=True, rng=rng)
        if res_shf["predicted_val"] == b:
            correct_shuffled_arith += 1

        res_bln = engine.wave_invert(c, a, blind_input=True, rng=rng)
        if res_bln["predicted_val"] == b:
            correct_blind_arith += 1

        res_uns = engine.wave_invert(c, a, ablate_soma=True, rng=rng)
        if res_uns["predicted_val"] == b:
            correct_unsettled_arith += 1

        if rng.randint(0, 51) == b:
            correct_random_arith += 1

    return {
        "seed": seed,
        "analogy_acc_active": 100.0 * correct_active_analogy / n_test_analogies,
        "analogy_acc_decoupled": 100.0 * correct_decoupled_analogy / n_test_analogies,
        "analogy_acc_shuffled": 100.0 * correct_shuffled_analogy / n_test_analogies,
        "analogy_acc_blind": 100.0 * correct_blind_analogy / n_test_analogies,
        "analogy_acc_unsettled": 100.0 * correct_unsettled_analogy / n_test_analogies,
        "analogy_acc_flat": 100.0 * correct_flat_analogy / n_test_analogies,
        "analogy_acc_random": 100.0 * correct_random_analogy / n_test_analogies,
        "transitive_acc_active": 100.0 * correct_active_trans / n_test_transitive,
        "transitive_acc_decoupled": 100.0 * correct_decoupled_trans / n_test_transitive,
        "transitive_acc_shuffled": 100.0 * correct_shuffled_trans / n_test_transitive,
        "transitive_acc_blind": 100.0 * correct_blind_trans / n_test_transitive,
        "transitive_acc_unsettled": 100.0 * correct_unsettled_trans / n_test_transitive,
        "transitive_acc_random": 100.0 * correct_random_trans / n_test_transitive,
        "arithmetic_acc_active": 100.0 * correct_active_arith / n_arithmetic_ops,
        "arithmetic_acc_decoupled": 100.0 * correct_decoupled_arith / n_arithmetic_ops,
        "arithmetic_acc_shuffled": 100.0 * correct_shuffled_arith / n_arithmetic_ops,
        "arithmetic_acc_blind": 100.0 * correct_blind_arith / n_arithmetic_ops,
        "arithmetic_acc_unsettled": 100.0 * correct_unsettled_arith / n_arithmetic_ops,
        "arithmetic_acc_random": 100.0 * correct_random_arith / n_arithmetic_ops,
    }


def run_benchmark() -> Dict[str, Any]:
    """Execute Card B-47 benchmark across 20 held-out test seeds [920..939]."""
    test_seeds = list(range(920, 940))
    print(f"Executing Card B-47 Benchmark across {len(test_seeds)} seeds: {test_seeds}...")

    seed_results: List[Dict[str, Any]] = []
    for s in test_seeds:
        res = evaluate_crfn_seed(s)
        seed_results.append(res)
        print(
            f"Seed {s}: Analogy={res['analogy_acc_active']:.1f}% (Flat={res['analogy_acc_flat']:.1f}%), "
            f"Transitive={res['transitive_acc_active']:.1f}%, Arithmetic={res['arithmetic_acc_active']:.1f}%"
        )

    metric_keys = [
        "analogy_acc_active", "analogy_acc_decoupled", "analogy_acc_shuffled",
        "analogy_acc_blind", "analogy_acc_unsettled", "analogy_acc_flat", "analogy_acc_random",
        "transitive_acc_active", "transitive_acc_decoupled", "transitive_acc_shuffled",
        "transitive_acc_blind", "transitive_acc_unsettled", "transitive_acc_random",
        "arithmetic_acc_active", "arithmetic_acc_decoupled", "arithmetic_acc_shuffled",
        "arithmetic_acc_blind", "arithmetic_acc_unsettled", "arithmetic_acc_random",
    ]

    summary_stats = {k: compute_stats([r[k] for r in seed_results]) for k in metric_keys}

    config = {
        "dim": 256,
        "n_entities": 16,
        "n_relations": 8,
        "noise_sigma": 0.25,
        "arith_noise": 0.15,
        "n_frequencies": 64,
        "omega_min": 0.01,
        "omega_max": 0.10,
        "mu": 1.0,
    }

    meta = results_meta.build_meta(
        exp_id="crfn_b47_weight_free_symbolic",
        seeds=test_seeds,
        config=config,
        command="python -m benchmarks.crfn_b47_symbolic_benchmark",
        tuning_seeds=[860, 861, 862, 863, 864],
        repo_root=REPO_ROOT,
    )

    payload = {
        "exp_id": "crfn_b47_weight_free_symbolic",
        "summary": summary_stats,
        "seed_results": seed_results,
    }

    out_file = REPO_ROOT / "results" / "crfn_b47_weight_free_symbolic" / "results.json"
    results_meta.write_results(out_file, payload, meta)
    print(f"\n[OK] Results successfully serialized with metadata to: {out_file}")

    print("\n" + "=" * 80)
    print("CARD B-47: CRFN CONTINUOUS SYMBOLIC REASONING SCORECARD (20 SEEDS)")
    print("=" * 80)
    print(f"Task 1: Relational Analogy Accuracy : {summary_stats['analogy_acc_active']['mean']:.2f}% "
          f"+/- {summary_stats['analogy_acc_active']['ci95']:.2f}% (Flat Baseline: {summary_stats['analogy_acc_flat']['mean']:.2f}%)")
    print(f"Task 2: Multi-Hop Transitive Acc    : {summary_stats['transitive_acc_active']['mean']:.2f}% "
          f"+/- {summary_stats['transitive_acc_active']['ci95']:.2f}% (Decoupled: {summary_stats['transitive_acc_decoupled']['mean']:.2f}%)")
    print(f"Task 3: Reversible Wavefield Arith  : {summary_stats['arithmetic_acc_active']['mean']:.2f}% "
          f"+/- {summary_stats['arithmetic_acc_active']['ci95']:.2f}% (Shuffled: {summary_stats['arithmetic_acc_shuffled']['mean']:.2f}%)")
    print("=" * 80)

    return payload


if __name__ == "__main__":
    run_benchmark()
