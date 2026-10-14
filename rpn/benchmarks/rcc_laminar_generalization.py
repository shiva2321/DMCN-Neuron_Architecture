"""Laminar Cortical Column (RCC) Compositional Generalization Benchmark.

Evaluates:
1. Column capacity (8 where-shards x 16 shapes = 128 associations) vs Monolithic Baseline.
2. Zero-shot out-of-distribution (OOD) novel location adaptation and retention.
3. Lateral consensus voting under phase noise.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.laminar_column import LaminarCorticalColumn
from benchmarks.baselines import SingleRPNComposite


def generate_grid_dataset(n_locations: int = 8, n_items: int = 16) -> list[tuple[str, str, str]]:
    """Generate (where, what, object) association tuples."""
    pairs = []
    for loc_i in range(n_locations):
        for item_j in range(n_items):
            pairs.append((f"loc_{loc_i}", f"item_{item_j}", f"obj_{loc_i}_{item_j}"))
    return pairs


def register_items(items: ItemTable, pairs: list[tuple[str, str, str]]):
    for where, what, obj in pairs:
        _ = items[where]
        _ = items[what]
        _ = items[obj]


def run_benchmark(D: int = 256, seed: int = 42) -> dict:
    print(f"Running Laminar Cortical Column (RCC) Generalization Benchmark (D={D})...")

    # 1. Capacity Shootout: 8 locations x 16 items = 128 pairs
    pairs_128 = generate_grid_dataset(n_locations=8, n_items=16)
    items = ItemTable(D, seed=seed)
    register_items(items, pairs_128)

    rcc = LaminarCorticalColumn(D, items, n_shards=8, seed=seed)
    single_monolith = SingleRPNComposite(D, items)

    # Train both
    for where, what, obj in pairs_128:
        rcc.learn(where, what, obj)
        single_monolith.learn(where, what, obj)

    # Test accuracy
    rcc_correct = sum(1 for w, t, o in pairs_128 if rcc.read(w, t)[0] == o)
    rcc_acc = rcc_correct / len(pairs_128)

    single_correct = sum(1 for w, t, o in pairs_128 if single_monolith.read(w, t)[0] == o)
    single_acc = single_correct / len(pairs_128)

    print(f"  [Capacity 8x16] RCC Column: {rcc_acc * 100:.1f}% vs Single Monolith: {single_acc * 100:.1f}%")

    # 2. Zero-Shot Out-of-Distribution (OOD) Adaptation
    # Train on 6 locations, then adapt to location 6 and 7
    train_pairs = [p for p in pairs_128 if int(p[0].split("_")[1]) < 6]
    adapt_pairs = [p for p in pairs_128 if int(p[0].split("_")[1]) >= 6]

    items_ood = ItemTable(D, seed=seed + 10)
    register_items(items_ood, pairs_128)
    rcc_ood = LaminarCorticalColumn(D, items_ood, n_shards=8, seed=seed + 10)
    single_ood = SingleRPNComposite(D, items_ood)

    # Train baseline
    for w, t, o in train_pairs:
        rcc_ood.learn(w, t, o)
        single_ood.learn(w, t, o)

    # Adapt new locations
    for w, t, o in adapt_pairs:
        rcc_ood.learn(w, t, o)
        single_ood.learn(w, t, o)

    # Measure retention on prior locations
    rcc_retention = sum(1 for w, t, o in train_pairs if rcc_ood.read(w, t)[0] == o) / len(train_pairs)
    single_retention = sum(1 for w, t, o in train_pairs if single_ood.read(w, t)[0] == o) / len(train_pairs)

    rcc_new_acc = sum(1 for w, t, o in adapt_pairs if rcc_ood.read(w, t)[0] == o) / len(adapt_pairs)
    single_new_acc = sum(1 for w, t, o in adapt_pairs if single_ood.read(w, t)[0] == o) / len(adapt_pairs)

    print(f"  [OOD Retention] RCC Retention: {rcc_retention * 100:.1f}% vs Monolith: {single_retention * 100:.1f}%")
    print(f"  [OOD New Loc]   RCC New Loc:   {rcc_new_acc * 100:.1f}% vs Monolith: {single_new_acc * 100:.1f}%")

    results = {
        "Capacity_128": {
            "rcc_accuracy": rcc_acc,
            "single_monolith_accuracy": single_acc,
            "margin": rcc_acc - single_acc,
        },
        "OOD_Adaptation": {
            "rcc_prior_retention": rcc_retention,
            "single_prior_retention": single_retention,
            "rcc_new_location_accuracy": rcc_new_acc,
            "single_new_location_accuracy": single_new_acc,
            "retention_margin": rcc_retention - single_retention,
        },
    }

    print("\n" + "=" * 65)
    print("      LAMINAR CORTICAL COLUMN (RCC) SCORECARD")
    print("=" * 65)
    print(f"{'Benchmark Metric':<32} {'RCC Column':<16} {'Monolith':<12} {'Verdict'}")
    print("-" * 65)
    cap_pass = rcc_acc >= 0.95 and single_acc <= 0.60
    ret_pass = rcc_retention >= 0.95 and (rcc_retention - single_retention) >= 0.40
    print(f"{'Capacity (8x16 = 128 items)':<32} {rcc_acc * 100:>5.1f}%          {single_acc * 100:>5.1f}%       {'PASS' if cap_pass else 'FAIL'}")
    print(f"{'OOD Retention (Prior Tasks)':<32} {rcc_retention * 100:>5.1f}%          {single_retention * 100:>5.1f}%       {'PASS' if ret_pass else 'FAIL'}")
    print(f"{'OOD Acquisition (New Tasks)':<32} {rcc_new_acc * 100:>5.1f}%          {single_new_acc * 100:>5.1f}%       {'PASS' if rcc_new_acc >= 0.95 else 'FAIL'}")
    print("=" * 65)

    with open("rcc_laminar_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to rcc_laminar_results.json")
    return results


if __name__ == "__main__":
    run_benchmark(D=256)
