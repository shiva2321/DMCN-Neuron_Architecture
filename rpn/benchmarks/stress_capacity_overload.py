"""Stress Benchmark 3: Cortical Shard Overload & Capacity Breakdown.

Systematically measures accuracy degradation as the number of spatial reference
frames (locations) scales from 8 up to 512, forcing multiple coordinate systems
to collide within the same 8 columnar shards (16 to 1024 items per shard in D=256).
Finds the exact mathematical breakdown threshold where fast-weight capacity saturates.
"""
from __future__ import annotations

import json
import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.laminar_column import LaminarCorticalColumn
from benchmarks.baselines import SingleRPNComposite


def run_capacity_stress_benchmark(
    D: int = 256,
    n_shards: int = 8,
    seed: int = 42,
) -> dict:
    print("=" * 85)
    print("      BENCHMARK 3: CORTICAL SHARD OVERLOAD & CAPACITY BREAKDOWN")
    print("=" * 85)

    location_levels = [8, 16, 32, 64, 128, 192, 256, 384, 512]
    results = {}

    print(f"{'Locations':<11} {'Total Items':<14} {'Items/Shard':<13} {'RCC Column':<16} {'Monolith':<14} {'Regime':<14}")
    print("-" * 85)

    for n_locs in location_levels:
        pairs = [
            (f"loc_{i}", f"item_{j}", f"obj_{i}_{j}")
            for i in range(n_locs)
            for j in range(16)
        ]
        items = ItemTable(D, seed=seed)
        for w, t, o in pairs:
            _ = items[w]
            _ = items[t]
            _ = items[o]

        rcc = LaminarCorticalColumn(D, items, n_shards=n_shards, seed=seed)
        mono = SingleRPNComposite(D, items)

        # Train both
        for w, t, o in pairs:
            rcc.learn(w, t, o)
            mono.learn(w, t, o)

        # Evaluate on up to 500 random pairs to keep benchmark fast and unbiased
        rng = np.random.RandomState(seed)
        eval_indices = rng.permutation(len(pairs))[:500] if len(pairs) > 500 else range(len(pairs))
        eval_subset = [pairs[idx] for idx in eval_indices]

        rcc_corr = sum(1 for w, t, o in eval_subset if rcc.read(w, t)[0] == o)
        mono_corr = sum(1 for w, t, o in eval_subset if mono.read(w, t)[0] == o)

        acc_rcc = float(rcc_corr / len(eval_subset))
        acc_mono = float(mono_corr / len(eval_subset))
        load_per_shard = len(pairs) // n_shards

        if load_per_shard <= D:
            regime = "Sub-Rank (<D)"
        elif load_per_shard <= int(1.5 * D):
            regime = "Near-Rank (~1.5D)"
        elif load_per_shard <= int(2.5 * D):
            regime = "Over-Rank (~2D)"
        else:
            regime = "Saturated (>3D)"

        results[str(n_locs)] = {
            "locations": n_locs,
            "total_associations": len(pairs),
            "items_per_shard": load_per_shard,
            "rcc_accuracy": acc_rcc,
            "monolith_accuracy": acc_mono,
            "regime": regime,
        }

        print(f"{n_locs:<11} {len(pairs):<14} {load_per_shard:<13} {acc_rcc * 100:>5.1f}%          {acc_mono * 100:>5.1f}%          {regime:<14}")

    print("=" * 85)
    with open("stress_capacity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to stress_capacity_results.json\n")
    return results


if __name__ == "__main__":
    run_capacity_stress_benchmark()
