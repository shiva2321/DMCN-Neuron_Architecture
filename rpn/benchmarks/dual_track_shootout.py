"""Dual-Track Shootout Benchmark: Track 1 (RCPN/RCC) vs Track 2 (HPR) vs Baselines.

Synthesizes experimental results across all verified runs and generates the unified scorecard.
"""
from __future__ import annotations

import json
from pathlib import Path


def generate_dual_track_shootout() -> dict:
    # 1. Load Track 1 results
    with open("rcpn_xor_results.json") as f:
        rcpn_xor = json.load(f)
    with open("rcpn_timing_results.json") as f:
        rcpn_timing = json.load(f)
    with open("rcc_laminar_results.json") as f:
        rcc_laminar = json.load(f)

    # 2. Load Track 2 results
    with open("hpr_benchmark_results.json") as f:
        hpr_bench = json.load(f)

    shootout = {
        "Track1_RCPN_RCC": {
            "paradigm": "Biophysical Compartmental Microcircuit",
            "autonomous_xor_acc": rcpn_xor["RCPN"]["mean"],
            "autonomous_xor_std": rcpn_xor["RCPN"]["std"],
            "continuous_delay_mse": rcpn_timing["RCPN_ContinuousState"]["mean_mse"],
            "column_capacity_128": rcc_laminar["Capacity_128"]["rcc_accuracy"],
            "ood_prior_retention": rcc_laminar["OOD_Adaptation"]["rcc_prior_retention"],
            "ood_new_location": rcc_laminar["OOD_Adaptation"]["rcc_new_location_accuracy"],
            "state_dimension": 6,
        },
        "Track2_HPR": {
            "paradigm": "First-Principles Holomorphic Wave-Interference",
            "autonomous_xor_acc": hpr_bench["xor_benchmark"]["mean_accuracy"],
            "autonomous_xor_std": hpr_bench["xor_benchmark"]["std_accuracy"],
            "anti_phase_alignment_cos": hpr_bench["xor_benchmark"]["mean_cos_dtheta"],
            "continuous_delay_mse": hpr_bench["timing_benchmark"]["hpr_spectral_mse"],
            "state_dimension": 16,
        },
        "Baselines": {
            "point_lif_xor_acc": rcpn_xor["PointLIF"]["mean"],
            "linear_perceptron_xor_acc": rcpn_xor["LinearPerceptron"]["mean"],
            "two_layer_mlp_xor_acc": rcpn_xor["TwoLayerMLP"]["mean"],
            "tapped_delay_line_mse": rcpn_timing["TappedDelayLine"]["mean_mse"],
            "memoryless_mse": rcpn_timing["MemorylessBaseline"]["mean_mse"],
            "monolith_capacity_128": rcc_laminar["Capacity_128"]["single_monolith_accuracy"],
            "monolith_ood_retention": rcc_laminar["OOD_Adaptation"]["single_prior_retention"],
        },
    }

    with open("dual_track_shootout_results.json", "w") as f:
        json.dump(shootout, f, indent=2)

    print("\n" + "=" * 80)
    print("                 STAGE 6 DUAL-TRACK GRAND SHOOTOUT SCORECARD")
    print("=" * 80)
    print(f"{'Task / Capability':<30} {'Track 1 (RCPN/RCC)':<20} {'Track 2 (HPR)':<18} {'Best Baseline':<12}")
    print("-" * 80)
    print(
        f"{'Autonomous XOR (20 seeds)':<30} "
        f"{shootout['Track1_RCPN_RCC']['autonomous_xor_acc']*100:>5.1f}% +/- {shootout['Track1_RCPN_RCC']['autonomous_xor_std']*100:>4.1f}%   "
        f"{shootout['Track2_HPR']['autonomous_xor_acc']*100:>5.1f}% +/- {shootout['Track2_HPR']['autonomous_xor_std']*100:>4.1f}%  "
        f"{shootout['Baselines']['two_layer_mlp_xor_acc']*100:>5.1f}% (MLP)"
    )
    print(
        f"{'Continuous Delay (tau=20)':<30} "
        f"{shootout['Track1_RCPN_RCC']['continuous_delay_mse']:>10.4f} (6 params) "
        f"{shootout['Track2_HPR']['continuous_delay_mse']:>10.6f}      "
        f"{shootout['Baselines']['tapped_delay_line_mse']:>10.4f} (TDL)"
    )
    print(
        f"{'Column Capacity (128 items)':<30} "
        f"{shootout['Track1_RCPN_RCC']['column_capacity_128']*100:>5.1f}%               "
        f"{'N/A (Column)':<18} "
        f"{shootout['Baselines']['monolith_capacity_128']*100:>5.1f}% (Mono)"
    )
    print(
        f"{'OOD Task Retention':<30} "
        f"{shootout['Track1_RCPN_RCC']['ood_prior_retention']*100:>5.1f}%               "
        f"{'N/A (Column)':<18} "
        f"{shootout['Baselines']['monolith_ood_retention']*100:>5.1f}% (Mono)"
    )
    print("=" * 80)
    print("Dual-Track Shootout saved to dual_track_shootout_results.json\n")
    return shootout


if __name__ == "__main__":
    generate_dual_track_shootout()
