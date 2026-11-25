import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.biophysics.dcaap_neuron import GidonDCaAPNeuron
from benchmarks.track1_dcaap_benchmark import generate_stimulus


def test_dcaap_xor_intact_accuracy():
    """Verify intact model solves XOR across multiple seeds without label branching."""
    patterns = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
    for seed in [100, 101, 102, 103, 104]:
        correct = 0
        for in_a, in_b, target in patterns:
            i_syn, time_ms = generate_stimulus(in_a, in_b, seed=seed)
            neuron = GidonDCaAPNeuron()
            res = neuron.simulate(i_syn, time_ms)
            pred = 1 if res["spikes"] >= 1 else 0
            if pred == target:
                correct += 1
        assert correct == 4, f"Seed {seed} failed XOR: {correct}/4"


def test_dcaap_kv_shunting_ablation_fails_xor():
    """Verify that ablating Kv shunting causes (1, 1) to spike and fail XOR."""
    # On (1, 1), intact neuron must not spike (pred=0), but ablated MUST spike (pred=1)
    i_syn, time_ms = generate_stimulus(1, 1, seed=100)
    intact_neuron = GidonDCaAPNeuron(ablate_shunting=False)
    ablated_neuron = GidonDCaAPNeuron(ablate_shunting=True)

    intact_res = intact_neuron.simulate(i_syn, time_ms)
    ablated_res = ablated_neuron.simulate(i_syn, time_ms)

    assert intact_res["spikes"] == 0, "Intact neuron should not spike on (1, 1)"
    assert ablated_res["spikes"] >= 1, "Kv-ablated neuron must spike on (1, 1)"
    assert ablated_res["k_factor"] == 1.0


def test_dcaap_passive_control_fails_xor():
    """Verify that a passive dendrite (no dCaAP) cannot drive somatic spikes."""
    for in_a, in_b, _ in [(1, 0, 1), (0, 1, 1)]:
        i_syn, time_ms = generate_stimulus(in_a, in_b, seed=100)
        passive_neuron = GidonDCaAPNeuron(passive_dendrite=True)
        res = passive_neuron.simulate(i_syn, time_ms)
        assert res["spikes"] == 0, f"Passive dendrite spiked on ({in_a}, {in_b})"
        assert res["k_factor"] == 0.0


def test_dcaap_non_monotonic_attenuation():
    """Verify dCaAP attenuation K(I_peak) decreases monotonically with suprathreshold I."""
    neuron = GidonDCaAPNeuron(i_th_na=0.6, i_opt_na=1.0, d_param=0.35)
    k_sub = neuron.compute_attenuation(0.4)
    k_opt = neuron.compute_attenuation(1.0)
    k_high = neuron.compute_attenuation(1.6)
    k_supra = neuron.compute_attenuation(2.2)

    assert k_sub == 0.0, "Subthreshold attenuation must be 0"
    assert k_opt == 1.0, "Optimal threshold attenuation must be 1.0"
    assert k_high < k_opt, "High current attenuation must be less than optimal"
    assert k_supra < k_high, "Suprathreshold current attenuation must be strictly less than high"
    assert k_supra < 0.15, f"Dual input current must be heavily attenuated, got {k_supra}"
