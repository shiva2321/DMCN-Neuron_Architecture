"""Tests for Kuramoto relaxation and KuramotoRPN."""
import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.kuramoto import band_order_parameter, kuramoto_relax
from neuron_v2.core.kuramoto_rpn import KuramotoRPN
from neuron_v2.core.rpn import RPN
from neuron_v2.substrate.phasor import normalise, random_hv


def test_kuramoto_relax_changes_phases():
    bands = [normalise(random_hv(64, seed=i)) for i in range(3)]
    before = [band_order_parameter(b) for b in bands]
    out = kuramoto_relax(bands, 4, coupling=0.5, dt=0.3)
    after = [band_order_parameter(b) for b in out]
    assert any(abs(a - b) > 1e-4 for a, b in zip(before, after))


def test_kuramoto_rpn_smoke():
    items = ItemTable(256, seed=1)
    pairs = [(f"c{i}", f"t{i}") for i in range(8)]
    for c, t in pairs:
        _ = items[c]
        _ = items[t]
    km = KuramotoRPN(256, items, K=3, k=4)
    for c, t in pairs:
        km.learn(c, t)
    for c, t in pairs:
        assert km.read(c)[0] == t


def test_kuramoto_beats_rpn_on_noisy_read():
    from benchmarks.kuramoto_denoise import KuramotoConfig, run_benchmark

    cfg = KuramotoConfig(M=64, min_noisy_margin=0.12, max_noisy_rpn=0.75)
    km, pred, rpn = run_benchmark(cfg)
    assert km.noisy_acc > rpn.noisy_acc
    assert km.noisy_acc - rpn.noisy_acc >= cfg.min_noisy_margin
    assert km.noisy_acc >= pred.noisy_acc + cfg.min_vs_predictive_margin


def test_kuramoto_benchmark_smoke():
    from benchmarks.kuramoto_denoise import KuramotoConfig, evaluate_verdict, run_benchmark

    cfg = KuramotoConfig(M=64, min_noisy_margin=0.12, max_noisy_rpn=0.75)
    assert evaluate_verdict(run_benchmark(cfg), cfg).passed
