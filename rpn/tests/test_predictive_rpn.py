"""Tests for PredictiveRPN settling and surprise gating."""
import numpy as np

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.predictive_rpn import PredictiveRPN
from neuron_v2.core.settling import settle_predict
from neuron_v2.substrate.phasor import random_hv, sim


def test_surprise_gate_skips_redundant_writes():
    D = 256
    items = ItemTable(D, seed=1)
    _ = items["a"]
    _ = items["b"]
    rpn = PredictiveRPN(D, items, k=1, surprise_threshold=0.05)
    rpn.learn("a", "b")
    assert rpn.n_writes_applied == 1
    for _ in range(10):
        rpn.learn("a", "b")
    assert rpn.n_writes_skipped == 10
    assert rpn.read("a")[0] == "b"


def test_settling_k1_matches_plain_predict():
    D = 128
    items = ItemTable(D, seed=2)
    ctx = random_hv(D, seed=3)
    tgt = random_hv(D, seed=4)

    from neuron_v2.core.delta_memory import DeltaMemory

    mem = DeltaMemory(D, mode="delta")
    mem.write(ctx, tgt)
    plain = mem.predict(ctx)
    settled = settle_predict(mem, ctx, items, k=1)
    assert sim(plain, settled) > 0.99


def test_predictive_k4_beats_k1_on_noisy_read():
    from benchmarks.predictive_loop import LoopConfig, evaluate_verdict, run_benchmark

    cfg = LoopConfig(M=48, stream_epochs=2)
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    by = {r.name: r for r in results}
    k1 = by["rpn_k1"]
    k4 = by[f"predictive_k{cfg.settle_k}"]
    assert k4.noisy_acc > k1.noisy_acc
    assert k4.noisy_acc - k1.noisy_acc >= 0.10
    assert verdict.passed or k4.clean_acc >= cfg.min_clean_acc
