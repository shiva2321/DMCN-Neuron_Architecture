"""Unit tests for Stress Benchmark 1: Frozen-weight continuous delay."""
import numpy as np
import pytest

from benchmarks.stress_frozen_timing import run_frozen_timing_benchmark


def test_frozen_timing_benchmark_integrity():
    res = run_frozen_timing_benchmark(n_train_steps=1200, n_test_steps=400, tau=20.0, seed=42)

    # 1. On novel frequencies, all models beat memoryless baseline
    b_res = res["B_NovelFrequencies"]
    assert b_res["rcpn_mse"] < b_res["memoryless_mse"]
    assert b_res["hpr_mse"] < b_res["memoryless_mse"]
    assert b_res["tdl_mse"] < b_res["memoryless_mse"]

    # 2. Verify that frozen MSE is realistically bounded (not artificially 0.0009)
    assert b_res["hpr_mse"] > 0.01  # Acknowledges spectral leakage / lack of online tracking
    assert b_res["rcpn_mse"] > 0.005

    # 3. On discontinuous square wave, all memory models beat memoryless
    d_res = res["D_SquareWave"]
    assert d_res["rcpn_mse"] < d_res["memoryless_mse"]
    assert d_res["hpr_mse"] < d_res["memoryless_mse"]
