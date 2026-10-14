"""Unit tests for Stress Benchmark 2: Bandwidth Sensitivity."""
import pytest

from benchmarks.stress_bandwidth_sensitivity import run_bandwidth_sensitivity


def test_bandwidth_sensitivity_roll_off():
    res = run_bandwidth_sensitivity(tau=20.0, dt=1.0, n_steps=800, seed=42)

    # 1. In-band performance is strong (< 0.010)
    in_band_mse = res["0.05"]["hpr_mse"]
    assert in_band_mse < 0.010

    # 2. Out-of-band roll-off occurs as predicted by physics
    out_band_mse = res["2.0"]["hpr_mse"]
    assert out_band_mse > 0.10  # Acknowledges real physical bandwidth limitation
    assert out_band_mse > in_band_mse * 20.0
