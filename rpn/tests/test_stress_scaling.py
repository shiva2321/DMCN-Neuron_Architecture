"""Unit tests for Stress Benchmark 4: Scaling and Efficiency."""
import pytest

from benchmarks.stress_scaling_efficiency import run_scaling_benchmark


def test_scaling_efficiency_metrics():
    res = run_scaling_benchmark()

    # 1. Throughput bounds
    hpr_lat = res["throughput"]["HPR_Harmonic_Resonator"]["latency_us"]
    assert hpr_lat < 50.0  # Under 50 microseconds per continuous step

    wave_lat = res["throughput"]["HPR_Wave_Gate"]["latency_us"]
    assert wave_lat < 50.0  # Under 50 microseconds per logic gate step

    # 2. Memory scaling verification
    assert res["memory_scaling"]["D256_S8"]["mb"] == 4.0  # Exactly 4 MB for 8 shards at D=256
    assert res["memory_scaling"]["D1024_S8"]["mb"] == 64.0  # Exactly 64 MB for 8 shards at D=1024

    # 3. Vocabulary clean-up scaling
    assert res["vocabulary_cleanup_scaling"]["100"]["latency_us"] < res["vocabulary_cleanup_scaling"]["10000"]["latency_us"]
