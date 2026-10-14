"""Stress Benchmark 1: Frozen-Weight Continuous Delay Evaluation (Zero Test-Time Adaptation).

Evaluates whether models truly learned a general delay operator or merely tracked
a stationary periodic wave. Weights are trained on a multi-sine signal, FROZEN,
and evaluated zero-shot across 4 challenging, unseen test signals:
1. Signal A (Phase Shift): In-distribution frequencies with random phase offsets.
2. Signal B (Novel Frequencies): Out-of-distribution frequencies (0.015, 0.045, 0.12).
3. Signal C (Chirp Sweep): Non-stationary linear frequency chirp (0.005 -> 0.150).
4. Signal D (Square Wave): Discontinuous step transitions (period = 100).
"""
from __future__ import annotations

import json
import numpy as np

from neuron_v2.core.hpr import HarmonicSpectralMemory
from neuron_v2.core.rcpn import ContinuousTimeDelay


def run_frozen_timing_benchmark(
    n_train_steps: int = 1500,
    n_test_steps: int = 600,
    tau: float = 20.0,
    dt: float = 1.0,
    seed: int = 42,
) -> dict:
    print("=" * 80)
    print("      BENCHMARK 1: FROZEN-WEIGHT DELAY STRESS TEST (ZERO TEST-TIME ADAPTATION)")
    print("=" * 80)

    # 1. Training Phase on standard multi-sine wave
    t_train = np.arange(n_train_steps, dtype=np.float64) * dt
    x_train = np.sin(0.01 * t_train) + 0.5 * np.cos(0.03 * t_train) + 0.25 * np.sin(0.07 * t_train)

    hpr = HarmonicSpectralMemory(K=16, dt=dt, lr=0.05, seed=seed)
    rcpn = ContinuousTimeDelay(order=6, theta=100.0, dt=dt)
    tdl_buf = np.zeros(50, dtype=np.float64)
    tdl_w = np.zeros(50, dtype=np.float64)

    for t in range(n_train_steps):
        xt = x_train[t]
        hpr.step(xt)
        rcpn.step(xt)
        tdl_buf = np.roll(tdl_buf, 1)
        tdl_buf[0] = xt

        if t >= int(tau):
            target = x_train[t - int(tau)]
            hpr.learn_delay(target)
            rcpn.learn_readout(target)
            pred_tdl = float(np.dot(tdl_w, tdl_buf))
            err_tdl = target - pred_tdl
            tdl_w += (0.05 / (float(np.dot(tdl_buf, tdl_buf)) + 1e-4)) * err_tdl * tdl_buf

    print(f"Training completed on {n_train_steps} steps. Readout weights are now FROZEN.\n")

    # 2. Test Signals (Zero learning permitted during test)
    t_test = np.arange(n_test_steps, dtype=np.float64) * dt
    test_signals = {
        "A_PhaseShift": np.sin(0.01 * t_test + 1.2) + 0.5 * np.cos(0.03 * t_test + 0.8) + 0.25 * np.sin(0.07 * t_test + 2.1),
        "B_NovelFrequencies": 0.7 * np.sin(0.015 * t_test) + 0.5 * np.cos(0.045 * t_test) + 0.3 * np.sin(0.12 * t_test),
        "C_ChirpSweep": np.sin(0.005 * t_test + 0.00012 * t_test**2),
        "D_SquareWave": np.sign(np.sin(2 * np.pi * t_test / 100.0)),
    }

    results = {}

    print(f"{'Test Signal Regime':<24} {'HPR (Frozen)':<14} {'RCPN (Frozen)':<14} {'TDL (Frozen)':<14} {'Memoryless':<12}")
    print("-" * 80)

    for name, x_sig in test_signals.items():
        hpr.reset()
        rcpn.reset()
        t_buf = np.zeros(50, dtype=np.float64)

        preds_hpr = []
        preds_rcpn = []
        preds_tdl = []
        preds_none = []
        targets = []

        for t in range(n_test_steps):
            xt = x_sig[t]
            hpr.step(xt)
            rcpn.step(xt)
            t_buf = np.roll(t_buf, 1)
            t_buf[0] = xt

            p_hpr = hpr.predict_delay()
            p_rcpn = float((rcpn.W_readout @ rcpn.h)[0, 0])
            p_tdl = float(np.dot(tdl_w, t_buf))

            if t >= int(tau):
                target = x_sig[t - int(tau)]
                preds_hpr.append(p_hpr)
                preds_rcpn.append(p_rcpn)
                preds_tdl.append(p_tdl)
                preds_none.append(xt)
                targets.append(target)

        tar = np.array(targets)
        mse_hpr = float(np.mean((np.array(preds_hpr) - tar) ** 2))
        mse_rcpn = float(np.mean((np.array(preds_rcpn) - tar) ** 2))
        mse_tdl = float(np.mean((np.array(preds_tdl) - tar) ** 2))
        mse_none = float(np.mean((np.array(preds_none) - tar) ** 2))

        results[name] = {
            "hpr_mse": mse_hpr,
            "rcpn_mse": mse_rcpn,
            "tdl_mse": mse_tdl,
            "memoryless_mse": mse_none,
        }

        print(f"{name:<24} {mse_hpr:<14.5f} {mse_rcpn:<14.5f} {mse_tdl:<14.5f} {mse_none:<12.5f}")

    print("=" * 80)
    with open("stress_frozen_timing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to stress_frozen_timing_results.json\n")
    return results


if __name__ == "__main__":
    run_frozen_timing_benchmark()
