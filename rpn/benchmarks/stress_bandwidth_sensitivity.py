"""Stress Benchmark 2: Bandwidth Sensitivity Analysis.

Systematically measures delay reconstruction error as input frequency sweeps
from ultra-slow drift (0.0005 rad/s) through the tuned passband (0.005 - 0.20 rad/s)
up to the Nyquist limit (2.00 rad/s).
Documents exact roll-off, spectral attenuation, and bandwidth limits of both HPR and RCPN.
"""
from __future__ import annotations

import json
import numpy as np

from neuron_v2.core.hpr import HarmonicSpectralMemory
from neuron_v2.core.rcpn import ContinuousTimeDelay


def run_bandwidth_sensitivity(
    tau: float = 20.0,
    dt: float = 1.0,
    n_steps: int = 1000,
    seed: int = 42,
) -> dict:
    print("=" * 80)
    print("      BENCHMARK 2: BANDWIDTH SENSITIVITY & FREQUENCY ROLL-OFF ANALYSIS")
    print("=" * 80)

    sweep_frequencies = [
        0.0005, 0.0010, 0.0030, 0.0050, 0.0100, 0.0200, 0.0500,
        0.1000, 0.1500, 0.2000, 0.3000, 0.5000, 0.8000, 1.2000, 2.0000
    ]

    results = {}

    print(f"{'Input Frequency (rad/s)':<24} {'HPR MSE':<14} {'RCPN MSE':<14} {'Memoryless':<14} {'Regime':<14}")
    print("-" * 80)

    for freq in sweep_frequencies:
        t = np.arange(n_steps, dtype=np.float64) * dt
        x = np.sin(freq * t)

        hpr = HarmonicSpectralMemory(K=16, omega_min=0.005, omega_max=0.20, dt=dt, lr=0.05, seed=seed)
        rcpn = ContinuousTimeDelay(order=6, theta=100.0, dt=dt)

        preds_hpr = []
        preds_rcpn = []
        targets = []
        preds_none = []

        for step in range(n_steps):
            xt = x[step]
            hpr.step(xt)
            rcpn.step(xt)

            pred_h = hpr.predict_delay()
            pred_r = float((rcpn.W_readout @ rcpn.h)[0, 0])

            if step >= int(tau):
                target = x[step - int(tau)]
                hpr.learn_delay(target)
                rcpn.learn_readout(target)
                preds_hpr.append(pred_h)
                preds_rcpn.append(pred_r)
                targets.append(target)
                preds_none.append(xt)

        # Measure after convergence on last 300 steps
        eval_h = np.array(preds_hpr[-300:])
        eval_r = np.array(preds_rcpn[-300:])
        eval_t = np.array(targets[-300:])
        eval_n = np.array(preds_none[-300:])

        mse_h = float(np.mean((eval_h - eval_t) ** 2))
        mse_r = float(np.mean((eval_r - eval_t) ** 2))
        mse_n = float(np.mean((eval_n - eval_t) ** 2))

        if freq < 0.005:
            regime = "Sub-Band"
        elif freq <= 0.20:
            regime = "In-Band"
        else:
            regime = "Out-Of-Band"

        results[str(freq)] = {
            "frequency_rad_s": freq,
            "regime": regime,
            "hpr_mse": mse_h,
            "rcpn_mse": mse_r,
            "memoryless_mse": mse_n,
            "hpr_to_memoryless_ratio": mse_h / (mse_n + 1e-8),
        }

        print(f"{freq:<24.4f} {mse_h:<14.6f} {mse_r:<14.6f} {mse_n:<14.6f} {regime:<14}")

    print("=" * 80)
    with open("stress_bandwidth_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to stress_bandwidth_results.json\n")
    return results


if __name__ == "__main__":
    run_bandwidth_sensitivity()
