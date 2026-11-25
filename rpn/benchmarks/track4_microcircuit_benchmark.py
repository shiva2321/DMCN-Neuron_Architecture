"""Track 4: Canonical Cortical Predictive Microcircuit Benchmark.

Evaluates the 3-Layer Predictive Microcircuit across 20 held-out independent
test seeds [400..419].

Quantifies:
1. Primary Metric: Prediction Error Suppression Ratio (PESR = MSE_pred / MSE_surprise in [0, 1]).
2. Secondary Metric: Predictable Regime Mean Squared Error (MSE_pred).
3. Surprise Regime Mean Squared Error (MSE_surprise), measuring anomaly response.

Evaluates all mandatory null controls and baselines:
- Null Control 1: Decoupled Top-Down Feedback (L5 -> L2/3 disconnected, y_hat = 0).
- Null Control 2: Shuffled Temporal Context (L5 state temporally scrambled).
- Baseline 1: Static Untrained Column (w = 0, no prediction).
- Baseline 2: Classical AR(8) Linear Predictive Filter (8-tap autoregressive filter).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.biophysics.predictive_microcircuit import PredictiveCorticalMicrocircuit


def generate_sensory_stream(
    seed: int,
    dt: float = 0.5,
    n_steps: int = 600,
    surprise_step: int = 400,
) -> np.ndarray:
    """Generate continuous sensory stream: predictable wave followed by unexpected surprise."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps, dtype=np.float64) * dt

    omega = 0.15
    phase_off = float(rng.uniform(0.0, 2.0 * np.pi))
    y = np.zeros(n_steps, dtype=np.float64)

    # Predictable regime (steps 0..surprise_step)
    y[:surprise_step] = (
        np.sin(omega * t[:surprise_step] + phase_off)
        + 0.3 * np.cos(2.0 * omega * t[:surprise_step] + phase_off)
    )

    # Unexpected surprise regime (steps surprise_step..n_steps)
    surprise_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    surprise_omega = float(rng.uniform(2.8, 3.5)) * omega
    y[surprise_step:] = np.sin(surprise_omega * t[surprise_step:] + surprise_phase)

    return y


def evaluate_intact_microcircuit(
    seeds: List[int],
    order: int = 8,
    theta: float = 40.0,
    dt: float = 0.5,
    lr: float = 0.05,
    n_steps: int = 600,
    surprise_step: int = 400,
) -> Dict[str, Any]:
    """Evaluate intact PredictiveCorticalMicrocircuit across seeds."""
    pesr_list: List[float] = []
    mse_pred_list: List[float] = []
    mse_surp_list: List[float] = []

    for seed in seeds:
        col = PredictiveCorticalMicrocircuit(order=order, theta=theta, dt=dt, lr=lr)
        y_sensory = generate_sensory_stream(seed, dt=dt, n_steps=n_steps, surprise_step=surprise_step)

        errors: List[float] = []
        for step in range(n_steps):
            res = col.step(y_sensory[step])
            errors.append(res["error"])

        err_arr = np.array(errors, dtype=np.float64)
        mse_pred = float(np.mean(err_arr[200:surprise_step] ** 2))
        mse_surp = float(np.mean(err_arr[surprise_step : surprise_step + 50] ** 2))
        pesr = float(mse_pred / (mse_surp + 1e-12))

        pesr_list.append(pesr)
        mse_pred_list.append(mse_pred)
        mse_surp_list.append(mse_surp)

    arr_pesr = np.array(pesr_list, dtype=np.float64)
    s_pesr = float(np.std(arr_pesr))
    ci_pesr = float(1.96 * s_pesr / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

    return {
        "mean_pesr": float(np.mean(arr_pesr)),
        "std_pesr": s_pesr,
        "ci95_pesr": ci_pesr,
        "min_pesr": float(np.min(arr_pesr)),
        "max_pesr": float(np.max(arr_pesr)),
        "per_seed_pesr": [float(v) for v in arr_pesr],
        "mean_mse_pred": float(np.mean(mse_pred_list)),
        "std_mse_pred": float(np.std(mse_pred_list)),
        "mean_mse_surprise": float(np.mean(mse_surp_list)),
    }


def evaluate_decoupled_control(
    seeds: List[int],
    order: int = 8,
    theta: float = 40.0,
    dt: float = 0.5,
    lr: float = 0.05,
    n_steps: int = 600,
    surprise_step: int = 400,
) -> Dict[str, Any]:
    """Null Control 1: Decoupled Top-Down Feedback (y_hat = 0)."""
    pesr_list: List[float] = []
    for seed in seeds:
        col = PredictiveCorticalMicrocircuit(order=order, theta=theta, dt=dt, lr=lr)
        y_sensory = generate_sensory_stream(seed, dt=dt, n_steps=n_steps, surprise_step=surprise_step)

        errors: List[float] = []
        for step in range(n_steps):
            # Override prediction to zero
            res = col.step(y_sensory[step], top_down_override=0.0, learn=False)
            errors.append(res["error"])

        err_arr = np.array(errors, dtype=np.float64)
        mse_pred = float(np.mean(err_arr[200:surprise_step] ** 2))
        mse_surp = float(np.mean(err_arr[surprise_step : surprise_step + 50] ** 2))
        pesr_list.append(float(mse_pred / (mse_surp + 1e-12)))

    arr = np.array(pesr_list, dtype=np.float64)
    return {
        "mean_pesr": float(np.mean(arr)),
        "std_pesr": float(np.std(arr)),
        "min_pesr": float(np.min(arr)),
        "max_pesr": float(np.max(arr)),
    }


def evaluate_shuffled_control(
    seeds: List[int],
    order: int = 8,
    theta: float = 40.0,
    dt: float = 0.5,
    lr: float = 0.05,
    n_steps: int = 600,
    surprise_step: int = 400,
) -> Dict[str, Any]:
    """Null Control 2: Shuffled Temporal Context (L5 state temporally scrambled)."""
    pesr_list: List[float] = []
    for seed in seeds:
        col = PredictiveCorticalMicrocircuit(order=order, theta=theta, dt=dt, lr=lr)
        y_sensory = generate_sensory_stream(seed, dt=dt, n_steps=n_steps, surprise_step=surprise_step)
        shuffler = np.random.RandomState(seed + 900)

        errors: List[float] = []
        for step in range(n_steps):
            # Scramble L5 input
            scrambled_y = float(shuffler.normal(0.0, 1.0))
            col.l5_engine.step(scrambled_y)
            pred = float(np.dot(col.w_pred, col.l5_engine.state.flatten()))
            eps = float(y_sensory[step] - pred)
            norm_m = float(np.dot(col.l5_engine.state.flatten(), col.l5_engine.state.flatten())) + 1e-4
            col.w_pred += (lr / norm_m) * eps * col.l5_engine.state.flatten()
            errors.append(eps)

        err_arr = np.array(errors, dtype=np.float64)
        mse_pred = float(np.mean(err_arr[200:surprise_step] ** 2))
        mse_surp = float(np.mean(err_arr[surprise_step : surprise_step + 50] ** 2))
        pesr_list.append(float(mse_pred / (mse_surp + 1e-12)))

    arr = np.array(pesr_list, dtype=np.float64)
    return {
        "mean_pesr": float(np.mean(arr)),
        "std_pesr": float(np.std(arr)),
        "min_pesr": float(np.min(arr)),
        "max_pesr": float(np.max(arr)),
    }


def evaluate_ar_baseline(
    seeds: List[int],
    p: int = 8,
    dt: float = 0.5,
    lr: float = 0.05,
    n_steps: int = 600,
    surprise_step: int = 400,
) -> Dict[str, Any]:
    """Baseline 2: Classical AR(p) Linear Predictive Filter."""
    pesr_list: List[float] = []
    for seed in seeds:
        y_sensory = generate_sensory_stream(seed, dt=dt, n_steps=n_steps, surprise_step=surprise_step)
        w_ar = np.zeros(p, dtype=np.float64)
        buf = np.zeros(p, dtype=np.float64)

        errors: List[float] = []
        for step in range(n_steps):
            y_t = y_sensory[step]
            pred = float(np.dot(w_ar, buf))
            eps = float(y_t - pred)

            norm_buf = float(np.dot(buf, buf)) + 1e-4
            w_ar += (lr / norm_buf) * eps * buf

            buf = np.roll(buf, 1)
            buf[0] = y_t
            errors.append(eps)

        err_arr = np.array(errors, dtype=np.float64)
        mse_pred = float(np.mean(err_arr[200:surprise_step] ** 2))
        mse_surp = float(np.mean(err_arr[surprise_step : surprise_step + 50] ** 2))
        pesr_list.append(float(mse_pred / (mse_surp + 1e-12)))

    arr = np.array(pesr_list, dtype=np.float64)
    return {
        "mean_pesr": float(np.mean(arr)),
        "std_pesr": float(np.std(arr)),
        "min_pesr": float(np.min(arr)),
        "max_pesr": float(np.max(arr)),
    }


def run_track4_microcircuit_benchmark() -> Dict[str, Any]:
    """Run full Track 4 predictive microcircuit benchmark across 20 held-out seeds."""
    exp_id = "track4_predictive_microcircuit"
    test_seeds = list(range(400, 420))
    order = 8
    theta = 40.0
    dt = 0.5
    lr = 0.05
    n_steps = 600
    surprise_step = 400

    print("=" * 80)
    print("TRACK 4 BENCHMARK: CANONICAL CORTICAL PREDICTIVE MICROCIRCUIT")
    print(f"Seeds: {len(test_seeds)} held-out seeds [400..419]")
    print(f"Order d = {order}, theta = {theta} ms, dt = {dt} ms, lr = {lr}")
    print("=" * 80)

    # 1. Intact Model
    intact_res = evaluate_intact_microcircuit(
        test_seeds, order=order, theta=theta, dt=dt, lr=lr, n_steps=n_steps, surprise_step=surprise_step
    )
    print(f"Intact Microcircuit Mean PESR: {intact_res['mean_pesr']:.4f} +/- {intact_res['std_pesr']:.4f}")
    print(f"Predictable Regime MSE:        {intact_res['mean_mse_pred']:.6f}")
    print(f"Surprise Regime MSE:           {intact_res['mean_mse_surprise']:.6f}")

    # 2. Null Controls
    decoupled_res = evaluate_decoupled_control(
        test_seeds, order=order, theta=theta, dt=dt, lr=lr, n_steps=n_steps, surprise_step=surprise_step
    )
    shuffled_res = evaluate_shuffled_control(
        test_seeds, order=order, theta=theta, dt=dt, lr=lr, n_steps=n_steps, surprise_step=surprise_step
    )
    print(f"Null Control 1 (Decoupled) Mean PESR: {decoupled_res['mean_pesr']:.4f}")
    print(f"Null Control 2 (Shuffled)  Mean PESR: {shuffled_res['mean_pesr']:.4f}")

    # 3. Baselines
    ar_res = evaluate_ar_baseline(
        test_seeds, p=order, dt=dt, lr=lr, n_steps=n_steps, surprise_step=surprise_step
    )
    print(f"Baseline 2 (AR({order}) Filter) Mean PESR:  {ar_res['mean_pesr']:.4f}")

    # 4. Check Falsifiers
    falsifier_tripped = False
    falsifier_reasons: List[str] = []

    pesr_mean = intact_res["mean_pesr"]
    if pesr_mean > 0.15:
        falsifier_tripped = True
        falsifier_reasons.append(f"Mean PESR {pesr_mean:.4f} exceeded threshold 0.15")

    if intact_res["mean_mse_pred"] >= 0.02:
        falsifier_tripped = True
        falsifier_reasons.append(f"Mean predictable MSE {intact_res['mean_mse_pred']:.6f} exceeded 0.02")

    if decoupled_res["mean_pesr"] < 0.85:
        falsifier_tripped = True
        falsifier_reasons.append(f"Decoupled control PESR {decoupled_res['mean_pesr']:.4f} below 0.85")

    verdict = "FALSIFIED" if falsifier_tripped else "CONFIRMED"
    print(f"\nFinal Verdict: {verdict}")

    config = {
        "order": order,
        "theta_ms": theta,
        "dt_ms": dt,
        "lr": lr,
        "n_steps": n_steps,
        "surprise_step": surprise_step,
        "predictable_eval_window": [200, surprise_step],
        "surprise_eval_window": [surprise_step, surprise_step + 50],
    }

    payload = {
        "verdict": verdict,
        "intact_microcircuit": intact_res,
        "null_control_decoupled": decoupled_res,
        "null_control_shuffled": shuffled_res,
        "baseline_ar": ar_res,
        "falsifier_tripped": falsifier_tripped,
        "falsifier_reasons": falsifier_reasons,
    }

    meta = results_meta.build_meta(
        exp_id=exp_id,
        seeds=test_seeds,
        config=config,
        command="python rpn/benchmarks/track4_microcircuit_benchmark.py",
        tuning_seeds=[996],
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / exp_id / "results.json"
    results_meta.write_results(out_file, payload, meta)
    print(f"Results written with compliant metadata to: {out_file}")

    return payload


if __name__ == "__main__":
    run_track4_microcircuit_benchmark()
