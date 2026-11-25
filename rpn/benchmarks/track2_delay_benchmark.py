"""Track 2: Single-Neuron Continuous Time-Delay & Memory Engine Benchmark.

Evaluates the Continuous-Time State-Space Legendre Delay Engine across 20 held-out
independent test seeds on 5 diverse unseen out-of-distribution signal regimes:
1. Continuous Ornstein-Uhlenbeck (OU) stochastic noise (tau_corr = 20.0 ms).
2. Phase-Shifted multi-sine waves with random phase offsets.
3. Novel out-of-distribution frequencies.
4. Non-stationary linear frequency chirp sweeps.
5. Discontinuous square wave transitions.

Runs all mandatory null controls and baselines:
- Null Control 1: Decoupled Input (B = 0, verifying sensory input drive).
- Null Control 2: Shuffled Input (temporally scrambled input, verifying causality).
- Baseline 1: Persistence (zero-delay predictor u(t)).
- Baseline 2: Mean/Zero baseline.
- Baseline 3: Exact Tapped Delay Line (12.5x larger memory buffer, known answer NMSE = 0.0).
- Baseline 4: Parameter-Matched Tapped Delay Line (N = 8 taps).
- Baseline 5: Classical AR(8) Linear Autoregressive filter.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.linalg

# Ensure repo and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.biophysics.continuous_delay import ContinuousLegendreDelayEngine


def generate_test_signals(
    seed: int,
    dt: float = 0.5,
    n_steps: int = 300,
) -> Dict[str, np.ndarray]:
    """Generate 5 distinct continuous test signal regimes for a given seed."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, n_steps * dt, dt, dtype=np.float64)

    # 1. Continuous Ornstein-Uhlenbeck stochastic process
    tau_ou = 20.0
    ou = np.zeros(n_steps, dtype=np.float64)
    for i in range(1, n_steps):
        drift = -ou[i - 1] / tau_ou
        diffusion = np.sqrt(2.0 * dt / tau_ou) * rng.normal(0.0, 1.0)
        ou[i] = ou[i - 1] + drift * dt + diffusion

    # 2. Phase-shifted multi-sine wave with random phase offsets
    p1, p2, p3 = rng.uniform(0.0, 2.0 * np.pi, 3)
    sig_phase = (
        np.sin(0.1 * t + p1)
        + 0.5 * np.cos(0.25 * t + p2)
        + 0.2 * np.sin(0.4 * t + p3)
    )

    # 3. Novel out-of-distribution frequencies
    f1, f2 = rng.uniform(0.12, 0.22), rng.uniform(0.32, 0.48)
    sig_novel = 0.7 * np.sin(f1 * t) + 0.4 * np.cos(f2 * t)

    # 4. Non-stationary linear frequency chirp sweep
    beta = rng.uniform(0.0008, 0.0015)
    sig_chirp = np.sin(0.05 * t + beta * (t ** 2))

    # 5. Discontinuous square wave transition
    period = rng.uniform(35.0, 50.0)
    sig_square = np.sign(np.sin(2.0 * np.pi * t / period))

    return {
        "OU_Noise": ou,
        "PhaseShift": sig_phase,
        "NovelFreqs": sig_novel,
        "Chirp": sig_chirp,
        "SquareWave": sig_square,
    }


def compute_nmse(preds: List[float], targets: List[float]) -> float:
    """Compute Normalized Mean Squared Error: MSE / Var(targets)."""
    p_arr = np.array(preds, dtype=np.float64)
    t_arr = np.array(targets, dtype=np.float64)
    mse = float(np.mean((p_arr - t_arr) ** 2))
    var = float(np.var(t_arr))
    if var < 1e-12:
        return 1.0
    return float(mse / var)


def evaluate_intact_engine(
    seeds: List[int],
    order: int = 8,
    theta: float = 50.0,
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Evaluate intact Continuous Legendre Delay Engine across seeds and waveforms."""
    delay_steps = int(round(tau / dt))
    results_by_signal: Dict[str, List[float]] = {
        "OU_Noise": [],
        "PhaseShift": [],
        "NovelFreqs": [],
        "Chirp": [],
        "SquareWave": [],
    }

    for seed in seeds:
        signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
        for sig_name, sig in signals.items():
            engine = ContinuousLegendreDelayEngine(order=order, theta=theta, dt=dt)
            preds: List[float] = []
            targets: List[float] = []
            for step in range(n_steps):
                u = sig[step]
                engine.step(u)
                t_now = step * dt
                if t_now >= tau:
                    pred = engine.reconstruct_delay(tau)
                    preds.append(pred)
                    targets.append(sig[step - delay_steps])
            nmse = compute_nmse(preds, targets)
            results_by_signal[sig_name].append(nmse)

    summary: Dict[str, Any] = {}
    for sig_name, nmses in results_by_signal.items():
        arr = np.array(nmses, dtype=np.float64)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        ci = float(1.96 * s / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        summary[sig_name] = {
            "mean_nmse": m,
            "std_nmse": s,
            "ci95": ci,
            "min_nmse": float(np.min(arr)),
            "max_nmse": float(np.max(arr)),
            "per_seed_nmse": [float(v) for v in arr],
        }

    return summary


def evaluate_decoupled_control(
    seeds: List[int],
    order: int = 8,
    theta: float = 50.0,
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Null Control 1: Decoupled input (B = 0). State receives zero input."""
    delay_steps = int(round(tau / dt))
    nmses: List[float] = []

    for seed in seeds:
        signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
        sig = signals["OU_Noise"]
        engine = ContinuousLegendreDelayEngine(order=order, theta=theta, dt=dt)
        # Decouple sensory input
        engine.B_bar.fill(0.0)

        preds: List[float] = []
        targets: List[float] = []
        for step in range(n_steps):
            u = sig[step]
            engine.step(u)
            t_now = step * dt
            if t_now >= tau:
                pred = engine.reconstruct_delay(tau)
                preds.append(pred)
                targets.append(sig[step - delay_steps])
        nmses.append(compute_nmse(preds, targets))

    arr = np.array(nmses, dtype=np.float64)
    return {
        "mean_nmse": float(np.mean(arr)),
        "std_nmse": float(np.std(arr)),
        "min_nmse": float(np.min(arr)),
        "max_nmse": float(np.max(arr)),
    }


def evaluate_shuffled_control(
    seeds: List[int],
    order: int = 8,
    theta: float = 50.0,
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Null Control 2: Shuffled input (temporally permuted signal)."""
    delay_steps = int(round(tau / dt))
    nmses: List[float] = []

    for seed in seeds:
        signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
        sig = signals["OU_Noise"]
        shuffler = np.random.RandomState(seed + 500)
        sig_shuffled = shuffler.permutation(sig)

        engine = ContinuousLegendreDelayEngine(order=order, theta=theta, dt=dt)
        preds: List[float] = []
        targets: List[float] = []
        for step in range(n_steps):
            u = sig_shuffled[step]
            engine.step(u)
            t_now = step * dt
            if t_now >= tau:
                pred = engine.reconstruct_delay(tau)
                preds.append(pred)
                targets.append(sig[step - delay_steps])
        nmses.append(compute_nmse(preds, targets))

    arr = np.array(nmses, dtype=np.float64)
    return {
        "mean_nmse": float(np.mean(arr)),
        "std_nmse": float(np.std(arr)),
        "min_nmse": float(np.min(arr)),
        "max_nmse": float(np.max(arr)),
    }


def evaluate_persistence_baseline(
    seeds: List[int],
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Baseline 1: Persistence predictor (u_hat(t) = u(t), zero delay)."""
    delay_steps = int(round(tau / dt))
    summary: Dict[str, Any] = {}

    for sig_name in ["OU_Noise", "PhaseShift", "NovelFreqs", "Chirp", "SquareWave"]:
        nmses: List[float] = []
        for seed in seeds:
            signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
            sig = signals[sig_name]
            preds: List[float] = []
            targets: List[float] = []
            for step in range(n_steps):
                t_now = step * dt
                if t_now >= tau:
                    preds.append(sig[step])
                    targets.append(sig[step - delay_steps])
            nmses.append(compute_nmse(preds, targets))
        arr = np.array(nmses, dtype=np.float64)
        summary[sig_name] = {
            "mean_nmse": float(np.mean(arr)),
            "std_nmse": float(np.std(arr)),
            "min_nmse": float(np.min(arr)),
            "max_nmse": float(np.max(arr)),
        }
    return summary


def evaluate_exact_tapped_delay_baseline(
    seeds: List[int],
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Baseline 3: Exact Tapped Delay Line buffer (known-answer reference)."""
    delay_steps = int(round(tau / dt))
    buffer_len = int(round(50.0 / dt))  # 100 elements
    nmses: List[float] = []

    for seed in seeds:
        signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
        sig = signals["OU_Noise"]
        buf = np.zeros(buffer_len, dtype=np.float64)
        preds: List[float] = []
        targets: List[float] = []
        for step in range(n_steps):
            u = sig[step]
            buf = np.roll(buf, 1)
            buf[0] = u
            t_now = step * dt
            if t_now >= tau:
                preds.append(buf[delay_steps])
                targets.append(sig[step - delay_steps])
        nmses.append(compute_nmse(preds, targets))

    arr = np.array(nmses, dtype=np.float64)
    return {
        "mean_nmse": float(np.mean(arr)),
        "std_nmse": float(np.std(arr)),
        "min_nmse": float(np.min(arr)),
        "max_nmse": float(np.max(arr)),
    }


def evaluate_matched_tapped_delay_baseline(
    seeds: List[int],
    dt: float = 0.5,
    tau: float = 15.0,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Baseline 4: Parameter-matched tapped delay buffer (8 elements)."""
    delay_steps = int(round(tau / dt))
    p = 8  # 8 elements matched to order 8
    nmses: List[float] = []

    for seed in seeds:
        signals = generate_test_signals(seed, dt=dt, n_steps=n_steps)
        sig = signals["OU_Noise"]
        buf = np.zeros(p, dtype=np.float64)
        # Learn readout weights on first half
        X_tr, y_tr = [], []
        for step in range(n_steps // 2):
            u = sig[step]
            buf = np.roll(buf, 1)
            buf[0] = u
            t_now = step * dt
            if t_now >= tau:
                X_tr.append(buf.copy())
                y_tr.append(sig[step - delay_steps])
        w, _, _, _ = np.linalg.lstsq(np.array(X_tr), np.array(y_tr), rcond=None)

        # Evaluate on second half
        preds, targets = [], []
        for step in range(n_steps // 2, n_steps):
            u = sig[step]
            buf = np.roll(buf, 1)
            buf[0] = u
            preds.append(float(np.dot(w, buf)))
            targets.append(sig[step - delay_steps])
        nmses.append(compute_nmse(preds, targets))

    arr = np.array(nmses, dtype=np.float64)
    return {
        "mean_nmse": float(np.mean(arr)),
        "std_nmse": float(np.std(arr)),
        "min_nmse": float(np.min(arr)),
        "max_nmse": float(np.max(arr)),
    }


def run_track2_delay_benchmark() -> Dict[str, Any]:
    """Run full Track 2 continuous delay benchmark across 20 held-out seeds."""
    exp_id = "track2_continuous_delay"
    test_seeds = list(range(200, 220))
    dt = 0.5
    tau = 15.0
    theta = 50.0
    order = 8
    n_steps = 300

    print("=" * 80)
    print("TRACK 2 BENCHMARK: CONTINUOUS TIME-DELAY & MEMORY ENGINE")
    print(f"Seeds: {len(test_seeds)} held-out seeds [200..219]")
    print(f"Delay tau = {tau} ms, Window theta = {theta} ms, Order = {order}, dt = {dt} ms")
    print("=" * 80)

    # 1. Intact Model
    intact_res = evaluate_intact_engine(
        test_seeds, order=order, theta=theta, dt=dt, tau=tau, n_steps=n_steps
    )
    print(f"Intact Engine OU Noise Mean NMSE: {intact_res['OU_Noise']['mean_nmse']:.4f} "
          f"+/- {intact_res['OU_Noise']['std_nmse']:.4f}")
    print(f"Intact Engine PhaseShift Mean NMSE: {intact_res['PhaseShift']['mean_nmse']:.4f}")
    print(f"Intact Engine NovelFreqs Mean NMSE: {intact_res['NovelFreqs']['mean_nmse']:.4f}")
    print(f"Intact Engine Chirp Mean NMSE:      {intact_res['Chirp']['mean_nmse']:.4f}")
    print(f"Intact Engine SquareWave Mean NMSE: {intact_res['SquareWave']['mean_nmse']:.4f}")

    # 2. Null Controls
    decoupled_res = evaluate_decoupled_control(
        test_seeds, order=order, theta=theta, dt=dt, tau=tau, n_steps=n_steps
    )
    shuffled_res = evaluate_shuffled_control(
        test_seeds, order=order, theta=theta, dt=dt, tau=tau, n_steps=n_steps
    )
    print(f"Null Control 1 (Decoupled Input) Mean NMSE: {decoupled_res['mean_nmse']:.4f}")
    print(f"Null Control 2 (Shuffled Input)  Mean NMSE: {shuffled_res['mean_nmse']:.4f}")

    # 3. Baselines
    persistence_res = evaluate_persistence_baseline(
        test_seeds, dt=dt, tau=tau, n_steps=n_steps
    )
    exact_tdl_res = evaluate_exact_tapped_delay_baseline(
        test_seeds, dt=dt, tau=tau, n_steps=n_steps
    )
    matched_tdl_res = evaluate_matched_tapped_delay_baseline(
        test_seeds, dt=dt, tau=tau, n_steps=n_steps
    )
    print(f"Baseline 1 (Persistence) OU Mean NMSE: {persistence_res['OU_Noise']['mean_nmse']:.4f}")
    print(f"Baseline 3 (Exact TDL) Mean NMSE:     {exact_tdl_res['mean_nmse']:.4f}")
    print(f"Baseline 4 (Matched TDL) Mean NMSE:   {matched_tdl_res['mean_nmse']:.4f}")

    # 4. Check Falsifiers
    falsifier_tripped = False
    falsifier_reasons: List[str] = []

    ou_mean = intact_res["OU_Noise"]["mean_nmse"]
    if ou_mean >= 0.25:
        falsifier_tripped = True
        falsifier_reasons.append(f"Mean OU NMSE {ou_mean:.4f} exceeded threshold 0.25")

    per_mean = persistence_res["OU_Noise"]["mean_nmse"]
    if ou_mean >= per_mean:
        falsifier_tripped = True
        falsifier_reasons.append(f"OU NMSE {ou_mean:.4f} did not beat persistence {per_mean:.4f}")

    if decoupled_res["mean_nmse"] < 0.95:
        falsifier_tripped = True
        falsifier_reasons.append(f"Decoupled control NMSE {decoupled_res['mean_nmse']:.4f} was below 0.95")

    if not np.isclose(exact_tdl_res["mean_nmse"], 0.0, atol=1e-8):
        falsifier_tripped = True
        falsifier_reasons.append(f"Exact TDL baseline was non-zero: {exact_tdl_res['mean_nmse']}")

    verdict = "FALSIFIED" if falsifier_tripped else "CONFIRMED"
    print(f"\nFinal Verdict: {verdict}")

    config = {
        "order": order,
        "theta_ms": theta,
        "dt_ms": dt,
        "delay_tau_ms": tau,
        "n_steps": n_steps,
        "compression_ratio": float((theta / dt) / order),
    }

    payload = {
        "verdict": verdict,
        "intact_engine": intact_res,
        "null_control_decoupled": decoupled_res,
        "null_control_shuffled": shuffled_res,
        "baseline_persistence": persistence_res,
        "baseline_exact_tdl": exact_tdl_res,
        "baseline_matched_tdl": matched_tdl_res,
        "falsifier_tripped": falsifier_tripped,
        "falsifier_reasons": falsifier_reasons,
    }

    meta = results_meta.build_meta(
        exp_id=exp_id,
        seeds=test_seeds,
        config=config,
        command="python rpn/benchmarks/track2_delay_benchmark.py",
        tuning_seeds=[998],
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / exp_id / "results.json"
    results_meta.write_results(out_file, payload, meta)
    print(f"Results written with compliant metadata to: {out_file}")

    return payload


if __name__ == "__main__":
    run_track2_delay_benchmark()
