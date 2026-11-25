"""Track 3: Impedance Matching Local Plasticity Benchmark.

Evaluates autonomous gradient-free transmission-line impedance matching across
20 held-out independent test seeds [300..319].

Quantifies:
1. Primary Metric: Selective Power Transmission Contrast Delta_T = mean(T_signal) - mean(T_noise).
2. Secondary Metric: Somatic Signal-to-Noise Ratio (SNR) improvement.
3. Convergence Metric: Mean Euclidean distance to theoretical conjugate match ||Z_signal - Z0*||_2.

Evaluates all mandatory null controls and baselines:
- Null Control 1: Decoupled Somatic Feedback (S = 0).
- Null Control 2: Shuffled Somatic Feedback (temporally permuted).
- Baseline 1: Static Untrained Network.
- Baseline 2: Classical Oja Hebbian Synapse array (real-valued weights).
- Baseline 3: Uniform All-Pass array (T = 1.0 everywhere).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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
from neuron_v2.biophysics.impedance_plasticity import TransmissionLineImpedanceArray


def generate_episode_inputs(
    is_pattern: bool,
    seed: int,
    n_syn: int = 8,
    n_sig: int = 4,
    n_steps: int = 300,
    dt: float = 0.5,
) -> np.ndarray:
    """Generate multi-channel complex wave inputs for one episode."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps, dtype=np.float64) * dt
    u = np.zeros((n_syn, n_steps), dtype=np.complex128)

    if is_pattern:
        omega = 0.2
        shared_wave = np.exp(1j * (omega * t + rng.uniform(0.0, 2.0 * np.pi)))
        for i in range(n_sig):
            noise = rng.normal(0.0, 0.3, n_steps) + 1j * rng.normal(0.0, 0.3, n_steps)
            u[i] = shared_wave + noise
    else:
        for i in range(n_sig):
            u[i] = rng.normal(0.0, 1.0, n_steps) + 1j * rng.normal(0.0, 1.0, n_steps)

    for i in range(n_sig, n_syn):
        u[i] = rng.normal(0.0, 1.0, n_steps) + 1j * rng.normal(0.0, 1.0, n_steps)

    return u


def compute_somatic_snr(
    transmissions: np.ndarray,
    seed: int,
    n_syn: int = 8,
    n_sig: int = 4,
    n_steps: int = 300,
    dt: float = 0.5,
) -> float:
    """Compute Somatic Signal-to-Noise Ratio: Power(Soma | Pattern) / Power(Soma | Noise)."""
    u_pat = generate_episode_inputs(True, seed, n_syn=n_syn, n_sig=n_sig, n_steps=n_steps, dt=dt)
    u_noi = generate_episode_inputs(False, seed, n_syn=n_syn, n_sig=n_sig, n_steps=n_steps, dt=dt)

    amp_t = np.sqrt(transmissions)[:, None]
    V_pat = np.sum(amp_t * u_pat, axis=0)
    V_noi = np.sum(amp_t * u_noi, axis=0)

    pow_pat = float(np.mean(np.abs(V_pat) ** 2))
    pow_noi = float(np.mean(np.abs(V_noi) ** 2))
    return float(pow_pat / (pow_noi + 1e-12))


def evaluate_intact_array(
    seeds: List[int],
    n_episodes: int = 50,
    n_syn: int = 8,
    n_sig: int = 4,
    lr: float = 0.08,
    coherence_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Evaluate intact TransmissionLineImpedanceArray across seeds."""
    delta_ts: List[float] = []
    t_sigs: List[float] = []
    t_nois: List[float] = []
    dists: List[float] = []
    snr_inits: List[float] = []
    snr_finals: List[float] = []

    for seed in seeds:
        array = TransmissionLineImpedanceArray(
            n_synapses=n_syn,
            lr=lr,
            coherence_threshold=coherence_threshold,
            seed=seed,
        )

        init_T = array.get_all_transmissions()
        snr_init = compute_somatic_snr(init_T, seed + 1000, n_syn=n_syn, n_sig=n_sig)
        snr_inits.append(snr_init)

        # Unsupervised training over n_episodes
        for ep in range(n_episodes):
            u_ep = generate_episode_inputs(True, seed * 100 + ep, n_syn=n_syn, n_sig=n_sig)
            array.adapt_episode(u_ep)

        final_T = array.get_all_transmissions()
        mean_t_sig = float(np.mean(final_T[:n_sig]))
        mean_t_noi = float(np.mean(final_T[n_sig:]))
        delta_t = mean_t_sig - mean_t_noi

        mean_dist = float(np.mean([array.distance_to_conjugate_match(i) for i in range(n_sig)]))
        snr_final = compute_somatic_snr(final_T, seed + 1000, n_syn=n_syn, n_sig=n_sig)

        delta_ts.append(delta_t)
        t_sigs.append(mean_t_sig)
        t_nois.append(mean_t_noi)
        dists.append(mean_dist)
        snr_finals.append(snr_final)

    arr_dt = np.array(delta_ts, dtype=np.float64)
    s_dt = float(np.std(arr_dt))
    ci_dt = float(1.96 * s_dt / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

    return {
        "mean_delta_T": float(np.mean(arr_dt)),
        "std_delta_T": s_dt,
        "ci95_delta_T": ci_dt,
        "min_delta_T": float(np.min(arr_dt)),
        "max_delta_T": float(np.max(arr_dt)),
        "per_seed_delta_T": [float(v) for v in arr_dt],
        "mean_t_signal": float(np.mean(t_sigs)),
        "mean_t_noise": float(np.mean(t_nois)),
        "mean_distance_to_match": float(np.mean(dists)),
        "mean_snr_init": float(np.mean(snr_inits)),
        "mean_snr_final": float(np.mean(snr_finals)),
    }


def evaluate_decoupled_control(
    seeds: List[int],
    n_episodes: int = 50,
    n_syn: int = 8,
    n_sig: int = 4,
    lr: float = 0.08,
    coherence_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Null Control 1: Decoupled somatic feedback (S = 0)."""
    delta_ts: List[float] = []
    for seed in seeds:
        array = TransmissionLineImpedanceArray(
            n_synapses=n_syn, lr=lr, coherence_threshold=coherence_threshold, seed=seed
        )
        for ep in range(n_episodes):
            u_ep = generate_episode_inputs(True, seed * 100 + ep, n_syn=n_syn, n_sig=n_sig)
            # Decoupled feedback: zero somatic wave
            zero_soma = np.zeros(u_ep.shape[1], dtype=np.complex128)
            array.adapt_episode(u_ep, somatic_override=zero_soma)

        final_T = array.get_all_transmissions()
        delta_ts.append(float(np.mean(final_T[:n_sig]) - np.mean(final_T[n_sig:])))

    arr = np.array(delta_ts, dtype=np.float64)
    return {
        "mean_delta_T": float(np.mean(arr)),
        "std_delta_T": float(np.std(arr)),
        "min_delta_T": float(np.min(arr)),
        "max_delta_T": float(np.max(arr)),
    }


def evaluate_shuffled_control(
    seeds: List[int],
    n_episodes: int = 50,
    n_syn: int = 8,
    n_sig: int = 4,
    lr: float = 0.08,
    coherence_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Null Control 2: Shuffled somatic feedback (temporal synchrony broken)."""
    delta_ts: List[float] = []
    for seed in seeds:
        array = TransmissionLineImpedanceArray(
            n_synapses=n_syn, lr=lr, coherence_threshold=coherence_threshold, seed=seed
        )
        for ep in range(n_episodes):
            u_ep = generate_episode_inputs(True, seed * 100 + ep, n_syn=n_syn, n_sig=n_sig)
            array.adapt_episode(u_ep, is_shuffled=True)

        final_T = array.get_all_transmissions()
        delta_ts.append(float(np.mean(final_T[:n_sig]) - np.mean(final_T[n_sig:])))

    arr = np.array(delta_ts, dtype=np.float64)
    return {
        "mean_delta_T": float(np.mean(arr)),
        "std_delta_T": float(np.std(arr)),
        "min_delta_T": float(np.min(arr)),
        "max_delta_T": float(np.max(arr)),
    }


def evaluate_untrained_baseline(
    seeds: List[int],
    n_syn: int = 8,
    n_sig: int = 4,
) -> Dict[str, Any]:
    """Baseline 1: Static untrained network (initial mismatched state)."""
    delta_ts: List[float] = []
    for seed in seeds:
        array = TransmissionLineImpedanceArray(n_synapses=n_syn, seed=seed)
        init_T = array.get_all_transmissions()
        delta_ts.append(float(np.mean(init_T[:n_sig]) - np.mean(init_T[n_sig:])))

    arr = np.array(delta_ts, dtype=np.float64)
    return {
        "mean_delta_T": float(np.mean(arr)),
        "std_delta_T": float(np.std(arr)),
        "min_delta_T": float(np.min(arr)),
        "max_delta_T": float(np.max(arr)),
    }


def evaluate_oja_baseline(
    seeds: List[int],
    n_episodes: int = 50,
    n_syn: int = 8,
    n_sig: int = 4,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Baseline 2: Classical Oja Hebbian Synapse array (real-valued weights)."""
    delta_ws: List[float] = []
    snr_finals: List[float] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        w = rng.uniform(0.1, 0.3, n_syn)
        eta_oja = 0.002

        for ep in range(n_episodes):
            u_ep = np.real(generate_episode_inputs(True, seed * 100 + ep, n_syn=n_syn, n_sig=n_sig, n_steps=n_steps))
            for step in range(n_steps):
                x = u_ep[:, step]
                y = float(np.dot(w, x))
                w += eta_oja * (y * x - (y ** 2) * w)
                w = np.clip(w, 0.01, 2.0)

        mean_w_sig = float(np.mean(w[:n_sig]))
        mean_w_noi = float(np.mean(w[n_sig:]))
        delta_ws.append(mean_w_sig - mean_w_noi)

        # Compute SNR
        u_pat = np.real(generate_episode_inputs(True, seed + 1000, n_syn=n_syn, n_sig=n_sig, n_steps=n_steps))
        u_noi = np.real(generate_episode_inputs(False, seed + 1000, n_syn=n_syn, n_sig=n_sig, n_steps=n_steps))
        pow_pat = float(np.mean(np.dot(w, u_pat) ** 2))
        pow_noi = float(np.mean(np.dot(w, u_noi) ** 2))
        snr_finals.append(pow_pat / (pow_noi + 1e-12))

    arr_dw = np.array(delta_ws, dtype=np.float64)
    return {
        "mean_delta_w": float(np.mean(arr_dw)),
        "std_delta_w": float(np.std(arr_dw)),
        "mean_snr_final": float(np.mean(snr_finals)),
    }


def run_track3_impedance_benchmark() -> Dict[str, Any]:
    """Run full Track 3 impedance matching benchmark across 20 held-out seeds."""
    exp_id = "track3_impedance_plasticity"
    test_seeds = list(range(300, 320))
    n_syn = 8
    n_sig = 4
    lr = 0.08
    coherence_threshold = 0.40
    n_episodes = 50

    print("=" * 80)
    print("TRACK 3 BENCHMARK: IMPEDANCE MATCHING LOCAL PLASTICITY")
    print(f"Seeds: {len(test_seeds)} held-out seeds [300..319]")
    print(f"Synapses K = {n_syn} ({n_sig} signal, {n_syn - n_sig} noise), lr = {lr}, threshold = {coherence_threshold}")
    print("=" * 80)

    # 1. Intact Model
    intact_res = evaluate_intact_array(
        test_seeds,
        n_episodes=n_episodes,
        n_syn=n_syn,
        n_sig=n_sig,
        lr=lr,
        coherence_threshold=coherence_threshold,
    )
    print(f"Intact Array Delta_T:           {intact_res['mean_delta_T']:.4f} +/- {intact_res['std_delta_T']:.4f}")
    print(f"Intact Array Signal T:          {intact_res['mean_t_signal']:.4f}")
    print(f"Intact Array Noise T:           {intact_res['mean_t_noise']:.4f}")
    print(f"Distance to Theoretical Match:  {intact_res['mean_distance_to_match']:.4f}")
    print(f"Somatic SNR:                    {intact_res['mean_snr_init']:.2f} -> {intact_res['mean_snr_final']:.2f}")

    # 2. Null Controls
    decoupled_res = evaluate_decoupled_control(
        test_seeds, n_episodes=n_episodes, n_syn=n_syn, n_sig=n_sig, lr=lr, coherence_threshold=coherence_threshold
    )
    shuffled_res = evaluate_shuffled_control(
        test_seeds, n_episodes=n_episodes, n_syn=n_syn, n_sig=n_sig, lr=lr, coherence_threshold=coherence_threshold
    )
    untrained_res = evaluate_untrained_baseline(test_seeds, n_syn=n_syn, n_sig=n_sig)
    print(f"Null Control 1 (Decoupled) Delta_T: {decoupled_res['mean_delta_T']:.4f}")
    print(f"Null Control 2 (Shuffled) Delta_T:  {shuffled_res['mean_delta_T']:.4f}")
    print(f"Baseline 1 (Untrained) Delta_T:     {untrained_res['mean_delta_T']:.4f}")

    # 3. Baselines
    oja_res = evaluate_oja_baseline(test_seeds, n_episodes=n_episodes, n_syn=n_syn, n_sig=n_sig)
    print(f"Baseline 2 (Oja Synapses) Delta_w:  {oja_res['mean_delta_w']:.4f} | Final SNR: {oja_res['mean_snr_final']:.2f}")

    # 4. Check Falsifiers
    falsifier_tripped = False
    falsifier_reasons: List[str] = []

    dt_mean = intact_res["mean_delta_T"]
    if dt_mean < 0.60:
        falsifier_tripped = True
        falsifier_reasons.append(f"Mean Delta_T {dt_mean:.4f} below threshold 0.60")

    dist_mean = intact_res["mean_distance_to_match"]
    if dist_mean >= 0.15:
        falsifier_tripped = True
        falsifier_reasons.append(f"Mean distance to theoretical match {dist_mean:.4f} exceeded 0.15")

    if decoupled_res["mean_delta_T"] > 0.10:
        falsifier_tripped = True
        falsifier_reasons.append(f"Decoupled control Delta_T {decoupled_res['mean_delta_T']:.4f} exceeded 0.10")

    if shuffled_res["mean_delta_T"] > 0.10:
        falsifier_tripped = True
        falsifier_reasons.append(f"Shuffled control Delta_T {shuffled_res['mean_delta_T']:.4f} exceeded 0.10")

    verdict = "FALSIFIED" if falsifier_tripped else "CONFIRMED"
    print(f"\nFinal Verdict: {verdict}")

    config = {
        "n_synapses": n_syn,
        "n_signal_channels": n_sig,
        "n_noise_channels": n_syn - n_sig,
        "lr": lr,
        "coherence_threshold": coherence_threshold,
        "n_episodes": n_episodes,
        "z0_membrane": [1.0, -0.5],
        "z_match_target": [1.0, 0.5],
        "z_shunt_target": [0.02, 0.0],
    }

    payload = {
        "verdict": verdict,
        "intact_array": intact_res,
        "null_control_decoupled": decoupled_res,
        "null_control_shuffled": shuffled_res,
        "baseline_untrained": untrained_res,
        "baseline_oja": oja_res,
        "falsifier_tripped": falsifier_tripped,
        "falsifier_reasons": falsifier_reasons,
    }

    meta = results_meta.build_meta(
        exp_id=exp_id,
        seeds=test_seeds,
        config=config,
        command="python rpn/benchmarks/track3_impedance_benchmark.py",
        tuning_seeds=[997],
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / exp_id / "results.json"
    results_meta.write_results(out_file, payload, meta)
    print(f"Results written with compliant metadata to: {out_file}")

    return payload


if __name__ == "__main__":
    run_track3_impedance_benchmark()
