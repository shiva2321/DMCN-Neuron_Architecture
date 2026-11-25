"""Universal Domain-Agnostic CAPN Benchmark (Claim B-44).

Evaluates the canonical UniversalCAPNColumn across 20 independent test seeds [800..819]
on 3 disjoint heterogeneous modalities:
1. Domain 1: Continuous 3D Embodied Navigation (obstacle avoidance via motor attractor).
2. Domain 2: Chaotic Non-Linear Lorenz Attractor Online State Prediction (NMSE vs persistence).
3. Domain 3: Streaming Multi-Channel Signal Anomaly Detection & STC Structural Plasticity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repo root and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.capn.universal_column import UniversalCAPNColumn, PlasticTransitionModel
from neuron_v2.capn.sensorimotor_arena import (
    Continuous3DArenaEnvironment,
    Blind3DSensorimotorAgent,
    ReactiveGreedy3DAgent,
)
from neuron_v2.capn.structural_plasticity import (
    StructuralPlasticityNetwork,
    TimerBasedBaselineNetwork,
)


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * sd / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95": ci95,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def evaluate_domain1_navigation(seed: int) -> Dict[str, Any]:
    """Domain 1: 3D Embodied Navigation with Universal CAPN Column."""
    env = Continuous3DArenaEnvironment(max_steps=220)
    col = UniversalCAPNColumn(
        d_sensory=14,
        d_state=14,
        d_action=3,
        n_action_basis=14,
        dt=0.1,
        eta_dyn=0.002,
        use_efference_prior=True,
        clip_range=(0.0, 5.0),
        rng=np.random.RandomState(seed),
    )

    obs = env.reset(seed=seed)
    col.reset()
    done = False
    reached = False

    def hazard_eval(sim_z: np.ndarray, h: int) -> float:
        min_val = float(np.min(sim_z))
        if min_val < 0.5:
            penetration = 0.5 - min_val
            return 200.0 * (1.0 / h) + 300.0 * (penetration ** 2)
        return 0.0

    while not done:
        dists = obs["distances"]
        dir_goal = obs["dir_to_goal"]
        z_curr = col.encode_state(dists)

        # Sensory hazard profile: non-linear proximity braking
        hazard_profile = np.where(dists < 1.8, ((1.8 - dists) / 1.8) ** 2, 0.0)

        act = col.select_action(
            dir_goal=dir_goal,
            sensory_hazard_profile=hazard_profile,
            z_curr=z_curr,
            hazard_eval_fn=hazard_eval,
            horizon=15,
        )

        obs, reward, done, reached, info = env.step(act)
        z_next = col.encode_state(obs["distances"])
        col.step_learn_transition(z_next)

    traj = np.array(env.trajectory)
    diffs = np.diff(traj, axis=0)
    traj_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    straight_line_dist = float(np.linalg.norm(env.goal - env.start))
    path_eff = float(straight_line_dist / traj_len) if traj_len > 0.1 else 0.0

    # Null Control 1: Blind agent (distances zeroed)
    blind_agent = Blind3DSensorimotorAgent(rng=np.random.RandomState(seed))
    obs_b = env.reset(seed=seed)
    blind_agent.reset()
    done_b = False
    reached_b = False
    while not done_b:
        act_b = blind_agent.select_action(obs_b)
        obs_b, _, done_b, reached_b, info_b = env.step(act_b)

    # Null Control 2: Reactive greedy agent (no forward rollouts)
    reactive_agent = ReactiveGreedy3DAgent(rng=np.random.RandomState(seed))
    obs_r = env.reset(seed=seed)
    reactive_agent.reset()
    done_r = False
    reached_r = False
    while not done_r:
        act_r = reactive_agent.select_action(obs_r)
        obs_r, _, done_r, reached_r, info_r = env.step(act_r)

    return {
        "reached": float(reached),
        "collisions": info["collision_count"],
        "steps": info["steps"],
        "path_efficiency": path_eff,
        "blind_reached": float(reached_b),
        "blind_collisions": info_b["collision_count"],
        "reactive_reached": float(reached_r),
        "reactive_collisions": info_r["collision_count"],
    }


def evaluate_domain2_lorenz(seed: int) -> Dict[str, Any]:
    """Domain 2: Chaotic Lorenz Attractor Non-Linear State Prediction."""
    rng = np.random.RandomState(seed)
    col = UniversalCAPNColumn(
        d_sensory=3,
        d_state=3,
        d_action=0,
        n_action_basis=0,
        eta_dyn=0.45,
        use_efference_prior=False,
        rng=rng,
    )
    col.dyn_model = PlasticTransitionModel(
        d_state=3,
        d_action=0,
        d_hid=64,
        eta=0.45,
        sigma_mod=0.01,
        B_prior=None,
        rng=np.random.RandomState(seed + 1000),
    )

    # Decoupled control (constant surprise M(t)=1.0, ablation of third factor)
    rng_ctl = np.random.RandomState(seed + 1000)
    model_decoupled = PlasticTransitionModel(
        d_state=3,
        d_action=0,
        d_hid=64,
        eta=0.45,
        sigma_mod=0.01,
        B_prior=None,
        rng=rng_ctl,
    )

    # Generate chaotic Lorenz trajectory
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt_lorenz = 0.01

    state = np.array([1.0, 1.0, 1.0]) + rng.uniform(-0.1, 0.1, size=3)
    lorenz_states = []
    for _ in range(600):
        dx = sigma * (state[1] - state[0]) * dt_lorenz
        dy = (state[0] * (rho - state[2]) - state[1]) * dt_lorenz
        dz = (state[0] * state[1] - beta * state[2]) * dt_lorenz
        state = state + np.array([dx, dy, dz])
        lorenz_states.append(state.copy())

    lorenz_states = np.array(lorenz_states)
    lorenz_norm = lorenz_states / (np.std(lorenz_states, axis=0) + 1e-6)

    pred_errors = []
    persistence_errors = []
    decoupled_errors = []
    deltas = []

    for t in range(len(lorenz_norm) - 1):
        s_curr = lorenz_norm[t]
        s_next = lorenz_norm[t + 1]
        delta_true = s_next - s_curr
        deltas.append(delta_true)

        # Forward predictions of delta before update
        _, delta_hat = col.dyn_model.forward(s_curr, np.zeros(0))
        _, delta_dec = model_decoupled.forward(s_curr, np.zeros(0))

        err_pred = float(np.sum((delta_true - delta_hat) ** 2))
        err_persist = float(np.sum(delta_true ** 2))
        err_dec = float(np.sum((delta_true - delta_dec) ** 2))

        # Intact model updates with three-factor gating
        col.dyn_model.learn_step(s_curr, np.zeros(0), s_next)

        # Decoupled control updates with constant M(t) = 1.0
        v = s_curr
        h = np.tanh(model_decoupled.W1 @ v + model_decoupled.b1)
        delta_z = np.clip(model_decoupled.W2 @ h + model_decoupled.b2, -0.4, 0.4)
        eps_z = delta_true - delta_z
        eps_norm = float(np.linalg.norm(eps_z))
        if eps_norm > 0.4:
            eps_z = (eps_z / eps_norm) * 0.4
        dh = 1.0 - h ** 2
        eps_h = dh * (model_decoupled.B @ eps_z)
        scale_v = 1.0 / np.sqrt(1.0 + float(np.sum(s_curr ** 2)))
        scale_h = 1.0 / np.sqrt(1.0 + float(np.sum(h ** 2)))
        decay = 0.0005
        model_decoupled.W2 += model_decoupled.eta * np.outer(eps_z, h) * 1.0 * scale_h - decay * model_decoupled.W2
        model_decoupled.b2 += model_decoupled.eta * eps_z * 1.0 * 0.5
        model_decoupled.W1 += model_decoupled.eta * np.outer(eps_h, v) * 1.0 * scale_v - decay * model_decoupled.W1
        model_decoupled.b1 += model_decoupled.eta * eps_h * 1.0 * 0.5

        if t > 200:  # After initial adaptation period
            pred_errors.append(err_pred)
            persistence_errors.append(err_persist)
            decoupled_errors.append(err_dec)

    delta_var = float(np.var(deltas[200:]) * 3.0) + 1e-9
    nmse_capn = float(np.mean(pred_errors)) / delta_var
    nmse_persist = float(np.mean(persistence_errors)) / delta_var
    nmse_decoupled = float(np.mean(decoupled_errors)) / delta_var

    return {
        "nmse_capn": nmse_capn,
        "nmse_persistence": nmse_persist,
        "nmse_decoupled": nmse_decoupled,
        "ratio_improvement": float(nmse_persist / (nmse_capn + 1e-9)),
    }


def evaluate_domain3_temporal_shift(seed: int) -> Dict[str, Any]:
    """Domain 3: Multi-Channel Signal Anomaly Detection & STC Structural Plasticity."""
    rng = np.random.RandomState(seed)
    d_in, d_out = 8, 4

    # 1. Stationary stream (300 steps)
    W_stat = rng.randn(d_out, d_in) * 0.5
    X_stat = rng.randn(300, d_in)
    Y_stat = X_stat @ W_stat.T + rng.normal(0, 0.05, size=(300, d_out))

    # 2. White noise stream (300 steps)
    X_noise = rng.randn(300, d_in)
    Y_noise = rng.randn(300, d_out) * 1.0

    # 3. Systematic distribution shift stream (600 steps)
    X_shift = np.zeros((600, d_in))
    Y_shift = np.zeros((600, d_out))
    for k in range(3):
        start = k * 200
        end = (k + 1) * 200
        w_k = rng.randn(d_out, d_in) * (0.8 + 0.3 * k)
        X_shift[start:end] = rng.randn(200, d_in) + (k * 2.0)
        Y_shift[start:end] = np.tanh(X_shift[start:end] @ w_k.T) + rng.normal(0, 0.05, size=(200, d_out))

    # Independent fresh instances for disjoint evaluation (no cross-stream contamination)
    net_stat = StructuralPlasticityNetwork(d_in=d_in, d_initial_hid=4, d_out=d_out, rng=np.random.RandomState(seed))
    res_stat = net_stat.process_stream(X_stat, Y_stat)

    net_noise = StructuralPlasticityNetwork(d_in=d_in, d_initial_hid=4, d_out=d_out, rng=np.random.RandomState(seed))
    res_noise = net_noise.process_stream(X_noise, Y_noise)

    net_shift = StructuralPlasticityNetwork(d_in=d_in, d_initial_hid=4, d_out=d_out, rng=np.random.RandomState(seed))
    res_shift = net_shift.process_stream(X_shift, Y_shift)

    # Periodic timer baseline on stationary stream
    timer_net = TimerBasedBaselineNetwork(d_in=d_in, d_initial_hid=4, d_out=d_out, spawn_period=50)
    res_timer = timer_net.process_stream(X_stat, Y_stat)

    return {
        "fp_stationary": res_stat["n_spawn"],
        "fp_noise": res_noise["n_spawn"],
        "tp_shift": res_shift["n_spawn"],
        "timer_spurious": res_timer["n_spawn"],
    }


def evaluate_single_seed(seed: int) -> Dict[str, Any]:
    """Evaluate Universal CAPN Column across all 3 heterogeneous domains on a single seed."""
    d1 = evaluate_domain1_navigation(seed)
    d2 = evaluate_domain2_lorenz(seed)
    d3 = evaluate_domain3_temporal_shift(seed)

    return {
        "seed": seed,
        "d1_reached": d1["reached"],
        "d1_collisions": d1["collisions"],
        "d1_steps": d1["steps"],
        "d1_path_efficiency": d1["path_efficiency"],
        "d1_blind_reached": d1["blind_reached"],
        "d1_blind_collisions": d1["blind_collisions"],
        "d1_reactive_reached": d1["reactive_reached"],
        "d1_reactive_collisions": d1["reactive_collisions"],
        "d2_nmse_capn": d2["nmse_capn"],
        "d2_nmse_persistence": d2["nmse_persistence"],
        "d2_nmse_decoupled": d2["nmse_decoupled"],
        "d2_ratio_improvement": d2["ratio_improvement"],
        "d3_fp_stationary": d3["fp_stationary"],
        "d3_fp_noise": d3["fp_noise"],
        "d3_tp_shift": d3["tp_shift"],
        "d3_timer_spurious": d3["timer_spurious"],
    }


def run_universal_benchmark() -> None:
    """Run frozen benchmark across 20 pre-registered test seeds [800..819]."""
    test_seeds = list(range(800, 820))
    n_seeds = len(test_seeds)

    per_seed_results: List[Dict[str, Any]] = []
    for seed in test_seeds:
        res = evaluate_single_seed(seed)
        per_seed_results.append(res)

    # Compute summary statistics
    d1_succ = compute_stats([r["d1_reached"] * 100.0 for r in per_seed_results])
    d1_coll = compute_stats([float(r["d1_collisions"]) for r in per_seed_results])
    d1_steps = compute_stats([float(r["d1_steps"]) for r in per_seed_results])
    d1_eff = compute_stats([r["d1_path_efficiency"] for r in per_seed_results])

    d1_blind_succ = compute_stats([r["d1_blind_reached"] * 100.0 for r in per_seed_results])
    d1_blind_coll = compute_stats([float(r["d1_blind_collisions"]) for r in per_seed_results])
    d1_reactive_succ = compute_stats([r["d1_reactive_reached"] * 100.0 for r in per_seed_results])
    d1_reactive_coll = compute_stats([float(r["d1_reactive_collisions"]) for r in per_seed_results])

    d2_nmse = compute_stats([r["d2_nmse_capn"] for r in per_seed_results])
    d2_persist = compute_stats([r["d2_nmse_persistence"] for r in per_seed_results])
    d2_dec = compute_stats([r["d2_nmse_decoupled"] for r in per_seed_results])
    d2_ratio = compute_stats([r["d2_ratio_improvement"] for r in per_seed_results])

    d3_fp_stat = compute_stats([float(r["d3_fp_stationary"]) for r in per_seed_results])
    d3_fp_noise = compute_stats([float(r["d3_fp_noise"]) for r in per_seed_results])
    d3_tp_shift = compute_stats([float(r["d3_tp_shift"]) for r in per_seed_results])
    d3_timer = compute_stats([float(r["d3_timer_spurious"]) for r in per_seed_results])

    exit_criterion_met = bool(
        (d1_succ["mean"] >= 95.0)
        and (d1_coll["mean"] == 0.0)
        and (d2_nmse["mean"] < 0.20)
        and (d2_ratio["mean"] > 3.0)
        and (d3_fp_stat["mean"] == 0.0)
        and (d3_fp_noise["mean"] == 0.0)
        and (d3_tp_shift["mean"] >= 2.0)
    )

    out_data = {
        "headline_metric": "universal_multi_domain_success",
        "direction": "higher_is_better",
        "n_seeds": n_seeds,
        "summary": {
            "domain1_3d_navigation": {
                "success_rate_pct": d1_succ,
                "collisions": d1_coll,
                "steps": d1_steps,
                "path_efficiency": d1_eff,
                "blind_control_success_pct": d1_blind_succ,
                "blind_control_collisions": d1_blind_coll,
                "reactive_control_success_pct": d1_reactive_succ,
                "reactive_control_collisions": d1_reactive_coll,
            },
            "domain2_chaotic_lorenz": {
                "nmse_capn": d2_nmse,
                "nmse_persistence": d2_persist,
                "nmse_decoupled_control": d2_dec,
                "ratio_improvement": d2_ratio,
            },
            "domain3_temporal_shift": {
                "fp_spawns_stationary": d3_fp_stat,
                "fp_spawns_white_noise": d3_fp_noise,
                "tp_spawns_shift": d3_tp_shift,
                "timer_spurious_spawns": d3_timer,
            },
            "exit_criterion_met": exit_criterion_met,
        },
        "per_seed": per_seed_results,
    }

    # Attach results metadata
    meta = results_meta.build_meta(
        exp_id="capn_universal_domain_agnostic",
        seeds=test_seeds,
        config={
            "n_seeds": n_seeds,
            "domains": ["3D_embodied_navigation", "chaotic_lorenz_tracking", "temporal_signal_shift"],
            "architecture": "UniversalCAPNColumn",
        },
        command="python rpn/benchmarks/capn_universal_benchmark.py",
        repo_root=REPO_ROOT,
    )
    out_data["_meta"] = meta

    out_dir = REPO_ROOT / "results" / "capn_universal_domain_agnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print("=" * 60)
    print("UNIVERSAL DOMAIN-AGNOSTIC CAPN BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Test Seeds: {n_seeds} ({test_seeds[0]}..{test_seeds[-1]})")
    print(f"Domain 1 (3D Navigation) Success: {d1_succ['mean']:.2f}% +/- {d1_succ['ci95']:.2f}% (Collisions: {d1_coll['mean']:.2f})")
    print(f"Domain 1 Path Efficiency: {d1_eff['mean']:.4f} +/- {d1_eff['ci95']:.4f} (Steps: {d1_steps['mean']:.2f})")
    print(f"Domain 1 Blind Control: {d1_blind_succ['mean']:.2f}% (Collisions: {d1_blind_coll['mean']:.2f})")
    print(f"Domain 1 Reactive Control: {d1_reactive_succ['mean']:.2f}% (Collisions: {d1_reactive_coll['mean']:.2f})")
    print(f"Domain 2 (Chaotic Lorenz) NMSE: {d2_nmse['mean']:.4f} +/- {d2_nmse['ci95']:.4f} vs Persistence: {d2_persist['mean']:.4f} ({d2_ratio['mean']:.2f}x better)")
    print(f"Domain 2 Decoupled Control NMSE: {d2_dec['mean']:.4f} +/- {d2_dec['ci95']:.4f}")
    print(f"Domain 3 (Signal Shift) FP Stationary: {d3_fp_stat['mean']:.4f}, FP Noise: {d3_fp_noise['mean']:.4f}")
    print(f"Domain 3 TP Shift Spawns: {d3_tp_shift['mean']:.2f} +/- {d3_tp_shift['ci95']:.2f} (Timer Baseline: {d3_timer['mean']:.2f})")
    print(f"Exit Criterion Met: {exit_criterion_met}")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run_universal_benchmark()
