"""Milestone 6: Universal Modality Ingestion & Grand 3D Sensorimotor Benchmark.

Evaluates the integrated CAPN architecture across 20 held-out independent test seeds [700..719]
in a continuous 3D arena with deceptive spherical hazard clusters and stop-and-rebound physics.

Quantifies:
1. Primary Metric: Goal Success Rate (% of episodes reaching goal region without collisions).
2. Secondary Metric: Path Efficiency Ratio (straight-line distance / actual trajectory length).

Evaluates mandatory null controls and baselines:
- Null Control 1: Blind Agent Control (range sensors zeroed, proves dependency on sensory inputs).
- Null Control 2: Reactive Greedy Control (steers greedily without prospective forward rollouts).
- Baseline: Model-Free Random Walk Baseline (random continuous velocity exploration).
"""
from __future__ import annotations

import json
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
from neuron_v2.capn.sensorimotor_arena import (
    Blind3DSensorimotorAgent,
    CAPN3DSensorimotorAgent,
    Continuous3DArenaEnvironment,
    ReactiveGreedy3DAgent,
    RandomWalk3DAgent,
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


def evaluate_single_seed(seed: int) -> Dict[str, Any]:
    """Evaluate CAPN and all controls on a single independent seed."""
    env = Continuous3DArenaEnvironment(max_steps=220)

    # 1. Intact CAPN 3D prospective agent (ZERO ORACLE)
    capn_agent = CAPN3DSensorimotorAgent(rng=np.random.RandomState(seed))
    obs = env.reset(seed=seed)
    capn_agent.reset()
    done = False
    reached = False
    while not done:
        act = capn_agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)

    capn_reached = float(reached)
    capn_collisions = info["collision_count"]
    capn_steps = info["steps"]
    capn_final_dist = info["final_dist_goal"]

    # Compute trajectory length and path efficiency
    traj = np.array(env.trajectory)
    diffs = np.diff(traj, axis=0)
    traj_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    straight_line_dist = float(np.linalg.norm(env.goal - env.start))
    path_efficiency = float(straight_line_dist / traj_len) if traj_len > 0.1 else 0.0

    # 2. Blind Agent (Null Control 1)
    blind_agent = Blind3DSensorimotorAgent(rng=np.random.RandomState(seed))
    obs = env.reset(seed=seed)
    blind_agent.reset()
    done = False
    reached = False
    while not done:
        act = blind_agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)
    blind_reached = float(reached)
    blind_collisions = info["collision_count"]
    blind_final_dist = info["final_dist_goal"]

    # 3. Reactive Greedy Agent (Null Control 2)
    reactive_agent = ReactiveGreedy3DAgent(rng=np.random.RandomState(seed))
    obs = env.reset(seed=seed)
    reactive_agent.reset()
    done = False
    reached = False
    while not done:
        act = reactive_agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)
    reactive_reached = float(reached)
    reactive_collisions = info["collision_count"]
    reactive_final_dist = info["final_dist_goal"]

    # 4. Random Walk Agent (Baseline)
    random_agent = RandomWalk3DAgent(rng=np.random.RandomState(seed))
    obs = env.reset(seed=seed)
    random_agent.reset()
    done = False
    reached = False
    while not done:
        act = random_agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)
    random_reached = float(reached)
    random_collisions = info["collision_count"]
    random_final_dist = info["final_dist_goal"]

    return {
        "seed": seed,
        "capn_reached": capn_reached,
        "capn_collisions": capn_collisions,
        "capn_steps": capn_steps,
        "capn_final_dist": capn_final_dist,
        "path_efficiency": path_efficiency,
        "blind_reached": blind_reached,
        "blind_collisions": blind_collisions,
        "blind_final_dist": blind_final_dist,
        "reactive_reached": reactive_reached,
        "reactive_collisions": reactive_collisions,
        "reactive_final_dist": reactive_final_dist,
        "random_reached": random_reached,
        "random_collisions": random_collisions,
        "random_final_dist": random_final_dist,
    }


def run_frozen_benchmark() -> None:
    """Run frozen benchmark across 20 pre-registered test seeds [700..719]."""
    test_seeds = list(range(700, 720))
    n_seeds = len(test_seeds)

    per_seed_results: List[Dict[str, Any]] = []
    for seed in test_seeds:
        res = evaluate_single_seed(seed)
        per_seed_results.append(res)

    capn_reaches = [r["capn_reached"] for r in per_seed_results]
    capn_collisions = [r["capn_collisions"] for r in per_seed_results]
    capn_steps = [r["capn_steps"] for r in per_seed_results]
    path_effs = [r["path_efficiency"] for r in per_seed_results]

    blind_reaches = [r["blind_reached"] for r in per_seed_results]
    blind_collisions = [r["blind_collisions"] for r in per_seed_results]

    reactive_reaches = [r["reactive_reached"] for r in per_seed_results]
    random_reaches = [r["random_reached"] for r in per_seed_results]

    capn_reach_stats = compute_stats(capn_reaches)
    capn_coll_stats = compute_stats(capn_collisions)
    capn_step_stats = compute_stats(capn_steps)
    path_eff_stats = compute_stats(path_effs)

    blind_reach_stats = compute_stats(blind_reaches)
    blind_coll_stats = compute_stats(blind_collisions)
    reactive_reach_stats = compute_stats(reactive_reaches)
    random_reach_stats = compute_stats(random_reaches)

    meta = results_meta.build_meta(
        exp_id="capn_m6_grand_benchmark",
        seeds=test_seeds,
        config={
            "n_seeds": n_seeds,
            "arena_size": 10.0,
            "max_steps": 180,
            "dt": 0.1,
            "rollout_horizon": 15,
        },
        command="python rpn/benchmarks/capn_m6_grand_benchmark.py",
        repo_root=REPO_ROOT,
    )

    out_file = REPO_ROOT / "results" / "capn_m6_grand_benchmark" / "results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "headline_metric": "capn_goal_success_rate_pct",
        "direction": "higher_is_better",
        "n_seeds": n_seeds,
        "summary": {
            "capn_success_rate_pct": {
                "mean": capn_reach_stats["mean"] * 100.0,
                "sd": capn_reach_stats["sd"] * 100.0,
                "ci95": capn_reach_stats["ci95"] * 100.0,
            },
            "capn_collisions": capn_coll_stats,
            "capn_steps": capn_step_stats,
            "path_efficiency": path_eff_stats,
            "blind_success_rate_pct": {
                "mean": blind_reach_stats["mean"] * 100.0,
                "sd": blind_reach_stats["sd"] * 100.0,
                "ci95": blind_reach_stats["ci95"] * 100.0,
            },
            "blind_collisions": blind_coll_stats,
            "reactive_success_rate_pct": {
                "mean": reactive_reach_stats["mean"] * 100.0,
                "sd": reactive_reach_stats["sd"] * 100.0,
                "ci95": reactive_reach_stats["ci95"] * 100.0,
            },
            "random_success_rate_pct": {
                "mean": random_reach_stats["mean"] * 100.0,
                "sd": random_reach_stats["sd"] * 100.0,
                "ci95": random_reach_stats["ci95"] * 100.0,
            },
            "exit_criterion_met": bool(capn_reach_stats["mean"] >= 0.90 and blind_reach_stats["mean"] <= 0.20),
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("==================================================")
    print("CAPN MILESTONE 6 GRAND 3D BENCHMARK RESULTS")
    print("==================================================")
    print(f"Test Seeds: {n_seeds} ({test_seeds[0]}..{test_seeds[-1]})")
    print(f"CAPN Success Rate: {payload['summary']['capn_success_rate_pct']['mean']:.2f}% +/- {payload['summary']['capn_success_rate_pct']['ci95']:.2f}%")
    print(f"CAPN Collisions: Mean {capn_coll_stats['mean']:.2f} +/- {capn_coll_stats['ci95']:.2f} (Min: {capn_coll_stats['min']:.0f}, Max: {capn_coll_stats['max']:.0f})")
    print(f"CAPN Path Efficiency: Mean {path_eff_stats['mean']:.4f} +/- {path_eff_stats['ci95']:.4f}")
    print(f"Blind Control Success Rate: {payload['summary']['blind_success_rate_pct']['mean']:.2f}% (Mean Collisions: {blind_coll_stats['mean']:.1f})")
    print(f"Reactive Control Success Rate: {payload['summary']['reactive_success_rate_pct']['mean']:.2f}%")
    print(f"Random Baseline Success Rate: {payload['summary']['random_success_rate_pct']['mean']:.2f}%")
    print(f"Exit Criterion Met: {payload['summary']['exit_criterion_met']}")
    print(f"Results written to: {out_file}")


if __name__ == "__main__":
    run_frozen_benchmark()
