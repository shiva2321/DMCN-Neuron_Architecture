"""Pre-registered 20-Seed Collective Swarm Intelligence Benchmark (Boids + DMCN).

Evaluates SwarmBoidNetwork across two regimes:
  Regime A: Extreme per-agent sensor noise (sigma=2.0) migrating target tracking.
  Regime B: Same noisy migrating target + unseen mid-flight crosswind disturbance (t >= 50).

Baselines & Null Controls (identical noise realizations per seed):
  - Swarm+DMCN (intact: Boids + DMCN spine traces + local proprioceptive slip plasticity)
  - Pure Reynolds Boids (self-propelled Separation + Alignment + Cohesion, no DMCN)
  - EMA-Filtered Boids (Reynolds Boids + classical per-agent exponential moving-average filter)
  - Swarm+DMCN (ablate_learning=True)
  - Swarm+DMCN (ablate_spine=True)
  - Decoupled Neighbors (16 independent DMCN agents, zero flocking rules)
  - Scrambled Neighbors (neighbor positions & velocities randomly permuted)
  - Single Isolated DMCN Agent
  - Blind Swarm (target bearing zeroed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import SwarmBoidNetwork, SwarmConfig  # noqa: E402


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
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


def run_regime(seed: int, with_wind: bool) -> Dict[str, float]:
    rng = np.random.RandomState(seed + (500 if with_wind else 0))
    n_boids = 16
    n_steps = 150
    wind_step = 50
    noise_sigma = 2.0
    dt = 0.2
    wind_vec = np.array([0.0, -1.1], dtype=np.float64) if with_wind else np.zeros(2, dtype=np.float64)

    targets = np.zeros((n_steps, 2), dtype=np.float64)
    for t in range(n_steps):
        targets[t] = np.array([0.22 * t, 3.5 * np.sin(0.05 * t)], dtype=np.float64)
    sensor_noise = rng.randn(n_steps, n_boids, 2) * noise_sigma

    def eval_condition(
        cfg: SwarmConfig,
        decouple: bool = False,
        scramble: bool = False,
        blind: bool = False,
    ) -> Tuple[float, float, float]:
        sw = SwarmBoidNetwork(cfg)
        errs, pols, cols = [], [], []
        for t in range(n_steps):
            diffs = targets[t][None, :] - sw.pos
            dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9
            noisy_b = (diffs / dists) + sensor_noise[t, : cfg.n_boids]
            w_t = wind_vec if (with_wind and t >= wind_step) else None
            info = sw.step(
                noisy_b,
                dt=dt,
                external_wind=w_t,
                decouple_neighbors=decouple,
                scramble_neighbors=scramble,
                blind=blind,
                learn=True,
            )
            if t >= n_steps - 50:
                errs.append(float(np.linalg.norm(info["centroid"] - targets[t])))
                pols.append(float(info["polarization"]))
            cols.append(float(info["n_collisions"]))
        return float(np.mean(errs)), float(np.mean(pols)), float(np.sum(cols))

    tag = "wind" if with_wind else "nowind"
    c_intact = SwarmConfig(n_boids=n_boids, use_dmcn_memory=True, seed=seed)
    c_pure = SwarmConfig(n_boids=n_boids, use_dmcn_memory=False, use_ema_filter=False, seed=seed)
    c_ema = SwarmConfig(n_boids=n_boids, use_dmcn_memory=False, use_ema_filter=True, ema_alpha=0.25, seed=seed)
    c_nolearn = SwarmConfig(n_boids=n_boids, use_dmcn_memory=True, ablate_learning=True, seed=seed)
    c_nospine = SwarmConfig(n_boids=n_boids, use_dmcn_memory=True, ablate_spine=True, seed=seed)
    c_single = SwarmConfig(n_boids=1, use_dmcn_memory=True, seed=seed)

    e_dmcn, p_dmcn, col_dmcn = eval_condition(c_intact)
    e_pure, p_pure, col_pure = eval_condition(c_pure)
    e_ema, p_ema, col_ema = eval_condition(c_ema)
    e_nol, _, _ = eval_condition(c_nolearn)
    e_nos, _, _ = eval_condition(c_nospine)
    e_dec, p_dec, _ = eval_condition(c_intact, decouple=True)
    e_scr, p_scr, _ = eval_condition(c_intact, scramble=True)
    e_bld, _, _ = eval_condition(c_intact, blind=True)
    e_sng, _, _ = eval_condition(c_single, decouple=True)

    return {
        f"{tag}_swarm_dmcn_err": e_dmcn,
        f"{tag}_swarm_dmcn_pol": p_dmcn,
        f"{tag}_swarm_dmcn_col": col_dmcn,
        f"{tag}_pure_boids_err": e_pure,
        f"{tag}_pure_boids_pol": p_pure,
        f"{tag}_pure_boids_col": col_pure,
        f"{tag}_ema_boids_err": e_ema,
        f"{tag}_ema_boids_pol": p_ema,
        f"{tag}_ema_boids_col": col_ema,
        f"{tag}_ablate_learning_err": e_nol,
        f"{tag}_ablate_spine_err": e_nos,
        f"{tag}_decoupled_err": e_dec,
        f"{tag}_decoupled_pol": p_dec,
        f"{tag}_scrambled_err": e_scr,
        f"{tag}_scrambled_pol": p_scr,
        f"{tag}_blind_err": e_bld,
        f"{tag}_single_agent_err": e_sng,
    }


def evaluate_seed(seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seed": seed}
    out.update(run_regime(seed, with_wind=False))
    out.update(run_regime(seed, with_wind=True))
    return out


def profile_swarm_latency() -> Dict[str, float]:
    sw = SwarmBoidNetwork(SwarmConfig(n_boids=16, use_dmcn_memory=True, seed=42))
    rng = np.random.RandomState(42)
    b_seq = rng.randn(200, 16, 2)
    t0 = time.perf_counter()
    for i in range(200):
        sw.step(b_seq[i], dt=0.2)
    us_per_step = (time.perf_counter() - t0) * 1e6 / 200.0
    ops_per_boid = sw.brains[0].ops_report()
    return {
        "swarm_16_boids_us_per_step": float(us_per_step),
        "per_boid_dmcn_flops": float(ops_per_boid.total_flops_per_step),
        "per_boid_dmcn_params": float(ops_per_boid.parameter_count),
        "per_boid_dmcn_ram_fp32_bytes": float(ops_per_boid.memory_bytes_fp32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_swarm_boids_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        res = evaluate_seed(999)
        prof = profile_swarm_latency()
        log_path = out_dir / "tuning_log.json"
        if log_path.exists():
            raw = json.loads(log_path.read_text(encoding="utf-8"))
            entries = raw.get("tuning_entries", []) if isinstance(raw, dict) else []
        else:
            entries = []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_swarm_boids_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999"},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"val_seed_999": res, "compute_profile": prof}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    per_seed = [evaluate_seed(s) for s in test_seeds]
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}
    prof = profile_swarm_latency()

    payload = {
        "headline_metric": "wind_swarm_dmcn_err",
        "direction": "lower_is_better",
        "n_seeds": len(test_seeds),
        "summary": summary,
        "compute_profile": prof,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_swarm_boids_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_boids": 16, "n_steps": 150, "noise_sigma": 2.0, "wind_vec": [0.0, -1.1]},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"summary": summary, "compute_profile": prof}, indent=2))


if __name__ == "__main__":
    main()
