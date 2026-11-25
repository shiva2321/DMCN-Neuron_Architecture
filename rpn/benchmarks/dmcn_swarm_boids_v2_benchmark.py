"""Pre-registered 20-Seed Swarm Boids + DMCN vs. Classical Integral/PID Disturbance Observer Benchmark.

Evaluates decentralized Boids + per-boid local nonlinear DMCN proprioceptive wind adaptation across 3 regimes:
  - Regime 1: Uniform mid-flight crosswind (honest check against classical PID/Integral observer)
  - Regime 2: Spatially-varying nonlinear wind shear field w(x, y) + moving target (where scalar PID lags)
  - Regime 3: 31% blind / sensor-failed boids + shifting wind gusts (emergent consensus rescue + adaptation)

Includes strong baselines (Boids + Classical Integral/PID Observer, Pure Boids, EMA-Boids) and
strict null controls (Decoupled, Scrambled, Blind, ablate_learning).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNLayer  # noqa: E402


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


def _wind_at(pos: np.ndarray, t: int, regime: str) -> np.ndarray:
    if regime == "regime1_uniform_wind":
        return np.array([0.0, -1.1], dtype=np.float64) if t >= 50 else np.zeros(2, dtype=np.float64)
    if regime == "regime2_spatial_wind":
        if t < 20:
            return np.zeros(2, dtype=np.float64)
        wx = 1.2 * np.sin(0.35 * pos[1])
        wy = -1.2 * np.cos(0.25 * pos[0])
        return np.array([wx, wy], dtype=np.float64)
    if regime == "regime3_blind31_gusts":
        if 40 <= t < 110:
            return np.array([0.0, -1.1], dtype=np.float64)
        if t >= 110:
            return np.array([0.9, 0.8], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)
    return np.zeros(2, dtype=np.float64)


def run_swarm_episode(
    seed: int,
    regime: str,
    mode: str,
    n_boids: int = 16,
    t_steps: int = 240,
    dt: float = 0.2,
    bearing_sigma: float = 2.0,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-3.0, 3.0, size=(n_boids, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_boids)
    v_cruise = 1.5
    v_max = 2.5
    vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * v_cruise

    blind_mask = np.zeros(n_boids, dtype=bool)
    if regime == "regime3_blind31_gusts":
        blind_mask[:5] = True  # 5 / 16 = 31.25% blind boids

    # Per-boid controllers
    ema_bear = np.zeros((n_boids, 2), dtype=np.float64)
    pid_wind_hat = np.zeros((n_boids, 2), dtype=np.float64)
    pid_err_int = np.zeros((n_boids, 2), dtype=np.float64)
    alpha_pid = 0.65

    # Per-boid DMCN: nonlinear spatial wind adapter ([pos/8, vel/2, 1] -> 24 tanh -> 2) + bearing smoother
    dmcn_exp = []
    dmcn_wind = []
    dmcn_bear = []
    for i in range(n_boids):
        e_l = DMCNLayer(DMCNConfig(
            d_in=5, d_out=24, n_branches=1, n_spine_scales=0,
            soma_activation="tanh", weight_init_scale=2.2, seed=seed * 100 + i,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))
        e_l.b_dend[0] = rng.normal(0.0, 0.4, size=24)
        w_l = DMCNLayer(DMCNConfig(
            d_in=25, d_out=2, n_branches=1, n_spine_scales=0,
            soma_activation="linear", eta_fast=0.95, gamma_forget=0.0, eta_consol=0.0,
            use_unit_key_norm=True,
            weight_init_scale=0.0 if mode != "ablate_learning" else 0.35,
            seed=seed * 100 + 50 + i,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))
        b_l = DMCNLayer(DMCNConfig(
            d_in=4, d_out=2, n_branches=1, n_spine_scales=1, tau_min=2.5,
            soma_activation="linear", eta_fast=0.35, gamma_forget=0.0, eta_consol=0.0,
            weight_init_scale=0.1, seed=seed * 100 + 80 + i,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))
        dmcn_exp.append(e_l)
        dmcn_wind.append(w_l)
        dmcn_bear.append(b_l)

    prev_cmd_vel = vel.copy()
    prev_wind_est = np.zeros((n_boids, 2), dtype=np.float64)
    prev_w_raw = np.zeros((n_boids, 2), dtype=np.float64)

    errs_post = []
    blind_errs_post = []
    pols = []
    collisions = 0

    decouple = mode == "decoupled"
    scramble = mode == "scrambled"
    all_blind = mode == "blind"

    for t in range(t_steps):
        target = np.array([7.0 * np.sin(0.025 * t), 4.5 * np.sin(0.05 * t)], dtype=np.float64)

        # Pairwise distances for Reynolds Boids forces
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        collisions += int(np.sum((dist < 0.8) & (dist > 1e-6))) // 2

        next_cmd_vel = np.zeros_like(vel)
        next_wind_est = np.zeros_like(vel)

        for i in range(n_boids):
            # 1. Local proprioceptive slip: observed velocity minus commanded velocity from t-1
            slip = vel[i] - prev_cmd_vel[i]

            # 2. Local target bearing sensor
            if all_blind or blind_mask[i]:
                raw_b = np.zeros(2, dtype=np.float64)
            else:
                vec_t = target - pos[i]
                nt = float(np.linalg.norm(vec_t))
                true_b = (vec_t / nt) if nt > 1e-8 else np.zeros(2, dtype=np.float64)
                raw_b = true_b + rng.normal(0.0, bearing_sigma, size=2)

            # 3. Reynolds neighbor forces
            f_sep = np.zeros(2, dtype=np.float64)
            f_ali = np.zeros(2, dtype=np.float64)
            f_coh = np.zeros(2, dtype=np.float64)
            if not decouple:
                nbrs = np.flatnonzero((dist[i] < 6.0) & (np.arange(n_boids) != i))
                if len(nbrs) > 0:
                    close = nbrs[dist[i, nbrs] < 1.5]
                    if len(close) > 0:
                        for j in close:
                            f_sep += diff[i, j] / (dist[i, j] ** 2)
                        f_sep /= len(close)
                    if scramble:
                        rand_ang = rng.uniform(0.0, 2.0 * np.pi, size=len(nbrs))
                        speeds = np.linalg.norm(vel[nbrs], axis=1, keepdims=True)
                        v_nbr = np.stack([np.cos(rand_ang), np.sin(rand_ang)], axis=1) * speeds
                    else:
                        v_nbr = vel[nbrs]
                    f_ali = np.mean(v_nbr, axis=0) - vel[i]
                    f_coh = np.mean(pos[nbrs], axis=0) - pos[i]

            # 4. Bearing filtering + disturbance estimation by mode
            if mode in ("dmcn", "decoupled", "scrambled", "blind", "ablate_learning"):
                ctx_b = np.concatenate([raw_b, f_ali])
                b_pred = dmcn_bear[i].step(ctx_b)
                if mode != "ablate_learning" and not (all_blind or blind_mask[i]):
                    dmcn_bear[i].learn_local(raw_b - b_pred)
                steer_b = b_pred if not (all_blind or blind_mask[i]) else np.zeros(2, dtype=np.float64)

                # 1) Update previous step's spatial wind prediction from exact proprioceptive slip
                if t > 0 and mode != "ablate_learning":
                    true_w_obs = prev_wind_est[i] + slip
                    dmcn_wind[i].learn_local(true_w_obs - prev_w_raw[i])

                # 2) Predict wind at current local position/velocity
                x_w = np.array([pos[i, 0] / 8.0, pos[i, 1] / 8.0, vel[i, 0] / 2.0, vel[i, 1] / 2.0, 1.0], dtype=np.float64)
                h_w = np.append(dmcn_exp[i].step(x_w), 1.0)
                w_est = dmcn_wind[i].step(h_w)
                prev_w_raw[i] = w_est
            elif mode == "pid_observer":
                ema_bear[i] = 0.70 * ema_bear[i] + 0.30 * raw_b
                steer_b = ema_bear[i] if not (all_blind or blind_mask[i]) else np.zeros(2, dtype=np.float64)
                if t > 0:
                    true_w_obs = prev_wind_est[i] + slip
                    err_w = true_w_obs - pid_wind_hat[i]
                    pid_err_int[i] = 0.90 * pid_err_int[i] + err_w
                    pid_wind_hat[i] = pid_wind_hat[i] + alpha_pid * err_w + 0.08 * pid_err_int[i]
                w_est = pid_wind_hat[i].copy()
            elif mode == "ema_boids":
                ema_bear[i] = 0.70 * ema_bear[i] + 0.30 * raw_b
                steer_b = ema_bear[i] if not (all_blind or blind_mask[i]) else np.zeros(2, dtype=np.float64)
                w_est = np.zeros(2, dtype=np.float64)
            else:  # pure_boids
                steer_b = raw_b
                w_est = np.zeros(2, dtype=np.float64)

            # 5. Reynolds self-propelled cruising-speed command + disturbance cancellation
            acc = 1.6 * f_sep + 1.3 * f_ali + 0.65 * f_coh + 0.85 * steer_b
            v_cmd = vel[i] + dt * acc
            sp = float(np.linalg.norm(v_cmd))
            if sp > 1e-8:
                target_sp = sp + 0.65 * (v_cruise - sp)
                v_cmd = v_cmd * (min(v_max, target_sp) / sp)
            # Apply learned disturbance cancellation
            v_cmd_comp = v_cmd - np.clip(w_est, -2.5, 2.5)
            next_cmd_vel[i] = v_cmd
            next_wind_est[i] = np.clip(w_est, -2.5, 2.5)

            # Physics step: environment applies true local wind field w(pos_i, t)
            w_true = _wind_at(pos[i], t, regime)
            vel[i] = v_cmd_comp + w_true

        pos = pos + dt * vel
        prev_cmd_vel = next_cmd_vel
        prev_wind_est = next_wind_est

        centroid = np.mean(pos, axis=0)
        err = float(np.linalg.norm(centroid - target))
        speeds = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        pol = float(np.linalg.norm(np.mean(vel / speeds, axis=0)))
        pols.append(pol)
        if t >= 90:
            errs_post.append(err)
            if regime == "regime3_blind31_gusts":
                blind_c = np.mean(pos[blind_mask], axis=0)
                blind_errs_post.append(float(np.linalg.norm(blind_c - target)))

    out = {
        "post_err": float(np.mean(errs_post)),
        "collisions": float(collisions),
        "polarization": float(np.mean(pols[-50:])),
    }
    if regime == "regime3_blind31_gusts":
        out["blind_subset_err"] = float(np.mean(blind_errs_post))
    return out


def evaluate_seed(seed: int) -> Dict[str, Any]:
    regimes = ["regime1_uniform_wind", "regime2_spatial_wind", "regime3_blind31_gusts"]
    modes = ["dmcn", "pid_observer", "pure_boids", "ema_boids", "decoupled", "scrambled", "blind", "ablate_learning"]
    res: Dict[str, Any] = {"seed": seed}
    for reg in regimes:
        for m in modes:
            ep = run_swarm_episode(seed, reg, m)
            res[f"{reg}__{m}__err"] = ep["post_err"]
            if m in ("dmcn", "pid_observer", "pure_boids"):
                res[f"{reg}__{m}__col"] = ep["collisions"]
            if reg == "regime3_blind31_gusts" and m in ("dmcn", "pid_observer", "pure_boids", "decoupled"):
                res[f"{reg}__{m}__blind_err"] = ep["blind_subset_err"]
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_swarm_boids_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_swarm_boids_v2_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "n_boids": 16, "t_steps": 180},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed = [evaluate_seed(s) for s in test_seeds]
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    payload = {
        "headline_metric": "regime2_spatial_wind__dmcn__err",
        "direction": "lower_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_swarm_boids_v2",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_boids": 16, "t_steps": 180, "dt": 0.2, "bearing_sigma": 1.5},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
