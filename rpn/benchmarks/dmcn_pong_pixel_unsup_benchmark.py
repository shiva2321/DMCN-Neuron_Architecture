"""Pre-registered 20-Seed Unsupervised Motion-Driven Oja Layer-1 Raw-Pixel Pong Benchmark.

Tests whether UnsupOjaDendriticLayer (strictly local k-WTA competitive Oja plasticity driven by
frame-to-frame pixel motion, zero backprop, zero coordinates, zero reward shaping) improves
Raw 16x16 Pixel Pong over ablate_unsup (frozen Layer 1) and how close it gets to PyTorch Backprop A2C.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNLayer, UnsupOjaDendriticLayer  # noqa: E402
from benchmarks.dmcn_pong_rl_benchmark import PongEnv, _softmax, eval_trivial_policy, summarize  # noqa: E402
from benchmarks.dmcn_pong_rl_v2_benchmark import TorchA2CMLP  # noqa: E402


class UnsupOjaPongAgent:
    """Unsupervised Oja/k-WTA Layer-1 + Local TD(lambda) Actor-Critic on raw 16x16 pixels (zero backprop)."""

    def __init__(
        self,
        d_hid: int = 128,
        seed: int = 42,
        ablate_unsup: bool = False,
        eta_actor: float = 0.25,
        eta_critic: float = 0.35,
        gamma: float = 0.95,
        lam: float = 0.85,
        beta_ent: float = 0.02,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.gamma = gamma
        self.beta_ent = beta_ent
        self.vis = UnsupOjaDendriticLayer(d_in=256, d_hid=d_hid, k_wta=8, eta_oja=0.10, seed=seed + 101, ablate_unsup=ablate_unsup)
        d_pol = 256 + 2 * d_hid + 1
        self.actor = DMCNLayer(DMCNConfig(
            d_in=d_pol, d_out=3, n_branches=1, n_spine_scales=0, soma_activation="linear",
            weight_init_scale=0.0, gamma_elig=gamma * lam, eta_rl=eta_actor, gamma_forget=0.0, eta_consol=0.0,
            seed=seed, ablation=AblationConfig(ablate_homeostasis=True),
        ))
        self.critic = DMCNLayer(DMCNConfig(
            d_in=d_pol, d_out=1, n_branches=1, n_spine_scales=0, soma_activation="linear",
            weight_init_scale=0.0, gamma_elig=gamma * lam, eta_rl=eta_critic, gamma_forget=0.0, eta_consol=0.0,
            seed=seed + 202, ablation=AblationConfig(ablate_homeostasis=True),
        ))

    def reset_episode(self) -> None:
        self.vis.reset_temporal_state()
        self.actor.reset_temporal_state()
        self.critic.reset_temporal_state()

    def encode(self, frame: np.ndarray, learn_unsup: bool = True) -> np.ndarray:
        h_vis = self.vis.step(frame, learn_unsup=learn_unsup)
        return np.concatenate([frame, h_vis, np.array([1.0], dtype=np.float64)])

    def select_action(self, h: np.ndarray, greedy: bool = False) -> Tuple[int, np.ndarray, float]:
        logits = self.actor.step(h)
        val = float(self.critic.step(h)[0])
        probs = _softmax(logits)
        if greedy:
            action = int(np.argmax(logits))
        else:
            probs_exp = 0.96 * probs + 0.04 / 3.0
            action = int(self.rng.choice(3, p=probs_exp))
        return action, probs, val

    def td_update(self, action: int, probs: np.ndarray, v_curr: float, reward: float, v_next: float, done: bool) -> None:
        td_clip = float(np.clip(reward + (0.0 if done else self.gamma * v_next) - v_curr, -2.0, 2.0))
        log_p = np.log(np.clip(probs, 1e-8, 1.0))
        ent = -float(np.dot(probs, log_p))
        ent_grad = -probs * (log_p + ent)
        act_grad = -probs.copy()
        act_grad[action] += 1.0
        self.actor.accumulate_eligibility(act_grad + self.beta_ent * ent_grad)
        self.actor.apply_rpe(td_clip)
        self.critic.accumulate_eligibility(np.array([1.0], dtype=np.float64))
        self.critic.apply_rpe(td_clip)


def _train_agent(agent: UnsupOjaPongAgent, rng: np.random.RandomState, n_serves: int = 2000, shuffle_r: bool = False) -> None:
    env = PongEnv()
    for _ in range(n_serves):
        env.reset_serve(rng, ood=False)
        agent.reset_episode()
        h = agent.encode(env.obs_pixels(), learn_unsup=True)
        done = False
        steps = 0
        while not done and steps < 40:
            act, probs, v_curr = agent.select_action(h, greedy=False)
            r_raw, done = env.step(act)
            if done:
                r_learn = float(rng.choice([-1.0, 1.0])) if shuffle_r else r_raw
                agent.td_update(act, probs, v_curr, r_learn, 0.0, True)
            else:
                h_next = agent.encode(env.obs_pixels(), learn_unsup=True)
                v_next = float(agent.critic.step(h_next)[0])
                agent.actor.step(h)
                agent.critic.step(h)
                agent.td_update(act, probs, v_curr, 0.0, v_next, False)
                h = h_next
            steps += 1


def _eval_agent(agent: UnsupOjaPongAgent, rng: np.random.RandomState, n_serves: int = 100, blind: bool = False) -> Tuple[float, float]:
    env = PongEnv()
    hits = 0
    dir_ok, dir_tot = 0, 0
    for _ in range(n_serves):
        env.reset_serve(rng, ood=False)
        agent.reset_episode()
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            obs = np.zeros(256, dtype=np.float64) if blind else env.obs_pixels()
            h = agent.encode(obs, learn_unsup=False)
            act, _, _ = agent.select_action(h, greedy=True)
            if abs(env.ball_y - env.paddle_y) > 0.05:
                dir_tot += 1
                if (env.ball_y > env.paddle_y and act == 2) or (env.ball_y < env.paddle_y and act == 0):
                    dir_ok += 1
            reward, done = env.step(act)
            steps += 1
        if reward > 0.0:
            hits += 1
    return 100.0 * hits / float(n_serves), 100.0 * dir_ok / float(max(1, dir_tot))


def evaluate_seed(seed: int, n_train: int = 2000, n_eval: int = 100) -> Dict[str, Any]:
    ag_unsup = UnsupOjaPongAgent(d_hid=128, seed=seed, ablate_unsup=False)
    ag_frozen = UnsupOjaPongAgent(d_hid=128, seed=seed, ablate_unsup=True)
    ag_untr = UnsupOjaPongAgent(d_hid=128, seed=seed, ablate_unsup=False)
    ag_shuf = UnsupOjaPongAgent(d_hid=128, seed=seed, ablate_unsup=False)

    _train_agent(ag_unsup, np.random.RandomState(seed + 1000), n_train)
    _train_agent(ag_frozen, np.random.RandomState(seed + 1000), n_train)
    _train_agent(ag_shuf, np.random.RandomState(seed + 1000), n_train, shuffle_r=True)

    hit_u, dir_u = _eval_agent(ag_unsup, np.random.RandomState(seed + 5000), n_eval)
    hit_f, dir_f = _eval_agent(ag_frozen, np.random.RandomState(seed + 5000), n_eval)
    hit_untr, _ = _eval_agent(ag_untr, np.random.RandomState(seed + 5000), n_eval)
    hit_blind, _ = _eval_agent(ag_unsup, np.random.RandomState(seed + 5000), n_eval, blind=True)
    hit_shuf, _ = _eval_agent(ag_shuf, np.random.RandomState(seed + 5000), n_eval)

    # Measure receptive-field concentration (mean max-pixel weight^2 across 128 detectors)
    rf_conc_unsup = float(np.mean(np.max(ag_unsup.vis.W ** 2, axis=1)))
    rf_conc_frozen = float(np.mean(np.max(ag_frozen.vis.W ** 2, axis=1)))

    bp = TorchA2CMLP(d_in=512, d_hid=128, seed=seed, lr=1.5e-3)
    env_bp = PongEnv()
    rng_bp = np.random.RandomState(seed + 1000)
    for _ in range(n_train):
        env_bp.reset_serve(rng_bp, ood=False)
        bp.train_serve(env_bp, pixel_mode=True)
    hit_bp = bp.eval_serves(np.random.RandomState(seed + 5000), n_eval, ood=False, pixel_mode=True)

    return {
        "seed": seed,
        "pixel_unsup_oja_dmcn_hit_rate": hit_u,
        "pixel_unsup_oja_dmcn_dir_align": dir_u,
        "pixel_ablate_unsup_hit_rate": hit_f,
        "pixel_ablate_unsup_dir_align": dir_f,
        "pixel_bp_a2c_hit_rate": hit_bp,
        "pixel_untrained_hit_rate": hit_untr,
        "pixel_blind_hit_rate": hit_blind,
        "pixel_shuffled_reward_hit_rate": hit_shuf,
        "pixel_random_hit_rate": eval_trivial_policy("random", np.random.RandomState(seed + 5000), n_eval),
        "rf_peak_concentration_unsup": rf_conc_unsup,
        "rf_peak_concentration_frozen": rf_conc_frozen,
    }


def _worker(seed: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    return evaluate_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_pong_pixel_unsup_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_pong_pixel_unsup_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "n_train": 2000, "d_hid": 128},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        per_seed = list(ex.map(_worker, test_seeds))
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    payload = {
        "headline_metric": "pixel_unsup_oja_dmcn_hit_rate",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_pong_pixel_unsup_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_train": 2000, "n_eval": 100, "d_hid": 128, "k_wta": 8, "eta_oja": 0.06},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
