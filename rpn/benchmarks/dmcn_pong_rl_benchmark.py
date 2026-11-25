"""Pre-registered 20-Seed Zero-Hardcoding Pong RL Benchmark (Tier 1 Position-Only + Tier 2 Raw Pixels).

Tests whether DMCN (Pillar 2 Spine ODEs + nonlinear dendritic layer + Pillar 4 3-factor eligibility
traces, ZERO backpropagation, ZERO hardcoded policy, ZERO distance reward shaping) can learn to play
Pong from strictly sparse terminal outcomes (+1 hit, -1 miss, 0 every other frame).

Includes:
  - Tier 1: Position-Only Partially-Observable Pong ([ball_x, ball_y, paddle_y], NO velocity given)
  - Tier 1 OOD Generalization: Unseen faster ball + narrower paddle (frozen weights)
  - Tier 2: Raw-Pixel Pong (16x16 = 256 grayscale pixels ONLY, ZERO coordinates)
  - Strong Baseline: PyTorch Backprop REINFORCE MLP (Adam, 3-frame stack so velocity is observable)
  - Ablations: ablate_spine_traces, ablate_eligibility
  - Null Controls: Untrained, Blind (obs=0), Shuffled Reward, Random, Constant-Stay, Constant-Up
  - Mechanistic Audit: Directional tracking alignment & velocity-inference probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -20.0, 0.0))
    return e / (float(np.sum(e)) + 1e-12)


class PongEnv:
    """Clean deterministic 2D Pong physics (sparse +1 hit / -1 miss reward, zero shaping)."""

    def __init__(self, paddle_speed: float = 0.08, paddle_x: float = 0.96) -> None:
        self.paddle_speed = paddle_speed
        self.paddle_x = paddle_x
        self.ball_x = 0.08
        self.ball_y = 0.5
        self.vx = 0.065
        self.vy = 0.0
        self.paddle_y = 0.5
        self.paddle_half_h = 0.12

    def reset_serve(self, rng: np.random.RandomState, ood: bool = False) -> None:
        self.paddle_half_h = 0.09 if ood else 0.12
        self.ball_x = 0.08
        self.ball_y = float(rng.uniform(0.15, 0.85))
        self.paddle_y = float(rng.uniform(0.20, 0.80))
        self.vx = float(rng.uniform(0.075, 0.095) if ood else rng.uniform(0.055, 0.075))
        self.vy = float(rng.uniform(-0.075, 0.075) if ood else rng.uniform(-0.06, 0.06))

    def obs_pos(self) -> np.ndarray:
        """Position-only observation: [ball_x, ball_y, paddle_y] centered in [-0.5, 0.5]. NO velocity!"""
        return np.array([self.ball_x - 0.5, self.ball_y - 0.5, self.paddle_y - 0.5], dtype=np.float64)

    def obs_pixels(self, size: int = 16) -> np.ndarray:
        """Raw 16x16 grayscale pixel screen (256 dims). ZERO coordinates."""
        grid = np.zeros((size, size), dtype=np.float64)
        bx = int(np.clip(round(self.ball_x * (size - 1)), 0, size - 1))
        by = int(np.clip(round(self.ball_y * (size - 1)), 0, size - 1))
        grid[max(0, by - 1) : min(size, by + 1), max(0, bx - 1) : min(size, bx + 1)] = 1.0
        py_lo = int(np.clip(round((self.paddle_y - self.paddle_half_h) * (size - 1)), 0, size - 1))
        py_hi = int(np.clip(round((self.paddle_y + self.paddle_half_h) * (size - 1)), 0, size - 1))
        grid[py_lo : py_hi + 1, size - 1] = 1.0
        return grid.ravel()

    def step(self, action: int) -> Tuple[float, bool]:
        """Action in {0: Down, 1: Stay, 2: Up}. Returns (sparse_reward, done)."""
        move = float(action - 1) * self.paddle_speed
        self.paddle_y = float(np.clip(self.paddle_y + move, self.paddle_half_h, 1.0 - self.paddle_half_h))
        self.ball_x += self.vx
        self.ball_y += self.vy

        if self.ball_y < 0.0:
            self.ball_y = -self.ball_y
            self.vy = -self.vy
        elif self.ball_y > 1.0:
            self.ball_y = 2.0 - self.ball_y
            self.vy = -self.vy

        if self.ball_x >= self.paddle_x:
            hit = abs(self.ball_y - self.paddle_y) <= self.paddle_half_h
            return (1.0 if hit else -1.0), True
        return 0.0, False


class DMCNPongAgent:
    """2-layer gradient-free DMCN RL agent (Pillar 2 spines + nonlinear dendrites + Pillar 4 3-factor RPE)."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        n_spine_scales: int,
        seed: int,
        ablate_spine: bool = False,
        ablate_elig: bool = False,
        eta_rl: float = 0.22,
        exp_scale: float = 2.2,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.enc = DMCNLayer(DMCNConfig(
            d_in=d_in,
            d_out=d_hid,
            n_branches=1,
            n_spine_scales=n_spine_scales,
            tau_min=1.0,
            tau_max=6.0,
            soma_activation="tanh",
            weight_init_scale=exp_scale,
            seed=seed + 101,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_spine_traces=ablate_spine),
        ))
        self.enc.b_dend[0] = self.rng.normal(0.0, 0.15 if d_in > 16 else 0.35, size=d_hid)
        self.policy = DMCNLayer(DMCNConfig(
            d_in=d_hid + 1,
            d_out=3,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            weight_init_scale=0.0,
            gamma_elig=0.92,
            eta_rl=eta_rl,
            gamma_forget=0.0,
            eta_consol=0.0,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_eligibility=ablate_elig),
        ))
        self.r_baseline = 0.0

    def reset_episode(self) -> None:
        self.enc.reset_temporal_state()
        self.policy.reset_temporal_state()

    def act(self, obs: np.ndarray, greedy: bool = False) -> Tuple[int, np.ndarray]:
        h = np.append(self.enc.step(obs), 1.0)
        logits = self.policy.step(h)
        probs = _softmax(logits)
        if greedy:
            action = int(np.argmax(logits))
        else:
            probs_exp = 0.94 * probs + 0.06 / 3.0
            action = int(self.rng.choice(3, p=probs_exp))
            grad = -probs.copy()
            grad[action] += 1.0
            self.policy.accumulate_eligibility(grad)
        return action, probs

    def finish_episode(self, reward: float) -> None:
        rpe = reward - self.r_baseline
        self.policy.apply_rpe(rpe)
        self.r_baseline = 0.95 * self.r_baseline + 0.05 * reward
        self.reset_episode()


class TorchReinforceMLP(nn.Module):
    """Standard PyTorch Backprop REINFORCE MLP baseline (given 3-frame history stack for fairness)."""

    def __init__(self, d_in: int, d_hid: int, seed: int, lr: float = 3e-3) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh(), nn.Linear(d_hid, 3))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.r_baseline = 0.0
        self.rng = np.random.RandomState(seed)

    def train_serve(self, env: PongEnv, pixel_mode: bool = False) -> float:
        obs_dim = 256 if pixel_mode else 3
        hist = [np.zeros(obs_dim, dtype=np.float32) for _ in range(3)]
        log_probs: List[torch.Tensor] = []
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            cur = (env.obs_pixels() if pixel_mode else env.obs_pos()).astype(np.float32)
            hist = hist[1:] + [cur]
            x = torch.tensor(np.concatenate(hist), dtype=torch.float32).unsqueeze(0)
            logits = self.net(x).squeeze(0)
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs=probs)
            act = dist.sample()
            log_probs.append(dist.log_prob(act))
            reward, done = env.step(int(act.item()))
            steps += 1

        adv = reward - self.r_baseline
        self.r_baseline = 0.95 * self.r_baseline + 0.05 * reward
        if log_probs:
            loss = torch.tensor(0.0)
            g = 1.0
            for lp in reversed(log_probs):
                loss = loss - lp * (adv * g)
                g *= 0.92
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return reward

    @torch.no_grad()
    def eval_serves(self, rng: np.random.RandomState, n_serves: int, ood: bool = False, pixel_mode: bool = False) -> float:
        env = PongEnv()
        obs_dim = 256 if pixel_mode else 3
        hits = 0
        for _ in range(n_serves):
            env.reset_serve(rng, ood=ood)
            hist = [np.zeros(obs_dim, dtype=np.float32) for _ in range(3)]
            done = False
            reward = 0.0
            steps = 0
            while not done and steps < 40:
                cur = (env.obs_pixels() if pixel_mode else env.obs_pos()).astype(np.float32)
                hist = hist[1:] + [cur]
                x = torch.tensor(np.concatenate(hist), dtype=torch.float32).unsqueeze(0)
                act = int(torch.argmax(self.net(x), dim=1).item())
                reward, done = env.step(act)
                steps += 1
            if reward > 0.0:
                hits += 1
        return 100.0 * hits / float(n_serves)


def eval_dmcn_serves(
    agent: DMCNPongAgent,
    rng: np.random.RandomState,
    n_serves: int,
    ood: bool = False,
    pixel_mode: bool = False,
    blind: bool = False,
    record_traces: int = 0,
) -> Dict[str, Any]:
    env = PongEnv()
    obs_dim = 256 if pixel_mode else 3
    hits = 0
    dir_correct = 0
    dir_total = 0
    traces: List[Dict[str, Any]] = []

    for s_idx in range(n_serves):
        env.reset_serve(rng, ood=ood)
        agent.reset_episode()
        done = False
        reward = 0.0
        steps = 0
        tr_bx: List[float] = []
        tr_by: List[float] = []
        tr_py: List[float] = []
        while not done and steps < 40:
            if s_idx < record_traces:
                tr_bx.append(env.ball_x)
                tr_by.append(env.ball_y)
                tr_py.append(env.paddle_y)
            obs = np.zeros(obs_dim, dtype=np.float64) if blind else (env.obs_pixels() if pixel_mode else env.obs_pos())
            act, _ = agent.act(obs, greedy=True)
            if abs(env.ball_y - env.paddle_y) > 0.05:
                dir_total += 1
                if (env.ball_y > env.paddle_y and act == 2) or (env.ball_y < env.paddle_y and act == 0):
                    dir_correct += 1
            reward, done = env.step(act)
            steps += 1
        if s_idx < record_traces:
            tr_bx.append(env.ball_x)
            tr_by.append(env.ball_y)
            tr_py.append(env.paddle_y)
            traces.append({"bx": tr_bx, "by": tr_by, "py": tr_py, "hit": bool(reward > 0.0)})
        if reward > 0.0:
            hits += 1

    return {
        "hit_rate": 100.0 * hits / float(n_serves),
        "dir_align_acc": 100.0 * dir_correct / float(max(1, dir_total)),
        "traces": traces,
    }


def probe_velocity_sensitivity(agent: DMCNPongAgent) -> float:
    """Controlled probe: identical final position (0.5, 0.5, 0.5) preceded by upward vs downward ball motion."""
    deltas = []
    for py in (0.35, 0.50, 0.65):
        # Upward trajectory arriving at (0.5, py)
        agent.reset_episode()
        for k in range(4, -1, -1):
            obs_u = np.array([0.5 - 0.06 * k - 0.5, py - 0.05 * k - 0.5, py - 0.5], dtype=np.float64)
            _, probs_u = agent.act(obs_u, greedy=True)
        # Downward trajectory arriving at exact same (0.5, py)
        agent.reset_episode()
        for k in range(4, -1, -1):
            obs_d = np.array([0.5 - 0.06 * k - 0.5, py + 0.05 * k - 0.5, py - 0.5], dtype=np.float64)
            _, probs_d = agent.act(obs_d, greedy=True)
        deltas.append(float(probs_u[2] - probs_d[2]))  # P(Up | moving up) - P(Up | moving down)
    return 100.0 * float(np.mean(deltas))


def train_dmcn(
    agent: DMCNPongAgent,
    rng: np.random.RandomState,
    n_serves: int = 600,
    pixel_mode: bool = False,
    shuffle_reward: bool = False,
    record_curve: bool = False,
) -> List[float]:
    env = PongEnv()
    curve: List[float] = []
    bin_hits = 0
    bin_sz = max(1, n_serves // 10)
    for s in range(n_serves):
        env.reset_serve(rng, ood=False)
        agent.reset_episode()
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            obs = env.obs_pixels() if pixel_mode else env.obs_pos()
            act, _ = agent.act(obs, greedy=False)
            reward, done = env.step(act)
            steps += 1
        if reward > 0.0:
            bin_hits += 1
        r_learn = float(rng.choice([-1.0, 1.0])) if shuffle_reward else reward
        agent.finish_episode(r_learn)
        if record_curve and (s + 1) % bin_sz == 0:
            curve.append(100.0 * bin_hits / float(bin_sz))
            bin_hits = 0
    return curve


def eval_trivial_policy(policy_name: str, rng: np.random.RandomState, n_serves: int = 100) -> float:
    env = PongEnv()
    hits = 0
    for _ in range(n_serves):
        env.reset_serve(rng, ood=False)
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            if policy_name == "random":
                act = int(rng.randint(0, 3))
            elif policy_name == "const_up":
                act = 2
            else:
                act = 1
            reward, done = env.step(act)
            steps += 1
        if reward > 0.0:
            hits += 1
    return 100.0 * hits / float(n_serves)


def evaluate_seed(seed: int, n_train: int = 600, n_eval: int = 100, record_extras: bool = False) -> Dict[str, Any]:
    # --- Tier 1: Position-Only Partially-Observable Pong (NO velocity given) ---
    dmcn_t1 = DMCNPongAgent(d_in=3, d_hid=64, n_spine_scales=3, seed=seed, ablate_spine=False, ablate_elig=False, eta_rl=0.25, exp_scale=2.2)
    dmcn_nospine = DMCNPongAgent(d_in=3, d_hid=64, n_spine_scales=3, seed=seed, ablate_spine=True, ablate_elig=False, eta_rl=0.25, exp_scale=2.2)
    dmcn_noelig = DMCNPongAgent(d_in=3, d_hid=64, n_spine_scales=3, seed=seed, ablate_spine=False, ablate_elig=True, eta_rl=0.25, exp_scale=2.2)
    dmcn_untr = DMCNPongAgent(d_in=3, d_hid=64, n_spine_scales=3, seed=seed, ablate_spine=False, ablate_elig=False, eta_rl=0.25, exp_scale=2.2)
    dmcn_shuf = DMCNPongAgent(d_in=3, d_hid=64, n_spine_scales=3, seed=seed, ablate_spine=False, ablate_elig=False, eta_rl=0.25, exp_scale=2.2)

    untr_eval = eval_dmcn_serves(dmcn_untr, np.random.RandomState(seed + 5000), n_eval, record_traces=4 if record_extras else 0)
    curve_t1 = train_dmcn(dmcn_t1, np.random.RandomState(seed + 1000), n_train, pixel_mode=False, record_curve=record_extras)
    train_dmcn(dmcn_nospine, np.random.RandomState(seed + 1000), n_train, pixel_mode=False)
    train_dmcn(dmcn_noelig, np.random.RandomState(seed + 1000), n_train, pixel_mode=False)
    train_dmcn(dmcn_shuf, np.random.RandomState(seed + 1000), n_train, pixel_mode=False, shuffle_reward=True)

    t1_id = eval_dmcn_serves(dmcn_t1, np.random.RandomState(seed + 5000), n_eval, ood=False, record_traces=4 if record_extras else 0)
    t1_ood = eval_dmcn_serves(dmcn_t1, np.random.RandomState(seed + 6000), n_eval, ood=True)
    t1_blind = eval_dmcn_serves(dmcn_t1, np.random.RandomState(seed + 5000), n_eval, blind=True)
    t1_nospine = eval_dmcn_serves(dmcn_nospine, np.random.RandomState(seed + 5000), n_eval)
    t1_noelig = eval_dmcn_serves(dmcn_noelig, np.random.RandomState(seed + 5000), n_eval)
    t1_shuf = eval_dmcn_serves(dmcn_shuf, np.random.RandomState(seed + 5000), n_eval)

    # PyTorch Backprop REINFORCE MLP baseline (Tier 1, given 3-frame stack)
    bp_t1 = TorchReinforceMLP(d_in=9, d_hid=64, seed=seed, lr=4e-3)
    env_bp = PongEnv()
    rng_bp = np.random.RandomState(seed + 1000)
    for _ in range(n_train):
        env_bp.reset_serve(rng_bp, ood=False)
        bp_t1.train_serve(env_bp, pixel_mode=False)
    bp_t1_id = bp_t1.eval_serves(np.random.RandomState(seed + 5000), n_eval, ood=False, pixel_mode=False)
    bp_t1_ood = bp_t1.eval_serves(np.random.RandomState(seed + 6000), n_eval, ood=True, pixel_mode=False)

    # --- Tier 2: Raw-Pixel Pong (16x16 = 256 grayscale pixels ONLY, ZERO coordinates) ---
    dmcn_t2 = DMCNPongAgent(d_in=256, d_hid=64, n_spine_scales=2, seed=seed + 77, ablate_spine=False, ablate_elig=False, eta_rl=0.20, exp_scale=8.0)
    curve_t2 = train_dmcn(dmcn_t2, np.random.RandomState(seed + 2000), n_train, pixel_mode=True, record_curve=record_extras)
    t2_id = eval_dmcn_serves(dmcn_t2, np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True)
    t2_blind = eval_dmcn_serves(dmcn_t2, np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True, blind=True)

    bp_t2 = TorchReinforceMLP(d_in=256 * 3, d_hid=128, seed=seed + 77, lr=2e-3)
    rng_bp2 = np.random.RandomState(seed + 2000)
    for _ in range(n_train):
        env_bp.reset_serve(rng_bp2, ood=False)
        bp_t2.train_serve(env_bp, pixel_mode=True)
    bp_t2_id = bp_t2.eval_serves(np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True)

    res: Dict[str, Any] = {
        "seed": seed,
        "t1_dmcn_hit_rate": t1_id["hit_rate"],
        "t1_dmcn_ood_hit_rate": t1_ood["hit_rate"],
        "t1_dmcn_dir_align_acc": t1_id["dir_align_acc"],
        "t1_dmcn_vel_sensitivity_pp": probe_velocity_sensitivity(dmcn_t1),
        "t1_ablate_spine_hit_rate": t1_nospine["hit_rate"],
        "t1_ablate_spine_vel_sens_pp": probe_velocity_sensitivity(dmcn_nospine),
        "t1_ablate_elig_hit_rate": t1_noelig["hit_rate"],
        "t1_bp_reinforce_hit_rate": bp_t1_id,
        "t1_bp_reinforce_ood_hit_rate": bp_t1_ood,
        "t1_untrained_hit_rate": untr_eval["hit_rate"],
        "t1_blind_hit_rate": t1_blind["hit_rate"],
        "t1_shuffled_reward_hit_rate": t1_shuf["hit_rate"],
        "t1_random_hit_rate": eval_trivial_policy("random", np.random.RandomState(seed + 5000), n_eval),
        "t1_const_stay_hit_rate": eval_trivial_policy("const_stay", np.random.RandomState(seed + 5000), n_eval),
        "t1_const_up_hit_rate": eval_trivial_policy("const_up", np.random.RandomState(seed + 5000), n_eval),
        "t2_pixel_dmcn_hit_rate": t2_id["hit_rate"],
        "t2_pixel_dmcn_dir_align_acc": t2_id["dir_align_acc"],
        "t2_pixel_dmcn_blind_hit_rate": t2_blind["hit_rate"],
        "t2_pixel_bp_reinforce_hit_rate": bp_t2_id,
    }
    if record_extras:
        res["_extras"] = {
            "curve_t1": curve_t1,
            "curve_t2": curve_t2,
            "traces_untrained": untr_eval["traces"],
            "traces_trained": t1_id["traces"],
        }
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_pong_rl_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999, record_extras=False)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_pong_rl_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "n_train": 600, "n_eval": 100},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed_raw = [evaluate_seed(s, record_extras=(s == 2000)) for s in test_seeds]
    elapsed = time.perf_counter() - t0

    extras_2000 = per_seed_raw[0].pop("_extras")
    keys = [k for k in per_seed_raw[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed_raw]) for k in keys}

    payload = {
        "headline_metric": "t1_dmcn_hit_rate",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "seed_2000_analysis": extras_2000,
        "per_seed": per_seed_raw,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_pong_rl_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_train": 600, "n_eval": 100, "d_hid_t1": 64, "d_hid_t2": 128, "eta_rl_t1": 0.25, "eta_rl_t2": 0.20},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
