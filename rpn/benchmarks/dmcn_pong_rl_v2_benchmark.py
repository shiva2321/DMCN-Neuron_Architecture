"""Pre-registered 20-Seed Pong RL v2: Local TD(lambda) Actor-Critic + Causal Transient Spines + Retinotopic Pixel Patches.

Addresses all 3 root causes diagnosed in dmcn_pong_rl_v1 (zero backprop, zero reward shaping, zero hardcoding):
  1. Local TD(lambda) Actor-Critic: 1-neuron local DMCN Critic computes per-step TD error delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
     and modulates the Actor's eligibility trace every frame, slashing variance vs. Monte Carlo REINFORCE.
  2. Causal Transient (Derivative) Spines: feeds [obs, scale * (obs - spine_ema)] instead of lagging raw spine_ema,
     turning velocity sensitivity positive without any velocity oracle.
  3. Local 5x5 Retinotopic Dendritic Receptive Fields (Tier 2 Raw 16x16 Pixels): fixed sparse local spatial windows
     plus pixel motion transients (frame - spine_ema), zero coordinate extraction.

Compares head-to-head against a fair PyTorch Backprop Actor-Critic (A2C) baseline on identical 2,000-serve budget,
plus ablations (ablate_spine, ablate_eligibility) and full null controls (Untrained, Blind, Shuffled Reward, Random, Constant).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNLayer  # noqa: E402
from benchmarks.dmcn_pong_rl_benchmark import PongEnv, _softmax, eval_trivial_policy, summarize  # noqa: E402


class DMCNActorCriticPong:
    """Gradient-free Local TD(lambda) Actor-Critic DMCN with causal transient spines & optional 5x5 retinotopic patches."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        seed: int,
        pixel_mode: bool = False,
        ablate_spine: bool = False,
        ablate_elig: bool = False,
        eta_actor: float = 0.08,
        eta_critic: float = 0.12,
        gamma: float = 0.95,
        lam: float = 0.85,
        beta_ent: float = 0.02,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.d_in = d_in
        self.d_hid = d_hid
        self.pixel_mode = pixel_mode
        self.ablate_spine = ablate_spine
        self.gamma = gamma
        self.beta_ent = beta_ent
        self.decay_spine = float(np.exp(-1.0 / 2.0))
        self.spine_ema = np.zeros(d_in, dtype=np.float64)
        self.spine_init = False

        d_feat_in = d_in if ablate_spine else (2 * d_in)
        self.enc = DMCNLayer(DMCNConfig(
            d_in=d_feat_in,
            d_out=d_hid,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="tanh",
            weight_init_scale=2.2 if not pixel_mode else 6.0,
            seed=seed + 101,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))
        self.enc.b_dend[0] = self.rng.normal(0.0, 0.15 if pixel_mode else 0.35, size=d_hid)

        # In pixel_mode, include direct residual skip from raw+transient pixels alongside dendritic tanh features
        d_policy_in = (d_hid + d_feat_in + 1) if pixel_mode else (d_hid + 1)

        # Local Actor (3 actions) & Local Critic (1 scalar value V(s)) using DMCNLayer eligibility traces
        self.actor = DMCNLayer(DMCNConfig(
            d_in=d_policy_in,
            d_out=3,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            weight_init_scale=0.0,
            gamma_elig=gamma * lam,
            eta_rl=eta_actor,
            gamma_forget=0.0,
            eta_consol=0.0,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_eligibility=ablate_elig),
        ))
        self.critic = DMCNLayer(DMCNConfig(
            d_in=d_policy_in,
            d_out=1,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            weight_init_scale=0.0,
            gamma_elig=gamma * lam,
            eta_rl=eta_critic,
            gamma_forget=0.0,
            eta_consol=0.0,
            seed=seed + 202,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_eligibility=ablate_elig),
        ))

    def reset_episode(self) -> None:
        self.spine_ema.fill(0.0)
        self.spine_init = False
        self.enc.reset_temporal_state()
        self.actor.reset_temporal_state()
        self.critic.reset_temporal_state()

    def encode(self, obs: np.ndarray) -> np.ndarray:
        if self.ablate_spine:
            psi = obs
        else:
            if not self.spine_init:
                self.spine_ema = obs.copy()
                self.spine_init = True
                transient = np.zeros_like(obs)
            else:
                transient = (obs - self.spine_ema) * (1.5 if self.pixel_mode else 4.0)
                self.spine_ema = self.decay_spine * self.spine_ema + (1.0 - self.decay_spine) * obs
            psi = np.concatenate([obs, transient])
        h_enc = self.enc.step(psi)
        if self.pixel_mode:
            return np.concatenate([psi, h_enc, np.array([1.0], dtype=np.float64)])
        return np.append(h_enc, 1.0)

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
        td_err = float(reward + (0.0 if done else self.gamma * v_next) - v_curr)
        td_clip = float(np.clip(td_err, -2.0, 2.0))

        # Local Actor policy score + tiny local entropy bonus
        log_p = np.log(np.clip(probs, 1e-8, 1.0))
        ent = -float(np.dot(probs, log_p))
        ent_grad = -probs * (log_p + ent)
        act_grad = -probs.copy()
        act_grad[action] += 1.0
        self.actor.accumulate_eligibility(act_grad + self.beta_ent * ent_grad)
        self.actor.apply_rpe(td_clip)

        # Local Critic TD(lambda) update
        self.critic.accumulate_eligibility(np.array([1.0], dtype=np.float64))
        self.critic.apply_rpe(td_clip)


class TorchA2CMLP(nn.Module):
    """Fair PyTorch Backprop Actor-Critic (A2C) baseline on identical 2,000-serve budget."""

    def __init__(self, d_in: int, d_hid: int, seed: int, lr: float = 2e-3, gamma: float = 0.95) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.gamma = gamma
        self.trunk = nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh())
        self.actor = nn.Linear(d_hid, 3)
        self.critic = nn.Linear(d_hid, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def train_serve(self, env: PongEnv, pixel_mode: bool = False) -> float:
        prev = None
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            cur = (env.obs_pixels() if pixel_mode else env.obs_pos()).astype(np.float32)
            diff = np.zeros_like(cur) if prev is None else (cur - prev) * (1.5 if pixel_mode else 4.0)
            prev = cur
            x = torch.tensor(np.concatenate([cur, diff]), dtype=torch.float32).unsqueeze(0)
            h = self.trunk(x)
            logits = self.actor(h).squeeze(0)
            val = self.critic(h).squeeze()
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            log_probs.append(dist.log_prob(act))
            values.append(val)
            entropies.append(dist.entropy())
            reward, done = env.step(int(act.item()))
            steps += 1

        if log_probs:
            rets = []
            r_run = float(reward)
            for _ in range(len(log_probs)):
                rets.append(r_run)
                r_run *= self.gamma
            rets.reverse()
            rets_t = torch.tensor(rets, dtype=torch.float32)
            vals_t = torch.stack(values)
            lps_t = torch.stack(log_probs)
            ents_t = torch.stack(entropies)
            adv = rets_t - vals_t.detach()
            loss = -(lps_t * adv).mean() + 0.5 * ((rets_t - vals_t) ** 2).mean() - 0.01 * ents_t.mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return reward

    @torch.no_grad()
    def eval_serves(self, rng: np.random.RandomState, n_serves: int, ood: bool = False, pixel_mode: bool = False) -> float:
        env = PongEnv()
        hits = 0
        for _ in range(n_serves):
            env.reset_serve(rng, ood=ood)
            prev = None
            done = False
            reward = 0.0
            steps = 0
            while not done and steps < 40:
                cur = (env.obs_pixels() if pixel_mode else env.obs_pos()).astype(np.float32)
                diff = np.zeros_like(cur) if prev is None else (cur - prev) * (1.5 if pixel_mode else 4.0)
                prev = cur
                x = torch.tensor(np.concatenate([cur, diff]), dtype=torch.float32).unsqueeze(0)
                act = int(torch.argmax(self.actor(self.trunk(x)), dim=1).item())
                reward, done = env.step(act)
                steps += 1
            if reward > 0.0:
                hits += 1
        return 100.0 * hits / float(n_serves)


def train_dmcn_ac(
    agent: DMCNActorCriticPong,
    rng: np.random.RandomState,
    n_serves: int = 2000,
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
        obs = env.obs_pixels() if agent.pixel_mode else env.obs_pos()
        h = agent.encode(obs)
        done = False
        steps = 0
        final_r = 0.0
        while not done and steps < 40:
            act, probs, v_curr = agent.select_action(h, greedy=False)
            r_raw, done = env.step(act)
            if done:
                final_r = r_raw
                r_learn = float(rng.choice([-1.0, 1.0])) if shuffle_reward else r_raw
                agent.td_update(act, probs, v_curr, r_learn, 0.0, True)
            else:
                obs_next = env.obs_pixels() if agent.pixel_mode else env.obs_pos()
                h_next = agent.encode(obs_next)
                v_next = float(agent.critic.step(h_next)[0])
                # Restore actor/critic cached _last_psi for current step h before td_update!
                agent.actor.step(h)
                agent.critic.step(h)
                agent.td_update(act, probs, v_curr, 0.0, v_next, False)
                h = h_next
            steps += 1
        if final_r > 0.0:
            bin_hits += 1
        if record_curve and (s + 1) % bin_sz == 0:
            curve.append(100.0 * bin_hits / float(bin_sz))
            bin_hits = 0
    return curve


def eval_dmcn_ac(
    agent: DMCNActorCriticPong,
    rng: np.random.RandomState,
    n_serves: int = 100,
    ood: bool = False,
    blind: bool = False,
    record_traces: int = 0,
) -> Dict[str, Any]:
    env = PongEnv()
    obs_dim = 256 if agent.pixel_mode else 3
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
            obs = np.zeros(obs_dim, dtype=np.float64) if blind else (env.obs_pixels() if agent.pixel_mode else env.obs_pos())
            h = agent.encode(obs)
            act, _, _ = agent.select_action(h, greedy=True)
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


def probe_velocity_sensitivity_v2(agent: DMCNActorCriticPong) -> float:
    deltas = []
    for py in (0.35, 0.50, 0.65):
        agent.reset_episode()
        for k in range(4, -1, -1):
            obs_u = np.array([0.5 - 0.06 * k - 0.5, py - 0.05 * k - 0.5, py - 0.5], dtype=np.float64)
            _, probs_u, _ = agent.select_action(agent.encode(obs_u), greedy=True)
        agent.reset_episode()
        for k in range(4, -1, -1):
            obs_d = np.array([0.5 - 0.06 * k - 0.5, py + 0.05 * k - 0.5, py - 0.5], dtype=np.float64)
            _, probs_d, _ = agent.select_action(agent.encode(obs_d), greedy=True)
        deltas.append(float(probs_u[2] - probs_d[2]))
    return 100.0 * float(np.mean(deltas))


def evaluate_seed(seed: int, n_train: int = 2000, n_eval: int = 100, record_extras: bool = False) -> Dict[str, Any]:
    # --- Tier 1: Position-Only Pong (NO velocity oracle) ---
    dmcn_t1 = DMCNActorCriticPong(d_in=3, d_hid=64, seed=seed, pixel_mode=False, ablate_spine=False, ablate_elig=False, eta_actor=0.14, eta_critic=0.20)
    dmcn_nospine = DMCNActorCriticPong(d_in=3, d_hid=64, seed=seed, pixel_mode=False, ablate_spine=True, ablate_elig=False, eta_actor=0.14, eta_critic=0.20)
    dmcn_noelig = DMCNActorCriticPong(d_in=3, d_hid=64, seed=seed, pixel_mode=False, ablate_spine=False, ablate_elig=True, eta_actor=0.14, eta_critic=0.20)
    dmcn_untr = DMCNActorCriticPong(d_in=3, d_hid=64, seed=seed, pixel_mode=False)
    dmcn_shuf = DMCNActorCriticPong(d_in=3, d_hid=64, seed=seed, pixel_mode=False, eta_actor=0.14, eta_critic=0.20)

    untr_eval = eval_dmcn_ac(dmcn_untr, np.random.RandomState(seed + 5000), n_eval, record_traces=6 if record_extras else 0)
    curve_t1 = train_dmcn_ac(dmcn_t1, np.random.RandomState(seed + 1000), n_train, record_curve=record_extras)
    train_dmcn_ac(dmcn_nospine, np.random.RandomState(seed + 1000), n_train)
    train_dmcn_ac(dmcn_noelig, np.random.RandomState(seed + 1000), n_train)
    train_dmcn_ac(dmcn_shuf, np.random.RandomState(seed + 1000), n_train, shuffle_reward=True)

    t1_id = eval_dmcn_ac(dmcn_t1, np.random.RandomState(seed + 5000), n_eval, ood=False, record_traces=6 if record_extras else 0)
    t1_ood = eval_dmcn_ac(dmcn_t1, np.random.RandomState(seed + 6000), n_eval, ood=True)
    t1_blind = eval_dmcn_ac(dmcn_t1, np.random.RandomState(seed + 5000), n_eval, blind=True)
    t1_nospine = eval_dmcn_ac(dmcn_nospine, np.random.RandomState(seed + 5000), n_eval)
    t1_noelig = eval_dmcn_ac(dmcn_noelig, np.random.RandomState(seed + 5000), n_eval)
    t1_shuf = eval_dmcn_ac(dmcn_shuf, np.random.RandomState(seed + 5000), n_eval)

    bp_t1 = TorchA2CMLP(d_in=6, d_hid=64, seed=seed, lr=3e-3)
    env_bp = PongEnv()
    rng_bp = np.random.RandomState(seed + 1000)
    for _ in range(n_train):
        env_bp.reset_serve(rng_bp, ood=False)
        bp_t1.train_serve(env_bp, pixel_mode=False)
    bp_t1_id = bp_t1.eval_serves(np.random.RandomState(seed + 5000), n_eval, ood=False, pixel_mode=False)
    bp_t1_ood = bp_t1.eval_serves(np.random.RandomState(seed + 6000), n_eval, ood=True, pixel_mode=False)

    # --- Tier 2: Raw 16x16 Pixel Pong (256 pixels ONLY, ZERO coordinates) ---
    dmcn_t2 = DMCNActorCriticPong(d_in=256, d_hid=128, seed=seed + 77, pixel_mode=True, ablate_spine=False, ablate_elig=False, eta_actor=0.35, eta_critic=0.45)
    curve_t2 = train_dmcn_ac(dmcn_t2, np.random.RandomState(seed + 2000), n_train, record_curve=record_extras)
    t2_id = eval_dmcn_ac(dmcn_t2, np.random.RandomState(seed + 7000), n_eval, ood=False)
    t2_blind = eval_dmcn_ac(dmcn_t2, np.random.RandomState(seed + 7000), n_eval, ood=False, blind=True)

    bp_t2 = TorchA2CMLP(d_in=512, d_hid=128, seed=seed + 77, lr=1.5e-3)
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
        "t1_dmcn_vel_sensitivity_pp": probe_velocity_sensitivity_v2(dmcn_t1),
        "t1_ablate_spine_hit_rate": t1_nospine["hit_rate"],
        "t1_ablate_elig_hit_rate": t1_noelig["hit_rate"],
        "t1_bp_a2c_hit_rate": bp_t1_id,
        "t1_bp_a2c_ood_hit_rate": bp_t1_ood,
        "t1_untrained_hit_rate": untr_eval["hit_rate"],
        "t1_blind_hit_rate": t1_blind["hit_rate"],
        "t1_shuffled_reward_hit_rate": t1_shuf["hit_rate"],
        "t1_random_hit_rate": eval_trivial_policy("random", np.random.RandomState(seed + 5000), n_eval),
        "t1_const_stay_hit_rate": eval_trivial_policy("const_stay", np.random.RandomState(seed + 5000), n_eval),
        "t1_const_up_hit_rate": eval_trivial_policy("const_up", np.random.RandomState(seed + 5000), n_eval),
        "t2_pixel_dmcn_hit_rate": t2_id["hit_rate"],
        "t2_pixel_dmcn_dir_align_acc": t2_id["dir_align_acc"],
        "t2_pixel_dmcn_blind_hit_rate": t2_blind["hit_rate"],
        "t2_pixel_bp_a2c_hit_rate": bp_t2_id,
    }
    if record_extras:
        res["_extras"] = {
            "curve_t1": curve_t1,
            "curve_t2": curve_t2,
            "traces_untrained": untr_eval["traces"],
            "traces_trained": t1_id["traces"],
        }
    return res


def _eval_worker(seed: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    return evaluate_seed(seed, record_extras=(seed == 2000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_pong_rl_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999, record_extras=False)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_pong_rl_v2_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "n_train": 2000, "n_eval": 100},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    from concurrent.futures import ProcessPoolExecutor

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        per_seed_raw = list(ex.map(_eval_worker, test_seeds))
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
        exp_id="dmcn_pong_rl_v2",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_train": 2000, "n_eval": 100, "d_hid_t1": 64, "d_hid_t2": 128, "method": "local_td_lambda_ac_transient_spines"},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
