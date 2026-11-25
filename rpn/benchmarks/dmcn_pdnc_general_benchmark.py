"""Pre-registered 20-Seed General PDNC Benchmark (`dmcn_pdnc_general_v1`).

Tests one unified gradient-free & weight-transport-free learning algorithm —
Prospective Dendritic Natural-Credit (PDNC) — across three distinct regimes:
  - Suite A: Raw 16x16 Pixel Pong RL (sparse +1/-1, 256 pixels, zero coordinates)
  - Suite B: 24D Noisy XOR Nonlinear Feature Discovery (2 signal dims in 22 noise dims)
  - Suite C: Boundary-Free Continual Learning (3-task 24D subspace stream + compact
    256-unit Split-FashionMNIST without replay or task boundaries)

Includes every pre-registered ablation (`ablate_l1`, `ablate_kp`, `ablate_elig`,
`shuffle_apical`, `ablate_prospective_settle`, `ablate_subspace`), PyTorch Backprop
baselines on equal data/budgets, and full null controls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from neuron_v2.dmcn import (  # noqa: E402
    PDNCActorCritic,
    PDNCConfig,
    PDNCNetwork,
    PDNCTorchNetwork,
)
from benchmarks.dmcn_cl_literature_baselines_benchmark import get_fmnist_flat  # noqa: E402
from benchmarks.dmcn_pong_rl_benchmark import PongEnv, eval_trivial_policy, summarize  # noqa: E402
from benchmarks.dmcn_pong_rl_v2_benchmark import TorchA2CMLP  # noqa: E402


# ---------------- Suite A: Raw 16x16 Pixel Pong RL ----------------


def _make_pong_pdnc(
    seed: int,
    *,
    freeze_l1: bool = False,
    freeze_kp: bool = False,
    shuffle_apical: bool = False,
    ablate_elig: bool = False,
) -> PDNCActorCritic:
    return PDNCActorCritic(
        enc_cfg=PDNCConfig(
            d_in=256,
            d_out=128,
            activation="tanh",
            use_transient_spine=True,
            tau_spine=2.0,
            transient_scale=1.5,
            precision_mode="diagonal",
            eta=0.002,
            weight_decay=0.005,
            beta_mom=0.90,
            beta_rms=0.99,
            eps_prec=1e-8,
            lambda_kp=0.001,
            weight_init_scale=0.8,
            bias_init_scale=0.10,
            fb_init_scale=0.3,
            freeze_weights=freeze_l1,
            freeze_random_feedback=freeze_kp,
            shuffle_apical=shuffle_apical,
            ablate_eligibility=ablate_elig,
            seed=seed + 77,
        ),
        n_actions=3,
        eta_actor=0.002,
        eta_critic=0.003,
        gamma=0.95,
        lam=0.0 if ablate_elig else 0.995,
        critic_apical_weight=0.5,
        beta_ent=0.01,
        eps_explore=0.0,
        seed=seed,
    )


def _train_pong_pdnc(
    agent: PDNCActorCritic,
    rng: np.random.RandomState,
    n_serves: int = 2000,
    shuffle_reward: bool = False,
) -> List[float]:
    env = PongEnv()
    curve: List[float] = []
    bin_hits = 0
    bin_sz = max(1, n_serves // 10)
    for s in range(n_serves):
        env.reset_serve(rng, ood=False)
        agent.reset_episode()
        obs = env.obs_pixels()
        done = False
        steps = 0
        final_r = 0.0
        while not done and steps < 40:
            act, v_curr = agent.step_policy(obs, greedy=False)
            r_raw, done = env.step(act)
            if done:
                final_r = r_raw
                r_learn = float(rng.choice([-1.0, 1.0])) if shuffle_reward else r_raw
                agent.apply_td_update(v_curr, r_learn, 0.0, True)
            else:
                obs_n = env.obs_pixels()
                v_next = agent.value_only(obs_n)
                agent.apply_td_update(v_curr, 0.0, v_next, False)
                obs = obs_n
            steps += 1
        if final_r > 0.0:
            bin_hits += 1
        if (s + 1) % bin_sz == 0:
            curve.append(100.0 * bin_hits / float(bin_sz))
            bin_hits = 0
    return curve


def _eval_pong_pdnc(
    agent: PDNCActorCritic,
    rng: np.random.RandomState,
    n_serves: int = 100,
    ood: bool = False,
    blind: bool = False,
) -> Tuple[float, float]:
    env = PongEnv()
    hits = 0
    dir_ok, dir_tot = 0, 0
    for _ in range(n_serves):
        env.reset_serve(rng, ood=ood)
        agent.reset_episode()
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 40:
            obs = np.zeros(256, dtype=np.float64) if blind else env.obs_pixels()
            act, _ = agent.step_policy(obs, greedy=True)
            if abs(env.ball_y - env.paddle_y) > 0.05:
                dir_tot += 1
                if (env.ball_y > env.paddle_y and act == 2) or (env.ball_y < env.paddle_y and act == 0):
                    dir_ok += 1
            reward, done = env.step(act)
            steps += 1
        if reward > 0.0:
            hits += 1
    return 100.0 * hits / float(n_serves), 100.0 * dir_ok / float(max(1, dir_tot))


def run_suite_a(seed: int, n_train: int = 2000, n_eval: int = 100) -> Dict[str, Any]:
    ag_full = _make_pong_pdnc(seed)
    ag_nol1 = _make_pong_pdnc(seed, freeze_l1=True)
    ag_nokp = _make_pong_pdnc(seed, freeze_kp=True)
    ag_noelig = _make_pong_pdnc(seed, ablate_elig=True)
    ag_shuf_ap = _make_pong_pdnc(seed, shuffle_apical=True)
    ag_untr = _make_pong_pdnc(seed)
    ag_shuf_r = _make_pong_pdnc(seed)

    curve_full = _train_pong_pdnc(ag_full, np.random.RandomState(seed + 2000), n_train)
    _train_pong_pdnc(ag_nol1, np.random.RandomState(seed + 2000), n_train)
    _train_pong_pdnc(ag_nokp, np.random.RandomState(seed + 2000), n_train)
    _train_pong_pdnc(ag_noelig, np.random.RandomState(seed + 2000), n_train)
    _train_pong_pdnc(ag_shuf_ap, np.random.RandomState(seed + 2000), n_train)
    _train_pong_pdnc(ag_shuf_r, np.random.RandomState(seed + 2000), n_train, shuffle_reward=True)

    hit_full, dir_full = _eval_pong_pdnc(ag_full, np.random.RandomState(seed + 7000), n_eval)
    hit_ood, _ = _eval_pong_pdnc(ag_full, np.random.RandomState(seed + 8000), n_eval, ood=True)
    hit_blind, _ = _eval_pong_pdnc(ag_full, np.random.RandomState(seed + 7000), n_eval, blind=True)
    hit_untr, _ = _eval_pong_pdnc(ag_untr, np.random.RandomState(seed + 7000), n_eval)
    hit_nol1, dir_nol1 = _eval_pong_pdnc(ag_nol1, np.random.RandomState(seed + 7000), n_eval)
    hit_nokp, _ = _eval_pong_pdnc(ag_nokp, np.random.RandomState(seed + 7000), n_eval)
    hit_noelig, _ = _eval_pong_pdnc(ag_noelig, np.random.RandomState(seed + 7000), n_eval)
    hit_shuf_ap, _ = _eval_pong_pdnc(ag_shuf_ap, np.random.RandomState(seed + 7000), n_eval)
    hit_shuf_r, _ = _eval_pong_pdnc(ag_shuf_r, np.random.RandomState(seed + 7000), n_eval)

    bp = TorchA2CMLP(d_in=512, d_hid=128, seed=seed + 77, lr=1.5e-3)
    env_bp = PongEnv()
    rng_bp = np.random.RandomState(seed + 2000)
    for _ in range(n_train):
        env_bp.reset_serve(rng_bp, ood=False)
        bp.train_serve(env_bp, pixel_mode=True)
    hit_bp = bp.eval_serves(np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True)
    hit_bp_ood = bp.eval_serves(np.random.RandomState(seed + 8000), n_eval, ood=True, pixel_mode=True)

    return {
        "metrics": {
            "t2_pdnc_pixel_hit_rate": hit_full,
            "t2_pdnc_pixel_ood_hit_rate": hit_ood,
            "t2_pdnc_pixel_dir_align": dir_full,
            "t2_pdnc_kp_align_cos": ag_full.actor.feedback_alignment_cosine(),
            "t2_ablate_l1_learning_hit_rate": hit_nol1,
            "t2_ablate_l1_dir_align": dir_nol1,
            "t2_ablate_kp_alignment_hit_rate": hit_nokp,
            "t2_ablate_elig_trace_hit_rate": hit_noelig,
            "t2_shuffle_apical_hit_rate": hit_shuf_ap,
            "t2_bp_a2c_pixel_hit_rate": hit_bp,
            "t2_bp_a2c_pixel_ood_hit_rate": hit_bp_ood,
            "t2_pixel_untrained_hit_rate": hit_untr,
            "t2_pixel_blind_hit_rate": hit_blind,
            "t2_pixel_shuffled_reward_hit_rate": hit_shuf_r,
            "t2_pixel_random_hit_rate": eval_trivial_policy("random", np.random.RandomState(seed + 7000), n_eval),
        },
        "curve_pdnc": curve_full,
    }


# ---------------- Suite B: 24D Noisy XOR Feature Discovery ----------------


def run_suite_b(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed + 300)
    n_tr, n_te, d_in = 1600, 400, 24
    X_tr = rng.randn(n_tr, d_in)
    Y_tr = np.sign(X_tr[:, 0] * X_tr[:, 1]).reshape(-1, 1)
    X_te = rng.randn(n_te, d_in)
    Y_te = np.sign(X_te[:, 0] * X_te[:, 1]).reshape(-1, 1)

    def make_net(freeze_l1: bool = False, ablate_settle: bool = False) -> PDNCNetwork:
        return PDNCNetwork(
            [
                PDNCConfig(
                    d_in=d_in,
                    d_out=32,
                    activation="tanh",
                    precision_mode="diagonal",
                    eta=0.018,
                    weight_decay=0.50,
                    prospective_beta=0.8,
                    weight_init_scale=0.6,
                    bias_init_scale=0.3,
                    freeze_weights=freeze_l1,
                    seed=seed + 301,
                ),
                PDNCConfig(
                    d_in=32,
                    d_out=1,
                    activation="linear",
                    precision_mode="diagonal",
                    eta=0.015,
                    weight_decay=0.1,
                    weight_init_scale=0.0,
                    fb_init_scale=0.4,
                    seed=seed + 302,
                ),
            ],
            ablate_prospective_settle=ablate_settle,
        )

    net_pdnc = make_net(freeze_l1=False, ablate_settle=False)
    net_froz = make_net(freeze_l1=True, ablate_settle=False)
    net_noset = make_net(freeze_l1=False, ablate_settle=True)
    net_shuf = make_net(freeze_l1=False, ablate_settle=False)

    torch.manual_seed(seed + 301)
    bp = nn.Sequential(nn.Linear(d_in, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(bp.parameters(), lr=0.02, weight_decay=0.05)

    Y_shuf = Y_tr[rng.permutation(n_tr)]
    for _ in range(25):
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, 32):
            idx = perm[s : s + 32]
            net_pdnc.step(X_tr[idx])
            net_pdnc.learn_prospective(Y_tr[idx])
            net_froz.step(X_tr[idx])
            net_froz.learn_prospective(Y_tr[idx])
            net_noset.step(X_tr[idx])
            net_noset.learn_prospective(Y_tr[idx])
            net_shuf.step(X_tr[idx])
            net_shuf.learn_prospective(Y_shuf[idx])

            xb = torch.tensor(X_tr[idx], dtype=torch.float32)
            yb = torch.tensor(Y_tr[idx], dtype=torch.float32)
            loss = ((bp(xb) - yb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        bp_acc = 100.0 * float(
            (torch.sign(bp(torch.tensor(X_te, dtype=torch.float32))).numpy() == Y_te).mean()
        )
    sig_e = float(np.mean(net_pdnc.layers[0].W[:, :2] ** 2))
    noise_e = float(np.mean(net_pdnc.layers[0].W[:, 2:d_in] ** 2))
    return {
        "xor_pdnc_acc": 100.0 * float(np.mean(np.sign(net_pdnc.step(X_te)) == Y_te)),
        "xor_pdnc_signal_noise_ratio": sig_e / max(1e-9, noise_e),
        "xor_ablate_l1_acc": 100.0 * float(np.mean(np.sign(net_froz.step(X_te)) == Y_te)),
        "xor_ablate_settle_acc": 100.0 * float(np.mean(np.sign(net_noset.step(X_te)) == Y_te)),
        "xor_bp_mlp_acc": bp_acc,
        "xor_shuffled_label_acc": 100.0 * float(np.mean(np.sign(net_shuf.step(X_te)) == Y_te)),
    }


# ---------------- Suite C1: Boundary-Free 3-Task Orthogonal Subspace CL ----------------


def run_suite_c1(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed + 500)
    q_orth, _ = np.linalg.qr(rng.randn(24, 24))
    tasks_tr: List[Tuple[np.ndarray, np.ndarray]] = []
    tasks_te: List[Tuple[np.ndarray, np.ndarray]] = []
    for t_idx in range(3):
        qk = q_orth[:, 8 * t_idx : 8 * (t_idx + 1)]
        z_tr = rng.randn(256, 8)
        X_tr = z_tr @ qk.T
        Y_tr = np.sign(z_tr[:, 0] * z_tr[:, 1]).reshape(-1, 1)
        z_te = rng.randn(200, 8)
        X_te = z_te @ qk.T
        Y_te = np.sign(z_te[:, 0] * z_te[:, 1]).reshape(-1, 1)
        tasks_tr.append((X_tr, Y_tr))
        tasks_te.append((X_te, Y_te))

    def make_cl(p_mode: str) -> PDNCNetwork:
        return PDNCNetwork([
            PDNCConfig(
                d_in=24,
                d_out=48,
                activation="tanh",
                precision_mode=p_mode,
                alpha_null=0.04,
                scale_null_by_batch=True,
                eta=0.03,
                weight_decay=0.01,
                bias_init_scale=0.35,
                prospective_beta=0.8,
                lambda_kp=0.0,
                seed=seed + 501,
            ),
            PDNCConfig(
                d_in=48,
                d_out=1,
                activation="linear",
                precision_mode=p_mode,
                alpha_null=0.04,
                scale_null_by_batch=True,
                eta=0.03,
                weight_decay=0.01,
                lambda_kp=0.0,
                weight_init_scale=0.0,
                fb_init_scale=0.5,
                seed=seed + 502,
            ),
        ])

    net_sub = make_cl("subspace")
    net_diag = make_cl("diagonal")
    torch.manual_seed(seed + 501)
    bp = nn.Sequential(nn.Linear(24, 48), nn.Tanh(), nn.Linear(48, 1))
    opt = torch.optim.AdamW(bp.parameters(), lr=0.02, weight_decay=0.01)

    for X_tr, Y_tr in tasks_tr:
        for _ in range(25):
            for s in range(0, 256, 32):
                net_sub.step(X_tr[s : s + 32])
                net_sub.learn_prospective(Y_tr[s : s + 32])
                net_diag.step(X_tr[s : s + 32])
                net_diag.learn_prospective(Y_tr[s : s + 32])
                xb = torch.tensor(X_tr[s : s + 32], dtype=torch.float32)
                yb = torch.tensor(Y_tr[s : s + 32], dtype=torch.float32)
                loss = ((bp(xb) - yb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    sub_accs = [100.0 * float(np.mean(np.sign(net_sub.step(Xt)) == Yt)) for Xt, Yt in tasks_te]
    diag_accs = [100.0 * float(np.mean(np.sign(net_diag.step(Xt)) == Yt)) for Xt, Yt in tasks_te]
    with torch.no_grad():
        bp_accs = [
            100.0 * float((torch.sign(bp(torch.tensor(Xt, dtype=torch.float32))).numpy() == Yt).mean())
            for Xt, Yt in tasks_te
        ]

    return {
        "cl3_subspace_mean_acc": float(np.mean(sub_accs)),
        "cl3_subspace_t0_ret": sub_accs[0],
        "cl3_subspace_t2_acc": sub_accs[2],
        "cl3_diagonal_mean_acc": float(np.mean(diag_accs)),
        "cl3_diagonal_t0_ret": diag_accs[0],
        "cl3_bp_sgd_mean_acc": float(np.mean(bp_accs)),
        "cl3_bp_sgd_t0_ret": bp_accs[0],
    }


# ---------------- Suite C2: Compact 256-Unit Split-FashionMNIST (CUDA/CPU) ----------------


def run_suite_c2(seed: int, device: torch.device) -> Dict[str, float]:
    torch.manual_seed(seed + 700)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 700)
    x_tr, y_tr, x_te, y_te = get_fmnist_flat(device)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    p0_te = (y_te == 0) | (y_te == 1)

    def _run(mode_l1: str, freeze_l1: bool) -> Tuple[float, float]:
        net = PDNCTorchNetwork(
            [
                PDNCConfig(
                    d_in=784,
                    d_out=256,
                    activation="tanh",
                    precision_mode=mode_l1,
                    alpha_null=0.05,
                    scale_null_by_batch=True,
                    eta=0.004,
                    weight_decay=0.02,
                    weight_init_scale=1.2,
                    bias_init_scale=0.35,
                    prospective_beta=0.8,
                    lambda_kp=0.0,
                    freeze_weights=freeze_l1,
                    seed=seed + 701,
                ),
                PDNCConfig(
                    d_in=256,
                    d_out=10,
                    activation="linear",
                    precision_mode="woodbury",
                    precision_ridge=20.0,
                    alpha_null=1.0,
                    eta=1.0,
                    weight_init_scale=0.0,
                    fb_init_scale=0.5,
                    seed=seed + 702,
                ),
            ],
            device=device,
            dtype=torch.float32,
        )
        gen = torch.Generator(device=device).manual_seed(seed + 703)
        n_ep = 1 if freeze_l1 else 3
        for c0, c1 in pairs:
            m = (y_tr == c0) | (y_tr == c1)
            xp, yp = x_tr[m], y_tr[m]
            yp_oh = torch.zeros(xp.size(0), 10, device=device, dtype=torch.float32)
            yp_oh.scatter_(1, yp.unsqueeze(1), 1.0)
            for _ in range(n_ep):
                perm = torch.randperm(xp.size(0), generator=gen, device=device)
                for s in range(0, xp.size(0), 256):
                    idx = perm[s : s + 256]
                    net(xp[idx])
                    net.learn_prospective(yp_oh[idx])
        with torch.no_grad():
            logits = net(x_te)
            all_acc = 100.0 * float((logits.argmax(1) == y_te).float().mean().item())
            p0_acc = 100.0 * float((logits[p0_te].argmax(1) == y_te[p0_te]).float().mean().item())
        return all_acc, p0_acc

    sub_all, sub_p0 = _run("subspace", False)
    diag_all, diag_p0 = _run("diagonal", False)
    froz_all, froz_p0 = _run("diagonal", True)
    return {
        "fmnist_subspace_acc": sub_all,
        "fmnist_subspace_p0_ret": sub_p0,
        "fmnist_diagonal_acc": diag_all,
        "fmnist_diagonal_p0_ret": diag_p0,
        "fmnist_frozen_woodbury_acc": froz_all,
        "fmnist_frozen_woodbury_p0_ret": froz_p0,
    }


def _cpu_worker(seed: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    sa = run_suite_a(seed)
    sb = run_suite_b(seed)
    sc1 = run_suite_c1(seed)
    row: Dict[str, Any] = {"seed": seed}
    row.update(sa["metrics"])
    row.update(sb)
    row.update(sc1)
    if seed in (999, 2000):
        row["_curve_pdnc_pixel"] = sa["curve_pdnc"]
    return row


def evaluate_all_seeds(seeds: List[int]) -> Tuple[List[Dict[str, Any]], List[float]]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(max_workers=4) as ex:
        cpu_rows = list(ex.map(_cpu_worker, seeds))

    curve_2000: List[float] = []
    for row in cpu_rows:
        if "_curve_pdnc_pixel" in row:
            curve_2000 = row.pop("_curve_pdnc_pixel")
        row.update(run_suite_c2(int(row["seed"]), dev))
    return cpu_rows, curve_2000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_pdnc_general_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        rows, curve_val = evaluate_all_seeds([999])
        elapsed = time.perf_counter() - t0
        res = rows[0]
        res["curve_val_999"] = curve_val
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_pdnc_general_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999"},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed, curve_2000 = evaluate_all_seeds(test_seeds)
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    # Evaluate pre-registered falsifiers honestly in-script
    falsifiers = {
        "F1_pixel_hits_ge_65_and_beats_frozen_by_12pp": bool(
            summary["t2_pdnc_pixel_hit_rate"]["mean"] >= 65.0
            and (summary["t2_pdnc_pixel_hit_rate"]["mean"] - summary["t2_ablate_l1_learning_hit_rate"]["mean"]) >= 12.0
        ),
        "F2_pixel_within_8pp_of_bp_a2c": bool(
            summary["t2_pdnc_pixel_hit_rate"]["mean"] >= summary["t2_bp_a2c_pixel_hit_rate"]["mean"] - 8.0
        ),
        "F3_pixel_null_controls_le_38": bool(
            max(
                summary["t2_pixel_untrained_hit_rate"]["mean"],
                summary["t2_pixel_blind_hit_rate"]["mean"],
                summary["t2_pixel_shuffled_reward_hit_rate"]["mean"],
                summary["t2_pixel_random_hit_rate"]["mean"],
            )
            <= 38.0
        ),
        "F4_xor_ge_85_and_beats_frozen_by_20pp": bool(
            summary["xor_pdnc_acc"]["mean"] >= 85.0
            and (summary["xor_pdnc_acc"]["mean"] - summary["xor_ablate_l1_acc"]["mean"]) >= 20.0
        ),
        "F5_continual_subspace_ret_and_fmnist_gain": bool(
            summary["cl3_subspace_t0_ret"]["mean"] >= 80.0
            and (summary["cl3_subspace_mean_acc"]["mean"] - summary["cl3_diagonal_mean_acc"]["mean"]) >= 15.0
            and (summary["fmnist_subspace_acc"]["mean"] - summary["fmnist_diagonal_acc"]["mean"]) >= 25.0
        ),
    }

    payload = {
        "headline_metric": "t2_pdnc_pixel_hit_rate",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "falsifiers": falsifiers,
        "all_passed": bool(all(falsifiers.values())),
        "seed_2000_pixel_curve": curve_2000,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_pdnc_general_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={
            "suite_a": "Raw_16x16_Pixel_Pong_2000serves",
            "suite_b": "Noisy_24D_XOR_1600tr_400te",
            "suite_c1": "BoundaryFree_3Task_24D_Subspace_CL",
            "suite_c2": "Split_FashionMNIST_d256_0Replay",
            "algorithm": "PDNC",
        },
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "falsifiers": falsifiers, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
