"""Pre-registered 20-Seed DMCN v4 Efficiency & Local-Patch PDNC v2 Benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import (  # noqa: E402
    AblationConfig,
    DMCNConfig,
    DMCNTorchLayer,
    PDNCActorCritic,
    PDNCConfig,
    make_patch_receptive_field_mask,
)
from benchmarks.dmcn_cl_literature_baselines_benchmark import get_mnist_flat  # noqa: E402
from benchmarks.dmcn_pong_rl_benchmark import PongEnv, eval_trivial_policy, summarize  # noqa: E402
from benchmarks.dmcn_pong_rl_v2_benchmark import TorchA2CMLP  # noqa: E402
from benchmarks.dmcn_reviewer_controls_benchmark import eval_acc, get_cifar100_mobilenet_features  # noqa: E402


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_part_a_cuda(seed: int, device: torch.device) -> Dict[str, float]:
    """Compare Woodbury vs Gram-Cholesky vs 95% Sparse Ternary+Gram-Cholesky vs Adam MLP on Split-MNIST & Split-CIFAR-100."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)

    # ---- 1. Split-MNIST 5 pairs (784 -> 2048 -> 10, 1 pass, batch_size=256) ----
    x_tr, y_tr, x_te, y_te = get_mnist_flat(device)
    y_tr_oh = F.one_hot(y_tr, 10).float()
    pairs_m = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    task_indices_m = []
    for c0, c1 in pairs_m:
        idx_np = np.flatnonzero(((y_tr == c0) | (y_tr == c1)).cpu().numpy())
        rng.shuffle(idx_np)
        task_indices_m.append(torch.tensor(idx_np, device=device, dtype=torch.long))

    def _run_mnist_variant(mode: str, sparse_ternary: bool) -> Tuple[float, float]:
        rng_b = np.random.RandomState(seed + 555)
        exp_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=784, d_out=2048, n_branches=1, n_spine_scales=0,
                soma_activation="tanh", weight_init_scale=2.0,
                use_sparse_ternary=sparse_ternary, ternary_density=0.05,
                seed=seed + 101, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        exp_l.b_dend[0].copy_(torch.tensor(rng_b.normal(0.0, 0.35, size=2048), device=device, dtype=torch.float32))
        head_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=2049, d_out=10, n_branches=1, n_spine_scales=0,
                soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
                weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=5.0,
                precision_update_mode=mode, seed=seed, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        # Warmup once
        with torch.no_grad():
            _ = exp_l(x_tr[:256])
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            ones_buf = torch.ones((256, 1), device=device, dtype=torch.float32)
            for idx_t in task_indices_m:
                for s in range(0, idx_t.shape[0], 256):
                    bi = idx_t[s : s + 256]
                    fb = torch.cat([exp_l(x_tr[bi]), ones_buf[: bi.shape[0]]], dim=1)
                    head_l.learn_local(y_tr_oh[bi] - head_l(fb))
                if mode == "gram_cholesky":
                    head_l.consolidate_gram()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            fte = torch.cat([exp_l(x_te), torch.ones((x_te.shape[0], 1), device=device)], dim=1)
            acc = eval_acc(head_l(fte), y_te)
        del exp_l, head_l
        return acc, float(len(y_tr)) / max(1e-6, dt)

    acc_m_w, sps_m_w = _run_mnist_variant("woodbury", False)
    _cleanup_cuda(device)
    acc_m_g, sps_m_g = _run_mnist_variant("gram_cholesky", False)
    _cleanup_cuda(device)
    acc_m_tg, sps_m_tg = _run_mnist_variant("gram_cholesky", True)
    _cleanup_cuda(device)

    # PyTorch Backprop Adam reference throughput (1 pass 60k)
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(784, 2048), nn.Tanh(), nn.Linear(2048, 10)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for idx_t in task_indices_m:
        for s in range(0, idx_t.shape[0], 256):
            bi = idx_t[s : s + 256]
            opt.zero_grad(set_to_none=True)
            ce(mlp(x_tr[bi]), y_tr[bi]).backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sps_m_adam = float(len(y_tr)) / max(1e-6, time.perf_counter() - t0)
    del mlp, opt
    _cleanup_cuda(device)

    # ---- 2. Split-CIFAR-100 on MobileNetV2 1280-D (10 tasks x 10 classes) ----
    cx_tr, cy_tr, cx_te, cy_te = get_cifar100_mobilenet_features(device)
    cx_tr_n = F.normalize(cx_tr, dim=1) * np.sqrt(1280.0)
    cx_te_n = F.normalize(cx_te, dim=1) * np.sqrt(1280.0)
    cy_tr_oh = F.one_hot(cy_tr, 100).float()
    c_tasks = []
    for t in range(10):
        idx_np = np.flatnonzero(((cy_tr >= 10 * t) & (cy_tr < 10 * (t + 1))).cpu().numpy())
        rng.shuffle(idx_np)
        c_tasks.append(torch.tensor(idx_np, device=device, dtype=torch.long))
    shuf_perm = torch.tensor(rng.permutation(len(cy_tr)), device=device, dtype=torch.long)
    cy_tr_shuf_oh = cy_tr_oh[shuf_perm]

    def _run_c100_variant(mode: str, sparse_ternary: bool, shuffled: bool = False) -> Tuple[float, float, float]:
        rng_b = np.random.RandomState(seed + 666)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        exp_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=1280, d_out=4096, n_branches=1, n_spine_scales=0,
                soma_activation="tanh", weight_init_scale=1.0,
                use_sparse_ternary=sparse_ternary, ternary_density=0.05,
                seed=seed + 201, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        exp_l.b_dend[0].copy_(torch.tensor(rng_b.normal(0.0, 0.35, size=4096), device=device, dtype=torch.float32))
        head_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=1280 + 4096 + 1, d_out=100, n_branches=1, n_spine_scales=0,
                soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
                weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=20.0,
                precision_update_mode=mode, seed=seed, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        with torch.no_grad():
            _ = exp_l(cx_tr_n[:256])
        if device.type == "cuda":
            torch.cuda.synchronize()

        tgt_src = cy_tr_shuf_oh if shuffled else cy_tr_oh
        t0 = time.perf_counter()
        with torch.no_grad():
            for idx_t in c_tasks:
                for s in range(0, idx_t.shape[0], 256):
                    bi = idx_t[s : s + 256]
                    xb = cx_tr_n[bi]
                    fb = torch.cat([xb, exp_l(xb), torch.ones((xb.shape[0], 1), device=device)], dim=1)
                    head_l.learn_local(tgt_src[bi] - head_l(fb))
                if mode == "gram_cholesky":
                    head_l.consolidate_gram()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            vram_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
            fte = torch.cat([cx_te_n, exp_l(cx_te_n), torch.ones((cx_te_n.shape[0], 1), device=device)], dim=1)
            acc = eval_acc(head_l(fte), cy_te)
        del exp_l, head_l
        return acc, dt, vram_mb

    acc_c_w, dt_c_w, vram_c_w = _run_c100_variant("woodbury", False)
    _cleanup_cuda(device)
    acc_c_g, dt_c_g, vram_c_g = _run_c100_variant("gram_cholesky", False)
    _cleanup_cuda(device)
    acc_c_tg, dt_c_tg, vram_c_tg = _run_c100_variant("gram_cholesky", True)
    _cleanup_cuda(device)
    acc_c_shuf, _, _ = _run_c100_variant("gram_cholesky", False, shuffled=True)
    _cleanup_cuda(device)

    return {
        "mnist__woodbury_acc": acc_m_w,
        "mnist__woodbury_sps": sps_m_w,
        "mnist__gram_acc": acc_m_g,
        "mnist__gram_sps": sps_m_g,
        "mnist__ternary_gram_acc": acc_m_tg,
        "mnist__ternary_gram_sps": sps_m_tg,
        "mnist__adam_mlp_sps": sps_m_adam,
        "c100__woodbury_acc": acc_c_w,
        "c100__woodbury_time_s": dt_c_w,
        "c100__woodbury_vram_mb": vram_c_w,
        "c100__gram_acc": acc_c_g,
        "c100__gram_time_s": dt_c_g,
        "c100__gram_vram_mb": vram_c_g,
        "c100__ternary_gram_acc": acc_c_tg,
        "c100__ternary_gram_time_s": dt_c_tg,
        "c100__shuffled_acc": acc_c_shuf,
    }


def _eval_patch_pdnc_seed(seed: int, n_train: int = 2000, n_eval: int = 100) -> Dict[str, Any]:
    """Suite B: 4x4 Local-Patch PDNC vs Frozen Patch Ablation & Patch/Flat BP A2C on 16x16 Pixel Pong."""
    torch.set_num_threads(1)
    rf_mask = make_patch_receptive_field_mask(
        16, 16, patch_size=4, stride=2, compartments_per_patch=3, include_transient=True, include_bias=True
    )
    d_h = int(rf_mask.shape[0])  # 49 * 3 = 147

    def _make_agent(
        *,
        freeze_l1: bool = False,
        ablate_elig: bool = False,
        shuffle_apical: bool = False,
    ) -> PDNCActorCritic:
        ag = PDNCActorCritic(
            enc_cfg=PDNCConfig(
                d_in=256,
                d_out=d_h,
                activation="tanh",
                use_transient_spine=True,
                tau_spine=2.0,
                transient_scale=1.5,
                precision_mode="diagonal",
                eta=0.004,
                weight_decay=0.005,
                beta_mom=0.90,
                beta_rms=0.99,
                eps_prec=1e-8,
                lambda_kp=0.002,
                weight_init_scale=1.6,
                bias_init_scale=0.10,
                fb_init_scale=0.3,
                receptive_field_mask=rf_mask,
                freeze_weights=freeze_l1,
                ablate_eligibility=ablate_elig,
                shuffle_apical=shuffle_apical,
                seed=seed + 77,
            ),
            n_actions=3,
            eta_actor=0.003,
            eta_critic=0.004,
            gamma=0.95,
            lam=0.0 if ablate_elig else 0.995,
            critic_apical_weight=0.5,
            beta_ent=0.01,
            eps_explore=0.0,
            seed=seed,
        )
        # Receptive-field variance scaling (16 active pixels vs 256 global)
        ag.enc.W[:, :-1] *= np.sqrt(256.0 / 16.0) * 0.5
        return ag

    def _train_pdnc(ag: PDNCActorCritic, shuffle_reward: bool = False, record_curve: bool = False) -> List[float]:
        rng = np.random.RandomState(seed + 2000)
        env = PongEnv()
        curve: List[float] = []
        bin_hits = 0
        bin_sz = max(1, n_train // 10)
        for s in range(n_train):
            env.reset_serve(rng, ood=False)
            ag.reset_episode()
            obs = env.obs_pixels()
            done = False
            steps = 0
            final_r = 0.0
            while not done and steps < 40:
                act, vc = ag.step_policy(obs, greedy=False)
                r, done = env.step(act)
                if done:
                    final_r = r
                    r_use = float(rng.choice([-1.0, 1.0])) if shuffle_reward else r
                    ag.apply_td_update(vc, r_use, 0.0, True)
                else:
                    on = env.obs_pixels()
                    ag.apply_td_update(vc, 0.0, ag.value_only(on), False)
                    obs = on
                steps += 1
            if final_r > 0.0:
                bin_hits += 1
            if record_curve and (s + 1) % bin_sz == 0:
                curve.append(100.0 * bin_hits / float(bin_sz))
                bin_hits = 0
        return curve

    def _eval_pdnc(ag: PDNCActorCritic, ood: bool = False, blind: bool = False) -> Tuple[float, float]:
        rng = np.random.RandomState(seed + (8000 if ood else 7000))
        env = PongEnv()
        hits = 0
        dir_ok, dir_tot = 0, 0
        for _ in range(n_eval):
            env.reset_serve(rng, ood=ood)
            ag.reset_episode()
            done = False
            rew = 0.0
            st = 0
            while not done and st < 40:
                obs = np.zeros(256, dtype=np.float64) if blind else env.obs_pixels()
                act, _ = ag.step_policy(obs, greedy=True)
                if abs(env.ball_y - env.paddle_y) > 0.05:
                    dir_tot += 1
                    if (env.ball_y > env.paddle_y and act == 2) or (env.ball_y < env.paddle_y and act == 0):
                        dir_ok += 1
                rew, done = env.step(act)
                st += 1
            if rew > 0.0:
                hits += 1
        return 100.0 * hits / float(n_eval), 100.0 * dir_ok / float(max(1, dir_tot))

    ag_full = _make_agent()
    ag_froz = _make_agent(freeze_l1=True)
    ag_noel = _make_agent(ablate_elig=True)
    ag_shuf_a = _make_agent(shuffle_apical=True)
    ag_untr = _make_agent()
    ag_shuf_r = _make_agent()

    curve_full = _train_pdnc(ag_full, record_curve=(seed in (999, 2000)))
    _train_pdnc(ag_froz)
    _train_pdnc(ag_noel)
    _train_pdnc(ag_shuf_a)
    _train_pdnc(ag_shuf_r, shuffle_reward=True)

    hit_full, dir_full = _eval_pdnc(ag_full)
    hit_ood, _ = _eval_pdnc(ag_full, ood=True)
    hit_blind, _ = _eval_pdnc(ag_full, blind=True)
    hit_froz, _ = _eval_pdnc(ag_froz)
    hit_noel, _ = _eval_pdnc(ag_noel)
    hit_shuf_a, _ = _eval_pdnc(ag_shuf_a)
    hit_untr, _ = _eval_pdnc(ag_untr)
    hit_shuf_r, _ = _eval_pdnc(ag_shuf_r)

    # Fair Patch & Flat Backprop A2C baselines on identical 2,000-serve budget
    mask_t = torch.tensor(rf_mask[:, :-1], dtype=torch.float32)
    bp_patch = TorchA2CMLP(d_in=512, d_hid=d_h, seed=seed + 77, lr=1.5e-3)
    with torch.no_grad():
        bp_patch.trunk[0].weight.mul_(mask_t)
    env_p = PongEnv()
    rp = np.random.RandomState(seed + 2000)
    for _ in range(n_train):
        env_p.reset_serve(rp, ood=False)
        bp_patch.train_serve(env_p, pixel_mode=True)
        with torch.no_grad():
            bp_patch.trunk[0].weight.mul_(mask_t)
    hit_bp_patch = bp_patch.eval_serves(np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True)

    bp_flat = TorchA2CMLP(d_in=512, d_hid=128, seed=seed + 77, lr=1.5e-3)
    env_f = PongEnv()
    rf = np.random.RandomState(seed + 2000)
    for _ in range(n_train):
        env_f.reset_serve(rf, ood=False)
        bp_flat.train_serve(env_f, pixel_mode=True)
    hit_bp_flat = bp_flat.eval_serves(np.random.RandomState(seed + 7000), n_eval, ood=False, pixel_mode=True)

    return {
        "metrics": {
            "pong__patch_pdnc_hit": hit_full,
            "pong__patch_pdnc_ood_hit": hit_ood,
            "pong__patch_pdnc_dir_align": dir_full,
            "pong__patch_pdnc_kp_cos": ag_full.actor.feedback_alignment_cosine(),
            "pong__patch_frozen_hit": hit_froz,
            "pong__ablate_elig_hit": hit_noel,
            "pong__shuffle_apical_hit": hit_shuf_a,
            "pong__patch_bp_a2c_hit": hit_bp_patch,
            "pong__flat_bp_a2c_hit": hit_bp_flat,
            "pong__blind_hit": hit_blind,
            "pong__untrained_hit": hit_untr,
            "pong__shuffled_reward_hit": hit_shuf_r,
            "pong__random_hit": eval_trivial_policy("random", np.random.RandomState(seed + 7000), n_eval),
        },
        "curve": curve_full,
    }


def evaluate_all_seeds(seeds: List[int], device: torch.device) -> Tuple[List[Dict[str, Any]], List[float]]:
    with ProcessPoolExecutor(max_workers=4) as ex:
        cpu_res = list(ex.map(_eval_patch_pdnc_seed, seeds))

    curve_2000: List[float] = []
    per_seed: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, (s, pong_out) in enumerate(zip(seeds, cpu_res), 1):
        ts = time.perf_counter()
        if s == 2000 and pong_out["curve"]:
            curve_2000 = pong_out["curve"]
        row: Dict[str, Any] = {"seed": s}
        row.update(pong_out["metrics"])
        row.update(run_part_a_cuda(s, device))
        _cleanup_cuda(device)
        per_seed.append(row)
        dt = time.perf_counter() - ts
        eta = ((time.perf_counter() - t0) / idx) * (len(seeds) - idx)
        print(
            f"[dmcn_v4] seed {s} ({idx}/{len(seeds)}) {dt:.1f}s | eta={eta:.0f}s | "
            f"mnist_gram={row['mnist__gram_sps']:.0f}/s c100_tern={row['c100__ternary_gram_acc']:.2f}% "
            f"pong_patch={row['pong__patch_pdnc_hit']:.1f}%",
            flush=True,
        )
    return per_seed, curve_2000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / "results" / "dmcn_v4_efficiency_and_patch_pdnc_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        rows, _ = evaluate_all_seeds([999], device)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": rows[0]})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_v4_efficiency_and_patch_pdnc_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "evaluated_val_seed": 999, "device": str(device)},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": rows[0]}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed, curve_2000 = evaluate_all_seeds(test_seeds, device)
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    falsifiers = {
        "F1_mnist_gram_cholesky_parity_and_3x_speedup": bool(
            summary["mnist__gram_acc"]["mean"] >= 95.5
            and abs(summary["mnist__gram_acc"]["mean"] - summary["mnist__woodbury_acc"]["mean"]) <= 0.05
            and summary["mnist__gram_sps"]["mean"] >= 3.0 * summary["mnist__woodbury_sps"]["mean"]
        ),
        "F2_c100_ternary_and_gram_preserve_accuracy": bool(
            summary["c100__gram_acc"]["mean"] >= 68.0
            and abs(summary["c100__gram_acc"]["mean"] - summary["c100__woodbury_acc"]["mean"]) <= 0.05
            and summary["c100__ternary_gram_acc"]["mean"] >= 67.5
            and summary["c100__gram_time_s"]["mean"] <= 0.5 * summary["c100__woodbury_time_s"]["mean"]
        ),
        "F3_patch_pdnc_pong_ge_74_and_beats_frozen_by_25pp": bool(
            summary["pong__patch_pdnc_hit"]["mean"] >= 74.0
            and (summary["pong__patch_pdnc_hit"]["mean"] - summary["pong__patch_frozen_hit"]["mean"]) >= 25.0
            and summary["pong__patch_pdnc_hit"]["mean"] >= (summary["pong__patch_bp_a2c_hit"]["mean"] - 2.0)
        ),
        "F4_nulls_below_chance": bool(
            summary["c100__shuffled_acc"]["mean"] <= 3.0
            and max(
                summary["pong__blind_hit"]["mean"],
                summary["pong__untrained_hit"]["mean"],
                summary["pong__shuffled_reward_hit"]["mean"],
                summary["pong__random_hit"]["mean"],
            ) <= 35.0
        ),
    }

    payload = {
        "headline_metric": "pong__patch_pdnc_hit",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "device": str(device),
        "elapsed_test_sec": float(elapsed),
        "seed_2000_pong_curve": curve_2000,
        "summary": summary,
        "falsifiers": falsifiers,
        "all_passed": bool(all(falsifiers.values())),
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_v4_efficiency_and_patch_pdnc_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"ternary_density": 0.05, "patch_rf": "4x4_s2_3comp_147h", "precision_mode": "gram_cholesky"},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "falsifiers": falsifiers, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
