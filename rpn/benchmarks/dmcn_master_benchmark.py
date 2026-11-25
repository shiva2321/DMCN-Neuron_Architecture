"""Pre-registered Master Evaluation & Compute Profiling Suite for DMCN.

Runs ONE unified architecture (DMCNLayer / DMCNNetwork / DMCNTorchNetwork) across:
  E1a: Single-Cell Online Causal Noisy XOR
  E1b: Single-Cell Delayed Temporal XOR
  E2:  In-Stream 1-Shot Associative Recall & Overwrite Under Distractors
  E3:  Task-Free 4-Regime Continual Nonlinear Classification (Zero Task-IDs)
  E4:  Strictly Causal Chaotic Lorenz Forecasting Under 10x Gain Shock & Irregular dt
  E5:  Delayed-Cue Memory T-Maze RL (Zero Goal/Physics Oracles)
  E6a: Real Handwritten Digits Recognition (sklearn load_digits, 10 classes, Held-Out)
  E6b: Task-Free 5-Pair Continual Handwritten Digit Recognition (Zero Task-IDs)
  E7:  Exact Analytical MAC/FLOP, Memory, and CPU/CUDA Wall-Clock Latency Profiling

Usage:
    python rpn/benchmarks/dmcn_master_benchmark.py --mode val
    python rpn/benchmarks/dmcn_master_benchmark.py --mode test
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
from sklearn.datasets import load_digits

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import (  # noqa: E402
    AblationConfig,
    DMCNConfig,
    DMCNLayer,
    DMCNNetwork,
    DMCNTorchNetwork,
    SwarmBoidNetwork,
    SwarmConfig,
)


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


# ---------------------------------------------------------------------------
# Strong, Conventional Baselines (same data & online step budget)
# ---------------------------------------------------------------------------
class OnlineBackpropMLP:
    """Standard 2-layer tanh MLP trained online via exact backprop (SGD or Adam)."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        lr: float = 0.05,
        use_adam: bool = False,
        out_act: str = "linear",
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out
        self.lr = lr
        self.use_adam = use_adam
        self.out_act = out_act

        self.W1 = rng.randn(d_hid, d_in) * (1.0 / np.sqrt(d_in))
        self.b1 = np.zeros(d_hid, dtype=np.float64)
        self.W2 = rng.randn(d_out, d_hid) * (1.0 / np.sqrt(d_hid))
        self.b2 = np.zeros(d_out, dtype=np.float64)

        self.t = 0
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.W1 @ x + self.b1)
        z = self.W2 @ h + self.b2
        if self.out_act == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        elif self.out_act == "tanh":
            y = np.tanh(z)
        else:
            y = z
        return h, y

    def step_train(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        h, y = self.forward(x)
        err = target - y
        if self.out_act == "sigmoid":
            d_out = err * y * (1.0 - y)
        elif self.out_act == "tanh":
            d_out = err * (1.0 - y * y)
        else:
            d_out = err

        d_hid = (self.W2.T @ d_out) * (1.0 - h * h)
        grads = [np.outer(d_hid, x), d_hid, np.outer(d_out, h), d_out]
        params = [self.W1, self.b1, self.W2, self.b2]

        if not self.use_adam:
            for p_arr, g_arr in zip(params, grads):
                p_arr += self.lr * g_arr
        else:
            self.t += 1
            b1_c, b2_c, eps = 0.9, 0.999, 1e-8
            for idx, (p_arr, g_arr) in enumerate(zip(params, grads)):
                self.m[idx] = b1_c * self.m[idx] + (1.0 - b1_c) * g_arr
                self.v[idx] = b2_c * self.v[idx] + (1.0 - b2_c) * (g_arr * g_arr)
                m_hat = self.m[idx] / (1.0 - b1_c ** self.t)
                v_hat = self.v[idx] / (1.0 - b2_c ** self.t)
                p_arr += self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, y = self.forward(x)
        return y


class FixedFAMLP:
    """2-layer tanh MLP with fixed Random Feedback Alignment."""

    def __init__(self, d_in: int, d_hid: int, d_out: int, lr: float = 0.05, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.W1 = rng.randn(d_hid, d_in) * (1.0 / np.sqrt(d_in))
        self.b1 = np.zeros(d_hid, dtype=np.float64)
        self.W2 = rng.randn(d_out, d_hid) * (1.0 / np.sqrt(d_hid))
        self.b2 = np.zeros(d_out, dtype=np.float64)
        self.B = rng.randn(d_hid, d_out) * (1.0 / np.sqrt(d_out))

    def step_train(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        y = self.W2 @ h + self.b2
        err = target - y
        d_hid = (self.B @ err) * (1.0 - h * h)
        self.W2 += self.lr * np.outer(err, h)
        self.b2 += self.lr * err
        self.W1 += self.lr * np.outer(d_hid, x)
        self.b1 += self.lr * d_hid
        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2


# ---------------------------------------------------------------------------
# E1a & E1b: Single-Cell Causal Noisy XOR & Delayed Temporal XOR
# ---------------------------------------------------------------------------
def eval_e1_xor(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    base_x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    base_y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    # Disjoint train (400 noisy) and held-out test (120 noisy)
    tr_idx = rng.randint(0, 4, size=400)
    te_idx = rng.randint(0, 4, size=120)
    x_tr = base_x[tr_idx] + rng.normal(0.0, 0.06, size=(400, 2))
    y_tr = base_y[tr_idx]
    x_te = base_x[te_idx] + rng.normal(0.0, 0.06, size=(120, 2))
    y_te = base_y[te_idx]

    def make_xor_cell(ablate_nm: bool = False) -> DMCNLayer:
        cfg = DMCNConfig(
            d_in=2,
            d_out=1,
            n_branches=1,
            n_spine_scales=0,
            use_nonmonotonic=True,
            beta_dend=5.0,
            init_rho=0.85,
            init_theta_on=0.45,
            init_delta_theta=1.1,
            soma_activation="linear",
            eta_fast=0.25,
            gamma_forget=0.0,
            eta_consol=0.05,
            eta_dend_params=0.08,
            weight_init_scale=0.35,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_nonmonotonic=ablate_nm),
        )
        return DMCNLayer(cfg)

    cell_intact = make_xor_cell(ablate_nm=False)
    cell_untrained = make_xor_cell(ablate_nm=False)
    cell_ablated = make_xor_cell(ablate_nm=True)
    cell_shuf = make_xor_cell(ablate_nm=False)
    y_shuf = rng.permutation(y_tr)

    # Baselines: Point Logistic & 2-Layer Backprop MLP
    w_log = rng.randn(2) * 0.1
    b_log = 0.0
    mlp_ref = OnlineBackpropMLP(2, 8, 1, lr=0.15, out_act="sigmoid", seed=seed)

    for i in range(len(x_tr)):
        x_i, y_i = x_tr[i], np.array([y_tr[i]])
        cell_intact.step(x_i)
        cell_intact.learn_local(y_i - cell_intact._last_y)

        cell_ablated.step(x_i)
        cell_ablated.learn_local(y_i - cell_ablated._last_y)

        cell_shuf.step(x_i)
        cell_shuf.learn_local(np.array([y_shuf[i]]) - cell_shuf._last_y)

        p_l = 1.0 / (1.0 + np.exp(-np.clip(float(np.dot(w_log, x_i) + b_log), -30.0, 30.0)))
        w_log += 0.15 * (y_tr[i] - p_l) * x_i
        b_log += 0.15 * (y_tr[i] - p_l)
        mlp_ref.step_train(x_i, y_i)

    def acc_cell(c: DMCNLayer, blind: bool = False) -> float:
        hits = 0
        for i in range(len(x_te)):
            inp = np.zeros(2) if blind else x_te[i]
            pred = float(c.step(inp)[0]) >= 0.5
            hits += int(pred == bool(y_te[i] >= 0.5))
        return 100.0 * hits / len(x_te)

    acc_log = 100.0 * float(np.mean([(1.0 / (1.0 + np.exp(-float(np.dot(w_log, x_te[i]) + b_log))) >= 0.5) == (y_te[i] >= 0.5) for i in range(len(x_te))]))
    acc_mlp = 100.0 * float(np.mean([(float(mlp_ref.predict(x_te[i])[0]) >= 0.5) == (y_te[i] >= 0.5) for i in range(len(x_te))]))

    # E1b: Delayed Temporal XOR (bit A at t=0, silent gap t=1..3, bit B at t=4)
    def make_delayed_cell(ablate_spine: bool = False) -> DMCNLayer:
        cfg = DMCNConfig(
            d_in=1,
            d_out=1,
            n_branches=1,
            n_spine_scales=4,
            tau_min=1.0,
            tau_max=10.0,
            use_nonmonotonic=True,
            beta_dend=6.0,
            init_rho=0.95,
            init_theta_on=0.5,
            init_delta_theta=1.0,
            soma_activation="linear",
            eta_fast=0.18,
            gamma_forget=0.0,
            eta_consol=0.04,
            eta_dend_params=0.05,
            seed=seed + 50,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_spine_traces=ablate_spine),
        )
        c = DMCNLayer(cfg)
        c.W_slow[0, 0, :] = 0.5
        return c

    d_intact = make_delayed_cell(ablate_spine=False)
    d_nospine = make_delayed_cell(ablate_spine=True)

    def run_delay_trial(c: DMCNLayer, bit_a: float, bit_b: float, learn: bool, target_val: float) -> float:
        c.reset_temporal_state()
        seq = [bit_a, 0.0, 0.0, 0.0, bit_b]
        out = 0.0
        for val in seq:
            out = float(c.step(np.array([val]), dt=1.0)[0])
        if learn:
            c.learn_local(np.array([target_val - out]))
        return out

    for _ in range(320):
        a_b, b_b = float(rng.randint(0, 2)), float(rng.randint(0, 2))
        tgt_b = float(int(a_b) ^ int(b_b))
        run_delay_trial(d_intact, a_b + rng.normal(0.0, 0.03), b_b + rng.normal(0.0, 0.03), True, tgt_b)
        run_delay_trial(d_nospine, a_b + rng.normal(0.0, 0.03), b_b + rng.normal(0.0, 0.03), True, tgt_b)

    hits_di, hits_dn = 0, 0
    for _ in range(100):
        a_b, b_b = float(rng.randint(0, 2)), float(rng.randint(0, 2))
        tgt_b = int(a_b) ^ int(b_b)
        o_i = run_delay_trial(d_intact, a_b + rng.normal(0.0, 0.03), b_b + rng.normal(0.0, 0.03), False, 0.0)
        o_n = run_delay_trial(d_nospine, a_b + rng.normal(0.0, 0.03), b_b + rng.normal(0.0, 0.03), False, 0.0)
        hits_di += int((o_i >= 0.5) == bool(tgt_b))
        hits_dn += int((o_n >= 0.5) == bool(tgt_b))

    return {
        "e1a_dmcn_acc": acc_cell(cell_intact),
        "e1a_ablate_nonmonotonic_acc": acc_cell(cell_ablated),
        "e1a_untrained_acc": acc_cell(cell_untrained),
        "e1a_blind_acc": acc_cell(cell_intact, blind=True),
        "e1a_shuffled_acc": acc_cell(cell_shuf),
        "e1a_logistic_baseline_acc": acc_log,
        "e1a_mlp_ref_acc": acc_mlp,
        "e1b_delayed_dmcn_acc": float(hits_di),
        "e1b_ablate_spine_acc": float(hits_dn),
    }


# ---------------------------------------------------------------------------
# E2: In-Stream 1-Shot Associative Recall & Mid-Stream Overwrite
# ---------------------------------------------------------------------------
def eval_e2_associative_recall(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    dim = 32
    n_pairs = 16
    n_overwrite = 4

    # Orthonormal/well-separated unit keys and random unit values
    q_mat, _ = np.linalg.qr(rng.randn(dim, dim))
    keys = q_mat[:n_pairs]
    vals_init = rng.randn(n_pairs, dim)
    vals_init /= np.linalg.norm(vals_init, axis=1, keepdims=True) + 1e-9

    vals_new = rng.randn(n_overwrite, dim)
    vals_new /= np.linalg.norm(vals_new, axis=1, keepdims=True) + 1e-9
    # Ensure new value is distinct from old value
    for j in range(n_overwrite):
        vals_new[j] -= np.dot(vals_new[j], vals_init[j]) * vals_init[j]
        vals_new[j] /= np.linalg.norm(vals_new[j]) + 1e-9

    target_vals = vals_init.copy()
    target_vals[:n_overwrite] = vals_new

    # Candidate pool for nearest-neighbor evaluation (both old & new values included!)
    pool = np.vstack([vals_init, vals_new])

    def make_assoc_dmcn(ablate_fast: bool = False) -> DMCNLayer:
        cfg = DMCNConfig(
            d_in=dim,
            d_out=dim,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            use_nonmonotonic=False,
            use_unit_key_norm=True,
            eta_fast=1.0,
            eta_slow=0.05,
            gamma_forget=0.0,
            eta_consol=0.0,
            weight_init_scale=0.0,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_fast_delta=ablate_fast),
        )
        return DMCNLayer(cfg)

    dmcn = make_assoc_dmcn(ablate_fast=False)
    dmcn_nofast = make_assoc_dmcn(ablate_fast=True)
    dmcn_untrained = make_assoc_dmcn(ablate_fast=False)

    # Baselines: Additive Hebbian & Online SGD Linear
    W_hebb = np.zeros((dim, dim), dtype=np.float64)
    W_sgd = np.zeros((dim, dim), dtype=np.float64)
    last_val = np.zeros(dim, dtype=np.float64)

    # Phase 1: 1-shot stream of 16 pairs
    for j in range(n_pairs):
        y_d = dmcn.step(keys[j])
        dmcn.learn_local(vals_init[j] - y_d)

        y_nf = dmcn_nofast.step(keys[j])
        dmcn_nofast.learn_local(vals_init[j] - y_nf)

        W_hebb += np.outer(vals_init[j], keys[j])
        W_sgd += 0.3 * np.outer(vals_init[j] - W_sgd @ keys[j], keys[j])
        last_val = vals_init[j].copy()

    # Phase 2: Mid-stream 1-shot overwrite of first 4 keys
    for j in range(n_overwrite):
        y_d = dmcn.step(keys[j])
        dmcn.learn_local(vals_new[j] - y_d)

        y_nf = dmcn_nofast.step(keys[j])
        dmcn_nofast.learn_local(vals_new[j] - y_nf)

        W_hebb += np.outer(vals_new[j], keys[j])
        W_sgd += 0.3 * np.outer(vals_new[j] - W_sgd @ keys[j], keys[j])
        last_val = vals_new[j].copy()

    # Phase 3: 16 random distractor steps (read-only/unlabelled interference)
    for _ in range(16):
        dist_k = rng.randn(dim)
        dist_k /= np.linalg.norm(dist_k) + 1e-9
        dmcn.step(dist_k)
        dmcn_nofast.step(dist_k)

    def score_fn(pred_fn) -> Tuple[float, float, float]:
        hits_all, hits_ov = 0, 0
        cos_list = []
        for j in range(n_pairs):
            pred = pred_fn(keys[j])
            norm_p = float(np.linalg.norm(pred))
            expected_idx = (n_pairs + j) if j < n_overwrite else j
            if norm_p < 1e-9:
                cos_list.append(0.0)
                continue
            p_hat = pred / norm_p
            cos_list.append(float(np.dot(p_hat, target_vals[j])))
            best_idx = int(np.argmax(pool @ p_hat))
            ok = int(best_idx == expected_idx)
            hits_all += ok
            if j < n_overwrite:
                hits_ov += ok
        return 100.0 * hits_all / n_pairs, 100.0 * hits_ov / n_overwrite, float(np.mean(cos_list))

    all_dmcn, ov_dmcn, cos_dmcn = score_fn(lambda k: dmcn.step(k))
    all_nofast, ov_nofast, _ = score_fn(lambda k: dmcn_nofast.step(k))
    all_untr, _, _ = score_fn(lambda k: dmcn_untrained.step(k))
    all_blind, _, _ = score_fn(lambda k: dmcn.step(np.zeros(dim)))
    all_hebb, ov_hebb, cos_hebb = score_fn(lambda k: W_hebb @ k)
    all_sgd, ov_sgd, _ = score_fn(lambda k: W_sgd @ k)
    all_pers, _, _ = score_fn(lambda k: last_val)

    return {
        "e2_dmcn_all_acc": all_dmcn,
        "e2_dmcn_overwrite_acc": ov_dmcn,
        "e2_dmcn_cos": cos_dmcn,
        "e2_ablate_fast_all_acc": all_nofast,
        "e2_ablate_fast_overwrite_acc": ov_nofast,
        "e2_untrained_acc": all_untr,
        "e2_blind_acc": all_blind,
        "e2_hebbian_all_acc": all_hebb,
        "e2_hebbian_overwrite_acc": ov_hebb,
        "e2_hebbian_cos": cos_hebb,
        "e2_sgd_all_acc": all_sgd,
        "e2_sgd_overwrite_acc": ov_sgd,
        "e2_persistence_acc": all_pers,
    }


# ---------------------------------------------------------------------------
# E3: Task-Free 4-Regime Continual Nonlinear Classification (Zero Task-IDs)
# ---------------------------------------------------------------------------
def eval_e3_continual_regimes(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    d_in, n_classes, n_regimes = 16, 4, 4
    n_train_per, n_test_per = 160, 60

    # 4 disjoint orthonormal regime subspaces with distinct class prototypes per regime
    basis, _ = np.linalg.qr(rng.randn(d_in, d_in))
    regimes_tr: List[Tuple[np.ndarray, np.ndarray]] = []
    regimes_te: List[Tuple[np.ndarray, np.ndarray]] = []

    for r in range(n_regimes):
        sub_b = basis[r * 4 : (r + 1) * 4]  # (4, 16) orthonormal block per regime
        # Regime anchor + class offsets inside that regime's subspace
        anchor = sub_b[0] * 1.8
        class_dirs = rng.randn(n_classes, 4)
        class_dirs /= np.linalg.norm(class_dirs, axis=1, keepdims=True) + 1e-9
        class_centers = anchor[None, :] + (class_dirs @ sub_b) * 1.2

        y_tr_r = rng.randint(0, n_classes, size=n_train_per)
        x_tr_r = class_centers[y_tr_r] + rng.normal(0.0, 0.25, size=(n_train_per, d_in))
        y_te_r = rng.randint(0, n_classes, size=n_test_per)
        x_te_r = class_centers[y_te_r] + rng.normal(0.0, 0.25, size=(n_test_per, d_in))
        regimes_tr.append((x_tr_r, y_tr_r))
        regimes_te.append((x_te_r, y_te_r))

    def make_dmcn(n_branches: int = 4, ablate_cascade: bool = False) -> DMCNLayer:
        cfg = DMCNConfig(
            d_in=d_in,
            d_out=n_classes,
            n_branches=n_branches,
            top_k_branches=1,
            n_spine_scales=0,
            beta_route=12.0,
            eta_proto=0.03,
            theta_novel=0.35,
            theta_err_novel=0.02,
            theta_err=0.15,
            theta_coh=0.40,
            recruit_sustained=2,
            recruit_cooldown=10,
            soma_activation="linear",
            use_nonmonotonic=False,
            eta_fast=0.20,
            gamma_forget=0.002,
            eta_consol=0.06,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_cascade=ablate_cascade),
        )
        return DMCNLayer(cfg)

    dmcn = make_dmcn(n_branches=4, ablate_cascade=False)
    dmcn_b1 = make_dmcn(n_branches=1, ablate_cascade=False)
    dmcn_nocasc = make_dmcn(n_branches=4, ablate_cascade=True)
    dmcn_untr = make_dmcn(n_branches=4, ablate_cascade=False)
    dmcn_shuf = make_dmcn(n_branches=4, ablate_cascade=False)

    mlp_sgd = OnlineBackpropMLP(d_in, 32, n_classes, lr=0.05, use_adam=False, seed=seed)
    mlp_adam = OnlineBackpropMLP(d_in, 32, n_classes, lr=0.01, use_adam=True, seed=seed)
    mlp_fa = FixedFAMLP(d_in, 32, n_classes, lr=0.05, seed=seed)

    def eval_regime_acc(pred_fn, r_idx: int) -> float:
        x_t, y_t = regimes_te[r_idx]
        preds = [int(np.argmax(pred_fn(x_t[i]))) for i in range(len(x_t))]
        return 100.0 * float(np.mean(np.array(preds) == y_t))

    r0_after_r0_dmcn = 0.0
    r0_after_r0_sgd = 0.0

    # Strictly sequential online learning (ZERO task-IDs or boundary signals!)
    for r in range(n_regimes):
        x_r, y_r = regimes_tr[r]
        y_r_shuf = rng.permutation(y_r)
        for i in range(len(x_r)):
            tgt = np.zeros(n_classes, dtype=np.float64)
            tgt[y_r[i]] = 1.0
            tgt_s = np.zeros(n_classes, dtype=np.float64)
            tgt_s[y_r_shuf[i]] = 1.0

            y_d = dmcn.step(x_r[i])
            dmcn.learn_local(tgt - y_d)

            y_b1 = dmcn_b1.step(x_r[i])
            dmcn_b1.learn_local(tgt - y_b1)

            y_nc = dmcn_nocasc.step(x_r[i])
            dmcn_nocasc.learn_local(tgt - y_nc)

            y_sh = dmcn_shuf.step(x_r[i])
            dmcn_shuf.learn_local(tgt_s - y_sh)

            mlp_sgd.step_train(x_r[i], tgt)
            mlp_adam.step_train(x_r[i], tgt)
            mlp_fa.step_train(x_r[i], tgt)

        if r == 0:
            r0_after_r0_dmcn = eval_regime_acc(lambda x: dmcn.step(x), 0)
            r0_after_r0_sgd = eval_regime_acc(mlp_sgd.predict, 0)

    dmcn_accs = [eval_regime_acc(lambda x: dmcn.step(x), r) for r in range(n_regimes)]
    b1_accs = [eval_regime_acc(lambda x: dmcn_b1.step(x), r) for r in range(n_regimes)]
    nc_accs = [eval_regime_acc(lambda x: dmcn_nocasc.step(x), r) for r in range(n_regimes)]
    untr_accs = [eval_regime_acc(lambda x: dmcn_untr.step(x), r) for r in range(n_regimes)]
    shuf_accs = [eval_regime_acc(lambda x: dmcn_shuf.step(x), r) for r in range(n_regimes)]
    sgd_accs = [eval_regime_acc(mlp_sgd.predict, r) for r in range(n_regimes)]
    adam_accs = [eval_regime_acc(mlp_adam.predict, r) for r in range(n_regimes)]
    fa_accs = [eval_regime_acc(mlp_fa.predict, r) for r in range(n_regimes)]

    return {
        "e3_dmcn_final_avg_acc": float(np.mean(dmcn_accs)),
        "e3_dmcn_r0_retention": float(dmcn_accs[0]),
        "e3_dmcn_bwt_r0": float(dmcn_accs[0] - r0_after_r0_dmcn),
        "e3_dmcn_branches_recruited": float(np.sum(dmcn.recruited)),
        "e3_ablate_b1_avg_acc": float(np.mean(b1_accs)),
        "e3_ablate_cascade_avg_acc": float(np.mean(nc_accs)),
        "e3_untrained_avg_acc": float(np.mean(untr_accs)),
        "e3_shuffled_avg_acc": float(np.mean(shuf_accs)),
        "e3_mlp_sgd_avg_acc": float(np.mean(sgd_accs)),
        "e3_mlp_sgd_r0_retention": float(sgd_accs[0]),
        "e3_mlp_sgd_bwt_r0": float(sgd_accs[0] - r0_after_r0_sgd),
        "e3_mlp_adam_avg_acc": float(np.mean(adam_accs)),
        "e3_mlp_fa_avg_acc": float(np.mean(fa_accs)),
    }


# ---------------------------------------------------------------------------
# E4: Strictly Causal Chaotic Lorenz Forecasting Under 10x Gain Shock & Irregular dt
# ---------------------------------------------------------------------------
def eval_e4_nonstationary_chaos(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    state = np.array([1.0, 1.0, 1.0], dtype=np.float64) + rng.uniform(-0.5, 0.5, size=3)

    n_steps = 600
    shock_step = 300
    dts = rng.uniform(0.006, 0.014, size=n_steps)
    obs_seq = np.zeros((n_steps, 3), dtype=np.float64)

    for t in range(n_steps):
        dt = dts[t]
        dx = sigma * (state[1] - state[0]) * dt
        dy = (state[0] * (rho - state[2]) - state[1]) * dt
        dz = (state[0] * state[1] - beta * state[2]) * dt
        state = state + np.array([dx, dy, dz])
        # Strictly raw sensor observation (NO offline std normalization!)
        # Sudden 10x gain shock + offset at t >= shock_step
        if t >= shock_step:
            obs_seq[t] = state * 1.0 + np.array([20.0, -15.0, 10.0])
        else:
            obs_seq[t] = state * 0.1

    def make_chaos_dmcn(ablate_homeo: bool = False, ablate_spine: bool = False) -> DMCNNetwork:
        return DMCNNetwork([
            DMCNConfig(
                d_in=3,
                d_out=24,
                n_branches=1,
                n_spine_scales=3,
                tau_min=0.01,
                tau_max=0.10,
                alpha_homeo=0.08,
                center_homeostasis=False,
                eta_theta=0.0,
                soma_activation="tanh",
                eta_fast=0.04,
                gamma_forget=0.005,
                eta_consol=0.02,
                seed=seed,
                ablation=AblationConfig(ablate_homeostasis=ablate_homeo, ablate_spine_traces=ablate_spine),
            ),
            DMCNConfig(
                d_in=24,
                d_out=3,
                n_branches=1,
                n_spine_scales=0,
                eta_theta=0.0,
                adapt_output_scale=True,
                alpha_out_scale=0.12,
                soma_activation="linear",
                eta_fast=0.25,
                gamma_forget=0.002,
                eta_consol=0.02,
                seed=seed + 1,
                ablation=AblationConfig(ablate_homeostasis=ablate_homeo),
            ),
        ])

    dmcn = make_chaos_dmcn(ablate_homeo=False, ablate_spine=False)
    dmcn_nohomeo = make_chaos_dmcn(ablate_homeo=True, ablate_spine=False)
    dmcn_nospine = make_chaos_dmcn(ablate_homeo=False, ablate_spine=True)
    mlp_bp = OnlineBackpropMLP(3, 24, 3, lr=0.005, use_adam=False, seed=seed)
    w_nlms = np.zeros((3, 3), dtype=np.float64)

    errs = {k: [] for k in ("dmcn", "nohomeo", "nospine", "blind", "mlp", "nlms", "persist")}
    true_deltas = []

    for t in range(n_steps - 1):
        s_curr = obs_seq[t]
        s_next = obs_seq[t + 1]
        delta_true = s_next - s_curr
        dt = float(dts[t])

        # Predict strictly BEFORE observing s_next
        p_dmcn = dmcn.step(s_curr, dt=dt)
        p_noh = dmcn_nohomeo.step(s_curr, dt=dt)
        p_nos = dmcn_nospine.step(s_curr, dt=dt)
        p_mlp = mlp_bp.predict(s_curr)
        p_nlms = w_nlms @ s_curr

        if t >= shock_step + 40:
            true_deltas.append(delta_true)
            errs["dmcn"].append(float(np.sum((delta_true - p_dmcn) ** 2)))
            errs["nohomeo"].append(float(np.sum((delta_true - p_noh) ** 2)))
            errs["nospine"].append(float(np.sum((delta_true - p_nos) ** 2)))
            errs["blind"].append(float(np.sum(delta_true ** 2)))
            errs["mlp"].append(float(np.sum((delta_true - p_mlp) ** 2)))
            errs["nlms"].append(float(np.sum((delta_true - p_nlms) ** 2)))
            errs["persist"].append(float(np.sum(delta_true ** 2)))

        # Online causal updates
        dmcn.learn_local(delta_true)
        dmcn_nohomeo.learn_local(delta_true)
        dmcn_nospine.learn_local(delta_true)
        mlp_bp.step_train(s_curr, delta_true)
        w_nlms += 0.2 * np.outer(delta_true - p_nlms, s_curr) / (1.0 + float(np.dot(s_curr, s_curr)))

    delta_var = float(np.var(np.array(true_deltas), axis=0).sum()) + 1e-9
    return {
        "e4_dmcn_nmse": float(np.mean(errs["dmcn"]) / delta_var),
        "e4_ablate_homeostasis_nmse": float(np.mean(errs["nohomeo"]) / delta_var),
        "e4_ablate_spine_nmse": float(np.mean(errs["nospine"]) / delta_var),
        "e4_persistence_nmse": float(np.mean(errs["persist"]) / delta_var),
        "e4_nlms_baseline_nmse": float(np.mean(errs["nlms"]) / delta_var),
        "e4_mlp_bp_nmse": float(np.mean(errs["mlp"]) / delta_var),
    }


# ---------------------------------------------------------------------------
# E5: Delayed-Cue Memory T-Maze RL (Zero Goal/Physics Oracles)
# ---------------------------------------------------------------------------
def eval_e5_delayed_cue_tmaze(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n_train_ep = 350
    n_test_ep = 100
    ep_len = 10  # steps 0..2 cue, steps 3..8 silent corridor, step 9 decision

    def make_rl_agent(ablate_spine: bool = False, ablate_elig: bool = False) -> DMCNLayer:
        cfg = DMCNConfig(
            d_in=2,
            d_out=2,
            n_branches=1,
            n_spine_scales=4,
            tau_min=1.0,
            tau_max=20.0,
            alpha_homeo=0.05,
            eta_theta=0.0,
            soma_activation="linear",
            use_nonmonotonic=False,
            eta_rl=0.25,
            gamma_forget=0.0,
            gamma_elig=0.90,
            eta_consol=0.02,
            weight_init_scale=0.02,
            seed=seed,
            ablation=AblationConfig(
                ablate_homeostasis=False,
                ablate_spine_traces=ablate_spine,
                ablate_eligibility=ablate_elig,
            ),
        )
        return DMCNLayer(cfg)

    def run_rl_agent(
        agent: DMCNLayer,
        shuffle_reward: bool = False,
        train: bool = True,
        blind: bool = False,
    ) -> float:
        r_baseline = 0.0
        local_rng = np.random.RandomState(seed + (777 if not train else 0))

        def run_ep(cue_sign: float, is_train: bool) -> int:
            agent.reset_temporal_state()
            logits = np.zeros(2, dtype=np.float64)
            for t in range(ep_len):
                if blind:
                    obs = np.zeros(2, dtype=np.float64)
                else:
                    c_val = cue_sign if t <= 2 else 0.0
                    obs = np.array([c_val, float(t) / float(ep_len - 1)], dtype=np.float64)
                logits = agent.step(obs, dt=1.0)
                if is_train and t == ep_len - 1:
                    probs = np.exp(logits - np.max(logits))
                    probs /= float(np.sum(probs))
                    act = int(local_rng.choice(2, p=probs))
                    grad = -probs
                    grad[act] += 1.0
                    agent.accumulate_eligibility(grad)
                    target_act = 1 if cue_sign > 0 else 0
                    rew = 1.0 if act == target_act else -1.0
                    if shuffle_reward:
                        rew = float(local_rng.choice([-1.0, 1.0]))
                    return int(rew > 0), rew  # type: ignore[return-value]
            probs = np.exp(logits - np.max(logits))
            probs /= float(np.sum(probs))
            act = int(np.argmax(probs))
            target_act = 1 if cue_sign > 0 else 0
            return int(act == target_act)

        if train:
            for _ in range(n_train_ep):
                cue = 1.0 if local_rng.rand() > 0.5 else -1.0
                _, rew = run_ep(cue, True)  # type: ignore[misc]
                rpe = rew - r_baseline
                r_baseline = 0.95 * r_baseline + 0.05 * rew
                agent.apply_rpe(rpe)

        hits = 0
        for _ in range(n_test_ep):
            cue = 1.0 if local_rng.rand() > 0.5 else -1.0
            hits += run_ep(cue, False)
        return 100.0 * hits / n_test_ep

    ag_intact = make_rl_agent(ablate_spine=False, ablate_elig=False)
    ag_untr = make_rl_agent(ablate_spine=False, ablate_elig=False)
    ag_nospine = make_rl_agent(ablate_spine=True, ablate_elig=False)
    ag_shuf = make_rl_agent(ablate_spine=False, ablate_elig=False)

    acc_dmcn = run_rl_agent(ag_intact, train=True)
    acc_blind = run_rl_agent(ag_intact, train=False, blind=True)
    acc_untr = run_rl_agent(ag_untr, train=False)
    acc_nospine = run_rl_agent(ag_nospine, train=True)
    acc_shuf = run_rl_agent(ag_shuf, shuffle_reward=True, train=True)

    # Memoryless REINFORCE MLP baseline (sees only junction obs [0, 1])
    w_mem = rng.randn(2, 2) * 0.05
    r_base_m = 0.0
    for _ in range(n_train_ep):
        cue = 1.0 if rng.rand() > 0.5 else -1.0
        obs_j = np.array([0.0, 1.0], dtype=np.float64)
        lg = w_mem @ obs_j
        pr = np.exp(lg - np.max(lg))
        pr /= float(np.sum(pr))
        a = int(rng.choice(2, p=pr))
        rew = 1.0 if a == (1 if cue > 0 else 0) else -1.0
        g = -pr
        g[a] += 1.0
        w_mem += 0.1 * (rew - r_base_m) * np.outer(g, obs_j)
        r_base_m = 0.95 * r_base_m + 0.05 * rew

    mem_hits, const_l_hits, rand_hits = 0, 0, 0
    for _ in range(n_test_ep):
        cue = 1.0 if rng.rand() > 0.5 else -1.0
        tgt = 1 if cue > 0 else 0
        mem_hits += int(int(np.argmax(w_mem @ np.array([0.0, 1.0]))) == tgt)
        const_l_hits += int(0 == tgt)
        rand_hits += int(int(rng.randint(0, 2)) == tgt)

    return {
        "e5_dmcn_acc": acc_dmcn,
        "e5_ablate_spine_acc": acc_nospine,
        "e5_untrained_acc": acc_untr,
        "e5_blind_acc": acc_blind,
        "e5_shuffled_reward_acc": acc_shuf,
        "e5_memoryless_reinforce_acc": 100.0 * mem_hits / n_test_ep,
        "e5_constant_left_acc": 100.0 * const_l_hits / n_test_ep,
        "e5_random_acc": 100.0 * rand_hits / n_test_ep,
    }


# ---------------------------------------------------------------------------
# E6a & E6b: Real Handwritten Digits Recognition & Continual Learning (sklearn)
# ---------------------------------------------------------------------------
_DIGITS_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def get_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    global _DIGITS_CACHE
    if _DIGITS_CACHE is None:
        ds = load_digits()
        X = (ds.data / 16.0).astype(np.float64)
        y = ds.target.astype(np.int64)
        _DIGITS_CACHE = (X, y)
    return _DIGITS_CACHE


def eval_e6_digits(seed: int) -> Dict[str, float]:
    X_all, y_all = get_digits_data()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_all))
    n_tr = int(0.8 * len(X_all))
    X_tr, y_tr = X_all[perm[:n_tr]], y_all[perm[:n_tr]]
    X_te, y_te = X_all[perm[n_tr:]], y_all[perm[n_tr:]]

    def make_digit_net(freeze_fb: bool = False, shuffle_ap: bool = False, n_branches: int = 1) -> DMCNNetwork:
        return DMCNNetwork([
            DMCNConfig(
                d_in=64,
                d_out=64,
                n_branches=n_branches,
                top_k_branches=1,
                n_spine_scales=0,
                soma_activation="tanh",
                eta_fast=0.06,
                gamma_forget=0.0,
                eta_consol=0.04,
                eta_fb=0.04,
                lambda_fb=0.002,
                seed=seed,
                ablation=AblationConfig(
                    ablate_homeostasis=True,
                    freeze_random_feedback=freeze_fb,
                    shuffle_apical=shuffle_ap,
                ),
            ),
            DMCNConfig(
                d_in=64,
                d_out=10,
                n_branches=n_branches,
                top_k_branches=1,
                n_spine_scales=0,
                soma_activation="linear",
                eta_fast=0.08,
                gamma_forget=0.0,
                eta_consol=0.04,
                eta_fb=0.04,
                lambda_fb=0.002,
                seed=seed + 1,
                ablation=AblationConfig(
                    ablate_homeostasis=True,
                    freeze_random_feedback=freeze_fb,
                    shuffle_apical=shuffle_ap,
                ),
            ),
        ])

    # E6a: Standard 2-epoch online recognition on held-out unseen digits
    net_dmcn = make_digit_net(freeze_fb=False, shuffle_ap=False)
    net_untr = make_digit_net(freeze_fb=False, shuffle_ap=False)
    net_shuf_ap = make_digit_net(freeze_fb=False, shuffle_ap=True)
    net_shuf_lbl = make_digit_net(freeze_fb=False, shuffle_ap=False)

    mlp_sgd = OnlineBackpropMLP(64, 64, 10, lr=0.03, use_adam=False, seed=seed)
    mlp_adam = OnlineBackpropMLP(64, 64, 10, lr=0.005, use_adam=True, seed=seed)
    mlp_fa = FixedFAMLP(64, 64, 10, lr=0.03, seed=seed)

    y_tr_shuf = rng.permutation(y_tr)
    for _ in range(3):
        ep_perm = rng.permutation(len(X_tr))
        for idx in ep_perm:
            x_i = X_tr[idx]
            tgt = np.zeros(10, dtype=np.float64)
            tgt[y_tr[idx]] = 1.0
            tgt_s = np.zeros(10, dtype=np.float64)
            tgt_s[y_tr_shuf[idx]] = 1.0

            net_dmcn.step(x_i)
            net_dmcn.learn_local(tgt)

            net_shuf_ap.step(x_i)
            net_shuf_ap.learn_local(tgt)

            net_shuf_lbl.step(x_i)
            net_shuf_lbl.learn_local(tgt_s)

            mlp_sgd.step_train(x_i, tgt)
            mlp_adam.step_train(x_i, tgt)
            mlp_fa.step_train(x_i, tgt)

    def acc_net(pred_fn, blind: bool = False) -> float:
        preds = [int(np.argmax(pred_fn(np.zeros(64) if blind else X_te[i]))) for i in range(len(X_te))]
        return 100.0 * float(np.mean(np.array(preds) == y_te))

    # Majority class baseline
    maj_cls = int(np.bincount(y_tr).argmax())
    maj_acc = 100.0 * float(np.mean(y_te == maj_cls))

    # E6b: Task-Free 5-Pair Continual Digit Recognition ({0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9})
    # Sparse k-WTA expansion (64 -> 256, k_wta=24) decorrelates overlapping pixel manifolds
    # so Boids-repulsion dendritic branches (B=5) isolate each digit pair without task IDs.
    X_tr_c = X_tr - 0.35
    X_te_c = X_te - 0.35
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    exp_layer = DMCNLayer(DMCNConfig(
        d_in=64,
        d_out=256,
        n_branches=1,
        n_spine_scales=0,
        k_wta=24,
        soma_activation="tanh",
        weight_init_scale=1.5,
        seed=seed + 10,
        ablation=AblationConfig(ablate_homeostasis=True),
    ))
    X_tr_sp = np.array([exp_layer.step(x) for x in X_tr_c], dtype=np.float64)
    X_te_sp = np.array([exp_layer.step(x) for x in X_te_c], dtype=np.float64)

    cfg_cont_dmcn = DMCNConfig(
        d_in=256,
        d_out=10,
        n_branches=5,
        top_k_branches=1,
        n_spine_scales=0,
        beta_route=14.0,
        eta_proto=0.04,
        boids_repulsion=0.18,
        theta_novel=0.35,
        theta_err_novel=0.03,
        theta_err=0.10,
        theta_coh=0.25,
        recruit_sustained=2,
        recruit_cooldown=20,
        soma_activation="linear",
        eta_fast=0.12,
        gamma_forget=0.0,
        eta_consol=0.06,
        seed=seed,
        ablation=AblationConfig(ablate_homeostasis=True),
    )
    cont_dmcn = DMCNLayer(cfg_cont_dmcn)
    cont_dmcn_b1 = DMCNLayer(DMCNConfig(
        d_in=256, d_out=10, n_branches=1, n_spine_scales=0, soma_activation="linear",
        eta_fast=0.12, gamma_forget=0.0, eta_consol=0.06, seed=seed,
        ablation=AblationConfig(ablate_homeostasis=True),
    ))
    cont_sgd = OnlineBackpropMLP(64, 64, 10, lr=0.03, use_adam=False, seed=seed)
    cont_adam = OnlineBackpropMLP(64, 64, 10, lr=0.005, use_adam=True, seed=seed)

    for c0, c1 in pairs:
        mask = (y_tr == c0) | (y_tr == c1)
        x_p_raw, x_p_sp, y_p = X_tr_c[mask], X_tr_sp[mask], y_tr[mask]
        for _ in range(2):
            p_ord = rng.permutation(len(y_p))
            for idx in p_ord:
                tgt = np.zeros(10, dtype=np.float64)
                tgt[y_p[idx]] = 1.0
                y_d = cont_dmcn.step(x_p_sp[idx])
                cont_dmcn.learn_local(tgt - y_d)
                y_b = cont_dmcn_b1.step(x_p_sp[idx])
                cont_dmcn_b1.learn_local(tgt - y_b)
                cont_sgd.step_train(x_p_raw[idx], tgt)
                cont_adam.step_train(x_p_raw[idx], tgt)

    def acc_net_sp(pred_fn) -> float:
        preds = [int(np.argmax(pred_fn(X_te_sp[i]))) for i in range(len(X_te_sp))]
        return 100.0 * float(np.mean(np.array(preds) == y_te))

    def acc_net_c(pred_fn) -> float:
        preds = [int(np.argmax(pred_fn(X_te_c[i]))) for i in range(len(X_te_c))]
        return 100.0 * float(np.mean(np.array(preds) == y_te))

    return {
        "e6a_dmcn_digits_acc": acc_net(net_dmcn.step),
        "e6a_untrained_acc": acc_net(net_untr.step),
        "e6a_blind_acc": acc_net(net_dmcn.step, blind=True),
        "e6a_shuffled_labels_acc": acc_net(net_shuf_lbl.step),
        "e6a_shuffle_apical_acc": acc_net(net_shuf_ap.step),
        "e6a_mlp_sgd_acc": acc_net(mlp_sgd.predict),
        "e6a_mlp_adam_acc": acc_net(mlp_adam.predict),
        "e6a_mlp_fa_acc": acc_net(mlp_fa.predict),
        "e6a_majority_baseline_acc": maj_acc,
        "e6b_continual_dmcn_acc": acc_net_sp(cont_dmcn.step),
        "e6b_continual_dmcn_b1_acc": acc_net_sp(cont_dmcn_b1.step),
        "e6b_continual_mlp_sgd_acc": acc_net_c(cont_sgd.predict),
        "e6b_continual_mlp_adam_acc": acc_net_c(cont_adam.predict),
        "e6b_branches_recruited": float(np.sum(cont_dmcn.recruited)),
    }


# ---------------------------------------------------------------------------
# E8: Collective Swarm Intelligence (Boids + DMCN Under Extreme Sensor Noise)
# ---------------------------------------------------------------------------
def eval_e8_swarm_boids(seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n_boids = 16
    n_steps = 120
    noise_sigma = 2.0
    dt = 0.2

    # Pre-generate migrating target trajectory & identical per-agent sensor noise across all conditions
    targets = np.zeros((n_steps, 2), dtype=np.float64)
    for t in range(n_steps):
        targets[t] = np.array([0.22 * t, 4.0 * np.sin(0.06 * t)], dtype=np.float64)
    sensor_noise = rng.randn(n_steps, n_boids, 2) * noise_sigma

    def run_swarm(
        use_dmcn: bool = True,
        decouple: bool = False,
        scramble: bool = False,
        blind: bool = False,
        n_agents: int = n_boids,
    ) -> Tuple[float, float, float]:
        sw = SwarmBoidNetwork(SwarmConfig(n_boids=n_agents, use_dmcn_memory=use_dmcn, seed=seed))
        errs, pols, cols = [], [], []
        for t in range(n_steps):
            diffs = targets[t][None, :] - sw.pos
            dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9
            true_bearings = diffs / dists
            noisy_b = true_bearings + sensor_noise[t, :n_agents]
            info = sw.step(
                noisy_b,
                dt=dt,
                decouple_neighbors=decouple,
                scramble_neighbors=scramble,
                blind=blind,
                learn=True,
            )
            if t >= n_steps - 40:
                errs.append(float(np.linalg.norm(info["centroid"] - targets[t])))
                pols.append(float(info["polarization"]))
            cols.append(float(info["n_collisions"]))
        return float(np.mean(errs)), float(np.mean(pols)), float(np.sum(cols))

    err_intact, pol_intact, col_intact = run_swarm(use_dmcn=True)
    err_pure, pol_pure, col_pure = run_swarm(use_dmcn=False)
    err_dec, pol_dec, _ = run_swarm(use_dmcn=True, decouple=True)
    err_scr, pol_scr, _ = run_swarm(use_dmcn=True, scramble=True)
    err_blind, pol_blind, _ = run_swarm(use_dmcn=True, blind=True)
    err_single, _, _ = run_swarm(use_dmcn=True, decouple=True, n_agents=1)

    return {
        "e8_swarm_dmcn_err": err_intact,
        "e8_swarm_dmcn_polarization": pol_intact,
        "e8_swarm_dmcn_collisions": col_intact,
        "e8_pure_boids_err": err_pure,
        "e8_pure_boids_polarization": pol_pure,
        "e8_pure_boids_collisions": col_pure,
        "e8_decoupled_err": err_dec,
        "e8_decoupled_polarization": pol_dec,
        "e8_scrambled_err": err_scr,
        "e8_scrambled_polarization": pol_scr,
        "e8_blind_err": err_blind,
        "e8_single_agent_err": err_single,
    }


# ---------------------------------------------------------------------------
# E7: Exact Computational Cost, Operations & CPU/CUDA Latency Profiling
# ---------------------------------------------------------------------------
def profile_compute_and_ops() -> Dict[str, Any]:
    cfgs_digits = [
        DMCNConfig(d_in=64, d_out=64, n_branches=4, top_k_branches=1, n_spine_scales=0, soma_activation="tanh", seed=1),
        DMCNConfig(d_in=64, d_out=10, n_branches=4, top_k_branches=1, n_spine_scales=0, soma_activation="linear", seed=2),
    ]
    np_net = DMCNNetwork(cfgs_digits)
    ops_dmcn = np_net.total_ops_report()

    # Exact MLP (64 -> 64 -> 10) analytical ops comparison
    mlp_fwd_macs = 64 * 64 + 64 * 10
    mlp_bwd_macs = 2 * (64 * 64 + 64 * 10)  # grad_h + outer products
    mlp_params = (64 * 64 + 64) + (64 * 10 + 10)

    # Measure real CPU NumPy wall-clock latency (500 online step+learn calls)
    rng = np.random.RandomState(42)
    xs = rng.randn(500, 64)
    ts = rng.randn(500, 10)

    t0 = time.perf_counter()
    for i in range(500):
        np_net.step(xs[i])
        np_net.learn_local(ts[i])
    dmcn_cpu_us = (time.perf_counter() - t0) * 1e6 / 500.0

    mlp_ref = OnlineBackpropMLP(64, 64, 10, lr=0.01, seed=42)
    t0 = time.perf_counter()
    for i in range(500):
        mlp_ref.step_train(xs[i], ts[i])
    mlp_cpu_us = (time.perf_counter() - t0) * 1e6 / 500.0

    # Measure PyTorch CPU & CUDA latency
    pt_cpu = DMCNTorchNetwork(cfgs_digits, device="cpu", dtype=torch.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32)
    ts_t = torch.tensor(ts, dtype=torch.float32)
    t0 = time.perf_counter()
    for i in range(200):
        pt_cpu(xs_t[i])
        pt_cpu.learn_local(ts_t[i])
    dmcn_pt_cpu_us = (time.perf_counter() - t0) * 1e6 / 200.0

    cuda_available = torch.cuda.is_available()
    dmcn_cuda_us = -1.0
    if cuda_available:
        pt_gpu = DMCNTorchNetwork(cfgs_digits, device="cuda", dtype=torch.float32)
        xs_g = xs_t.to("cuda")
        ts_g = ts_t.to("cuda")
        # Warmup
        for i in range(20):
            pt_gpu(xs_g[i])
            pt_gpu.learn_local(ts_g[i])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(200):
            pt_gpu(xs_g[i])
            pt_gpu.learn_local(ts_g[i])
        torch.cuda.synchronize()
        dmcn_cuda_us = (time.perf_counter() - t0) * 1e6 / 200.0

    # Single-cell (2 -> 1, K=4) ops
    single_ops = DMCNLayer(DMCNConfig(d_in=2, d_out=1, n_branches=1, n_spine_scales=4, use_nonmonotonic=True)).ops_report()

    return {
        "digits_net_64_64_10_b4": ops_dmcn,
        "single_cell_2_1_k4": {
            "forward_macs": single_ops.forward_macs,
            "learn_macs": single_ops.learn_macs,
            "total_flops_per_step": single_ops.total_flops_per_step,
            "parameter_count": single_ops.parameter_count,
            "memory_bytes_fp32": single_ops.memory_bytes_fp32,
        },
        "matched_mlp_64_64_10": {
            "forward_macs": mlp_fwd_macs,
            "learn_macs": mlp_bwd_macs,
            "total_flops_per_step": 2 * (mlp_fwd_macs + mlp_bwd_macs),
            "parameter_count": mlp_params,
            "memory_bytes_fp32": mlp_params * 4,
        },
        "latency_us_per_step": {
            "dmcn_numpy_cpu": float(dmcn_cpu_us),
            "mlp_numpy_cpu": float(mlp_cpu_us),
            "dmcn_torch_cpu": float(dmcn_pt_cpu_us),
            "dmcn_torch_cuda": float(dmcn_cuda_us),
            "cuda_available": bool(cuda_available),
        },
    }


def evaluate_seed(seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seed": seed}
    out.update(eval_e1_xor(seed))
    out.update(eval_e2_associative_recall(seed))
    out.update(eval_e3_continual_regimes(seed))
    out.update(eval_e4_nonstationary_chaos(seed))
    out.update(eval_e5_delayed_cue_tmaze(seed))
    out.update(eval_e6_digits(seed))
    out.update(eval_e8_swarm_boids(seed))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    parser.add_argument("--exp-id", default="dmcn_master_eval_v2")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        res = evaluate_seed(999)
        ops = profile_compute_and_ops()
        log_path = out_dir / "tuning_log.json"
        if log_path.exists():
            raw = json.loads(log_path.read_text(encoding="utf-8"))
            entries = raw.get("tuning_entries", []) if isinstance(raw, dict) else []
        else:
            entries = []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id=f"{args.exp_id}_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_tuning_seed_999"},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"val_seed_999": res, "compute_profile": ops}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    per_seed = [evaluate_seed(s) for s in test_seeds]
    metric_keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in metric_keys}
    compute_profile = profile_compute_and_ops()

    payload = {
        "headline_metric": "multi_suite_dmcn_evaluation",
        "n_seeds": len(test_seeds),
        "summary": summary,
        "compute_profile": compute_profile,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id=args.exp_id,
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"suite": "E1a,E1b,E2,E3,E4,E5,E6a,E6b,E7", "n_test_seeds": 20},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"summary": summary, "compute_profile": compute_profile}, indent=2))


if __name__ == "__main__":
    main()
