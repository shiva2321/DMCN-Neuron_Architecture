"""Pre-registered 20-Seed Zero-Shot Compositional Generalization Benchmark (dmcn_compositional_v1).

Tests FactorizedBilinearDMCNLayer (unsupervised sparse Oja/L1 dendritic factorization + additive &
bilinear coincidence binding + local Sherman-Morrison readout, ZERO hardcoded factor indices, ZERO backprop)
on TWO genuine zero-shot compositional tasks with randomly permuted input columns per seed:
  - Task 1: Attribute Composition (4 Colors x 4 Shapes = 16 pairs; 12 seen / 4 NEVER-SEEN pairs)
  - Task 2: Rule x Distributed-Operand Composition (4 Operators x 12 Operands = 48 pairs; 36 seen / 12 NEVER-SEEN)

Compares against ablate_bilinear, Dense Entangled DMCN, PyTorch Backprop MLPs (SGD & Adam), and Null Controls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNLayer, FactorizedBilinearDMCNLayer  # noqa: E402


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * sd / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": ci95, "min": float(np.min(arr)), "max": float(np.max(arr))}


class DenseEntangledDMCN:
    """Standard dense random-tanh DMCN + Woodbury readout (no factorization, no bilinear binding)."""

    def __init__(self, d_in: int, d_out: int, d_hid: int = 128, ridge: float = 0.05, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        self.enc = DMCNLayer(DMCNConfig(
            d_in=d_in, d_out=d_hid, n_branches=1, n_spine_scales=0,
            soma_activation="tanh", weight_init_scale=2.2, seed=seed + 11,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))
        self.enc.b_dend[0] = rng.normal(0.0, 0.35, size=d_hid)
        self.head = DMCNLayer(DMCNConfig(
            d_in=d_hid + 1, d_out=d_out, n_branches=1, n_spine_scales=0,
            soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
            weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=ridge,
            seed=seed + 22, ablation=AblationConfig(ablate_homeostasis=True),
        ))

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.head.step(np.append(self.enc.step(x), 1.0))

    def learn_local(self, target: np.ndarray) -> None:
        self.head.learn_local(target - self.head._last_y)


class TorchMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hid: int = 128, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh(), nn.Linear(d_hid, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_torch_mlp(X_tr: np.ndarray, Y_tr: np.ndarray, seed: int, use_adam: bool = True, epochs: int = 250) -> TorchMLP:
    m = TorchMLP(X_tr.shape[1], Y_tr.shape[1], d_hid=128, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2) if use_adam else torch.optim.SGD(m.parameters(), lr=0.15, momentum=0.9)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(Y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((m(xt) - yt) ** 2)
        loss.backward()
        opt.step()
    return m


def evaluate_seed(seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    # === TASK 1: Zero-Shot Attribute Composition (4 Colors x 4 Shapes; 12 seen / 4 NEVER-SEEN) ===
    perm1 = rng.permutation(8)
    all_p1 = [(c, s) for c in range(4) for s in range(4)]
    test_p1 = {(c, (c + (seed % 4)) % 4) for c in range(4)}
    train_p1 = [p for p in all_p1 if p not in test_p1]

    def sample_t1(pairs: List, n_per_pair: int, noise: float = 0.08):
        xs, ys = [], []
        for c, s in pairs:
            for _ in range(n_per_pair):
                raw = np.zeros(8, dtype=np.float64)
                raw[c] = 1.0
                raw[4 + s] = 1.0
                x = (raw + rng.normal(0.0, noise, size=8))[perm1]
                xs.append(x)
                ys.append(raw)
        return np.array(xs), np.array(ys)

    X1_tr, Y1_tr = sample_t1(train_p1, n_per_pair=20)
    X1_te, Y1_te = sample_t1(list(test_p1), n_per_pair=25)

    fb1 = FactorizedBilinearDMCNLayer(d_in=8, d_out=8, n_branches=16, ridge=0.1, ablate_bilinear=False, seed=seed)
    fb1_shuf = FactorizedBilinearDMCNLayer(d_in=8, d_out=8, n_branches=16, ridge=0.1, ablate_bilinear=False, seed=seed)
    fb1_untr = FactorizedBilinearDMCNLayer(d_in=8, d_out=8, n_branches=16, ridge=0.1, ablate_bilinear=False, seed=seed)
    dense1 = DenseEntangledDMCN(d_in=8, d_out=8, d_hid=128, ridge=0.1, seed=seed)

    for x in X1_tr:
        fb1.observe_unsupervised(x)
        fb1_shuf.observe_unsupervised(x)
    shuf1 = rng.permutation(len(Y1_tr))
    for i in range(len(X1_tr)):
        fb1.step(X1_tr[i])
        fb1.learn_local(Y1_tr[i])
        fb1_shuf.step(X1_tr[i])
        fb1_shuf.learn_local(Y1_tr[shuf1[i]])
        dense1.step(X1_tr[i])
        dense1.learn_local(Y1_tr[i])

    mlp1_adam = _train_torch_mlp(X1_tr, Y1_tr, seed=seed, use_adam=True)
    mlp1_sgd = _train_torch_mlp(X1_tr, Y1_tr, seed=seed, use_adam=False)

    def pair_acc(preds: np.ndarray, tgts: np.ndarray) -> float:
        ok = (np.argmax(preds[:, :4], axis=1) == np.argmax(tgts[:, :4], axis=1)) & (
            np.argmax(preds[:, 4:], axis=1) == np.argmax(tgts[:, 4:], axis=1)
        )
        return 100.0 * float(np.mean(ok))

    p1_fb = np.array([fb1.step(x) for x in X1_te])
    p1_dense = np.array([dense1.step(x) for x in X1_te])
    p1_untr = np.array([fb1_untr.step(x) for x in X1_te])
    p1_blind = np.array([fb1.step(np.zeros(8)) for _ in X1_te])
    p1_shuf = np.array([fb1_shuf.step(x) for x in X1_te])
    with torch.no_grad():
        p1_adam = mlp1_adam(torch.tensor(X1_te, dtype=torch.float32)).numpy()
        p1_sgd = mlp1_sgd(torch.tensor(X1_te, dtype=torch.float32)).numpy()

    # === TASK 2: Zero-Shot Rule x Distributed-Operand Composition (4 Ops x 12 Operands; 36 seen / 12 NEVER-SEEN) ===
    perm2 = rng.permutation(8)
    ops_M = [np.eye(4), -np.eye(4), np.roll(np.eye(4), 1, axis=0), np.roll(np.eye(4), 2, axis=0)]
    operands = rng.randn(12, 4)
    operands /= (np.linalg.norm(operands, axis=1, keepdims=True) + 1e-8)

    all_p2 = [(o, v) for o in range(4) for v in range(12)]
    test_p2 = {(o, (3 * o + k) % 12) for o in range(4) for k in range(3)}  # 12 unseen pairs!
    train_p2 = [p for p in all_p2 if p not in test_p2]

    def sample_t2(pairs: List, n_per_pair: int, noise: float = 0.03):
        xs, ys = [], []
        for o, v_i in pairs:
            for _ in range(n_per_pair):
                op_oh = np.zeros(4, dtype=np.float64)
                op_oh[o] = 1.0
                val = operands[v_i] + rng.normal(0.0, noise, size=4)
                x = np.concatenate([op_oh, val])[perm2]
                y = ops_M[o] @ operands[v_i]
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    X2_tr, Y2_tr = sample_t2(train_p2, n_per_pair=10)
    X2_te, Y2_te = sample_t2(list(test_p2), n_per_pair=10, noise=0.0)
    # Build prototype bank of all 48 true outputs for nearest-prototype accuracy
    proto_y = np.array([ops_M[o] @ operands[v_i] for o, v_i in all_p2])

    fb2 = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=False, seed=seed + 50)
    fb2_nob = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=True, seed=seed + 50)
    fb2_shuf = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=False, seed=seed + 50)
    fb2_untr = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=False, seed=seed + 50)
    dense2 = DenseEntangledDMCN(d_in=8, d_out=4, d_hid=128, ridge=0.05, seed=seed + 50)

    for x in X2_tr:
        fb2.observe_unsupervised(x)
        fb2_nob.observe_unsupervised(x)
        fb2_shuf.observe_unsupervised(x)
    shuf2 = rng.permutation(len(Y2_tr))
    for i in range(len(X2_tr)):
        fb2.step(X2_tr[i])
        fb2.learn_local(Y2_tr[i])
        fb2_nob.step(X2_tr[i])
        fb2_nob.learn_local(Y2_tr[i])
        fb2_shuf.step(X2_tr[i])
        fb2_shuf.learn_local(Y2_tr[shuf2[i]])
        dense2.step(X2_tr[i])
        dense2.learn_local(Y2_tr[i])

    mlp2_adam = _train_torch_mlp(X2_tr, Y2_tr, seed=seed + 50, use_adam=True)
    mlp2_sgd = _train_torch_mlp(X2_tr, Y2_tr, seed=seed + 50, use_adam=False)

    def eval_t2(preds: np.ndarray) -> tuple[float, float]:
        mse = float(np.mean((preds - Y2_te) ** 2))
        cos_ok = 0
        for p_v, y_v in zip(preds, Y2_te):
            d_p = np.linalg.norm(proto_y - p_v[None, :], axis=1)
            d_y = np.linalg.norm(proto_y - y_v[None, :], axis=1)
            cos_ok += int(np.argmin(d_p) == np.argmin(d_y))
        return mse, 100.0 * cos_ok / float(len(Y2_te))

    m2_fb, a2_fb = eval_t2(np.array([fb2.step(x) for x in X2_te]))
    m2_nob, a2_nob = eval_t2(np.array([fb2_nob.step(x) for x in X2_te]))
    m2_dense, a2_dense = eval_t2(np.array([dense2.step(x) for x in X2_te]))
    m2_untr, a2_untr = eval_t2(np.array([fb2_untr.step(x) for x in X2_te]))
    m2_blind, a2_blind = eval_t2(np.array([fb2.step(np.zeros(8)) for _ in X2_te]))
    m2_shuf, a2_shuf = eval_t2(np.array([fb2_shuf.step(x) for x in X2_te]))
    with torch.no_grad():
        m2_adam, a2_adam = eval_t2(mlp2_adam(torch.tensor(X2_te, dtype=torch.float32)).numpy())
        m2_sgd, a2_sgd = eval_t2(mlp2_sgd(torch.tensor(X2_te, dtype=torch.float32)).numpy())

    return {
        "seed": seed,
        "t1_attr_dmcn_fac_acc": pair_acc(p1_fb, Y1_te),
        "t1_attr_dense_dmcn_acc": pair_acc(p1_dense, Y1_te),
        "t1_attr_mlp_adam_acc": pair_acc(p1_adam, Y1_te),
        "t1_attr_mlp_sgd_acc": pair_acc(p1_sgd, Y1_te),
        "t1_attr_untrained_acc": pair_acc(p1_untr, Y1_te),
        "t1_attr_blind_acc": pair_acc(p1_blind, Y1_te),
        "t1_attr_shuffled_acc": pair_acc(p1_shuf, Y1_te),
        "t2_rule_dmcn_fac_mse": m2_fb,
        "t2_rule_dmcn_fac_acc": a2_fb,
        "t2_rule_ablate_bilinear_mse": m2_nob,
        "t2_rule_ablate_bilinear_acc": a2_nob,
        "t2_rule_dense_dmcn_mse": m2_dense,
        "t2_rule_dense_dmcn_acc": a2_dense,
        "t2_rule_mlp_adam_mse": m2_adam,
        "t2_rule_mlp_adam_acc": a2_adam,
        "t2_rule_mlp_sgd_mse": m2_sgd,
        "t2_rule_mlp_sgd_acc": a2_sgd,
        "t2_rule_untrained_acc": a2_untr,
        "t2_rule_blind_acc": a2_blind,
        "t2_rule_shuffled_acc": a2_shuf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_compositional_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_compositional_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999"},
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
        "headline_metric": "t2_rule_dmcn_fac_mse",
        "direction": "lower_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_compositional_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"n_branches": 16, "tasks": ["zero_shot_attr_4x4", "zero_shot_rule_4x12"]},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
