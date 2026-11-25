"""Pre-registered 20-Seed Task-Free, Replay-Free Class-Incremental Learning Benchmark (Real Digits).

Evaluates 2-layer DMCN (nonlinear dendritic expansion + local Sherman-Morrison synaptic
precision metaplasticity) on strictly sequential 5-pair handwritten digit recognition:
    {0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}
with ZERO replay buffers, ZERO task IDs, ZERO task boundaries, and ZERO backpropagation.

Baselines & Controls (identical splits & sequential stream):
  - DMCN (intact: 64 -> 256 nonlinear dendritic expansion -> 10 precision metaplasticity)
  - DMCN (ablate_precision=True: exact same architecture without synaptic precision matrix)
  - Raw-Pixel Linear RLS (64 -> 10: precision metaplasticity without nonlinear dendritic expansion)
  - Online Backprop MLP (SGD, 64 -> 256 -> 10)
  - Online Backprop MLP (Adam, 64 -> 256 -> 10)
  - Joint Offline Upper Bound (closed-form ridge on all 10 classes simultaneously)
  - Null Controls: Untrained, Blind (x=0), Shuffled Labels, Majority Class
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import load_digits

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNLayer  # noqa: E402
from benchmarks.dmcn_master_benchmark import OnlineBackpropMLP  # noqa: E402


_DIGITS_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def get_digits() -> Tuple[np.ndarray, np.ndarray]:
    global _DIGITS_CACHE
    if _DIGITS_CACHE is None:
        ds = load_digits()
        _DIGITS_CACHE = ((ds.data / 16.0).astype(np.float64), ds.target.astype(np.int64))
    return _DIGITS_CACHE


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


def evaluate_seed(seed: int, d_exp: int = 256, ridge: float = 2.0, exp_scale: float = 1.6) -> Dict[str, Any]:
    X_all, y_all = get_digits()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_all))
    n_tr = int(0.8 * len(X_all))
    X_tr, y_tr = X_all[perm[:n_tr]], y_all[perm[:n_tr]]
    X_te, y_te = X_all[perm[n_tr:]], y_all[perm[n_tr:]]

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # Layer 1: Nonlinear dendritic expansion (64 -> d_exp, tanh + bias feature)
    exp_layer = DMCNLayer(DMCNConfig(
        d_in=64,
        d_out=d_exp,
        n_branches=1,
        n_spine_scales=0,
        soma_activation="tanh",
        weight_init_scale=exp_scale,
        seed=seed + 101,
        ablation=AblationConfig(ablate_homeostasis=True),
    ))
    # Random dendritic thresholds for diverse nonlinear receptive fields
    exp_layer.b_dend[0] = rng.normal(0.0, 0.35, size=d_exp)

    def feat(x: np.ndarray) -> np.ndarray:
        h = exp_layer.step(x)
        return np.append(h, 1.0)  # include constant bias coordinate for exact affine precision

    def make_head(d_in_h: int, ablate_prec: bool = False) -> DMCNLayer:
        return DMCNLayer(DMCNConfig(
            d_in=d_in_h,
            d_out=10,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            eta_fast=1.0 if not ablate_prec else 0.10,
            gamma_forget=0.0,
            eta_consol=0.0,
            weight_init_scale=0.0,
            use_precision_plasticity=True,
            precision_ridge=ridge,
            seed=seed,
            ablation=AblationConfig(ablate_homeostasis=True, ablate_precision=ablate_prec),
        ))

    head_intact = make_head(d_exp + 1, ablate_prec=False)
    head_ablate = make_head(d_exp + 1, ablate_prec=True)
    head_untr = make_head(d_exp + 1, ablate_prec=False)
    head_shuf = make_head(d_exp + 1, ablate_prec=False)
    head_raw_rls = make_head(65, ablate_prec=False)

    mlp_sgd = OnlineBackpropMLP(64, d_exp, 10, lr=0.03, use_adam=False, seed=seed)
    mlp_adam = OnlineBackpropMLP(64, d_exp, 10, lr=0.002, use_adam=True, seed=seed)

    # Pre-extract features strictly sample-by-sample (identical mapping at train & test)
    F_tr = np.array([feat(x) for x in X_tr], dtype=np.float64)
    F_te = np.array([feat(x) for x in X_te], dtype=np.float64)
    X_tr_b = np.hstack([X_tr, np.ones((len(X_tr), 1), dtype=np.float64)])
    X_te_b = np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float64)])

    def eval_acc(pred_fn, X_eval: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        idx_e = np.arange(len(y_te)) if mask is None else np.flatnonzero(mask)
        preds = [int(np.argmax(pred_fn(X_eval[i]))) for i in idx_e]
        return 100.0 * float(np.mean(np.array(preds) == y_te[idx_e]))

    p0_mask_te = (y_te == 0) | (y_te == 1)
    p0_after_p0_dmcn = 0.0
    p0_after_p0_sgd = 0.0

    # Strictly sequential 1-pass-per-pair stream (ZERO replay, ZERO task IDs!)
    y_tr_shuf = rng.permutation(y_tr)
    for p_idx, (c0, c1) in enumerate(pairs):
        mask = (y_tr == c0) | (y_tr == c1)
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        for i in indices:
            tgt = np.zeros(10, dtype=np.float64)
            tgt[y_tr[i]] = 1.0
            tgt_s = np.zeros(10, dtype=np.float64)
            tgt_s[y_tr_shuf[i]] = 1.0

            y_h = head_intact.step(F_tr[i])
            head_intact.learn_local(tgt - y_h)

            y_ab = head_ablate.step(F_tr[i])
            head_ablate.learn_local(tgt - y_ab)

            y_sh = head_shuf.step(F_tr[i])
            head_shuf.learn_local(tgt_s - y_sh)

            y_raw = head_raw_rls.step(X_tr_b[i])
            head_raw_rls.learn_local(tgt - y_raw)

            mlp_sgd.step_train(X_tr[i], tgt)
            mlp_adam.step_train(X_tr[i], tgt)

        if p_idx == 0:
            p0_after_p0_dmcn = eval_acc(head_intact.step, F_te, p0_mask_te)
            p0_after_p0_sgd = eval_acc(mlp_sgd.predict, X_te, p0_mask_te)

    # Joint Offline Upper Bound (closed-form ridge on ALL 10 classes simultaneously)
    Y_tr_oh = np.eye(10, dtype=np.float64)[y_tr]
    A_joint = F_tr.T @ F_tr + ridge * np.eye(d_exp + 1, dtype=np.float64)
    W_joint = np.linalg.solve(A_joint, F_tr.T @ Y_tr_oh).T  # (10, d_exp + 1)

    p0_final_dmcn = eval_acc(head_intact.step, F_te, p0_mask_te)
    p0_final_sgd = eval_acc(mlp_sgd.predict, X_te, p0_mask_te)

    maj_cls = int(np.bincount(y_tr).argmax())
    maj_acc = 100.0 * float(np.mean(y_te == maj_cls))

    return {
        "seed": seed,
        "dmcn_cil_final_acc": eval_acc(head_intact.step, F_te),
        "dmcn_cil_p0_retention": p0_final_dmcn,
        "dmcn_cil_p0_bwt": p0_final_dmcn - p0_after_p0_dmcn,
        "joint_offline_upper_bound_acc": eval_acc(lambda f: W_joint @ f, F_te),
        "raw_linear_rls_acc": eval_acc(head_raw_rls.step, X_te_b),
        "ablate_precision_acc": eval_acc(head_ablate.step, F_te),
        "mlp_sgd_cil_acc": eval_acc(mlp_sgd.predict, X_te),
        "mlp_sgd_p0_retention": p0_final_sgd,
        "mlp_sgd_p0_bwt": p0_final_sgd - p0_after_p0_sgd,
        "mlp_adam_cil_acc": eval_acc(mlp_adam.predict, X_te),
        "untrained_acc": eval_acc(head_untr.step, F_te),
        "blind_acc": eval_acc(lambda _: head_intact.step(np.zeros(d_exp + 1)), F_te),
        "shuffled_labels_acc": eval_acc(head_shuf.step, F_te),
        "majority_acc": maj_acc,
    }


def profile_cil_ops(d_exp: int = 256, ridge: float = 2.0) -> Dict[str, Any]:
    exp_l = DMCNLayer(DMCNConfig(d_in=64, d_out=d_exp, n_branches=1, n_spine_scales=0, soma_activation="tanh"))
    head_l = DMCNLayer(DMCNConfig(
        d_in=d_exp + 1, d_out=10, n_branches=1, n_spine_scales=0, soma_activation="linear",
        eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0, use_precision_plasticity=True, precision_ridge=ridge,
        ablation=AblationConfig(ablate_homeostasis=True),
    ))
    rng = np.random.RandomState(42)
    xs = rng.randn(300, 64)
    ts = rng.randn(300, 10)
    t0 = time.perf_counter()
    for i in range(300):
        f = np.append(exp_l.step(xs[i]), 1.0)
        y = head_l.step(f)
        head_l.learn_local(ts[i] - y)
    us_step = (time.perf_counter() - t0) * 1e6 / 300.0
    r_exp = exp_l.ops_report()
    r_head = head_l.ops_report()
    return {
        "us_per_online_step_cpu": float(us_step),
        "forward_macs": int(r_exp.forward_macs + r_head.forward_macs),
        "learn_macs": int(r_head.learn_macs),
        "total_flops_per_step": int(r_exp.forward_flops + r_head.total_flops_per_step),
        "total_params": int(r_exp.parameter_count + r_head.parameter_count),
        "precision_matrix_fp32_bytes": int((d_exp + 1) * (d_exp + 1) * 4),
        "total_ram_fp32_bytes": int(r_exp.memory_bytes_fp32 + r_head.memory_bytes_fp32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_continual_cil_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        res = evaluate_seed(999)
        prof = profile_cil_ops()
        log_path = out_dir / "tuning_log.json"
        if log_path.exists():
            raw = json.loads(log_path.read_text(encoding="utf-8"))
            entries = raw.get("tuning_entries", []) if isinstance(raw, dict) else []
        else:
            entries = []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_continual_cil_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "d_exp": 256, "ridge": 2.0},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"val_seed_999": res, "compute_profile": prof}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    per_seed = [evaluate_seed(s) for s in test_seeds]
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}
    prof = profile_cil_ops()

    payload = {
        "headline_metric": "dmcn_cil_final_acc",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "summary": summary,
        "compute_profile": prof,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_continual_cil_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"d_exp": 256, "ridge": 2.0, "exp_scale": 1.6, "pairs": 5},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"summary": summary, "compute_profile": prof}, indent=2))


if __name__ == "__main__":
    main()
