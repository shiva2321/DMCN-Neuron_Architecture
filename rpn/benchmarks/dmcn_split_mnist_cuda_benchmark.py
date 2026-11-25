"""Pre-registered 20-Seed CUDA Split-MNIST Class-Incremental Learning & GPU Throughput Benchmark.

Evaluates native batched CUDA DMCNTorchLayer (784 -> 2048 nonlinear dendritic expansion ->
2049 -> 10 exact batched Woodbury local synaptic precision plasticity) on official Split-MNIST:
    {0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}
(60,000 train / 10,000 held-out test, 1 pass per pair, batch_size=256)
with ZERO backpropagation, ZERO replay buffers, ZERO exemplars, ZERO task IDs/boundaries.
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
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNTorchLayer, DMCNTorchNetwork  # noqa: E402


_MNIST_CACHE: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None


def get_mnist(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        root = Path.home() / ".cache" / "torchvision_mnist"
        tr = datasets.MNIST(root=str(root), train=True, download=True)
        te = datasets.MNIST(root=str(root), train=False, download=True)
        x_tr = (tr.data.reshape(-1, 784).float() / 255.0).to(device)
        y_tr = tr.targets.long().to(device)
        x_te = (te.data.reshape(-1, 784).float() / 255.0).to(device)
        y_te = te.targets.long().to(device)
        _MNIST_CACHE = (x_tr, y_tr, x_te, y_te)
    return _MNIST_CACHE


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


class TorchBackpropMLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _eval_logits_acc(logits: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        logits = logits[mask]
        y_true = y_true[mask]
    preds = torch.argmax(logits, dim=1)
    return 100.0 * float((preds == y_true).float().mean().item())


def evaluate_seed(
    seed: int,
    device: torch.device,
    d_exp: int = 2048,
    ridge: float = 10.0,
    exp_scale: float = 2.2,
    batch_size: int = 256,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)

    x_tr, y_tr, x_te, y_te = get_mnist(device)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    p0_mask_te = (y_te == 0) | (y_te == 1)

    # Layer 1: Nonlinear dendritic expansion (784 -> d_exp, tanh + random dendritic thresholds)
    exp_layer = DMCNTorchLayer(
        DMCNConfig(
            d_in=784,
            d_out=d_exp,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="tanh",
            weight_init_scale=exp_scale,
            seed=seed + 101,
            ablation=AblationConfig(ablate_homeostasis=True),
        ),
        device=device,
        dtype=torch.float32,
    )
    b_rand = torch.tensor(rng.normal(0.0, 0.35, size=d_exp), device=device, dtype=torch.float32)
    exp_layer.b_dend[0].copy_(b_rand)

    def extract_features(x_in: torch.Tensor) -> torch.Tensor:
        chunks = []
        for s in range(0, x_in.shape[0], 2048):
            h = exp_layer(x_in[s : s + 2048])
            ones = torch.ones((h.shape[0], 1), device=device, dtype=torch.float32)
            chunks.append(torch.cat([h, ones], dim=1))
        return torch.cat(chunks, dim=0)

    f_tr = extract_features(x_tr)
    f_te = extract_features(x_te)
    x_tr_b = torch.cat([x_tr, torch.ones((x_tr.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    x_te_b = torch.cat([x_te, torch.ones((x_te.shape[0], 1), device=device, dtype=torch.float32)], dim=1)

    def make_head(d_in_h: int, ablate_prec: bool = False) -> DMCNTorchLayer:
        return DMCNTorchLayer(
            DMCNConfig(
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
            ),
            device=device,
            dtype=torch.float32,
        )

    head_intact = make_head(d_exp + 1, ablate_prec=False)
    head_ablate = make_head(d_exp + 1, ablate_prec=True)
    head_untr = make_head(d_exp + 1, ablate_prec=False)
    head_shuf = make_head(d_exp + 1, ablate_prec=False)
    head_raw_rls = make_head(785, ablate_prec=False)

    torch.manual_seed(seed)
    mlp_sgd = TorchBackpropMLP(784, d_exp, 10).to(device)
    opt_sgd = torch.optim.SGD(mlp_sgd.parameters(), lr=0.1, momentum=0.9)

    torch.manual_seed(seed)
    mlp_adam = TorchBackpropMLP(784, d_exp, 10).to(device)
    opt_adam = torch.optim.Adam(mlp_adam.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    y_tr_oh = torch.zeros((y_tr.shape[0], 10), device=device, dtype=torch.float32)
    y_tr_oh.scatter_(1, y_tr.unsqueeze(1), 1.0)
    shuf_perm = torch.tensor(rng.permutation(y_tr.shape[0]), device=device, dtype=torch.long)
    y_tr_shuf_oh = y_tr_oh[shuf_perm]

    p0_after_p0_dmcn = 0.0
    p0_after_p0_sgd = 0.0
    p0_after_p0_adam = 0.0

    # Strictly sequential 1-pass-per-pair stream (ZERO replay, ZERO task IDs)
    for p_idx, (c0, c1) in enumerate(pairs):
        mask_np = ((y_tr == c0) | (y_tr == c1)).cpu().numpy()
        idx_np = np.flatnonzero(mask_np)
        rng.shuffle(idx_np)
        idx_t = torch.tensor(idx_np, device=device, dtype=torch.long)

        for start in range(0, idx_t.shape[0], batch_size):
            b_idx = idx_t[start : start + batch_size]
            fb = f_tr[b_idx]
            xb = x_tr[b_idx]
            xb_b = x_tr_b[b_idx]
            yb = y_tr[b_idx]
            tb = y_tr_oh[b_idx]
            tb_s = y_tr_shuf_oh[b_idx]

            # DMCN intact (batched exact Woodbury precision plasticity)
            yh = head_intact(fb)
            head_intact.learn_local(tb - yh)

            # DMCN ablate_precision
            yab = head_ablate(fb)
            head_ablate.learn_local(tb - yab)

            # DMCN shuffled-label null control
            ysh = head_shuf(fb)
            head_shuf.learn_local(tb_s - ysh)

            # Raw-pixel linear RLS baseline
            yraw = head_raw_rls(xb_b)
            head_raw_rls.learn_local(tb - yraw)

            # PyTorch Backprop MLP (SGD)
            opt_sgd.zero_grad(set_to_none=True)
            loss_sgd = ce_loss(mlp_sgd(xb), yb)
            loss_sgd.backward()
            opt_sgd.step()

            # PyTorch Backprop MLP (Adam)
            opt_adam.zero_grad(set_to_none=True)
            loss_adam = ce_loss(mlp_adam(xb), yb)
            loss_adam.backward()
            opt_adam.step()

        if p_idx == 0:
            p0_after_p0_dmcn = _eval_logits_acc(head_intact(f_te), y_te, p0_mask_te)
            with torch.no_grad():
                p0_after_p0_sgd = _eval_logits_acc(mlp_sgd(x_te), y_te, p0_mask_te)
                p0_after_p0_adam = _eval_logits_acc(mlp_adam(x_te), y_te, p0_mask_te)

    # FP64 Joint Offline Ridge Upper Bound on ALL 60,000 samples simultaneously
    f_tr_64 = f_tr.double()
    y_oh_64 = y_tr_oh.double()
    a_joint = f_tr_64.T @ f_tr_64 + ridge * torch.eye(d_exp + 1, device=device, dtype=torch.float64)
    w_joint = torch.linalg.solve(a_joint, f_tr_64.T @ y_oh_64)  # (d_exp+1, 10)
    joint_logits = (f_te.double() @ w_joint).float()

    dmcn_logits = head_intact(f_te)
    p0_final_dmcn = _eval_logits_acc(dmcn_logits, y_te, p0_mask_te)
    with torch.no_grad():
        sgd_logits = mlp_sgd(x_te)
        adam_logits = mlp_adam(x_te)
    p0_final_sgd = _eval_logits_acc(sgd_logits, y_te, p0_mask_te)
    p0_final_adam = _eval_logits_acc(adam_logits, y_te, p0_mask_te)

    dmcn_final = _eval_logits_acc(dmcn_logits, y_te)
    joint_final = _eval_logits_acc(joint_logits, y_te)

    maj_cls = int(torch.bincount(y_tr).argmax().item())
    maj_acc = 100.0 * float((y_te == maj_cls).float().mean().item())
    blind_logits = head_intact(torch.zeros_like(f_te))

    return {
        "seed": seed,
        "dmcn_cil_final_acc": dmcn_final,
        "dmcn_cil_p0_retention": p0_final_dmcn,
        "dmcn_cil_p0_bwt": p0_final_dmcn - p0_after_p0_dmcn,
        "joint_offline_fp64_upper_bound_acc": joint_final,
        "abs_gap_vs_joint_fp64": abs(dmcn_final - joint_final),
        "raw_linear_rls_acc": _eval_logits_acc(head_raw_rls(x_te_b), y_te),
        "ablate_precision_acc": _eval_logits_acc(head_ablate(f_te), y_te),
        "mlp_sgd_cil_acc": _eval_logits_acc(sgd_logits, y_te),
        "mlp_sgd_p0_retention": p0_final_sgd,
        "mlp_sgd_p0_bwt": p0_final_sgd - p0_after_p0_sgd,
        "mlp_adam_cil_acc": _eval_logits_acc(adam_logits, y_te),
        "mlp_adam_p0_retention": p0_final_adam,
        "mlp_adam_p0_bwt": p0_final_adam - p0_after_p0_adam,
        "untrained_acc": _eval_logits_acc(head_untr(f_te), y_te),
        "blind_acc": _eval_logits_acc(blind_logits, y_te),
        "shuffled_labels_acc": _eval_logits_acc(head_shuf(f_te), y_te),
        "majority_acc": maj_acc,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_gpu_throughput(
    device: torch.device,
    d_exp: int = 2048,
    ridge: float = 10.0,
    batch_size: int = 256,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    n_batches = 40
    xb = torch.randn((batch_size, 784), device=device, dtype=torch.float32)
    tb = torch.zeros((batch_size, 10), device=device, dtype=torch.float32)
    yb = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)
    tb.scatter_(1, yb.unsqueeze(1), 1.0)
    ones_b = torch.ones((batch_size, 1), device=device, dtype=torch.float32)

    # 1. DMCN Batched Woodbury Precision (784 -> d_exp -> 10)
    exp_l = DMCNTorchLayer(
        DMCNConfig(d_in=784, d_out=d_exp, n_branches=1, n_spine_scales=0, soma_activation="tanh",
                   ablation=AblationConfig(ablate_homeostasis=True)),
        device=device, dtype=torch.float32,
    )
    head_p = DMCNTorchLayer(
        DMCNConfig(d_in=d_exp + 1, d_out=10, n_branches=1, n_spine_scales=0, soma_activation="linear",
                   eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0, use_precision_plasticity=True,
                   precision_ridge=ridge, ablation=AblationConfig(ablate_homeostasis=True)),
        device=device, dtype=torch.float32,
    )
    for _ in range(5):
        fb = torch.cat([exp_l(xb), ones_b], dim=1)
        head_p.learn_local(tb - head_p(fb))
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        fb = torch.cat([exp_l(xb), ones_b], dim=1)
        head_p.learn_local(tb - head_p(fb))
    _sync(device)
    dt_prec = time.perf_counter() - t0
    sps_prec = (n_batches * batch_size) / dt_prec

    # 2. DMCN Batched General Local Plasticity (2-layer Kolen-Pollack, no P_prec)
    net_gen = DMCNTorchNetwork(
        [
            DMCNConfig(d_in=784, d_out=d_exp, n_branches=1, n_spine_scales=0, soma_activation="tanh"),
            DMCNConfig(d_in=d_exp, d_out=10, n_branches=1, n_spine_scales=0, soma_activation="linear"),
        ],
        device=device, dtype=torch.float32,
    )
    for _ in range(5):
        net_gen(xb)
        net_gen.learn_local(tb)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        net_gen(xb)
        net_gen.learn_local(tb)
    _sync(device)
    dt_gen = time.perf_counter() - t0
    sps_gen = (n_batches * batch_size) / dt_gen

    # 3. DMCN Single-Sample (batch=1) GPU baseline
    x1 = xb[0]
    t1 = tb[0]
    o1 = torch.ones(1, device=device, dtype=torch.float32)
    for _ in range(10):
        f1 = torch.cat([exp_l(x1), o1], dim=0)
        head_p.learn_local(t1 - head_p(f1))
    _sync(device)
    n_single = 200
    t0 = time.perf_counter()
    for _ in range(n_single):
        f1 = torch.cat([exp_l(x1), o1], dim=0)
        head_p.learn_local(t1 - head_p(f1))
    _sync(device)
    dt_b1 = time.perf_counter() - t0
    sps_b1 = n_single / dt_b1

    # 4. PyTorch Backprop MLP (SGD)
    mlp_s = TorchBackpropMLP(784, d_exp, 10).to(device)
    opt_s = torch.optim.SGD(mlp_s.parameters(), lr=0.1, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    for _ in range(5):
        opt_s.zero_grad(set_to_none=True)
        ce(mlp_s(xb), yb).backward()
        opt_s.step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        opt_s.zero_grad(set_to_none=True)
        ce(mlp_s(xb), yb).backward()
        opt_s.step()
    _sync(device)
    dt_sgd = time.perf_counter() - t0
    sps_sgd = (n_batches * batch_size) / dt_sgd

    # 5. PyTorch Backprop MLP (Adam)
    mlp_a = TorchBackpropMLP(784, d_exp, 10).to(device)
    opt_a = torch.optim.Adam(mlp_a.parameters(), lr=1e-3)
    for _ in range(5):
        opt_a.zero_grad(set_to_none=True)
        ce(mlp_a(xb), yb).backward()
        opt_a.step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        opt_a.zero_grad(set_to_none=True)
        ce(mlp_a(xb), yb).backward()
        opt_a.step()
    _sync(device)
    dt_adam = time.perf_counter() - t0
    sps_adam = (n_batches * batch_size) / dt_adam

    r_exp = exp_l.ops_report()
    r_head = head_p.ops_report()
    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "batch_size": batch_size,
        "dmcn_batched_precision_samples_per_sec": float(sps_prec),
        "dmcn_batched_precision_us_per_sample": float(1e6 / sps_prec),
        "dmcn_batched_general_samples_per_sec": float(sps_gen),
        "dmcn_batched_general_us_per_sample": float(1e6 / sps_gen),
        "dmcn_single_sample_b1_samples_per_sec": float(sps_b1),
        "dmcn_batched_vs_b1_speedup": float(sps_prec / sps_b1),
        "pytorch_mlp_sgd_samples_per_sec": float(sps_sgd),
        "pytorch_mlp_sgd_us_per_sample": float(1e6 / sps_sgd),
        "pytorch_mlp_adam_samples_per_sec": float(sps_adam),
        "pytorch_mlp_adam_us_per_sample": float(1e6 / sps_adam),
        "dmcn_precision_matrix_fp32_bytes": int((d_exp + 1) * (d_exp + 1) * 4),
        "dmcn_total_ram_fp32_bytes": int(r_exp.memory_bytes_fp32 + r_head.memory_bytes_fp32),
        "dmcn_total_params": int(r_exp.parameter_count + r_head.parameter_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / "results" / "dmcn_split_mnist_cuda_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999, device=device)
        elapsed_val = time.perf_counter() - t0
        prof = profile_gpu_throughput(device=device)
        log_path = out_dir / "tuning_log.json"
        if log_path.exists():
            raw = json.loads(log_path.read_text(encoding="utf-8"))
            entries = raw.get("tuning_entries", []) if isinstance(raw, dict) else []
        else:
            entries = []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed_val, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_split_mnist_cuda_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "d_exp": 2048, "ridge": 10.0, "exp_scale": 2.2, "batch_size": 256},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed_val, "val_seed_999": res, "gpu_profile": prof}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed = [evaluate_seed(s, device=device) for s in test_seeds]
    elapsed_test = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}
    prof = profile_gpu_throughput(device=device)

    payload = {
        "headline_metric": "dmcn_cil_final_acc",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed_test),
        "summary": summary,
        "gpu_profile": prof,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_split_mnist_cuda_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"d_exp": 2048, "ridge": 10.0, "exp_scale": 2.2, "batch_size": 256, "pairs": 5},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed_test, "summary": summary, "gpu_profile": prof}, indent=2))


if __name__ == "__main__":
    main()
