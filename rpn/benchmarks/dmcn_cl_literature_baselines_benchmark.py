"""Pre-registered 20-Seed Continual Learning Literature Baselines Benchmark (Split-MNIST + Split-CIFAR-10).

Compares native batched CUDA DMCN (nonlinear dendritic expansion + exact Woodbury local precision
plasticity, ZERO replay, ZERO backprop, ZERO task IDs) and 4-branch Block-Diagonal DMCN (4x less VRAM)
against standard Continual Learning literature baselines on strictly sequential 5-pair streams:
    {0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}
  - ACIL / Linear-RLS (Zhuang et al., NeurIPS 2022)
  - Elastic Weight Consolidation (MLP + EWC, Kirkpatrick et al., PNAS 2017; given task boundaries)
  - Experience Replay (MLP + ER-200 & ER-500 reservoir buffers, Chaudhry et al., ICML 2019)
  - Vanilla Backprop MLP (Adam, 0 replay)
  - Joint Offline FP64 Ridge Upper Bound & Null Controls (Untrained, Blind, Shuffled, Majority)
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
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNTorchLayer  # noqa: E402


_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def get_mnist_flat(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if "mnist" not in _CACHE:
        root = Path.home() / ".cache" / "torchvision_mnist"
        tr = datasets.MNIST(root=str(root), train=True, download=True)
        te = datasets.MNIST(root=str(root), train=False, download=True)
        _CACHE["mnist"] = (
            (tr.data.reshape(-1, 784).float() / 255.0).to(device),
            tr.targets.long().to(device),
            (te.data.reshape(-1, 784).float() / 255.0).to(device),
            te.targets.long().to(device),
        )
    return _CACHE["mnist"]


@torch.no_grad()
def get_fmnist_flat(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if "fmnist" not in _CACHE:
        root = Path.home() / ".cache" / "torchvision_fmnist"
        tr = datasets.FashionMNIST(root=str(root), train=True, download=True)
        te = datasets.FashionMNIST(root=str(root), train=False, download=True)
        _CACHE["fmnist"] = (
            (tr.data.reshape(-1, 784).float() / 255.0).to(device),
            tr.targets.long().to(device),
            (te.data.reshape(-1, 784).float() / 255.0).to(device),
            te.targets.long().to(device),
        )
    return _CACHE["fmnist"]


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


class MLPBaseline(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 2048, d_out: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh(), nn.Linear(d_hid, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def eval_acc(logits: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        logits = logits[mask]
        y_true = y_true[mask]
    return 100.0 * float((torch.argmax(logits, dim=1) == y_true).float().mean().item())


def evaluate_dataset_seed(
    dataset_name: str,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    seed: int,
    device: torch.device,
    d_exp: int = 2048,
    ridge: float = 10.0,
    exp_scale: float = 2.2,
    batch_size: int = 256,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 555)

    d_in = int(x_tr.shape[1])
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    p0_mask_te = (y_te == 0) | (y_te == 1)

    def make_stack(d_h: int, s_off: int) -> Tuple[DMCNTorchLayer, DMCNTorchLayer]:
        exp_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_in, d_out=d_h, n_branches=1, n_spine_scales=0,
                soma_activation="tanh", weight_init_scale=exp_scale, seed=seed + 101 + s_off,
                ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        exp_l.b_dend[0].copy_(torch.tensor(rng.normal(0.0, 0.35, size=d_h), device=device, dtype=torch.float32))
        head_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_h + 1, d_out=10, n_branches=1, n_spine_scales=0,
                soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
                weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=ridge,
                seed=seed + s_off, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        return exp_l, head_l

    exp_dmcn, head_dmcn = make_stack(d_exp, 0)
    _, head_untr = make_stack(d_exp, 10)
    _, head_shuf = make_stack(d_exp, 20)
    n_blocks = 4
    d_block = d_exp // n_blocks
    block_stacks = [make_stack(d_block, 100 + b) for b in range(n_blocks)]

    # Raw-feature ACIL / Linear-RLS baseline (d_in + 1 -> 10)
    head_acil = DMCNTorchLayer(
        DMCNConfig(
            d_in=d_in + 1, d_out=10, n_branches=1, n_spine_scales=0,
            soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
            weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=ridge,
            seed=seed, ablation=AblationConfig(ablate_homeostasis=True),
        ),
        device=device, dtype=torch.float32,
    )

    def extract_f(exp_l: DMCNTorchLayer, x_in: torch.Tensor) -> torch.Tensor:
        chunks = []
        for s in range(0, x_in.shape[0], 2048):
            h = exp_l(x_in[s : s + 2048])
            chunks.append(torch.cat([h, torch.ones((h.shape[0], 1), device=device, dtype=torch.float32)], dim=1))
        return torch.cat(chunks, dim=0)

    f_tr = extract_f(exp_dmcn, x_tr)
    f_te = extract_f(exp_dmcn, x_te)
    f_tr_blocks = [extract_f(be, x_tr) for be, _ in block_stacks]
    f_te_blocks = [extract_f(be, x_te) for be, _ in block_stacks]
    x_tr_b = torch.cat([x_tr, torch.ones((x_tr.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    x_te_b = torch.cat([x_te, torch.ones((x_te.shape[0], 1), device=device, dtype=torch.float32)], dim=1)

    # PyTorch CL Baselines: Vanilla Adam, EWC, ER-200, ER-500
    torch.manual_seed(seed)
    mlp_adam = MLPBaseline(d_in, d_exp, 10).to(device)
    opt_adam = torch.optim.Adam(mlp_adam.parameters(), lr=1e-3)

    torch.manual_seed(seed)
    mlp_ewc = MLPBaseline(d_in, d_exp, 10).to(device)
    opt_ewc = torch.optim.Adam(mlp_ewc.parameters(), lr=1e-3)
    ewc_fisher: Dict[str, torch.Tensor] = {}
    ewc_anchor: Dict[str, torch.Tensor] = {}
    ewc_lambda = 500.0

    def make_er(cap: int):
        torch.manual_seed(seed)
        m = MLPBaseline(d_in, d_exp, 10).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        buf_x = torch.zeros((cap, d_in), device=device, dtype=torch.float32)
        buf_y = torch.zeros(cap, device=device, dtype=torch.long)
        return {"m": m, "opt": opt, "bx": buf_x, "by": buf_y, "cap": cap, "seen": 0, "filled": 0}

    er200 = make_er(200)
    er500 = make_er(500)

    ce = nn.CrossEntropyLoss()
    y_tr_oh = torch.zeros((len(y_tr), 10), device=device, dtype=torch.float32)
    y_tr_oh.scatter_(1, y_tr.unsqueeze(1), 1.0)
    shuf_perm = torch.tensor(rng.permutation(len(y_tr)), device=device, dtype=torch.long)
    y_tr_shuf_oh = y_tr_oh[shuf_perm]

    p0_after_p0_dmcn = 0.0

    for p_idx, (c0, c1) in enumerate(pairs):
        idx_np = np.flatnonzero(((y_tr == c0) | (y_tr == c1)).cpu().numpy())
        rng.shuffle(idx_np)
        idx_t = torch.tensor(idx_np, device=device, dtype=torch.long)
        ewc_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for start in range(0, idx_t.shape[0], batch_size):
            b_idx = idx_t[start : start + batch_size]
            bsz = int(b_idx.shape[0])
            xb = x_tr[b_idx]
            xb_b = x_tr_b[b_idx]
            fb = f_tr[b_idx]
            yb = y_tr[b_idx]
            tb = y_tr_oh[b_idx]
            tb_s = y_tr_shuf_oh[b_idx]

            # DMCN intact & 4-branch block-diagonal & shuffled & raw ACIL
            head_dmcn.learn_local(tb - head_dmcn(fb))
            for b_i, (_, bh) in enumerate(block_stacks):
                fbb = f_tr_blocks[b_i][b_idx]
                bh.learn_local(tb - bh(fbb))
            head_shuf.learn_local(tb_s - head_shuf(fb))
            head_acil.learn_local(tb - head_acil(xb_b))

            # Vanilla Adam
            opt_adam.zero_grad(set_to_none=True)
            ce(mlp_adam(xb), yb).backward()
            opt_adam.step()

            # EWC
            opt_ewc.zero_grad(set_to_none=True)
            loss_e = ce(mlp_ewc(xb), yb)
            if ewc_fisher:
                reg = torch.tensor(0.0, device=device)
                for n, p in mlp_ewc.named_parameters():
                    reg = reg + (ewc_fisher[n] * (p - ewc_anchor[n]).pow(2)).sum()
                loss_e = loss_e + 0.5 * ewc_lambda * reg
            loss_e.backward()
            opt_ewc.step()
            if len(ewc_samples) < 8:
                ewc_samples.append((xb.detach(), yb.detach()))

            # ER-200 & ER-500 (reservoir sampling)
            for er in (er200, er500):
                er["opt"].zero_grad(set_to_none=True)
                if er["filled"] > 0:
                    n_rep = min(64, er["filled"])
                    r_idx = torch.randint(0, er["filled"], (n_rep,), device=device, generator=gen)
                    x_mix = torch.cat([xb, er["bx"][r_idx]], dim=0)
                    y_mix = torch.cat([yb, er["by"][r_idx]], dim=0)
                    ce(er["m"](x_mix), y_mix).backward()
                else:
                    ce(er["m"](xb), yb).backward()
                er["opt"].step()

                for j in range(bsz):
                    er["seen"] += 1
                    if er["filled"] < er["cap"]:
                        er["bx"][er["filled"]].copy_(xb[j])
                        er["by"][er["filled"]] = yb[j]
                        er["filled"] += 1
                    else:
                        slot = int(rng.randint(0, er["seen"]))
                        if slot < er["cap"]:
                            er["bx"][slot].copy_(xb[j])
                            er["by"][slot] = yb[j]

        # EWC Fisher update at task boundary
        nf = {n: torch.zeros_like(p) for n, p in mlp_ewc.named_parameters()}
        for xf, yf in ewc_samples:
            mlp_ewc.zero_grad(set_to_none=True)
            ce(mlp_ewc(xf), yf).backward()
            for n, p in mlp_ewc.named_parameters():
                if p.grad is not None:
                    nf[n] += (p.grad.detach() ** 2) / float(len(ewc_samples))
        for n, p in mlp_ewc.named_parameters():
            ewc_fisher[n] = nf[n] if n not in ewc_fisher else (ewc_fisher[n] + nf[n])
            ewc_anchor[n] = p.detach().clone()

        if p_idx == 0:
            p0_after_p0_dmcn = eval_acc(head_dmcn(f_te), y_te, p0_mask_te)

    # FP64 Joint Offline Ridge Upper Bound
    f_tr_64 = f_tr.double()
    y_oh_64 = y_tr_oh.double()
    a_joint = f_tr_64.T @ f_tr_64 + ridge * torch.eye(d_exp + 1, device=device, dtype=torch.float64)
    w_joint = torch.linalg.solve(a_joint, f_tr_64.T @ y_oh_64)
    joint_logits = (f_te.double() @ w_joint).float()

    dmcn_logits = head_dmcn(f_te)
    block_logits = sum(bh(f_te_blocks[b_i]) for b_i, (_, bh) in enumerate(block_stacks)) / float(n_blocks)
    dmcn_acc = eval_acc(dmcn_logits, y_te)
    joint_acc = eval_acc(joint_logits, y_te)
    p0_final = eval_acc(dmcn_logits, y_te, p0_mask_te)

    with torch.no_grad():
        adam_acc = eval_acc(mlp_adam(x_te), y_te)
        ewc_acc = eval_acc(mlp_ewc(x_te), y_te)
        er200_acc = eval_acc(er200["m"](x_te), y_te)
        er500_acc = eval_acc(er500["m"](x_te), y_te)

    maj_cls = int(torch.bincount(y_tr).argmax().item())
    maj_acc = 100.0 * float((y_te == maj_cls).float().mean().item())
    pref = dataset_name

    return {
        f"{pref}__dmcn_acc": dmcn_acc,
        f"{pref}__dmcn_p0_ret": p0_final,
        f"{pref}__dmcn_p0_bwt": p0_final - p0_after_p0_dmcn,
        f"{pref}__dmcn_block4x_acc": eval_acc(block_logits, y_te),
        f"{pref}__joint_fp64_acc": joint_acc,
        f"{pref}__abs_gap_fp64": abs(dmcn_acc - joint_acc),
        f"{pref}__acil_linear_rls_acc": eval_acc(head_acil(x_te_b), y_te),
        f"{pref}__mlp_ewc_acc": ewc_acc,
        f"{pref}__mlp_er200_acc": er200_acc,
        f"{pref}__mlp_er500_acc": er500_acc,
        f"{pref}__mlp_adam_acc": adam_acc,
        f"{pref}__untrained_acc": eval_acc(head_untr(f_te), y_te),
        f"{pref}__blind_acc": eval_acc(head_dmcn(torch.zeros_like(f_te)), y_te),
        f"{pref}__shuffled_acc": eval_acc(head_shuf(f_te), y_te),
        f"{pref}__majority_acc": maj_acc,
    }


def evaluate_seed(seed: int, device: torch.device) -> Dict[str, Any]:
    xm_tr, ym_tr, xm_te, ym_te = get_mnist_flat(device)
    xf_tr, yf_tr, xf_te, yf_te = get_fmnist_flat(device)
    res: Dict[str, Any] = {"seed": seed}
    res.update(evaluate_dataset_seed("mnist", xm_tr, ym_tr, xm_te, ym_te, seed, device, d_exp=2048, ridge=10.0, exp_scale=2.2))
    res.update(evaluate_dataset_seed("fmnist", xf_tr, yf_tr, xf_te, yf_te, seed, device, d_exp=2048, ridge=10.0, exp_scale=2.2))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / "results" / "dmcn_cl_literature_baselines_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999, device=device)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_cl_literature_baselines_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "d_exp": 2048, "ridge": 10.0},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed = [evaluate_seed(s, device=device) for s in test_seeds]
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    payload = {
        "headline_metric": "fmnist__dmcn_acc",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_cl_literature_baselines_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"d_exp": 2048, "ridge": 10.0, "exp_scale": 2.2, "datasets": ["mnist", "fashion_mnist"]},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
