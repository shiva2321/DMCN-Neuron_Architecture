"""Pre-registered 20-Seed Reviewer Controls Benchmark (`dmcn_reviewer_controls_v1`).

Addresses the two primary reviewer critiques directly:
  1. Streaming Temporal + 50x Gain-Shock Split-MNIST (`dmcn_unified_temporal_cil_v1` stream)
     with fair Causal-RMS-normalized baselines:
       - Pooled ACIL / RanPAC + CausalRMS (has Pillar 1, lacks Pillar 2 resonant spines)
       - GRU + CausalRMS + Adam (BPTT)
       - GRU + CausalRMS + EWC (given task boundaries)
       - GRU + CausalRMS + ER-200 (200-seq reservoir replay + causal gain normalization)
  2. Standard RGB 100-Class Split-CIFAR-100 (10 sequential tasks x 10 classes, 1 pass, 0 replay)
     on frozen ImageNet-pretrained MobileNetV2 features (`1280 -> 4096 -> 100`):
       - DMCN (Batched CUDA Woodbury, 0 replay)
       - DMCN Block-Diag (4 x 1024 branches = 4x less precision VRAM, 0 replay)
       - Joint Offline FP64 Ridge Upper Bound
       - ACIL / Linear-RLS (1280 -> 100, 0 replay)
       - MLP + ER-500, MLP + ER-200, MLP + EWC, MLP + Adam
       - Full Null Controls (Untrained, Blind, Shuffled Labels, Majority Class)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import AblationConfig, DMCNConfig, DMCNTorchLayer  # noqa: E402
from benchmarks.dmcn_unified_temporal_cil_benchmark import (  # noqa: E402
    GRUClassifier,
    dmcn_temporal_encode,
    eval_acc,
    get_mnist_2d,
    make_stream_batch,
    summarize,
)


@torch.no_grad()
def causal_rms_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Apply DMCN Pillar-1 causal fast-attack slow-release RMS normalization frame-by-frame to a sequence."""
    bsz, t_steps, _ = seq.shape
    out = torch.empty_like(seq)
    e_rms = torch.zeros((bsz, 1), device=seq.device, dtype=seq.dtype)
    for t in range(t_steps):
        xt = seq[:, t, :]
        energy = (xt * xt).mean(dim=1, keepdim=True)
        e_rms = energy.clone() if t == 0 else torch.maximum(0.98 * e_rms, energy)
        out[:, t, :] = torch.clamp(xt / torch.sqrt(e_rms + 1e-6), -8.0, 8.0)
    return out


_C100_CACHE: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None


@torch.no_grad()
def get_cifar100_mobilenet_features(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract & cache frozen ImageNet MobileNetV2 (1280-D) features for CIFAR-100 train/test."""
    global _C100_CACHE
    if _C100_CACHE is not None:
        return _C100_CACHE

    cache_file = Path.home() / ".cache" / "cifar100_mobilenetv2_1280.pt"
    if cache_file.exists():
        blob = torch.load(cache_file, map_location="cpu", weights_only=True)
        _C100_CACHE = (
            blob["x_tr"].to(device),
            blob["y_tr"].to(device),
            blob["x_te"].to(device),
            blob["y_te"].to(device),
        )
        return _C100_CACHE

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v2(weights=weights).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def _extract_split(split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = load_dataset("uoft-cs/cifar100", split=split_name)
        imgs_np = np.stack([np.asarray(ex["img"], dtype=np.uint8) for ex in ds], axis=0)  # (N, 32, 32, 3)
        labels_np = np.asarray(ds["fine_label"], dtype=np.int64)
        x_raw = torch.from_numpy(imgs_np).permute(0, 3, 1, 2)  # CPU uint8
        feats_list: List[torch.Tensor] = []
        bs = 256
        for start in range(0, x_raw.shape[0], bs):
            xb = x_raw[start : start + bs].to(device=device, dtype=torch.float32) / 255.0
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
            feat = backbone.features(xb).mean(dim=(2, 3))  # (B, 1280)
            feats_list.append(feat.cpu())
        return torch.cat(feats_list, dim=0), torch.from_numpy(labels_np)

    x_tr_c, y_tr_c = _extract_split("train")
    x_te_c, y_te_c = _extract_split("test")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x_tr": x_tr_c, "y_tr": y_tr_c, "x_te": x_te_c, "y_te": y_te_c}, cache_file)
    _C100_CACHE = (x_tr_c.to(device), y_tr_c.to(device), x_te_c.to(device), y_te_c.to(device))
    return _C100_CACHE


def eval_temporal_normalized_controls(
    seed: int,
    device: torch.device,
    d_exp: int = 2048,
    ridge: float = 5.0,
    exp_scale: float = 2.0,
    batch_size: int = 256,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 777)

    x_tr_2d, y_tr, x_te_2d, y_te = get_mnist_2d(device)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    pair_gains = [1.0, 6.0, 0.2, 10.0, 0.4]
    p0_te = (y_te == 0) | (y_te == 1)

    log_g_te = rng.uniform(np.log(0.2), np.log(10.0), size=len(y_te))
    gains_te = torch.tensor(np.exp(log_g_te), device=device, dtype=torch.float32)
    seq_te = make_stream_batch(x_te_2d, gains_te, gen)

    def make_stack(d_in_psi: int, s_off: int) -> Tuple[DMCNTorchLayer, DMCNTorchLayer]:
        exp_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_in_psi, d_out=d_exp, n_branches=1, n_spine_scales=0,
                soma_activation="tanh", weight_init_scale=exp_scale, seed=seed + 101 + s_off,
                ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        exp_l.b_dend[0].copy_(torch.tensor(rng.normal(0.0, 0.35, size=d_exp), device=device, dtype=torch.float32))
        head_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_exp + 1, d_out=10, n_branches=1, n_spine_scales=0,
                soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
                weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=ridge,
                seed=seed + s_off, ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        return exp_l, head_l

    exp_uni, head_uni = make_stack(112 * 11, 0)
    exp_pool_norm, head_pool_norm = make_stack(112, 75)

    def expand_bias(exp_l: DMCNTorchLayer, psi: torch.Tensor) -> torch.Tensor:
        h = exp_l(psi)
        return torch.cat([h, torch.ones((h.shape[0], 1), device=device, dtype=torch.float32)], dim=1)

    torch.manual_seed(seed)
    gru_n_adam = GRUClassifier(112, 256, 10).to(device)
    opt_adam = torch.optim.Adam(gru_n_adam.parameters(), lr=1e-3)

    torch.manual_seed(seed)
    gru_n_ewc = GRUClassifier(112, 256, 10).to(device)
    opt_ewc = torch.optim.Adam(gru_n_ewc.parameters(), lr=1e-3)
    ewc_fisher: Dict[str, torch.Tensor] = {}
    ewc_anchor: Dict[str, torch.Tensor] = {}

    torch.manual_seed(seed)
    gru_n_er = GRUClassifier(112, 256, 10).to(device)
    opt_er = torch.optim.Adam(gru_n_er.parameters(), lr=1e-3)
    er_cap = 200
    er_seqs = torch.zeros((er_cap, 11, 112), device=device, dtype=torch.float32)
    er_labels = torch.zeros(er_cap, device=device, dtype=torch.long)
    er_seen = 0
    er_filled = 0

    ce = nn.CrossEntropyLoss()
    y_tr_oh = torch.zeros((len(y_tr), 10), device=device, dtype=torch.float32)
    y_tr_oh.scatter_(1, y_tr.unsqueeze(1), 1.0)

    for p_idx, (c0, c1) in enumerate(pairs):
        idx_np = np.flatnonzero(((y_tr == c0) | (y_tr == c1)).cpu().numpy())
        rng.shuffle(idx_np)
        idx_t = torch.tensor(idx_np, device=device, dtype=torch.long)
        gain_val = pair_gains[p_idx]
        ewc_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for start in range(0, idx_t.shape[0], batch_size):
            b_idx = idx_t[start : start + batch_size]
            bsz = int(b_idx.shape[0])
            seq_b = make_stream_batch(x_tr_2d[b_idx], torch.full((bsz,), gain_val, device=device, dtype=torch.float32), gen)
            seq_norm = causal_rms_sequence(seq_b)
            yb = y_tr[b_idx]
            tb = y_tr_oh[b_idx]

            # 1. Unified DMCN (Homeostasis + Resonant Spines + Woodbury)
            psi_uni = dmcn_temporal_encode(seq_b, use_homeostasis=True, use_spine_traces=True)
            f_uni = expand_bias(exp_uni, psi_uni)
            head_uni.learn_local(tb - head_uni(f_uni))

            # 2. Temporal-Pooled ACIL / RanPAC + CausalRMS (has Pillar 1, lacks Pillar 2 resonant spines)
            f_pn = expand_bias(exp_pool_norm, seq_norm.mean(dim=1))
            head_pool_norm.learn_local(tb - head_pool_norm(f_pn))

            # 3. GRU + CausalRMS + Adam
            opt_adam.zero_grad(set_to_none=True)
            ce(gru_n_adam(seq_norm), yb).backward()
            opt_adam.step()

            # 4. GRU + CausalRMS + EWC
            opt_ewc.zero_grad(set_to_none=True)
            l_ewc = ce(gru_n_ewc(seq_norm), yb)
            if ewc_fisher:
                reg = torch.tensor(0.0, device=device)
                for name, param in gru_n_ewc.named_parameters():
                    reg = reg + (ewc_fisher[name] * (param - ewc_anchor[name]).pow(2)).sum()
                l_ewc = l_ewc + 200.0 * reg
            l_ewc.backward()
            opt_ewc.step()
            if len(ewc_batches) < 8:
                ewc_batches.append((seq_norm.detach(), yb.detach()))

            # 5. GRU + CausalRMS + ER-200
            opt_er.zero_grad(set_to_none=True)
            if er_filled > 0:
                n_rep = min(64, er_filled)
                r_idx = torch.randint(0, er_filled, (n_rep,), device=device, generator=gen)
                seq_mix = torch.cat([seq_norm, er_seqs[r_idx]], dim=0)
                y_mix = torch.cat([yb, er_labels[r_idx]], dim=0)
                ce(gru_n_er(seq_mix), y_mix).backward()
            else:
                ce(gru_n_er(seq_norm), yb).backward()
            opt_er.step()

            for j in range(bsz):
                er_seen += 1
                if er_filled < er_cap:
                    er_seqs[er_filled].copy_(seq_norm[j])
                    er_labels[er_filled] = yb[j]
                    er_filled += 1
                else:
                    slot = int(rng.randint(0, er_seen))
                    if slot < er_cap:
                        er_seqs[slot].copy_(seq_norm[j])
                        er_labels[slot] = yb[j]

        new_fisher = {n: torch.zeros_like(p) for n, p in gru_n_ewc.named_parameters()}
        for sq_f, y_f in ewc_batches:
            gru_n_ewc.zero_grad(set_to_none=True)
            ce(gru_n_ewc(sq_f), y_f).backward()
            for n, p in gru_n_ewc.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += (p.grad.detach() ** 2) / float(len(ewc_batches))
        for n, p in gru_n_ewc.named_parameters():
            ewc_fisher[n] = new_fisher[n] if n not in ewc_fisher else (ewc_fisher[n] + new_fisher[n])
            ewc_anchor[n] = p.detach().clone()

    with torch.no_grad():
        psi_te = dmcn_temporal_encode(seq_te, use_homeostasis=True, use_spine_traces=True)
        seq_te_norm = causal_rms_sequence(seq_te)
        log_uni = head_uni(expand_bias(exp_uni, psi_te))
        log_pn = head_pool_norm(expand_bias(exp_pool_norm, seq_te_norm.mean(dim=1)))
        log_ga = gru_n_adam(seq_te_norm)
        log_ge = gru_n_ewc(seq_te_norm)
        log_ger = gru_n_er(seq_te_norm)

    return {
        "temp__dmcn_unified_acc": eval_acc(log_uni, y_te),
        "temp__dmcn_unified_p0_ret": eval_acc(log_uni, y_te, p0_te),
        "temp__pooled_acil_causal_rms_acc": eval_acc(log_pn, y_te),
        "temp__gru_causal_rms_adam_acc": eval_acc(log_ga, y_te),
        "temp__gru_causal_rms_ewc_acc": eval_acc(log_ge, y_te),
        "temp__gru_causal_rms_er200_acc": eval_acc(log_ger, y_te),
        "temp__gru_causal_rms_er200_p0_ret": eval_acc(log_ger, y_te, p0_te),
        "temp__blind_acc": eval_acc(head_uni(expand_bias(exp_uni, torch.zeros_like(psi_te))), y_te),
        "temp__majority_acc": 100.0 * float((y_te == torch.mode(y_tr).values).float().mean().item()),
    }


def eval_cifar100_cil(
    seed: int,
    device: torch.device,
    d_exp: int = 4096,
    ridge: float = 20.0,
    exp_scale: float = 1.0,
    batch_size: int = 256,
    mlp_epochs: int = 10,
    mlp_lr: float = 3e-4,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 888)

    x_tr, y_tr, x_te, y_te = get_cifar100_mobilenet_features(device)
    # Normalize frozen MobileNetV2 representations (zero label peeking: scale by sqrt(d))
    x_tr_n = F.normalize(x_tr, dim=1) * np.sqrt(1280.0)
    x_te_n = F.normalize(x_te, dim=1) * np.sqrt(1280.0)
    d_in = int(x_tr_n.shape[1])
    n_classes = 100
    tasks = [(10 * t, 10 * (t + 1)) for t in range(10)]
    t0_te = y_te < 10

    def make_stack(d_h: int, s_off: int) -> Tuple[DMCNTorchLayer, DMCNTorchLayer]:
        exp_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_in, d_out=d_h, n_branches=1, n_spine_scales=0,
                soma_activation="tanh", weight_init_scale=exp_scale, seed=seed + 201 + s_off,
                ablation=AblationConfig(ablate_homeostasis=True),
            ),
            device=device, dtype=torch.float32,
        )
        exp_l.b_dend[0].copy_(torch.tensor(rng.normal(0.0, 0.35, size=d_h), device=device, dtype=torch.float32))
        head_l = DMCNTorchLayer(
            DMCNConfig(
                d_in=d_in + d_h + 1, d_out=n_classes, n_branches=1, n_spine_scales=0,
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

    # Linear-RLS / ACIL without nonlinear dendritic expansion (val-tuned ridge=1.0)
    head_lin = DMCNTorchLayer(
        DMCNConfig(
            d_in=d_in + 1, d_out=n_classes, n_branches=1, n_spine_scales=0,
            soma_activation="linear", eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0,
            weight_init_scale=0.0, use_precision_plasticity=True, precision_ridge=1.0,
            seed=seed + 30, ablation=AblationConfig(ablate_homeostasis=True),
        ),
        device=device, dtype=torch.float32,
    )

    def make_mlp() -> nn.Module:
        return nn.Sequential(nn.Linear(d_in, d_exp), nn.Tanh(), nn.Linear(d_exp, n_classes)).to(device)

    torch.manual_seed(seed)
    mlp_adam = make_mlp()
    opt_adam = torch.optim.Adam(mlp_adam.parameters(), lr=mlp_lr)

    torch.manual_seed(seed)
    mlp_ewc = make_mlp()
    opt_ewc = torch.optim.Adam(mlp_ewc.parameters(), lr=mlp_lr)
    ewc_fisher: Dict[str, torch.Tensor] = {}
    ewc_anchor: Dict[str, torch.Tensor] = {}

    def _run_er(cap: int) -> Tuple[nn.Module, Any, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        m = make_mlp()
        opt = torch.optim.Adam(m.parameters(), lr=mlp_lr)
        bx = torch.zeros((cap, d_in), device=device, dtype=torch.float32)
        by = torch.zeros(cap, device=device, dtype=torch.long)
        return m, opt, bx, by

    mlp_er200, opt_er200, er200_x, er200_y = _run_er(200)
    mlp_er500, opt_er500, er500_x, er500_y = _run_er(500)
    er_seen = 0
    er200_f = 0
    er500_f = 0
    ce = nn.CrossEntropyLoss()

    y_tr_oh = torch.zeros((len(y_tr), n_classes), device=device, dtype=torch.float32)
    y_tr_oh.scatter_(1, y_tr.unsqueeze(1), 1.0)
    shuf_perm = torch.tensor(rng.permutation(len(y_tr)), device=device, dtype=torch.long)
    y_tr_shuf_oh = y_tr_oh[shuf_perm]

    def expand_bias(exp_l: DMCNTorchLayer, x: torch.Tensor) -> torch.Tensor:
        h = exp_l(x)
        return torch.cat([x, h, torch.ones((h.shape[0], 1), device=device, dtype=torch.float32)], dim=1)

    for c_lo, c_hi in tasks:
        idx_np = np.flatnonzero(((y_tr >= c_lo) & (y_tr < c_hi)).cpu().numpy())
        ewc_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Single-pass closed-form RLS updates for DMCN / Block4x / Linear-RLS
        rng.shuffle(idx_np)
        idx_once = torch.tensor(idx_np, device=device, dtype=torch.long)
        for start in range(0, idx_once.shape[0], batch_size):
            b_idx = idx_once[start : start + batch_size]
            bsz = int(b_idx.shape[0])
            xb = x_tr_n[b_idx]
            tb = y_tr_oh[b_idx]
            tb_s = y_tr_shuf_oh[b_idx]

            fb = expand_bias(exp_dmcn, xb)
            head_dmcn.learn_local(tb - head_dmcn(fb))
            head_shuf.learn_local(tb_s - head_shuf(fb))
            for b_exp, b_head in block_stacks:
                fbb = expand_bias(b_exp, xb)
                b_head.learn_local(tb - b_head(fbb))
            xb_lin = torch.cat([xb, torch.ones((bsz, 1), device=device, dtype=torch.float32)], dim=1)
            head_lin.learn_local(tb - head_lin(xb_lin))

        # Fair multi-epoch training for gradient baselines (Adam, EWC, ER-200, ER-500)
        for ep in range(mlp_epochs):
            rng.shuffle(idx_np)
            idx_t = torch.tensor(idx_np, device=device, dtype=torch.long)
            for start in range(0, idx_t.shape[0], batch_size):
                b_idx = idx_t[start : start + batch_size]
                bsz = int(b_idx.shape[0])
                xb = x_tr_n[b_idx]
                yb = y_tr[b_idx]

                opt_adam.zero_grad(set_to_none=True)
                ce(mlp_adam(xb), yb).backward()
                opt_adam.step()

                opt_ewc.zero_grad(set_to_none=True)
                l_ewc = ce(mlp_ewc(xb), yb)
                if ewc_fisher:
                    reg = torch.tensor(0.0, device=device)
                    for n, p in mlp_ewc.named_parameters():
                        reg = reg + (ewc_fisher[n] * (p - ewc_anchor[n]).pow(2)).sum()
                    l_ewc = l_ewc + 250.0 * reg
                l_ewc.backward()
                opt_ewc.step()
                if ep == mlp_epochs - 1 and len(ewc_batches) < 6:
                    ewc_batches.append((xb.detach(), yb.detach()))

                for m_er, o_er, bx, by, fill_n in (
                    (mlp_er200, opt_er200, er200_x, er200_y, er200_f),
                    (mlp_er500, opt_er500, er500_x, er500_y, er500_f),
                ):
                    o_er.zero_grad(set_to_none=True)
                    if fill_n > 0:
                        ri = torch.randint(0, fill_n, (min(256, fill_n),), device=device, generator=gen)
                        ce(m_er(torch.cat([xb, bx[ri]], dim=0)), torch.cat([yb, by[ri]], dim=0)).backward()
                    else:
                        ce(m_er(xb), yb).backward()
                    o_er.step()

                if ep == 0:
                    for j in range(bsz):
                        er_seen += 1
                        if er200_f < 200:
                            er200_x[er200_f].copy_(xb[j])
                            er200_y[er200_f] = yb[j]
                            er200_f += 1
                        else:
                            s2 = int(rng.randint(0, er_seen))
                            if s2 < 200:
                                er200_x[s2].copy_(xb[j])
                                er200_y[s2] = yb[j]
                        if er500_f < 500:
                            er500_x[er500_f].copy_(xb[j])
                            er500_y[er500_f] = yb[j]
                            er500_f += 1
                        else:
                            s5 = int(rng.randint(0, er_seen))
                            if s5 < 500:
                                er500_x[s5].copy_(xb[j])
                                er500_y[s5] = yb[j]

        new_fisher = {n: torch.zeros_like(p) for n, p in mlp_ewc.named_parameters()}
        for xbf, ybf in ewc_batches:
            mlp_ewc.zero_grad(set_to_none=True)
            ce(mlp_ewc(xbf), ybf).backward()
            for n, p in mlp_ewc.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += (p.grad.detach() ** 2) / float(len(ewc_batches))
        for n, p in mlp_ewc.named_parameters():
            ewc_fisher[n] = new_fisher[n] if n not in ewc_fisher else (ewc_fisher[n] + new_fisher[n])
            ewc_anchor[n] = p.detach().clone()

    with torch.no_grad():
        # Accumulate FP64 normal equations in 2048-row chunks to avoid multi-GB VRAM allocation
        d_feat = d_in + d_exp + 1
        xtx = float(ridge) * torch.eye(d_feat, device=device, dtype=torch.float64)
        xty = torch.zeros((d_feat, n_classes), device=device, dtype=torch.float64)
        for s in range(0, x_tr_n.shape[0], 2048):
            fc = expand_bias(exp_dmcn, x_tr_n[s : s + 2048]).to(dtype=torch.float64)
            yc = y_tr_oh[s : s + 2048].to(dtype=torch.float64)
            xtx.add_(fc.T @ fc)
            xty.add_(fc.T @ yc)
        w_joint = torch.linalg.solve(xtx, xty)
        log_joint = (expand_bias(exp_dmcn, x_te_n).to(dtype=torch.float64) @ w_joint).float()

        f_te = expand_bias(exp_dmcn, x_te_n)
        log_dmcn = head_dmcn(f_te)
        log_blk = torch.zeros_like(log_dmcn)
        for b_exp, b_head in block_stacks:
            log_blk = log_blk + b_head(expand_bias(b_exp, x_te_n))
        log_blk = log_blk / float(n_blocks)
        log_lin = head_lin(torch.cat([x_te_n, torch.ones((x_te_n.shape[0], 1), device=device, dtype=torch.float32)], dim=1))

        dmcn_acc = eval_acc(log_dmcn, y_te)
        joint_acc = eval_acc(log_joint, y_te)

        out = {
            "c100__dmcn_acc": dmcn_acc,
            "c100__dmcn_t0_ret": eval_acc(log_dmcn, y_te, t0_te),
            "c100__dmcn_block4x_acc": eval_acc(log_blk, y_te),
            "c100__joint_fp64_acc": joint_acc,
            "c100__abs_gap_fp64": abs(dmcn_acc - joint_acc),
            "c100__acil_linear_rls_acc": eval_acc(log_lin, y_te),
            "c100__mlp_er500_acc": eval_acc(mlp_er500(x_te_n), y_te),
            "c100__mlp_er200_acc": eval_acc(mlp_er200(x_te_n), y_te),
            "c100__mlp_ewc_acc": eval_acc(mlp_ewc(x_te_n), y_te),
            "c100__mlp_adam_acc": eval_acc(mlp_adam(x_te_n), y_te),
            "c100__untrained_acc": eval_acc(head_untr(f_te), y_te),
            "c100__blind_acc": eval_acc(head_dmcn(expand_bias(exp_dmcn, torch.zeros_like(x_te_n))), y_te),
            "c100__shuffled_acc": eval_acc(head_shuf(f_te), y_te),
            "c100__majority_acc": 100.0 * float((y_te == torch.mode(y_tr).values).float().mean().item()),
        }
    del exp_dmcn, head_dmcn, head_untr, head_shuf, block_stacks, head_lin, mlp_adam, mlp_ewc, mlp_er200, mlp_er500, ewc_fisher, ewc_anchor, xtx, xty, w_joint
    return out


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def evaluate_seed(seed: int, device: torch.device) -> Dict[str, Any]:
    row: Dict[str, Any] = {"seed": seed}
    row.update(eval_temporal_normalized_controls(seed, device))
    _cleanup_cuda(device)
    row.update(eval_cifar100_cil(seed, device))
    _cleanup_cuda(device)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / "results" / "dmcn_reviewer_controls_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999, device)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_reviewer_controls_v1_val",
            seeds=list(range(980, 1000)),
            config={"purpose": "validation_seed_999", "device": str(device)},
            repo_root=REPO_ROOT,
        )
        results_meta.write_results(log_path, {"tuning_entries": entries}, meta_val)
        print(json.dumps({"elapsed_val_sec": elapsed, "val_seed_999": res}, indent=2))
        return

    test_seeds = list(range(2000, 2020))
    t0 = time.perf_counter()
    per_seed: List[Dict[str, Any]] = []
    for idx, s in enumerate(test_seeds, 1):
        ts = time.perf_counter()
        row = evaluate_seed(s, device)
        per_seed.append(row)
        dt_s = time.perf_counter() - ts
        elapsed_so_far = time.perf_counter() - t0
        eta = (elapsed_so_far / idx) * (len(test_seeds) - idx)
        print(
            f"[dmcn_reviewer_controls] seed {s} ({idx}/{len(test_seeds)}) "
            f"{dt_s:.1f}s | eta={eta:.0f}s | temp={row['temp__dmcn_unified_acc']:.2f}% "
            f"c100={row['c100__dmcn_acc']:.2f}%",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    keys = [k for k in per_seed[0].keys() if k != "seed"]
    summary = {k: summarize([float(r[k]) for r in per_seed]) for k in keys}

    falsifiers = {
        "F1_temp_unified_ge_88_and_beats_causal_rms_baselines_by_15pp": bool(
            summary["temp__dmcn_unified_acc"]["mean"] >= 88.0
            and (summary["temp__dmcn_unified_acc"]["mean"] - summary["temp__pooled_acil_causal_rms_acc"]["mean"]) >= 15.0
            and (summary["temp__dmcn_unified_acc"]["mean"] - summary["temp__gru_causal_rms_er200_acc"]["mean"]) >= 15.0
        ),
        "F2_c100_ge_60_and_matches_joint_fp64_within_0_20pp": bool(
            summary["c100__dmcn_acc"]["mean"] >= 60.0 and summary["c100__abs_gap_fp64"]["mean"] <= 0.20
        ),
        "F3_c100_beats_er500_and_linear_rls_by_2pp": bool(
            (summary["c100__dmcn_acc"]["mean"] - summary["c100__mlp_er500_acc"]["mean"]) >= 2.0
            and (summary["c100__dmcn_acc"]["mean"] - summary["c100__acil_linear_rls_acc"]["mean"]) >= 2.0
        ),
        "F4_nulls_below_chance_floor": bool(
            summary["temp__blind_acc"]["mean"] <= 15.0
            and summary["temp__majority_acc"]["mean"] <= 15.0
            and summary["c100__untrained_acc"]["mean"] <= 5.0
            and summary["c100__blind_acc"]["mean"] <= 5.0
            and summary["c100__shuffled_acc"]["mean"] <= 5.0
            and summary["c100__majority_acc"]["mean"] <= 5.0
        ),
    }

    payload = {
        "headline_metric": "c100__dmcn_acc",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "device": str(device),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "falsifiers": falsifiers,
        "all_passed": bool(all(falsifiers.values())),
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_reviewer_controls_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={
            "d_exp_temp": 2048,
            "d_exp_c100": 4096,
            "include_x_c100": True,
            "exp_scale_c100": 1.0,
            "ridge_c100": 20.0,
            "ridge_linear_rls": 1.0,
            "mlp_epochs_c100": 10,
            "mlp_lr_c100": 3e-4,
            "backbone": "MobileNetV2_IMAGENET1K_V1",
        },
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "falsifiers": falsifiers, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
