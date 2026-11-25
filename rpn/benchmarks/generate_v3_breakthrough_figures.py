"""Generate 300-DPI PNG + SVG figures for the 3 DMCN v3 Breakthrough Benchmarks strictly from results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig8_self_gating() -> None:
    s = json.loads((REPO_ROOT / "results" / "dmcn_self_gating_v1" / "results.json").read_text(encoding="utf-8"))["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), gridspec_kw={"wspace": 0.30})

    # Panel A: Lorenz Chaos NMSE (lower is better)
    items_a = [
        ("Self-Gating\nDMCN", "lorenz_self_gate_nmse", "#1b6ca8"),
        ("Uniform\nAblation", "lorenz_uniform_ablate_nmse", "#3a9ad9"),
        ("Raw\nOnly", "lorenz_raw_only_nmse", "#66a61e"),
        ("Homeo\nOnly", "lorenz_homeo_only_nmse", "#d95f02"),
        ("Spine\nOnly", "lorenz_spine_only_nmse", "#e6ab02"),
    ]
    x = np.arange(len(items_a))
    ma = [s[k]["mean"] for _, k, _ in items_a]
    ca = [s[k]["ci95"] for _, k, _ in items_a]
    b1 = axes[0].bar(x, ma, yerr=ca, color=[c for _, _, c in items_a], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([l for l, _, _ in items_a], fontsize=8.4)
    axes[0].set_ylabel("Unseen NMSE (lower is better)")
    axes[0].set_ylim(0, max(m + c for m, c in zip(ma, ca)) * 1.25)
    axes[0].set_title(f"Task A: Lorenz Chaos\n(Self-Gate Raw+Trans: {100*s['lorenz_gate_raw_plus_trans']['mean']:.1f}%)")
    for bar, m, c in zip(b1, ma, ca):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, m + c + 0.0006, f"{m:.4f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel B: 100x Gain-Shock NMSE (lower is better)
    items_b = [
        ("Self-Gating\nDMCN", "gainshock_self_gate_nmse", "#1b6ca8"),
        ("Uniform\nAblation", "gainshock_uniform_ablate_nmse", "#3a9ad9"),
        ("Spine\nOnly", "gainshock_spine_only_nmse", "#e6ab02"),
        ("Homeo\nOnly", "gainshock_homeo_only_nmse", "#66a61e"),
        ("Raw\nOnly", "gainshock_raw_only_nmse", "#d95f02"),
    ]
    mb = [s[k]["mean"] for _, k, _ in items_b]
    cb = [s[k]["ci95"] for _, k, _ in items_b]
    b2 = axes[1].bar(x, mb, yerr=cb, color=[c for _, _, c in items_b], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l for l, _, _ in items_b], fontsize=8.4)
    axes[1].set_ylabel("Unseen Gain [0.2, 20] NMSE (lower is better)")
    axes[1].set_ylim(0, max(m + c for m, c in zip(mb, cb)) * 1.22)
    axes[1].set_title(f"Task B: 100x Sensor Gain Shock\n(Self-Gate Homeo+Spine: {100*s['gainshock_gate_homeo_plus_spine']['mean']:.1f}%)")
    for bar, m, c in zip(b2, mb, cb):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, m + c + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    # Panel C: 4-Step Delayed Cue Accuracy (higher is better)
    items_c = [
        ("Self-Gating\nDMCN", "delayed_self_gate_acc", "#1b6ca8"),
        ("Spine\nOnly", "delayed_spine_only_acc", "#66a61e"),
        ("Raw\nOnly", "delayed_raw_only_acc", "#d95f02"),
        ("Homeo\nOnly", "delayed_homeo_only_acc", "#d95f02"),
        ("Shuffled\nNull", "delayed_shuffled_acc", "#888888"),
    ]
    mc = [s[k]["mean"] for _, k, _ in items_c]
    cc = [s[k]["ci95"] for _, k, _ in items_c]
    b3 = axes[2].bar(x, mc, yerr=cc, color=[c for _, _, c in items_c], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([l for l, _, _ in items_c], fontsize=8.4)
    axes[2].set_ylabel("Held-Out Accuracy (%)")
    axes[2].set_ylim(0, 118)
    axes[2].set_title(f"Task C: 4-Step Delayed Cue\n(Self-Gate Spine: {100*s['delayed_gate_spine']['mean']:.1f}%, Raw: {100*s['delayed_gate_raw']['mean']:.3f}%)")
    for bar, m, c in zip(b3, mc, cc):
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, m + c + 2.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    fig.suptitle("ONE Frozen Self-Gating DMCN Config Across 3 Conflicting Tasks (Zero Human Pillar Switching, 20 Seeds)", fontsize=12.0, fontweight="bold", y=1.03)
    _save(fig, "fig8_self_gating_unified")


def plot_fig9_compositional() -> None:
    s = json.loads((REPO_ROOT / "results" / "dmcn_compositional_v1" / "results.json").read_text(encoding="utf-8"))["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"wspace": 0.28})

    # Panel A: Task 1 Zero-Shot Attribute Composition Accuracy
    items1 = [
        ("Factorized\nBilinear DMCN", "t1_attr_dmcn_fac_acc", "#1b6ca8"),
        ("PyTorch MLP\n(Adam)", "t1_attr_mlp_adam_acc", "#7570b3"),
        ("PyTorch MLP\n(SGD)", "t1_attr_mlp_sgd_acc", "#9e9ac8"),
        ("Dense Entangled\nDMCN", "t1_attr_dense_dmcn_acc", "#d95f02"),
        ("Null:\nShuffled", "t1_attr_shuffled_acc", "#888888"),
    ]
    x1 = np.arange(len(items1))
    m1 = [s[k]["mean"] for _, k, _ in items1]
    c1 = [s[k]["ci95"] for _, k, _ in items1]
    b1 = ax1.bar(x1, m1, yerr=c1, color=[c for _, _, c in items1], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([l for l, _, _ in items1], fontsize=8.5)
    ax1.set_ylim(0, 116)
    ax1.set_ylabel("Zero-Shot Unseen-Pair Accuracy (%)")
    ax1.set_title("Task 1: Zero-Shot Color x Shape Composition\n(12 Seen / 4 NEVER-SEEN Pairs, Permuted Columns)")
    for bar, m, c in zip(b1, m1, c1):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, m + c + 2.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: Task 2 Zero-Shot Rule x Operand Composition (Accuracy + MSE annotation)
    items2 = [
        ("Factorized\nBilinear DMCN", "t2_rule_dmcn_fac_acc", "t2_rule_dmcn_fac_mse", "#1b6ca8"),
        ("PyTorch MLP\n(Adam)", "t2_rule_mlp_adam_acc", "t2_rule_mlp_adam_mse", "#7570b3"),
        ("PyTorch MLP\n(SGD)", "t2_rule_mlp_sgd_acc", "t2_rule_mlp_sgd_mse", "#9e9ac8"),
        ("Dense Entangled\nDMCN", "t2_rule_dense_dmcn_acc", "t2_rule_dense_dmcn_mse", "#d95f02"),
        ("Ablate\nBilinear", "t2_rule_ablate_bilinear_acc", "t2_rule_ablate_bilinear_mse", "#e6ab02"),
    ]
    x2 = np.arange(len(items2))
    m2 = [s[k]["mean"] for _, k, _, _ in items2]
    c2 = [s[k]["ci95"] for _, k, _, _ in items2]
    mses = [s[mk]["mean"] for _, _, mk, _ in items2]
    b2 = ax2.bar(x2, m2, yerr=c2, color=[c for _, _, _, c in items2], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([l for l, _, _, _ in items2], fontsize=8.5)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Zero-Shot Unseen-Pair Accuracy (%)")
    ax2.set_title("Task 2: Zero-Shot Rule x Operand Binding\n(36 Seen / 12 NEVER-SEEN Pairs, Zero Backprop)")
    for bar, m, c, mse_v in zip(b2, m2, c2, mses):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, m + c + 2.0, f"{m:.1f}%\n(MSE {mse_v:.4f})", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    _save(fig, "fig9_compositional_generalization")


def plot_fig10_pixel_unsup() -> None:
    s = json.loads((REPO_ROOT / "results" / "dmcn_pong_pixel_unsup_v1" / "results.json").read_text(encoding="utf-8"))["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"wspace": 0.28})

    items = [
        ("PyTorch BP\nA2C", "pixel_bp_a2c_hit_rate", "#7570b3"),
        ("DMCN + Oja\n(Unsup)", "pixel_unsup_oja_dmcn_hit_rate", "#1b6ca8"),
        ("Frozen L1\n(Ablate)", "pixel_ablate_unsup_hit_rate", "#d95f02"),
        ("Null:\nBlind", "pixel_blind_hit_rate", "#888888"),
        ("Null:\nShuffled", "pixel_shuffled_reward_hit_rate", "#888888"),
        ("Null:\nRandom", "pixel_random_hit_rate", "#888888"),
    ]
    x = np.arange(len(items))
    m = [s[k]["mean"] for _, k, _ in items]
    c = [s[k]["ci95"] for _, k, _ in items]
    b1 = ax1.bar(x, m, yerr=c, color=[cl for _, _, cl in items], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.58)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l for l, _, _ in items], fontsize=7.6, rotation=0, ha="center")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Held-Out Return Rate (%) [20 seeds]")
    ax1.set_title("Raw 16x16 Pixel Pong (256 Pixels, Zero Coordinates)\nUnsupervised Motion-Oja Layer 1 vs. Backprop A2C")
    for bar, mv, cv in zip(b1, m, c):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, mv + cv + 1.8, f"{mv:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")

    # Panel B: Layer-1 Receptive-Field Concentration
    rf_m = [s["rf_peak_concentration_unsup"]["mean"], s["rf_peak_concentration_frozen"]["mean"]]
    rf_c = [s["rf_peak_concentration_unsup"]["ci95"], s["rf_peak_concentration_frozen"]["ci95"]]
    b2 = ax2.bar([0, 1], rf_m, yerr=rf_c, color=["#1b6ca8", "#d95f02"], edgecolor="#222222", linewidth=0.6, capsize=3, width=0.45)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Unsup Motion-Oja\nLayer 1", "Frozen Random\nLayer 1"], fontsize=9.0)
    ax2.set_ylabel("Mean Peak-Pixel Weight Energy (max w^2)")
    ax2.set_ylim(0, max(v + e for v, e in zip(rf_m, rf_c)) * 1.30)
    ax2.set_title("Layer-1 Visual Specialization\n(Self-Organization onto Moving Pixels)")
    for bar, mv, cv in zip(b2, rf_m, rf_c):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, mv + cv + 0.004, f"{mv:.3f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    _save(fig, "fig10_pong_pixel_unsup_oja")


def main() -> None:
    _style()
    plot_fig8_self_gating()
    plot_fig9_compositional()
    plot_fig10_pixel_unsup()
    print("Generated fig8, fig9, fig10 in:", FIG_DIR)


if __name__ == "__main__":
    main()
