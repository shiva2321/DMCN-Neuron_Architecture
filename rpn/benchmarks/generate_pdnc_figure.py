"""Generate 300-DPI PNG + SVG summary figure for dmcn_pdnc_general_v1 strictly from results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_PATH = REPO_ROOT / "results" / "dmcn_pdnc_general_v1" / "results.json"
OUT_DIR = REPO_ROOT / "docs" / "figures"


def main() -> None:
    data = json.loads(RES_PATH.read_text(encoding="utf-8"))
    s = data["summary"]
    curve = data.get("seed_2000_pixel_curve", [])

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=300)

    # Panel A: Suite A Raw 16x16 Pixel Pong RL
    ax = axes[0, 0]
    labels_a = [
        "Backprop\nA2C",
        "PDNC\n(Local)",
        "Ablate KP\n(FA Only)",
        "Ablate\nElig.",
        "Shuffle\nApical",
        "Ablate L1\n(Frozen)",
        "Blind /\nUntrained",
    ]
    keys_a = [
        "t2_bp_a2c_pixel_hit_rate",
        "t2_pdnc_pixel_hit_rate",
        "t2_ablate_kp_alignment_hit_rate",
        "t2_ablate_elig_trace_hit_rate",
        "t2_shuffle_apical_hit_rate",
        "t2_ablate_l1_learning_hit_rate",
        "t2_pixel_blind_hit_rate",
    ]
    colors_a = ["#4C78A8", "#009E73", "#56B4E9", "#E69F00", "#CC79A7", "#D55E00", "#8C8C8C"]
    means_a = [s[k]["mean"] for k in keys_a]
    cis_a = [s[k]["ci95"] for k in keys_a]
    x_a = np.arange(len(labels_a))
    bars_a = ax.bar(x_a, means_a, yerr=cis_a, capsize=3, color=colors_a, edgecolor="#222222", linewidth=0.6)
    ax.axhline(25.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6, label="Random baseline")
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=8)
    ax.set_ylabel("Held-Out Hit Rate (%)")
    ax.set_ylim(0, 96)
    ax.set_title("A. Raw 16x16 Pixel Pong RL (n=20, 0 Coords, Sparse +1/-1)")
    for b, m, ci in zip(bars_a, means_a, cis_a):
        ax.text(b.get_x() + b.get_width() / 2.0, m + ci + 1.8, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Panel B: Suite B 24D Noisy XOR Feature Discovery
    ax = axes[0, 1]
    labels_b = ["Backprop\nAdamW", "PDNC\nSettle Abl.", "PDNC\nFull", "Shuffled\nLabels", "Ablate L1\n(Frozen)"]
    keys_b = ["xor_bp_mlp_acc", "xor_ablate_settle_acc", "xor_pdnc_acc", "xor_shuffled_label_acc", "xor_ablate_l1_acc"]
    colors_b = ["#4C78A8", "#56B4E9", "#009E73", "#8C8C8C", "#D55E00"]
    means_b = [s[k]["mean"] for k in keys_b]
    cis_b = [s[k]["ci95"] for k in keys_b]
    x_b = np.arange(len(labels_b))
    bars_b = ax.bar(x_b, means_b, yerr=cis_b, capsize=3, color=colors_b, edgecolor="#222222", linewidth=0.6)
    ax.axhline(50.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(35, 102)
    ax.set_title("B. 24D Noisy XOR (2 Signal + 22 Noise Dims, n=20)")
    for b, m in zip(bars_b, means_b):
        ax.text(b.get_x() + b.get_width() / 2.0, m + 2.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Panel C: Suite C1 Rotated 3-Task Boundary-Free CL
    ax = axes[1, 0]
    cats_c = ["3-Task Mean", "Task-0 Retention (after T2)"]
    sub_vals = [s["cl3_subspace_mean_acc"]["mean"], s["cl3_subspace_t0_ret"]["mean"]]
    sub_cis = [s["cl3_subspace_mean_acc"]["ci95"], s["cl3_subspace_t0_ret"]["ci95"]]
    bp_vals = [s["cl3_bp_sgd_mean_acc"]["mean"], s["cl3_bp_sgd_t0_ret"]["mean"]]
    bp_cis = [s["cl3_bp_sgd_mean_acc"]["ci95"], s["cl3_bp_sgd_t0_ret"]["ci95"]]
    diag_vals = [s["cl3_diagonal_mean_acc"]["mean"], s["cl3_diagonal_t0_ret"]["mean"]]
    diag_cis = [s["cl3_diagonal_mean_acc"]["ci95"], s["cl3_diagonal_t0_ret"]["ci95"]]
    x_c = np.arange(len(cats_c))
    w = 0.24
    b1 = ax.bar(x_c - w, sub_vals, width=w, yerr=sub_cis, capsize=3, label="PDNC Subspace (0 Boundaries)", color="#009E73", edgecolor="#222222", linewidth=0.6)
    b2 = ax.bar(x_c, bp_vals, width=w, yerr=bp_cis, capsize=3, label="Backprop AdamW", color="#4C78A8", edgecolor="#222222", linewidth=0.6)
    b3 = ax.bar(x_c + w, diag_vals, width=w, yerr=diag_cis, capsize=3, label="PDNC Diagonal (No Null)", color="#D55E00", edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x_c)
    ax.set_xticklabels(cats_c, fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 98)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("C. Boundary-Free 3-Task Rotated Subspace CL (n=20)")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 2.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Panel D: Suite C2 Compact 256-Unit Split-FashionMNIST (0 Replay) + Learning Curve Inset Note
    ax = axes[1, 1]
    labels_d = ["Frozen + Woodbury\n(Disclosure Ceiling)", "PDNC Subspace\n(Plastic L1, Overall)", "PDNC Subspace\n(Pair-0 Ret.)", "PDNC Diagonal\n(No Null, Overall)", "PDNC Diagonal\n(Pair-0 Ret.)"]
    keys_d = [
        "fmnist_frozen_woodbury_acc",
        "fmnist_subspace_acc",
        "fmnist_subspace_p0_ret",
        "fmnist_diagonal_acc",
        "fmnist_diagonal_p0_ret",
    ]
    colors_d = ["#4C78A8", "#009E73", "#56B4E9", "#D55E00", "#A65628"]
    means_d = [s[k]["mean"] for k in keys_d]
    cis_d = [s[k]["ci95"] for k in keys_d]
    x_d = np.arange(len(labels_d))
    bars_d = ax.bar(x_d, means_d, yerr=cis_d, capsize=3, color=colors_d, edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x_d)
    ax.set_xticklabels(labels_d, fontsize=6.7)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 96)
    ax.set_title("D. Split-FashionMNIST d=256 (0 Replay, 0 Task IDs, n=20)")
    for b, m, ci in zip(bars_d, means_d, cis_d):
        ax.text(b.get_x() + b.get_width() / 2.0, m + ci + 2.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        "PDNC General Benchmark (dmcn_pdnc_general_v1): Mean +/- 95% CI Across 20 Test Seeds [2000..2019]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    if curve:
        fig.text(
            0.5,
            0.01,
            f"Seed 2000 Raw-Pixel Pong 200-serve trajectory: {' -> '.join(f'{v:.0f}%' for v in curve)}",
            ha="center",
            fontsize=8,
            color="#444444",
        )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "fig11_pdnc_general.png"
    svg_path = OUT_DIR / "fig11_pdnc_general.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    main()
