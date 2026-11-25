"""Generate 300-DPI PNG + SVG architecture diagram (fig0) and reviewer controls chart (fig12)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x: float, y: float, w: float, h: float, title: str, sub: str, fc: str, ec: str = "#1f2937", lw: float = 1.2, title_fs: float = 8.6, sub_fs: float = 7.3) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(x + w / 2.0, y + h * 0.72, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#111827")
        ax.text(x + w / 2.0, y + h * 0.34, sub, ha="center", va="center", fontsize=sub_fs, color="#374151", linespacing=1.28)
    else:
        ax.text(x + w / 2.0, y + h / 2.0, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#111827")


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, label: str = "", color: str = "#374151", ls: str = "-", lw: float = 1.4, rad: float = 0.0, lx_off: float = 0.0, ly_off: float = 0.024) -> None:
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5,head_width=3",
        linestyle=ls, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2.0 + lx_off, (y1 + y2) / 2.0 + ly_off, label, ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")


def plot_fig0_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.0), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Section A container: Single-Layer DMCN 4-Pillar + Self-Gating / Bilinear Readout
    sec_a = FancyBboxPatch((0.010, 0.49), 0.980, 0.49, boxstyle="round,pad=0.0,rounding_size=0.015", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(sec_a)
    ax.text(0.022, 0.953, "A. Dendritic Multiplexed Cascade Neuron (DMCN): Single-Layer Local Precision & Multi-Timescale Circuit", fontsize=10.2, fontweight="bold", color="#0f172a", va="center")

    # Standard Point Neuron inset (top-left comparison)
    std_bg = FancyBboxPatch((0.022, 0.775), 0.285, 0.145, boxstyle="round,pad=0.0,rounding_size=0.012", facecolor="#fff1f2", edgecolor="#fda4af", linewidth=0.9)
    ax.add_patch(std_bg)
    ax.text(0.1645, 0.896, "Standard Point Neuron (Baseline)", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#9f1239")
    ax.text(0.1645, 0.842, r"$y = \phi(W x + b)$  (no scale invariance,", ha="center", va="center", fontsize=7.6, color="#881337")
    ax.text(0.1645, 0.802, "no memory state, overwrites old tasks)", ha="center", va="center", fontsize=7.4, color="#881337")

    # DMCN pipeline (left to right with clear arrow gaps)
    _box(ax, 0.020, 0.525, 0.120, 0.195, "Input Stream", r"$x_t \in \mathbb{R}^d$" + "\n(unscaled / delayed)", "#e2e8f0")
    _box(ax, 0.175, 0.525, 0.155, 0.195, "Pillar 1: Homeostasis", r"$x^{\mathrm{norm}}_t = \frac{x_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}$" + "\n(strictly causal EMA)", "#dbeafe", "#1d4ed8")
    _box(ax, 0.375, 0.515, 0.188, 0.215, "Pillar 2: Resonant Spines", r"$z_{t,k} = \alpha_k z_{t-1,k} + \beta_k x^{\mathrm{norm}}_t$" + "\n" + r"$\alpha_k = e^{(-1/\tau_k + i\omega_k)\Delta t},\ \mathrm{Re/Im}$" + "\n+ transient " + r"$\Delta x_t$", "#dcfce7", "#15803d")
    _box(ax, 0.602, 0.520, 0.176, 0.205, "Pillar 3: Dendritic Basis", r"branches $h_b = f(W_b \psi_t + b_b)$" + "\n" + r"+ Self-Gate $g_k(x_t)$ (chaos/shock)" + "\n" + r"OR Bilinear $(U x_A) \odot (V x_B)$", "#fef3c7", "#b45309")
    _box(ax, 0.818, 0.520, 0.158, 0.205, "Pillar 4: Woodbury RLS", r"$P \leftarrow P - \frac{P \psi \psi^\top P}{I + \psi^\top P \psi}$" + "\n" + r"$\Delta W = \delta\, (P\psi)^\top$" + "\n(0 grad / 0 replay)", "#ede9fe", "#6d28d9")

    # Orthogonal null-space note above Pillar 3/4
    _box(ax, 0.580, 0.775, 0.396, 0.145, "Exact Batch-Sequential Equivalence & Orthogonal Protection", r"$P_t = (X_{1:t}^\top X_{1:t} + \lambda I)^{-1} \Rightarrow$ new error updates only $P_t \psi_t$" + "\n" + r"Block-diagonal $B$ blocks cut VRAM by $1/B$ ($4\times$ tested on Split-CIFAR-100)", "#f3e8ff", "#7e22ce", title_fs=8.3, sub_fs=7.3)

    _arrow(ax, 0.140, 0.622, 0.175, 0.622)
    _arrow(ax, 0.330, 0.622, 0.375, 0.622, r"$O(K d_{\mathrm{in}})$")
    _arrow(ax, 0.563, 0.622, 0.602, 0.622, r"$\psi_t$")
    _arrow(ax, 0.778, 0.622, 0.818, 0.622, r"$\phi(\psi_t)$")
    _arrow(ax, 0.897, 0.725, 0.897, 0.775, "", "#7e22ce", "--")

    # Section B container: PDNC Multi-Layer Local Credit Circuit
    sec_b = FancyBboxPatch((0.010, 0.018), 0.980, 0.445, boxstyle="round,pad=0.0,rounding_size=0.015", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.1)
    ax.add_patch(sec_b)
    ax.text(0.022, 0.438, "B. Prospective Dendritic Natural-Credit (PDNC): Multi-Layer Local Learning Without .backward() or Weight Transport", fontsize=10.2, fontweight="bold", color="#0f172a", va="center")

    _box(ax, 0.022, 0.180, 0.155, 0.220, "Basal Input / Pixels", r"$x_t$ (e.g. $16\times 16$ Pong," + "\n24D noisy XOR, FMNIST)\n(0 hand-crafted coords)", "#e2e8f0")
    _box(ax, 0.220, 0.180, 0.225, 0.220, "Hidden Layer L1 (Local)", r"$h_1 = \tanh(W_1 x_t + b_1)$" + "\nUpdate via Apical-Basal Local Rule:\n" + r"$\Delta W_1 = \eta\, (\mathrm{sign}(\tilde{e}_{\mathrm{apical}})\, P_1 x_t^\top)$", "#dbeafe", "#1d4ed8")
    _box(ax, 0.490, 0.180, 0.215, 0.220, "Readout Layer L2", r"$\hat{y}_t = W_2 h_1 + b_2$" + "\nSupervised: " + r"$\delta_2 = y_t - \hat{y}_t$" + "\nRL: episode-rate " + r"$\delta_{\mathrm{TD}(\lambda)} = R - V(h)$", "#ede9fe", "#6d28d9")
    _box(ax, 0.750, 0.180, 0.227, 0.220, "Natural Metric + Memory", "Diagonal BONG-momentum $m_i$ (RL/XOR)\nOR Woodbury Subspace $P_1$ (CL)\n+ Episode Eligibility Trace $E_t$\n(Zero frame-RMS collapse)", "#dcfce7", "#15803d")

    # Bottom Apical Feedback loop box with clean clearance below L1/L2
    _box(ax, 0.255, 0.035, 0.420, 0.088, "Apical Feedback Alignment (Kolen-Pollack Symmetric Update)", r"$e_{\mathrm{apical}} = B_{\mathrm{fb}} \delta_2,\quad B_{\mathrm{fb}} \leftarrow (1 - \alpha)B_{\mathrm{fb}} + \alpha \Delta B(\delta_2, h_1)$  ($\cos\theta \approx 0.77$)", "#fee2e2", "#b91c1c", title_fs=8.3, sub_fs=7.3)

    _arrow(ax, 0.177, 0.290, 0.220, 0.290, r"$W_1$")
    _arrow(ax, 0.445, 0.290, 0.490, 0.290, r"$h_1$")
    _arrow(ax, 0.705, 0.290, 0.750, 0.290, r"$\delta_2, h_1$")
    _arrow(ax, 0.598, 0.180, 0.598, 0.123, r"$\delta_2$", "#b91c1c", "--", lx_off=0.022, ly_off=0.0)
    _arrow(ax, 0.332, 0.123, 0.332, 0.180, r"$e_{\mathrm{apical}}$", "#b91c1c", "--", lx_off=0.036, ly_off=0.0)

    fig.savefig(FIG_DIR / "fig0_dmcn_pdnc_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig0_dmcn_pdnc_architecture.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig12_reviewer_controls() -> None:
    res_path = REPO_ROOT / "results" / "dmcn_reviewer_controls_v1" / "results.json"
    if not res_path.exists():
        print("Skipping fig12_reviewer_controls (results.json not present yet).")
        return
    data = json.loads(res_path.read_text(encoding="utf-8"))
    s = data["summary"]
    n = data.get("n_seeds", 20)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2), gridspec_kw={"wspace": 0.24})

    # Panel A: Temporal + 50x Shock Split-MNIST with CausalRMS Strengthened Baselines
    items_a = [
        ("Unified\nDMCN", "temp__dmcn_unified_acc", "#0f766e"),
        ("Pooled ACIL\n+RMS", "temp__pooled_acil_causal_rms_acc", "#2563eb"),
        ("GRU+RMS\n+ER-200", "temp__gru_causal_rms_er200_acc", "#d97706"),
        ("GRU+RMS\n+EWC", "temp__gru_causal_rms_ewc_acc", "#e11d48"),
        ("GRU+RMS\n+Adam", "temp__gru_causal_rms_adam_acc", "#7c3aed"),
        ("Blind /\nMajority", "temp__majority_acc", "#64748b"),
    ]
    xa = np.arange(len(items_a))
    ma = [s[k]["mean"] for _, k, _ in items_a]
    ca = [s[k]["ci95"] for _, k, _ in items_a]
    b1 = ax1.bar(xa, ma, yerr=ca, color=[c for _, _, c in items_a], edgecolor="#1e293b", linewidth=0.7, capsize=3, width=0.56)
    ax1.axhline(10.0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax1.set_xticks(xa)
    ax1.set_xticklabels([l for l, _, _ in items_a], fontsize=8.0)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_ylim(0, 104)
    ax1.set_title(f"A. Streaming Temporal + 50x Gain-Shock Split-MNIST (n={n})\nEven with DMCN's CausalRMS, Baselines Miss Complex Spines")
    for bar, m, c in zip(b1, ma, ca):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, m + c + 1.5, f"{m:.2f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel B: RGB 10-Task Split-CIFAR-100 (MobileNetV2 1280-D)
    items_b = [
        ("DMCN\nWoodbury", "c100__dmcn_acc", "#0f766e"),
        ("Joint FP64\nBound", "c100__joint_fp64_acc", "#1d4ed8"),
        ("DMCN\nBlock-4x", "c100__dmcn_block4x_acc", "#0891b2"),
        ("Linear-RLS\n(ACIL)", "c100__acil_linear_rls_acc", "#6366f1"),
        ("MLP\nER-500", "c100__mlp_er500_acc", "#d97706"),
        ("MLP\nER-200", "c100__mlp_er200_acc", "#f59e0b"),
        ("EWC /\nAdam", "c100__mlp_ewc_acc", "#e11d48"),
        ("Null\n(Shuf)", "c100__shuffled_acc", "#64748b"),
    ]
    xb = np.arange(len(items_b))
    mb = [s[k]["mean"] for _, k, _ in items_b]
    cb = [s[k]["ci95"] for _, k, _ in items_b]
    b2 = ax2.bar(xb, mb, yerr=cb, color=[c for _, _, c in items_b], edgecolor="#1e293b", linewidth=0.7, capsize=3, width=0.58)
    ax2.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([l for l, _, _ in items_b], fontsize=7.8)
    ax2.set_ylabel("All-100-Class Test Accuracy (%)")
    ax2.set_ylim(0, 80)
    ax2.set_title(f"B. RGB 10-Task Split-CIFAR-100 on MobileNetV2 1280-D (n={n})\n0 Replay, 0 Task IDs; 10-Epoch Trained Gradient Baselines")
    for bar, m, c in zip(b2, mb, cb):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, m + c + 1.2, f"{m:.2f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    fig.savefig(FIG_DIR / "fig12_reviewer_controls.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig12_reviewer_controls.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fig13_v4_and_patch_pdnc() -> None:
    res_path = REPO_ROOT / "results" / "dmcn_v4_efficiency_and_patch_pdnc_v1" / "results.json"
    if not res_path.exists():
        print("Skipping fig13 (results.json not present yet).")
        return
    data = json.loads(res_path.read_text(encoding="utf-8"))
    s = data["summary"]
    n = data.get("n_seeds", 20)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw={"wspace": 0.27})

    # Panel A: Split-MNIST CUDA Throughput (k samp/s) + Accuracy callouts
    items_a = [
        ("Woodbury\n(95.77%)", "mnist__woodbury_sps", "#64748b"),
        ("Gram-Chol\n(95.77%)", "mnist__gram_sps", "#0f766e"),
        ("95% Ternary\n+Gram (95.85%)", "mnist__ternary_gram_sps", "#0891b2"),
        ("Backprop\nAdam MLP", "mnist__adam_mlp_sps", "#4f46e5"),
    ]
    xa = np.arange(len(items_a))
    ma = [s[k]["mean"] / 1000.0 for _, k, _ in items_a]
    ca = [s[k]["ci95"] / 1000.0 for _, k, _ in items_a]
    b1 = ax1.bar(xa, ma, yerr=ca, color=[c for _, _, c in items_a], edgecolor="#1e293b", linewidth=0.7, capsize=3, width=0.56)
    ax1.set_xticks(xa)
    ax1.set_xticklabels([l for l, _, _ in items_a], fontsize=8.0)
    ax1.set_ylabel("Throughput (k samples / sec)")
    ax1.set_ylim(0, max(m + c for m, c in zip(ma, ca)) * 1.20)
    ax1.set_title(f"A. Split-MNIST 60k Speed (n={n})\n3.86x Faster via Gram-Cholesky")
    for bar, m, c in zip(b1, ma, ca):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, m + c + 2.2, f"{m:.1f}k/s", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel B: Split-CIFAR-100 Accuracy & Time (Woodbury vs Gram-Chol vs 95% Ternary)
    items_b = [
        ("Woodbury\n(6.76s)", "c100__woodbury_acc", "#64748b"),
        ("Gram-Chol\n(2.58s)", "c100__gram_acc", "#0f766e"),
        ("95% Ternary\n+Gram (2.59s)", "c100__ternary_gram_acc", "#0891b2"),
        ("Null\n(Shuffled)", "c100__shuffled_acc", "#94a3b8"),
    ]
    xb = np.arange(len(items_b))
    mb = [s[k]["mean"] for _, k, _ in items_b]
    cb = [s[k]["ci95"] for _, k, _ in items_b]
    b2 = ax2.bar(xb, mb, yerr=cb, color=[c for _, _, c in items_b], edgecolor="#1e293b", linewidth=0.7, capsize=3, width=0.54)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([l for l, _, _ in items_b], fontsize=8.0)
    ax2.set_ylabel("All-100-Class Test Accuracy (%)")
    ax2.set_ylim(0, 80)
    ax2.set_title(f"B. Split-CIFAR-100 50k (n={n})\n2.62x Speedup & 20x Fewer L1 Weights")
    for bar, m, c in zip(b2, mb, cb):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, m + c + 1.3, f"{m:.2f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel C: 4x4 Local-Patch PDNC on Raw 16x16 Pixel Pong RL
    items_c = [
        ("Patch\nPDNC\n(4.9k)", "pong__patch_pdnc_hit", "#0f766e"),
        ("Flat\nBP\n(66k)", "pong__flat_bp_a2c_hit", "#2563eb"),
        ("Patch\nBP\n(4.9k)", "pong__patch_bp_a2c_hit", "#4f46e5"),
        ("Ablate\nElig", "pong__ablate_elig_hit", "#d97706"),
        ("Shuffle\nApical", "pong__shuffle_apical_hit", "#e11d48"),
        ("Frozen\nPatch", "pong__patch_frozen_hit", "#dc2626"),
        ("Nulls\n(Blind)", "pong__blind_hit", "#64748b"),
    ]
    xc = np.arange(len(items_c))
    mc = [s[k]["mean"] for _, k, _ in items_c]
    cc = [s[k]["ci95"] for _, k, _ in items_c]
    b3 = ax3.bar(xc, mc, yerr=cc, color=[c for _, _, c in items_c], edgecolor="#1e293b", linewidth=0.7, capsize=3, width=0.56)
    ax3.axhline(25.0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax3.set_xticks(xc)
    ax3.set_xticklabels([l for l, _, _ in items_c], fontsize=7.0)
    ax3.set_ylabel("Held-Out Pixel Pong Hit Rate (%)")
    ax3.set_ylim(0, 92)
    ax3.set_title(f"C. 16x16 Pixel Pong RL (n={n}, 0 Coords)\nLocal-Patch PDNC Matches Flat BP & +41.8pp > Frozen")
    for bar, m, c in zip(b3, mc, cc):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, m + c + 1.5, f"{m:.2f}%", ha="center", va="bottom", fontsize=7.1, fontweight="bold")

    fig.savefig(FIG_DIR / "fig13_dmcn_v4_efficiency_and_patch_pdnc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig13_dmcn_v4_efficiency_and_patch_pdnc.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_fig0_architecture()
    plot_fig12_reviewer_controls()
    plot_fig13_v4_and_patch_pdnc()
    print("Generated figures in", FIG_DIR)


if __name__ == "__main__":
    main()
