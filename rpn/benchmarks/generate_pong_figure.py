"""Generate publication-quality Pong RL analysis & rally trajectory figure strictly from results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="dmcn_pong_rl_v1")
    parser.add_argument("--out", default="fig6_pong_learning_and_rallies")
    args = parser.parse_args()

    data = json.loads((REPO_ROOT / "results" / args.exp / "results.json").read_text(encoding="utf-8"))
    s = data["summary"]
    ex = data["seed_2000_analysis"]
    is_v2 = "t1_bp_a2c_hit_rate" in s
    bp_id_k = "t1_bp_a2c_hit_rate" if is_v2 else "t1_bp_reinforce_hit_rate"
    bp_ood_k = "t1_bp_a2c_ood_hit_rate" if is_v2 else "t1_bp_reinforce_ood_hit_rate"
    bp_t2_k = "t2_pixel_bp_a2c_hit_rate" if is_v2 else "t2_pixel_bp_reinforce_hit_rate"
    bp_lbl = "BP A2C" if is_v2 else "BP REINFORCE"
    n_train = int(data.get("_meta", {}).get("config", {}).get("n_train", 600))

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

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.45, 1.0, 1.05], "wspace": 0.28})

    # Panel A: Held-out & OOD Return Rates (20 seeds)
    items = [
        ("T1 DMCN (Held-Out ID)", "t1_dmcn_hit_rate", "#1b6ca8"),
        (f"T1 {bp_lbl} (ID)", bp_id_k, "#7570b3"),
        ("T1 DMCN (Unseen OOD)", "t1_dmcn_ood_hit_rate", "#3a9ad9"),
        (f"T1 {bp_lbl} (OOD)", bp_ood_k, "#9e9ac8"),
        ("T1 Ablate Spine", "t1_ablate_spine_hit_rate", "#d95f02"),
        ("T1 Ablate Eligibility", "t1_ablate_elig_hit_rate", "#d95f02"),
        ("T2 Pixel DMCN (16x16)", "t2_pixel_dmcn_hit_rate", "#1b9e77"),
        (f"T2 Pixel {bp_lbl}", bp_t2_k, "#66a61e"),
        ("Null: Untrained", "t1_untrained_hit_rate", "#888888"),
        ("Null: Blind (obs=0)", "t1_blind_hit_rate", "#888888"),
        ("Null: Shuffled Reward", "t1_shuffled_reward_hit_rate", "#888888"),
        ("Null: Random Policy", "t1_random_hit_rate", "#888888"),
    ][::-1]
    labels = [it[0] for it in items]
    means = [s[it[1]]["mean"] for it in items]
    cis = [s[it[1]]["ci95"] for it in items]
    colors = [it[2] for it in items]

    y = np.arange(len(labels))
    bars = axes[0].barh(y, means, xerr=cis, color=colors, edgecolor="#222222", linewidth=0.5, capsize=2.5, height=0.72)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.3)
    axes[0].set_xlim(0, 108)
    axes[0].set_xlabel("Frozen-Weight Return Rate (%) [20 seeds]")
    axes[0].set_title("Pong Return Rate: DMCN vs. Backprop & Controls\n(Sparse +1/-1 Reward Only, Zero Hardcoding)")
    for bar, m, c in zip(bars, means, cis):
        axes[0].text(m + c + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{m:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold")

    # Panel B: Learning Curves over Training Serves (seed 2000)
    bin_sz = max(1, n_train // len(ex["curve_t1"]))
    x_bins = np.arange(1, len(ex["curve_t1"]) + 1) * bin_sz
    axes[1].plot(x_bins, ex["curve_t1"], marker="o", color="#1b6ca8", linewidth=2.2, label="Tier 1: Position-Only DMCN")
    axes[1].plot(x_bins, ex["curve_t2"], marker="s", color="#1b9e77", linewidth=2.2, label="Tier 2: Raw 16x16 Pixel DMCN")
    axes[1].axhline(s["t1_random_hit_rate"]["mean"], color="#888888", linestyle=":", linewidth=1.3, label=f"Random ({s['t1_random_hit_rate']['mean']:.1f}%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel(f"Training Serves ({bin_sz}-serve bins)")
    axes[1].set_ylabel("In-Training Return Rate (%)")
    axes[1].set_title("Online 3-Factor RL Learning Curve\n(Seed 2000, Zero Backprop)")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8.2)

    # Panel C: Real Frozen-Weight Rally Trajectories (Untrained vs Trained DMCN on exact same unseen serves)
    ax_r = axes[2]
    # Pick first serve where Untrained missed and Trained hit (or serve 0 if none)
    chosen = 0
    for idx in range(len(ex["traces_trained"])):
        if ex["traces_trained"][idx]["hit"] and not ex["traces_untrained"][idx]["hit"]:
            chosen = idx
            break
    tr_tr = ex["traces_trained"][chosen]
    tr_un = ex["traces_untrained"][chosen]

    ax_r.plot(tr_tr["bx"], tr_tr["by"], color="#111111", linewidth=2.0, marker=".", markersize=5, label="Ball Path (Unseen Serve)")
    # Plot paddle y-trajectory mapped along x=0.94 (untrained) and x=0.98 (trained) or versus time
    t_steps = np.linspace(0.08, 0.96, len(tr_tr["py"]))
    ax_r.plot(t_steps, tr_tr["py"], color="#1b6ca8", linewidth=2.2, linestyle="-", label=f"Trained DMCN Paddle ({'HIT' if tr_tr['hit'] else 'MISS'})")
    t_un = np.linspace(0.08, 0.96, len(tr_un["py"]))
    ax_r.plot(t_un, tr_un["py"], color="#d95f02", linewidth=1.8, linestyle="--", label=f"Untrained Paddle ({'HIT' if tr_un['hit'] else 'MISS'})")
    # Draw final paddle bars at x=0.96
    ax_r.plot([0.96, 0.96], [tr_tr["py"][-1] - 0.12, tr_tr["py"][-1] + 0.12], color="#1b6ca8", linewidth=6.0, alpha=0.85)
    ax_r.plot([0.99, 0.99], [tr_un["py"][-1] - 0.12, tr_un["py"][-1] + 0.12], color="#d95f02", linewidth=6.0, alpha=0.55)
    ax_r.set_xlim(0.0, 1.05)
    ax_r.set_ylim(0.0, 1.0)
    ax_r.set_xlabel("Court X (Ball X & Paddle Y over Serve Time)")
    ax_r.set_ylabel("Court Y")
    ax_r.set_title(f"Frozen-Weight Unseen Serve Trace (Seed 2000)\n(Dir. Tracking Align: {s['t1_dmcn_dir_align_acc']['mean']:.1f}%)")
    ax_r.legend(loc="lower left", frameon=True, fontsize=7.8)

    fig.savefig(FIG_DIR / f"{args.out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{args.out}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}.png/.svg")


if __name__ == "__main__":
    main()
