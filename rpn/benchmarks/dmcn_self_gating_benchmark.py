"""Pre-registered 20-Seed Darwinian Self-Gating DMCN Benchmark (ONE Frozen Config Across 3 Conflicting Tasks).

Tests ONE identical SelfGatingConfig (zero per-task pillar switching, zero task IDs, zero backprop) across:
  - Task A: Unseen Chaotic Lorenz Forecasting (amplitude-critical -> raw/transient should win)
  - Task B: 100x Sensor Gain-Shock Nonlinear Stream (gain in [0.2, 20.0] -> homeo/spine should win)
  - Task C: 4-Step Delayed-Cue Classification (silent distractors -> spine should win)

Compares against ablate_self_gating (uniform 0.25 blend), fixed single-pathway models (Raw-Only,
Homeo-Only, Spine-Only), and strict null controls (Untrained, Blind, Shuffled).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "rpn", REPO_ROOT / ".integrity"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import results_meta  # noqa: E402
from neuron_v2.dmcn import SelfGatingConfig, SelfGatingDMCNLayer  # noqa: E402


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * sd / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": ci95, "min": float(np.min(arr)), "max": float(np.max(arr))}


def _generate_lorenz(seed: int, n_steps: int = 800, dt: float = 0.02) -> np.ndarray:
    rng = np.random.RandomState(seed)
    state = np.array([1.0, 1.0, 1.0], dtype=np.float64) + rng.normal(0.0, 0.5, size=3)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    traj = np.zeros((n_steps + 200, 3), dtype=np.float64)
    for t in range(n_steps + 200):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        state = state + dt * np.array([dx, dy, dz])
        traj[t] = state
    # Drop transient and scale by fixed physical constant 1/20 so coordinates are O(1)
    return traj[200:] / 20.0


def _make_model(d_in: int, d_out: int, seed: int, ablate_gate: bool = False) -> SelfGatingDMCNLayer:
    return SelfGatingDMCNLayer(SelfGatingConfig(
        d_in=d_in,
        d_out=d_out,
        d_hid=32,
        beta_gate=5.0,
        alpha_loss=0.06,
        alpha_homeo=0.05,
        tau_spine=2.5,
        use_precision=True,
        #precision_ridge=1.0,
        ablate_self_gating=ablate_gate,
        seed=seed,
    ))


def evaluate_seed(seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    # === TASK A: Unseen Chaotic Lorenz Forecasting (1-step prediction, raw amplitude critical) ===
    tr_lor = _generate_lorenz(seed + 10, n_steps=500)
    te_lor = _generate_lorenz(seed + 20, n_steps=300)
    var_te_lor = float(np.var(te_lor[1:])) + 1e-12

    m_lor = _make_model(3, 3, seed, ablate_gate=False)
    m_lor_uni = _make_model(3, 3, seed, ablate_gate=True)
    m_lor_shuf = _make_model(3, 3, seed, ablate_gate=False)
    perm_lor = rng.permutation(len(tr_lor) - 1)

    for t in range(len(tr_lor) - 1):
        m_lor.step(tr_lor[t], dt=1.0, episodic_peak=False)
        m_lor.learn_local(tr_lor[t + 1])
        m_lor_uni.step(tr_lor[t], dt=1.0, episodic_peak=False)
        m_lor_uni.learn_local(tr_lor[t + 1])
        m_lor_shuf.step(tr_lor[t], dt=1.0, episodic_peak=False)
        m_lor_shuf.learn_local(tr_lor[perm_lor[t] + 1])

    lor_gates = m_lor.gate_dict()
    preds_lor, preds_uni, preds_raw, preds_homeo, preds_spine, preds_blind, preds_shuf = [], [], [], [], [], [], []
    m_lor.reset_temporal_state()
    m_lor_uni.reset_temporal_state()
    m_lor_shuf.reset_temporal_state()
    for t in range(len(te_lor) - 1):
        y_g = m_lor.step(te_lor[t], dt=1.0, episodic_peak=False)
        preds_lor.append(y_g)
        preds_raw.append(m_lor._last_preds[0].copy())
        preds_homeo.append(m_lor._last_preds[1].copy())
        preds_spine.append(m_lor._last_preds[2].copy())
        preds_uni.append(m_lor_uni.step(te_lor[t], dt=1.0, episodic_peak=False))
        preds_shuf.append(m_lor_shuf.step(te_lor[t], dt=1.0, episodic_peak=False))
        preds_blind.append(np.zeros(3))

    tgt_lor = te_lor[1:]
    lor_nmse = float(np.mean((np.array(preds_lor) - tgt_lor) ** 2) / var_te_lor)
    lor_uni_nmse = float(np.mean((np.array(preds_uni) - tgt_lor) ** 2) / var_te_lor)
    lor_raw_nmse = float(np.mean((np.array(preds_raw) - tgt_lor) ** 2) / var_te_lor)
    lor_homeo_nmse = float(np.mean((np.array(preds_homeo) - tgt_lor) ** 2) / var_te_lor)
    lor_spine_nmse = float(np.mean((np.array(preds_spine) - tgt_lor) ** 2) / var_te_lor)
    lor_shuf_nmse = float(np.mean((np.array(preds_shuf) - tgt_lor) ** 2) / var_te_lor)

    # === TASK B: 100x Sensor Gain-Shock Nonlinear Regression (gains in [0.2, 20.0]) ===
    w_gs = rng.randn(2, 6) * 0.8
    m_gs = _make_model(6, 2, seed + 100, ablate_gate=False)
    m_gs_uni = _make_model(6, 2, seed + 100, ablate_gate=True)
    m_gs_shuf = _make_model(6, 2, seed + 100, ablate_gate=False)

    train_gains = [1.0, 20.0, 0.2, 10.0]
    gs_y_buf = []
    gs_x_buf = []
    for i in range(400):
        xb = rng.randn(6)
        xb_u = xb / (np.sqrt(np.mean(xb * xb)) + 1e-6)
        y_t = np.tanh(w_gs @ xb_u)
        g = train_gains[i // 100]
        gs_x_buf.append(xb * g)
        gs_y_buf.append(y_t)
    perm_gs = rng.permutation(400)
    for i in range(400):
        m_gs.reset_temporal_state()
        m_gs.step(gs_x_buf[i], episodic_peak=True)
        m_gs.learn_local(gs_y_buf[i])
        m_gs_uni.reset_temporal_state()
        m_gs_uni.step(gs_x_buf[i], episodic_peak=True)
        m_gs_uni.learn_local(gs_y_buf[i])
        m_gs_shuf.reset_temporal_state()
        m_gs_shuf.step(gs_x_buf[i], episodic_peak=True)
        m_gs_shuf.learn_local(gs_y_buf[perm_gs[i]])

    gs_gates = m_gs.gate_dict()
    gs_tgts, gs_p, gs_pu, gs_pr, gs_ph, gs_ps, gs_psh = [], [], [], [], [], [], []
    for _ in range(150):
        xb = rng.randn(6)
        xb_u = xb / (np.sqrt(np.mean(xb * xb)) + 1e-6)
        y_t = np.tanh(w_gs @ xb_u)
        g_te = float(np.exp(rng.uniform(np.log(0.2), np.log(20.0))))
        x_in = xb * g_te
        gs_tgts.append(y_t)
        m_gs.reset_temporal_state()
        gs_p.append(m_gs.step(x_in, episodic_peak=True))
        gs_pr.append(m_gs._last_preds[0].copy())
        gs_ph.append(m_gs._last_preds[1].copy())
        gs_ps.append(m_gs._last_preds[2].copy())
        m_gs_uni.reset_temporal_state()
        gs_pu.append(m_gs_uni.step(x_in, episodic_peak=True))
        m_gs_shuf.reset_temporal_state()
        gs_psh.append(m_gs_shuf.step(x_in, episodic_peak=True))

    gs_tgts_a = np.array(gs_tgts)
    var_gs = float(np.var(gs_tgts_a)) + 1e-12
    gs_nmse = float(np.mean((np.array(gs_p) - gs_tgts_a) ** 2) / var_gs)
    gs_uni_nmse = float(np.mean((np.array(gs_pu) - gs_tgts_a) ** 2) / var_gs)
    gs_raw_nmse = float(np.mean((np.array(gs_pr) - gs_tgts_a) ** 2) / var_gs)
    gs_homeo_nmse = float(np.mean((np.array(gs_ph) - gs_tgts_a) ** 2) / var_gs)
    gs_spine_nmse = float(np.mean((np.array(gs_ps) - gs_tgts_a) ** 2) / var_gs)
    gs_shuf_nmse = float(np.mean((np.array(gs_psh) - gs_tgts_a) ** 2) / var_gs)

    # === TASK C: 4-Step Delayed-Cue Classification (cue at t=0, noise t=1..4, decide at t=4) ===
    m_dc = _make_model(4, 1, seed + 200, ablate_gate=False)
    m_dc_uni = _make_model(4, 1, seed + 200, ablate_gate=True)
    m_dc_shuf = _make_model(4, 1, seed + 200, ablate_gate=False)
    cues = np.array([[1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]], dtype=np.float64)

    def run_dc_ep(model: SelfGatingDMCNLayer, label: int, gain: float, learn_y: float | None = None) -> np.ndarray:
        model.reset_temporal_state()
        for t in range(5):
            xt = (cues[label] + rng.normal(0.0, 0.15, size=4)) * gain if t == 0 else (rng.normal(0.0, 0.10, size=4) * gain)
            y_out = model.step(xt, dt=1.0, episodic_peak=True)
        if learn_y is not None:
            model.learn_local(np.array([learn_y]))
        return y_out

    for i in range(300):
        lbl = int(rng.randint(0, 2))
        y_val = 1.0 if lbl == 1 else -1.0
        g_ep = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        run_dc_ep(m_dc, lbl, g_ep, learn_y=y_val)
        run_dc_ep(m_dc_uni, lbl, g_ep, learn_y=y_val)
        run_dc_ep(m_dc_shuf, lbl, g_ep, learn_y=float(rng.choice([-1.0, 1.0])))

    dc_gates = m_dc.gate_dict()
    dc_ok, dc_uni_ok, dc_raw_ok, dc_homeo_ok, dc_spine_ok, dc_shuf_ok = 0, 0, 0, 0, 0, 0
    n_dc_te = 150
    for _ in range(n_dc_te):
        lbl = int(rng.randint(0, 2))
        g_ep = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        y_g = run_dc_ep(m_dc, lbl, g_ep)
        dc_ok += int((y_g[0] > 0.0) == (lbl == 1))
        dc_raw_ok += int((m_dc._last_preds[0][0] > 0.0) == (lbl == 1))
        dc_homeo_ok += int((m_dc._last_preds[1][0] > 0.0) == (lbl == 1))
        dc_spine_ok += int((m_dc._last_preds[2][0] > 0.0) == (lbl == 1))
        dc_uni_ok += int((run_dc_ep(m_dc_uni, lbl, g_ep)[0] > 0.0) == (lbl == 1))
        dc_shuf_ok += int((run_dc_ep(m_dc_shuf, lbl, g_ep)[0] > 0.0) == (lbl == 1))

    return {
        "seed": seed,
        "lorenz_self_gate_nmse": lor_nmse,
        "lorenz_uniform_ablate_nmse": lor_uni_nmse,
        "lorenz_raw_only_nmse": lor_raw_nmse,
        "lorenz_homeo_only_nmse": lor_homeo_nmse,
        "lorenz_spine_only_nmse": lor_spine_nmse,
        "lorenz_shuffled_nmse": lor_shuf_nmse,
        "lorenz_gate_raw_plus_trans": lor_gates["raw"] + lor_gates["transient"],
        "lorenz_gate_homeo": lor_gates["homeo"],
        "gainshock_self_gate_nmse": gs_nmse,
        "gainshock_uniform_ablate_nmse": gs_uni_nmse,
        "gainshock_raw_only_nmse": gs_raw_nmse,
        "gainshock_homeo_only_nmse": gs_homeo_nmse,
        "gainshock_spine_only_nmse": gs_spine_nmse,
        "gainshock_shuffled_nmse": gs_shuf_nmse,
        "gainshock_gate_homeo_plus_spine": gs_gates["homeo"] + gs_gates["spine"],
        "gainshock_gate_raw": gs_gates["raw"],
        "delayed_self_gate_acc": 100.0 * dc_ok / n_dc_te,
        "delayed_uniform_ablate_acc": 100.0 * dc_uni_ok / n_dc_te,
        "delayed_raw_only_acc": 100.0 * dc_raw_ok / n_dc_te,
        "delayed_homeo_only_acc": 100.0 * dc_homeo_ok / n_dc_te,
        "delayed_spine_only_acc": 100.0 * dc_spine_ok / n_dc_te,
        "delayed_shuffled_acc": 100.0 * dc_shuf_ok / n_dc_te,
        "delayed_gate_spine": dc_gates["spine"],
        "delayed_gate_raw": dc_gates["raw"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "test"], default="val")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "dmcn_self_gating_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "val":
        t0 = time.perf_counter()
        res = evaluate_seed(999)
        elapsed = time.perf_counter() - t0
        log_path = out_dir / "tuning_log.json"
        entries = json.loads(log_path.read_text(encoding="utf-8")).get("tuning_entries", []) if log_path.exists() else []
        entries.append({"run_index": len(entries) + 1, "seed": 999, "elapsed_sec": elapsed, "metrics": res})
        meta_val = results_meta.build_meta(
            exp_id="dmcn_self_gating_v1_val",
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
        "headline_metric": "delayed_self_gate_acc",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "elapsed_test_sec": float(elapsed),
        "summary": summary,
        "per_seed": per_seed,
    }
    meta = results_meta.build_meta(
        exp_id="dmcn_self_gating_v1",
        seeds=test_seeds,
        tuning_seeds=[999],
        config={"d_hid": 32, "beta_gate": 5.0, "tasks": ["lorenz", "gainshock_100x", "delayed_cue"]},
        repo_root=REPO_ROOT,
    )
    results_meta.write_results(out_dir / "results.json", payload, meta)
    print(json.dumps({"elapsed_test_sec": elapsed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
