"""Gates that check the committed result artifacts against the claims made about them. GEMINI.md sections 2.4, 6, 10."""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import re
from pathlib import Path

import pytest
from controls import (Rate, assert_gate, gate_metric_improved, gate_not_worse_than_baseline, rate, seed_sweep)
from results_meta import validate_meta

RPN = Path(__file__).resolve().parents[2]
REPO = RPN.parent


def load(rel: str) -> dict:
    return json.loads((RPN / rel).read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-29 REFUTED (docs/CLAIMS.md): tracking RMSE 0.669 -> 0.670; the board reports an internal proxy (motor T)")
def test_tracking_task_metric_improves():
    r = load("crfn_learning_benchmark_results.json")
    assert_gate(gate_metric_improved(r["initial_tracking_rmse"], r["final_tracking_rmse"], higher_is_better=False,
                                     min_rel_change=0.10))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-18/B-20 REFUTED (docs/CLAIMS.md): frozen-weight HPR delay error is worse than the tapped-delay baseline")
def test_hpr_frozen_delay_not_worse_than_tapped_delay():
    r = load("stress_frozen_timing_results.json")
    failures = []
    for name, v in r.items():
        failures += gate_not_worse_than_baseline(v["hpr_mse"], v["tdl_mse"], f"tapped-delay baseline on {name}")
    assert_gate(failures)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-22 REFUTED (docs/CLAIMS.md): board claims 3.3x-50x speedup, measured wall-clock ~1.9x")
def test_hierarchical_speedup_claim_matches_measurement():
    board = (REPO / "docs" / "BOARD.md").read_text(encoding="utf-8")
    row = next((ln for ln in board.splitlines() if "B-22" in ln), "")
    m = re.search(r"([0-9.]+)x\s*-\s*[0-9.]+x\s*speedup", row)
    if not m:  # the over-claim has been removed from the board: nothing left to contradict
        return
    claimed_floor = float(m.group(1))
    measured = load("stress_upgraded_results.json")["hierarchical_cleanup"]["speedup_factor"]
    assert_gate([] if measured >= claimed_floor else
                [f"BOARD claims >= {claimed_floor}x speedup but the committed measurement is {measured:.2f}x"])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-32 REFUTED (docs/CLAIMS.md): the committed multi-maze result is a single unseeded run without run metadata")
def test_multi_maze_results_report_seed_distribution():
    data = load("benchmarks/crfn_multi_maze_results.json")
    assert_gate(validate_meta(data.get("_meta")))


# --------------------------------------------------------------------------- opt-in slow replication of B-32
SLOW = os.environ.get("RUN_SLOW_INTEGRITY") == "1"


@pytest.mark.skipif(not SLOW, reason="slow (~1.5 min per seed). Run: RUN_SLOW_INTEGRITY=1 INTEGRITY_SEEDS=20 pytest ...")
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-32 REFUTED (docs/CLAIMS.md): 'Central Box 3/3' does not replicate across seeds")
def test_multi_maze_claim_across_seeds(monkeypatch):
    import numpy as np
    from benchmarks.crfn_multi_maze_generalization_benchmark import run_multi_maze_generalization_benchmark

    real_write = pathlib.Path.write_text

    def guarded(self, *a, **k):  # never overwrite the committed evidence while replicating
        if self.name == "crfn_multi_maze_results.json":
            return 0
        return real_write(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", guarded)
    n = int(os.environ.get("INTEGRITY_SEEDS", "20"))

    def one(seed: int) -> bool:
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_multi_maze_generalization_benchmark(seed=seed)
        cb = [x["goal"] for x in r["unseen_eval_results"] if "Central" in x["maze"]]
        return len(cb) == 3 and all(cb)

    full: Rate = rate(seed_sweep(one, list(range(1000, 1000 + n))))
    print(f"Central Box 3/3 on {full}")
    assert_gate([] if full.p >= 0.9 else
                [f"'100% zero-shot on the unseen Central Box' holds on only {full}; the claim needs >= 90% of seeds"])
