"""Stage 5 hardening scorecard (B-15): scale, RULER probe, energy, refusal.

Pillar coverage (spec section 15)
---------------------------------
- Scale: 16x64 compositional grid — RPC vs monolithic single RPN.
- RULER proxy: long-context retrieval (768 pairs) — early vs late needle accuracy.
- Energy: memory footprint + relearn write-skip (surprise gate sparsity).
- Refusal: checkerboard holdout zero-shot must stay near 0 (Born anti-hallucination).

Run: python -m benchmarks.stage5_scorecard
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn
from benchmarks.baselines import SingleRPNComposite


@dataclass
class Stage5Config:
    D: int = 512
    seed: int = 42
    # scale_capacity
    scale_colors: int = 16
    scale_shapes: int = 64
    min_scale_rpc: float = 0.95
    max_scale_single: float = 0.20
    min_scale_margin: float = 0.70
    # ruler_retrieval
    ruler_colors: int = 16
    ruler_shapes: int = 48
    ruler_probe: int = 80
    min_ruler_rpc: float = 0.95
    max_ruler_single: float = 0.25
    # energy
    energy_colors: int = 8
    energy_shapes: int = 32
    min_relearn_skip_rate: float = 0.20
    max_footprint_mb: float = 64.0
    # refusal
    refusal_colors: int = 16
    refusal_shapes: int = 32
    min_refusal_train: float = 0.95
    max_refusal_holdout_zero: float = 0.10


@dataclass
class ScenarioResult:
    name: str
    metrics: dict[str, float]
    passed: bool
    detail: str


def _grid(n_colors: int, n_shapes: int) -> list[tuple[str, str, str]]:
    return [
        (f"color_{ci}", f"shape_{si}", f"obj_{ci}_{si}")
        for ci in range(n_colors)
        for si in range(n_shapes)
    ]


def _register(items: ItemTable, pairs: list[tuple[str, str, str]]) -> None:
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]


def _accuracy(model, pairs: list[tuple[str, str, str]]) -> float:
    if not pairs:
        return 0.0
    ok = sum(1 for c, s, o in pairs if model.read(c, s)[0] == o)
    return ok / len(pairs)


def _delta_footprint_bytes(rpc: ResonantPredictiveColumn) -> int:
    units = [rpc.integrator, *rpc.slot_units.values(), *rpc._where_shards.values()]
    total = 0
    for unit in units:
        mem = getattr(unit, "mem", None)
        if mem is not None and hasattr(mem, "W"):
            total += int(mem.W.nbytes)
        elif hasattr(unit, "bands"):
            for band in unit.bands:
                total += int(band.W.nbytes)
    return total


def scenario_scale_capacity(cfg: Stage5Config) -> ScenarioResult:
    pairs = _grid(cfg.scale_colors, cfg.scale_shapes)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    t0 = time.perf_counter()
    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)
    train_s = time.perf_counter() - t0

    rpc_acc = _accuracy(rpc, pairs)
    single_acc = _accuracy(single, pairs)
    margin = rpc_acc - single_acc
    passed = (
        rpc_acc >= cfg.min_scale_rpc
        and single_acc <= cfg.max_scale_single
        and margin >= cfg.min_scale_margin
    )
    return ScenarioResult(
        "scale_capacity",
        {
            "rpc": rpc_acc,
            "single": single_acc,
            "margin": margin,
            "pairs": float(len(pairs)),
            "train_s": train_s,
        },
        passed,
        f"16x64 grid ({len(pairs)} pairs): RPC {rpc_acc:.3f} vs single {single_acc:.3f}",
    )


def scenario_ruler_retrieval(cfg: Stage5Config) -> ScenarioResult:
    pairs = _grid(cfg.ruler_colors, cfg.ruler_shapes)
    probe = pairs[: cfg.ruler_probe]
    late = pairs[-cfg.ruler_probe :]

    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    single = SingleRPNComposite(cfg.D, items)
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
        single.learn(color, shape, obj)

    rpc_early = _accuracy(rpc, probe)
    rpc_late = _accuracy(rpc, late)
    single_early = _accuracy(single, probe)
    passed = (
        rpc_early >= cfg.min_ruler_rpc
        and rpc_late >= cfg.min_ruler_rpc
        and single_early <= cfg.max_ruler_single
    )
    return ScenarioResult(
        "ruler_retrieval",
        {
            "rpc_early": rpc_early,
            "rpc_late": rpc_late,
            "single_early": single_early,
            "pairs": float(len(pairs)),
        },
        passed,
        f"768-pair stream: RPC early {rpc_early:.3f} late {rpc_late:.3f}; single early {single_early:.3f}",
    )


def scenario_energy(cfg: Stage5Config) -> ScenarioResult:
    pairs = _grid(cfg.energy_colors, cfg.energy_shapes)
    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items, memory="predictive")
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
    applied1, skipped1 = rpc.write_stats()
    for color, shape, obj in pairs:
        rpc.learn(color, shape, obj)
    applied2, skipped2 = rpc.write_stats()

    relearn_applied = applied2 - applied1
    relearn_skipped = skipped2 - skipped1
    denom = relearn_applied + relearn_skipped
    skip_rate = relearn_skipped / denom if denom > 0 else 0.0

    footprint_b = _delta_footprint_bytes(rpc)
    footprint_mb = footprint_b / (1024 * 1024)
    n_units = 1 + len(rpc.slot_units) + rpc.n_where_shards()
    read_flops_proxy = len(pairs) * cfg.D * cfg.D * n_units

    passed = skip_rate >= cfg.min_relearn_skip_rate and footprint_mb <= cfg.max_footprint_mb
    return ScenarioResult(
        "energy_accounting",
        {
            "skip_rate": skip_rate,
            "footprint_mb": footprint_mb,
            "n_units": float(n_units),
            "read_flops_proxy": float(read_flops_proxy),
            "n_writes_total": float(applied2),
        },
        passed,
        f"footprint {footprint_mb:.1f} MB; relearn skip {skip_rate:.1%}; {n_units} delta units",
    )


def scenario_refusal_gap(cfg: Stage5Config) -> ScenarioResult:
    pairs = _grid(cfg.refusal_colors, cfg.refusal_shapes)
    train = [
        p
        for p in pairs
        if (int(p[0].split("_")[1]) + int(p[1].split("_")[1])) % 2 == 0
    ]
    holdout = [
        p
        for p in pairs
        if (int(p[0].split("_")[1]) + int(p[1].split("_")[1])) % 2 == 1
    ]

    items = ItemTable(cfg.D, seed=cfg.seed)
    _register(items, pairs)

    rpc = ResonantPredictiveColumn(cfg.D, items)
    for color, shape, obj in train:
        rpc.learn(color, shape, obj)

    train_acc = _accuracy(rpc, train)
    holdout_zero = _accuracy(rpc, holdout)
    passed = (
        train_acc >= cfg.min_refusal_train
        and holdout_zero <= cfg.max_refusal_holdout_zero
    )
    return ScenarioResult(
        "refusal_zero_shot",
        {
            "train_acc": train_acc,
            "holdout_zero": holdout_zero,
            "holdout_pairs": float(len(holdout)),
        },
        passed,
        f"train {train_acc:.3f}; holdout zero-shot {holdout_zero:.3f} (anti-hallucination)",
    )


def run_scorecard(cfg: Stage5Config) -> list[ScenarioResult]:
    return [
        scenario_scale_capacity(cfg),
        scenario_ruler_retrieval(cfg),
        scenario_energy(cfg),
        scenario_refusal_gap(cfg),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[ScenarioResult], cfg: Stage5Config) -> Verdict:
    lines = [
        f"Stage 5 hardening scorecard (D={cfg.D})",
        "",
        f"{'scenario':<22} {'status':>6}  detail",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.name:<22} {status:>6}  {r.detail}")
        metric_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.metrics.items())
        lines.append(f"{'':22}         {metric_str}")
    lines.append("")
    lines.append("Pass/fail gates:")
    for r in results:
        lines.append(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    passed = all(r.passed for r in results)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - scale: compositional column beats monolithic RPN at 1024-pair grid.")
    lines.append("  - ruler: long-stream retrieval stable (early + late probes).")
    lines.append("  - energy: bounded delta footprint; surprise gate skips redundant writes.")
    lines.append("  - refusal: checkerboard holdout stays near 0 before one-shot fill.")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = Stage5Config()
    verdict = evaluate_verdict(run_scorecard(cfg), cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
