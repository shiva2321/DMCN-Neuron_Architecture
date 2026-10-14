# neuron_v2 — Resonant Predictive Neuron (RPN)

A gradient-free, phasor-based cognitive unit. See `../docs/specs/2026-06-06-neuron-v2-rpn-design.md`
for the full concept; this package is **Stage 1 (RPN core, k=1)**.

**Direction:** Path **B** (column / new architecture). Path **A** (RXCA delta retrofit) is **pinned** — see [`docs/BOARD.md`](../docs/BOARD.md). Benchmarks: `compositional_bind` (B-1), `column_capacity` (B-2), `column_vote` (B-3), `continual_bwt` (B-4), `predictive_loop` (B-5), `compositional_generalization` (B-6).

## What this MVP proves

- Phasor VSA substrate with exact bind/unbind (G1–G5 tests).
- `DeltaMemory`: delta-rule writes (store only the residual) **do not saturate** under
  repeated writes, unlike additive bundling — fixing RXCA's `max_writes`/`ReSC+` problem at the root.
- `RPN`: stores context→target, recalls by resonance, and **refuses to hallucinate** via
  confidence gating (raw predict magnitude) plus the rectified Born gate when there is no signal.

## Install & test

```bash
pip install -e ".[dev]"
python -m pytest -v
python -m benchmarks.capacity_curve
python -m benchmarks.continual_assoc   # RPN vs RXCA-style vs online MLP (pass/fail)
```

## Stage 3 (B-5)

- `PredictiveRPN`: `k>1` settling read (ctx-key denoise + target snap) + surprise-gated delta writes.
- `python -m benchmarks.predictive_loop` — noisy-read margin vs plain RPN + stream write reduction.

## Stage 5 (B-6)

- `ReferenceFrame` + `ResonantPredictiveColumn`: where/what slots, integrator + slot specialists + where-shards; read routes by where-shard (color) then shape context.
- `python -m benchmarks.compositional_generalization` — capacity, new-where adaptation, and bind scorecard.

## Stage 5b (B-7)

- RPC units are `PredictiveRPN` by default (`memory="predictive"`): surprise-gated `learn_hv` on integrator, slot specialists, and where-shards; `k=1` read on clean grids (pass `k=4` for noisy settling).
- `memory="plain"` keeps one-shot `RPN` for A/B; `write_stats()` aggregates skip/applied writes across the column.

## Stage 2b (B-8)

- `memory="tiered"`: `TieredRPN` units with T0/T1/T2 + `end_task()` consolidation at task boundaries.
- `python -m benchmarks.continual_composition` — same where-shards, sequential shape blocks; consolidate vs episodic-reset retention.

## Stage 5c (B-9)

- `ReferenceFrame.agent_action` + `compositional_ood` scorecard: new what-fillers, checkerboard holdout one-shot, alternate frames.
- `python -m benchmarks.compositional_ood`

## Stage 4 (B-10)

- **Multi-band:** `MultiBandRPN` — K band-projected `DeltaMemory` dendrites; redundant (all bands) or sharded routing via `band_route`.
- **Sequences:** `ChainRPN.read_hops` — multi-hop associative walks (A→B→C).
- `python -m benchmarks.multiband_denoise`
- `python -m benchmarks.sequence_multihop`

## Stage 4b (B-11)

- **SequenceRPC:** `ResonantPredictiveColumn` + `ChainRPN` — `(color, shape) → object → action`.
- **RPC multiband:** `memory="multiband"` uses `MultiBandRPN` units (`multiband_K=3`).
- `python -m benchmarks.sequence_composition`
- `python -m benchmarks.rpc_multiband`

## Stage 4c (B-12)

- **Kuramoto:** `KuramotoRPN` — K-band adjacent coupling + ctx/tgt prefix denoise + memory pull (matches `predictive_k4` robustness).
- `python -m benchmarks.kuramoto_denoise`

## Stage 4d (B-13)

- **RPC Kuramoto:** `memory="kuramoto"` uses `KuramotoRPN` units (`kuramoto_K=3`, `k=4`).
- `python -m benchmarks.rpc_kuramoto`

## Stage 5a (B-14)

- **Continual stack:** `SequenceRPC` + tiered RPC `end_task()` — object and 2-hop action retention across tasks.
- `python -m benchmarks.continual_stack`

## Stage 5b (B-15) — hardening scorecard

Unified gates for scale, long-stream retrieval (RULER proxy), energy accounting, and refusal:

- **Scale:** 16×64 grid (1024 pairs) — RPC vs monolithic `SingleRPNComposite`
- **RULER proxy:** 768-pair stream — early/late needle accuracy
- **Energy:** delta footprint MB + relearn write-skip (surprise gate)
- **Refusal:** checkerboard holdout zero-shot stays ~0 before one-shot fill

```bash
python -m benchmarks.stage5_scorecard
```

## Not yet (later stages)

Path B demo packaging; Path A unpin when criteria met.
