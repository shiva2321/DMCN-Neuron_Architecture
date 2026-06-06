# neuron_v2 — Resonant Predictive Neuron (RPN)

A gradient-free, phasor-based cognitive unit. See `../docs/specs/2026-06-06-neuron-v2-rpn-design.md`
for the full concept; this package is **Stage 1 (RPN core, k=1)**.

**Direction:** Path **B** (column / new architecture). Path **A** (RXCA delta retrofit) is **pinned** — see [`docs/BOARD.md`](../docs/BOARD.md). Benchmarks: `compositional_bind` (B-1), `column_capacity` (B-2), `column_vote` (B-3), `continual_bwt` (B-4), `predictive_loop` (B-5). Next: **B-6** full RPC + reference frames.

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

## Not yet (later stages)

Multi-band dendrites (K>1), Kuramoto mode (④), full RPC + reference frames (B-6).
