# Neuron v2 — Project Board

> **Decision (2026-06-06):** **Path B** is active. **Path A** is **pinned** — visible, deferred, not in flight.

---

## Paths (two forks)

| Path | Meaning | Status |
|------|---------|--------|
| **A** | **Patch RXCA** — drop delta `DeltaMemory` into existing `PartitionedBrain`; rerun 187 tests | **PINNED** |
| **B** | **New architecture** — RPN → column (RPC) → composition; prove *N neurons* buy something one cell cannot | **ACTIVE** |

---

## Board

### PINNED — Path A (do not start until unpinned)

| ID | Card | Exit criterion | Notes |
|----|------|----------------|-------|
| **A-0** | Stage 0: RXCA delta retrofit | No regression on RXCA 187 tests + benches; saturation fixed in real brain | Lives in **Resonance_X** repo, not `rpn/`. Unpin when Path B column demo lands or you explicitly swap priority. |
| **A-0-prep** | Port plan only | `DeltaMemory` API documented for RXCA integrator | Already implemented in `rpn/neuron_v2/core/delta_memory.py` |

**Why pinned:** Path B must show *why* the new primitive exists beyond a write-rule patch. If column + composition fail, Path A may be the real win.

---

### ACTIVE — Path B

| ID | Card | Status | Exit criterion |
|----|------|--------|----------------|
| **B-0** | Stage 1 RPN core (`k=1`) | **DONE** | 16 tests pass; delta > additive on capacity curve; continual shootout PASS vs RXCA hologram + online MLP |
| **B-1** | **Mini RPC: column + binding** | **DONE** | `color_sharded_column` **100%** vs single_rpn **42%** / RXCA **53%** on 8×8 grid (`python -m benchmarks.compositional_bind` PASS). `CompositionalColumn` (3-RPN) = 4-pair disambiguation unit test. |
| **B-2** | Column capacity at scale | **DONE** | 8-shard `ColorShardedColumn` **100%** vs single **23%** at **M=256** (8×32 grid, D=512). `python -m benchmarks.column_capacity` PASS |
| **B-3** | Column vote (not just hash route) | **DONE** | Token hash vote **100%** = route; color vote **12.5%** vs route **100%**; composite hash vote **37%** vs route **95%**. `python -m benchmarks.column_vote` PASS |
| **B-4** | T0/T2 + consolidation (spec Stage 2) | **DONE** | `tiered_t0t2_consolidate` BWT **0.0** vs episodic-reset **-1.0**; task-0 retention **100%**. `python -m benchmarks.continual_bwt` PASS |
| **B-5** | Predictive loop `k>1` (spec Stage 3) | **DONE** | `predictive_k4` noisy margin **≥0.20** vs `rpn_k1`; surprise gate **≥75%** write skip on stream. `python -m benchmarks.predictive_loop` PASS |
| **B-6** | Full RPC + reference frames (spec Stage 5) | Backlog | Compositional generalization scorecard |

**Current focus:** **B-6** — full RPC + reference frames (compositional generalization scorecard).

---

### DONE (reference)

- Stage 1 MVP under `rpn/` (`feature/stage1-rpn-core`)
- Benchmarks: `capacity_curve`, `continual_assoc`
- Shootout result: delta RPN beats RXCA hologram + online MLP on 1-pass streaming recall; ties matrix additive at same `O(D²)` cost

---

## What “Path B active” means in practice

1. **Build in `rpn/` only** — JS NAD-Neuron at repo root stays off-limits.
2. **Next code milestone = B-1** (3 neurons, phasor bind for composition), not RXCA integration.
3. **Path A stays on the board** so we don’t forget the cheapest high-value fallback.

---

## Unpin Path A when

- [ ] B-1 passes (composition demo), **or**
- [ ] You explicitly reprioritize (“patch RXCA first”), **or**
- [ ] B-1 fails and we need to validate “delta-only” on real RXCA tests before more architecture

---

## Commands (today)

```bash
cd rpn
python -m pytest -q
python -m benchmarks.continual_assoc
python -m benchmarks.capacity_curve
python -m benchmarks.compositional_bind
python -m benchmarks.column_capacity
python -m benchmarks.column_vote
python -m benchmarks.continual_bwt
python -m benchmarks.predictive_loop
```

---

## One-line north star (Path B)

**Prove that multiple RPNs + binding do something one RPN cannot — then scale.**
