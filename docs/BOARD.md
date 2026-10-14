# Neuron v2 — Project Board

> **Decision (2026-10-14):** **Path B Dual-Track** is active. Both Track 1 (Concrete Engineering & Laminar Column) and Track 2 (First-Principles Holomorphic Wavefield) run in parallel.

---

## Paths

| Path | Meaning | Status |
|------|---------|--------|
| **A** | **Patch RXCA** — drop delta `DeltaMemory` into existing `PartitionedBrain`; rerun 187 tests | **PINNED** |
| **B** | **Dual-Track Architecture** — Track 1 (RCPN + 6-Layer RCC) & Track 2 (Holomorphic Phase-Resonator HPR) | **ACTIVE** |

---

## Board

### ACTIVE — Path B (Stage 6: Dual-Track Evolution)

| ID | Card | Status | Exit criterion |
|----|------|--------|----------------|
| **B-0 to B-15** | Stages 1 through 5 (RPN, RPC, Kuramoto, Continual, Stage 5 Scorecard) | **DONE** | 71 tests pass; 16×64 scale RPC **100%** vs single **10%**; BWT **0.0**; RULER **100%**. |
| **B-16** | **Track 1: RCPN Core (Autonomous Routing & Delay)** | **DONE** | `rcpn.py` with basal NMDA plateaus, Larkum apical nexus, astrocyte gain, and continuous state-space delay. Autonomous XOR: **98.8% ± 5.4%** vs LIF 52.5%; Delay MSE: **0.0303**. |
| **B-17** | **Track 1: Laminar Cortical Column (RCC)** | **DONE** | `laminar_column.py` with 6 layers (L1-L6) & TEM factorization. Capacity 128 items: **100.0%** vs monolith 28.1%; Zero-shot OOD retention: **100.0%** vs monolith 14.6%. |
| **B-18** | **Track 2: Holomorphic Phase-Resonator (HPR)** | **DONE** | `hpr.py` deriving non-linear logic (XOR) natively from wave interference: **100.0% ± 0.0%**; Continuous delay from spectral dispersion: **0.000918 MSE**; Unitary energy conservation: $\|\Psi\|=1$. |
| **B-19** | **Dual-Track Shootout & Baselines** | **DONE** | Shootout synthesized across both tracks and baselines; serialized to `dual_track_shootout_results.json`. All 91 tests passing. |

---

### Commands

```bash
cd rpn
python -m pytest -v
python -m benchmarks.rcpn_autonomous_xor
python -m benchmarks.rcpn_autonomous_timing
python -m benchmarks.rcc_laminar_generalization
python -m benchmarks.hpr_wave_interference
```

---

## One-line north star (Path B)

**Deliver the zero-cheat engineering column AND derive the first-principles wave-holomorphic neuron from physics.**
