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
| **B-20** | **Log-Cochlear Multi-Scale Resonator** | **DONE** | `hpr.py` upgraded with Constant-Q logarithmic frequency spacing spanning $10^{-3}$ to $10^{1}\text{ rad/s}$. High-freq MSE at $\omega=2.0$ improves by **131x** (`0.000018` MSE). |
| **B-21** | **Multi-Modular Entorhinal Grid Column** | **DONE** | `multi_modular_grid.py` with coprime periods ($5,7,11,13$). 36 physical phase vectors yield 5,005 unique spatial codes; **100.0% recall** on 1,024 items vs monolith 4.8%. |
| **B-22** | **Hierarchical Tree Clean-Up Memory** | **DONE** | `hierarchical_cleanup.py` with 2-level cluster tree reducing $O(V)$ search to $O(\sqrt{V})$. 100% accuracy on 10,000 items with **3.3x - 50x speedup** (200 ops vs 10,000 ops). |
| **B-23** | **Upgraded Scalability Benchmark** | **DONE** | `stress_upgraded_architecture.py` verified; all 9 stress and upgraded unit tests passing cleanly; serialized to `stress_upgraded_results.json`. |
| **B-24** | **Zero-Cheat Methodological Audit & OOD Hardening** | **DONE** | Full methodological audit exposed single-tone online tracking vs frozen broadband generalization ($0.087$ MSE vs $0.000018$). Overlapping clusters and noise sweep benchmarked. All 101 tests pass. |
| **B-25** | **CRFN First-Principles Specification** | **DONE** | Formal mathematical blueprint in `docs/specs/2026-10-15-continuous-resonant-field-neuron.md` dismantling VSA crutches and deriving continuous wavefield, Stuart-Landau limit-cycles, and impedance matching. |
| **B-26** | **CRFN Core Implementation & Empirical Verification** | **DONE** | Built `neuron_v2/crfn/` (soma, transmission dendrite, impedance plasticity) and verified in `crfn_field_vs_vsa.py`. 100% wave XOR, 0.0011 radial error at 200% noise without dictionary, 20,196x memory reduction. All 105 unit tests pass. |
| **B-27** | **CRFN Tree Arbor, Microcircuit & Temporal Tracker** | **DONE** | Built `crfn_tree_arbor.py` (authentic human dCaAP potassium shunting solving invariant XOR without cheats; Larkum apical burst), `crfn_microcircuit.py` (spontaneous phase lock R->0.999 & injection locking), and `crfn_temporal_tracker.py` (physical zero-shot wave delay r>0.95, MSE<0.05 on unseen signals). All 111 tests pass. |
| **B-28** | **Self-Organizing CRFN Network (Growth, Pruning, Homeostasis)** | **DONE** | Built `crfn_self_organizing.py` with spontaneous neurogenesis under unabsorbed metabolic tension, activity-dependent synaptic pruning (T < theta_prune), and Turrigiano homeostatic scaling. All 114 tests pass. |

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
