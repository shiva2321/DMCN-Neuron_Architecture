# DMCN Neuron Architecture (`Neuron_v2`)

**Dendritic Multiplexed Cascade Neuron (DMCN)** and **Prospective Dendritic Natural-Credit (PDNC)** — stateful dendritic neuron architectures and local backpropagation-free learning algorithms, evaluated under pre-registered multi-seed protocols with mandatory null controls.

> **Research Integrity Protocol:** [GEMINI.md](GEMINI.md)  
> **Legacy Claim Ledger:** [docs/CLAIMS.md](docs/CLAIMS.md)  
> **Mechanical Integrity Gate:** `python .integrity/run_gates.py --fast` (full test suite: `python .integrity/run_gates.py`)

---

## Overview

This repository (`main` branch = **v1 baseline**) houses the DMCN/PDNC implementations, pre-registered 20-seed benchmark suites, unit tests, and preprint manuscript alongside earlier historical tracks (`core/`, `crfn/`, `biophysics/`, `capn/`) preserved for traceability and integrity-gate parity.

What is implemented in `rpn/neuron_v2/dmcn/`:
- **DMCN** (*Dendritic Multiplexed Cascade Neuron*): A **stateful, multi-compartment neuron and layer architecture** combining causal divisive gain normalization (`CausalRMS`), complex Zero-Order-Hold resonant spine memory traces, active non-monotonic / self-gated / bilinear dendritic compartments, and orthogonal recursive least squares (`woodbury`) or checkpointed `gram_cholesky` readout plasticity with optional 95% sparse ternary (`{-1, 0, +1}`) encoders.
- **PDNC** (*Prospective Dendritic Natural-Credit*): A **local, `.backward()`-free and weight-transport-free multi-layer training algorithm** uniting symmetric Kolen–Pollack feedback alignment, two-pass prospective activity settling, diagonal or subspace local conditioning, episode-rate three-factor TD($\lambda$) eligibility traces, and overlapping `4x4` spatial receptive-field patches.

**Honest Architectural Status on `main` (Motivating Branch `v2`):**  
On `main`, the DMCN and PDNC mechanisms are split across specialized classes (`DMCNTorchLayer`, `SelfGatingDMCNLayer`, `FactorizedBilinearDMCNLayer`, `PDNCLayer`) with per-benchmark configurations rather than a single unified general-purpose neuron and plasticity law. Branch `v2` will focus on synthesizing these lessons into a unified general architecture and local learning algorithm.

---

## Key 20-Seed Observations (`results/dmcn_*/results.json`, Seeds `2000–2019`)

Every experiment below was pre-registered (`results/dmcn_*/PREREG.md`), tuned on validation `seed 999`, and run once across `20` held-out test seeds alongside matched baselines and null controls (`Untrained`, `Blind`, `Shuffled`, `Majority`, `Random`, `Shuffle Apical`, `Decoupled`, `Scrambled`). Note that DMCN and PDNC do **not** replace backpropagation for pretraining large vision backbones (`Split-CIFAR-100` uses frozen `MobileNetV2` embeddings) or frontier language models.

| Domain / Benchmark | Primary Observed Result (`mean +/- 95% CI`) | Reference Baseline | Source Artifact |
|---|---|---|---|
| **Streaming Delayed-Cue + `50x` Gain-Shock `Split-MNIST`** (1 pass, `0` replay) | **Unified DMCN:** `90.32% +/- 0.06%` (`P0 ret: 97.77%`) | Raw `Pooled ACIL`: `24.26%`; `GRU + ER-200`: `19.94%`; Ablate any pillar: `14.1–40.0%` | [results.json](results/dmcn_unified_temporal_cil_v1/results.json) |
| **Fairness Control (`CausalRMS`-Armed Baselines)** | **Unified DMCN:** `90.30% +/- 0.06%` | `Pooled ACIL + CausalRMS`: `68.66%`; `GRU + CausalRMS + ER-200`: `30.88%` | [results.json](results/dmcn_reviewer_controls_v1/results.json) |
| **10-Task RGB `Split-CIFAR-100`** (`MobileNetV2`, 1 pass, `0` replay, `0` task IDs) | **DMCN Woodbury:** `68.65% +/- 0.09%` (`0.0035%` vs. Joint FP64 ridge) | `Linear-RLS / ACIL`: `64.99%`; `10-ep MLP + ER-500`: `33.54%` | [results.json](results/dmcn_reviewer_controls_v1/results.json) |
| **GPU Acceleration & `95%` Sparse Ternary** (RTX 3060) | **Gram–Cholesky:** `81.9k` samp/s (`3.86x` MNIST), `2.58s` (`2.62x` C100); **95% Ternary:** `68.65%` C100 (`20x` fewer L1 nz) | Dense Woodbury: `21.2k/s` MNIST, `6.76s` C100; Backprop Adam: `120.7k/s` | [results.json](results/dmcn_v4_efficiency_and_patch_pdnc_v1/results.json) |
| **Causal Self-Gating (1 Frozen Config, 3 Conflicting Tasks)** | Lorenz NMSE `0.0054 +/- 0.0013`; `100x` Shock NMSE `0.0928 +/- 0.0103`; 4-Step Delay `100.00%` | Uniform gate: `0.0064` / `0.1111`; Raw-only `100x` Shock: `0.2021` | [results.json](results/dmcn_self_gating_v1/results.json) |
| **Zero-Shot `Rule x Operand` Composition** (permuted inputs) | **Factorized Bilinear DMCN:** `99.58% +/- 0.82%` (`MSE 0.0017`) | Ablate bilinear: `10.83%`; Dense DMCN: `35.42%`; BP Adam: `83.75%` | [results.json](results/dmcn_compositional_v1/results.json) |
| **Raw `16x16` Pixel Pong RL (`0` `.backward()`, `0` coordinates)** | **`4x4` Local-Patch PDNC (`4,851` L1 syns):** `72.25% +/- 4.68%` | Flat BP A2C (`65.5k` syns): `72.30% +/- 3.05%`; Patch BP A2C: `68.90%`; Frozen Patch: `30.50%` | [results.json](results/dmcn_v4_efficiency_and_patch_pdnc_v1/results.json) |
| **Boundary-Free 3-Task Rotated Subspace CL** (`PDNC`) | **PDNC Subspace:** `83.28% +/- 0.81%` (`T0 ret: 82.00%`) | Backprop SGD: `77.85% +/- 1.28%`; PDNC Diagonal: `71.70%` | [results.json](results/dmcn_pdnc_general_v1/results.json) |
| **Decentralized `16`-Boid Flock Wind Adaptation** (`640` FLOPs/agent) | **Spatial Wind Tracking Err:** `3.97 +/- 0.87` (`31%` blind subset: `4.32`) | Tuned PID Observer: `6.30 +/- 0.80`; Decoupled blind subset: `22.54` | [results.json](results/dmcn_swarm_boids_v2/results.json) |

### Explicit Negative Results & Open Limitations
- **Refuted Unsupervised Vision Hypothesis:** Unsupervised Motion-Oja Layer-1 plasticity on Pixel Pong *degraded* held-out hit rate below a frozen random Layer 1 (`39.05% +/- 4.02%` vs. `45.30% +/- 3.98%`) because unrewarded competition captured bright paddles and ignored the ball (`results/dmcn_pong_pixel_unsup_v1/`).
- **Hidden-Feature Drift vs. Orthogonal Readouts:** Plastic Layer-1 updates on compact `d = 256` continual `Split-FashionMNIST` caused hidden coordinates to drift away from previously inverted readout precision matrices (`70.13%` plastic vs. `80.72%` frozen Layer 1 + Woodbury; `results/dmcn_pdnc_general_v1/`).
- **Pre-Registered Pixel Pong Target & Variance:** `4x4` `Local-Patch PDNC` reached `72.25% +/- 4.68%`, falling `1.75 pp` short of the pre-registered `74.0%` falsifier bar and showing wider cross-seed variance (`SD = 10.69%`, `min = 36.0%`) than Flat Backprop A2C (`SD = 6.97%`).
- **Ternary Hardware Execution & Swarm Control Caveat:** PyTorch `torch.mm` executes dense FP32 multiply-adds on `95%` sparse ternary matrices (physical energy/latency savings require bit-packed sparse kernels). In Swarm Regimes 1–2, randomized neighbor headings (`Scrambled`: `2.19` / `3.07`) induced stabilizing velocity damping `-v_i`, isolating true consensus gain strictly to Regime 3's blind-boid rescue.

---

## Preprint & Documentation

- **Markdown Preprint:** [docs/PAPER_DRAFT_DMCN.md](docs/PAPER_DRAFT_DMCN.md)
- **PDF Preprint (15 pages):** [docs/preprint_dmcn_researchgate.pdf](docs/preprint_dmcn_researchgate.pdf) (source: [docs/preprint_dmcn_researchgate.tex](docs/preprint_dmcn_researchgate.tex))
- **Architecture Spec:** [docs/ARCHITECTURE_DMCN.md](docs/ARCHITECTURE_DMCN.md)
- **Legacy Audit Ledger:** [docs/CLAIMS.md](docs/CLAIMS.md) (tracks `2026-10-16` legacy cards `B-0`..`B-48`; `docs/BOARD.md` reconciliation remains pending)

---

## Repository Structure

```text
DMCN-Neuron_Architecture/
├── rpn/
│   ├── neuron_v2/
│   │   ├── dmcn/              # Active v1 DMCN & PDNC modules (NumPy + PyTorch/CUDA)
│   │   ├── core/, crfn/       # Historical / gate-pinned tracks (see docs/CLAIMS.md)
│   │   ├── biophysics/, capn/ # Track B-33..B-46 standalone modules
│   │   └── substrate/         # Complex unit phasor utilities
│   ├── benchmarks/            # Pre-registered DMCN, PDNC & figure generation scripts
│   └── tests/
│       ├── test_dmcn.py       # DMCN / PDNC unit tests
│       └── integrity/         # Locked domain null-control gate tests
├── docs/                      # Preprint PDF/TeX/MD, figures/, specs/, CLAIMS.md
├── results/                   # Immutable PREREG.md, tuning_log.json, results.json runs
├── archive/                   # Retired Stage-0 JS prototype & refuted extras
├── .integrity/                # Locked mechanical integrity gates (run_gates.py)
├── .githooks/                 # Pre-commit integrity hook
├── GEMINI.md                  # Research & engineering integrity protocol
├── LICENSE                    # MIT License
└── README.md
```

---

## Quickstart & Verification

```bash
# 1. Run DMCN & PDNC unit tests
python -m pytest rpn/tests/test_dmcn.py -q

# 2. Run static integrity gates (lint + results metadata + claims ledger + lock)
python .integrity/run_gates.py --fast

# 3. Example: reproduce the 20-seed GPU Gram-Cholesky + Local-Patch PDNC benchmark
python rpn/benchmarks/dmcn_v4_efficiency_and_patch_pdnc_benchmark.py --mode test
```

## License

Released under the [MIT License](LICENSE).
