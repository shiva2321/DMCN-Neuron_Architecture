# Pre-registration: dmcn_unified_temporal_cil_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A single unified `DMCN` configuration with ALL pillars simultaneously active (Pillar 1 Causal RMS Homeostasis + Pillar 2 Multi-Timescale Spine ODE Bank + Pillar 3 Nonlinear Dendritic Expansion + Exact Batched Woodbury Synaptic Precision Plasticity, zero backprop, zero BPTT, zero replay buffer, zero task IDs) solves Streaming Delayed-Cue + Sensor Gain-Shock Split-MNIST (`0.2x` to `10x` gain shocks, 7-step image stream + 4-step noise delay, 5 sequential pairs `{0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}`) with >= 88.0% final 10-class accuracy over 20 seeds `[2000..2019]`, beating every individual-pillar ablation (`ablate_homeostasis`, `ablate_spine_traces`, `ablate_precision`), Static ACIL/RanPAC, Temporal-Pooled ACIL/RanPAC, GRU+Adam, GRU+EWC, and GRU+ER(200).

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric:** Final 10-class accuracy on the `10,000`-image held-out MNIST test set streamed temporally with random unseen per-episode sensor gains in `[0.2, 10.0]` after 4 distractor delay steps (higher is better).
- **Secondary Metrics:** Pair-0 (`{0,1}`) retention, Unified Block-Diagonal DMCN (`4` branches, `4x` smaller precision RAM) accuracy & VRAM, and individual-pillar ablation drops proving every pillar is necessary.
- **Falsifiers (Claim Refuted If):**
  1. Unified DMCN final 10-class accuracy `< 88.0%` over 20 seeds `[2000..2019]`, OR
  2. Pair-0 retention `< 90.0%`, OR
  3. Any single-pillar ablation (`ablate_homeostasis`, `ablate_spine_traces`, `ablate_precision`) matches or comes within `5.0 pp` of Unified DMCN (every pillar must carry measurable weight), OR
  4. Unified DMCN fails to beat Static ACIL/RanPAC, Temporal-Pooled ACIL/RanPAC, GRU+Adam, GRU+EWC, and GRU+ER(200) by `>= 5.0 pp`, OR
  5. Any null control (`Untrained`, `Blind`, `Shuffled Labels`, `Majority`) exceeds `20.0%`.

## Frozen Protocol
- Dataset: Official MNIST (`60,000` train / `10,000` test). Each 28x28 image is streamed over `T = 11` steps: `t = 0..6` present 4-row horizontal strips (`112` dims/step + `0.02` relative sensor noise), followed by `t = 7..10` blank/noisy distractor steps (`0.05` relative sensor noise, zero digit pixels). Classification occurs strictly at `t = 10`.
- Non-stationary sensor gain shocks: Training pairs 0..4 use gains `[1.0, 6.0, 0.2, 10.0, 0.4]` (50x range); held-out test episodes draw independent log-uniform unseen gains in `[0.2, 10.0]`.
- Strictly sequential 1-pass-per-pair stream at `batch_size = 256` on CUDA GPU. Zero replay / zero task IDs for DMCN (note: GRU+EWC receives task boundary markers and GRU+ER uses a 200-sequence reservoir replay buffer, strictly favoring those baselines).
- Validation tuning seed: `[999]` (`results/dmcn_unified_temporal_cil_v1/tuning_log.json`, disjoint validation pool `[980..999]`, max 5 runs).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_unified_temporal_cil_v1/results.json`).
