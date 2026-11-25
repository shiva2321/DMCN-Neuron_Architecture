# Pre-registration: capn_m2_attention_routing

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A dual-column cortical routing engine combining Communication-Through-Coherence (CTC) oscillatory phase-locking with top-down apical dendritic shunting dynamically routes concurrent task streams with cross-talk suppression >= 20.0 dB and task-switching latency <= 15.0 ms across 20 held-out test seeds, whereas random phase controls (cross-talk < 3.0 dB) and un-shunted controls fail task isolation.
- **Headline task metric (and direction):**
  1. Primary Metric: Cross-Talk Suppression Ratio (dB) between concurrent task streams, higher is better.
  2. Secondary Metric: Dynamic Task-Switching Latency (ms) following contextual task inversion, lower is better.
- **Falsifier (result that would refute the claim):** Mean cross-talk suppression < 16.0 dB or mean switching latency > 20.0 ms across 20 held-out test seeds, or failure to significantly outperform random phase and un-shunted controls.

## Frozen protocol
- Data / environment / constants:
  - Input dimension D = 64 per stream.
  - Two concurrent continuous analog streams: Stream 1 (high-frequency carrier omega_1 = 0.25 rad/ms) and Stream 2 (low-frequency carrier omega_2 = 0.08 rad/ms).
  - Total simulation time T = 300.0 ms, integration time-step dt = 0.5 ms (600 time-steps).
  - Intrinsic gamma rhythm omega_gamma = 40.0 Hz (period ~ 25.0 ms).
  - Task switch event at t = 150.0 ms: top-down context vector c_task flips from [1.0, 0.0] (Focus Stream 1) to [0.0, 1.0] (Focus Stream 2).
- Inputs / sensors / features (and which are oracles):
  - Stream 1 and Stream 2 simultaneously presented to input dendritic tree.
  - Top-down task context c_task presented to apical arbor.
  - ZERO oracles: No post-hoc output selection; routing must occur dynamically through physical phase-locking and dendritic shunting.
- Reward / loss / objective:
  - Intrinsic phase synchronization via Kuramoto coupling and apical inhibitory gating.
- Training budget:
  - Direct biophysical phase and shunting execution without backpropagation.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [620..639].
  - Evaluated on both phases (pre-switch t in [50, 150] ms and post-switch t in [200, 300] ms).
  - Success criterion: Mean cross-talk suppression >= 20.0 dB and Mean switching latency <= 15.0 ms.
- Seeds: validation=[993], test=[620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Phase Control): Random asynchronous phase offsets between columns (ablation of CTC phase-locking).
- Null Control 2 (Apical Shunt Ablation): Gating interneurons disabled (broadcast transmission without apical shunting).
- Baseline 1 (Static Linear Weight Router): Fixed feedforward linear weights without phase or shunting dynamics.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 993.
- Log file: results/capn_m2_attention_routing/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m2_attention_routing/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-39
