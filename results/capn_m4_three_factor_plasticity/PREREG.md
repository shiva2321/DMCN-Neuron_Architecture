# Pre-registration: capn_m4_three_factor_plasticity

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A multi-layer gradient-free three-factor predictive plasticity engine ($\Delta W_{ij} = \eta \cdot x_j \cdot \epsilon_i \cdot M(t)$) converges on non-linear multi-stage hierarchical associative regression without backpropagation, achieving mean normalized test error (NMSE) < 0.10 and robust retention under noise drift (retained NMSE < 0.25) across 20 held-out test seeds, matching or outperforming standard backpropagation while decoupled neuromodulator and shuffled error controls suffer catastrophic drift (NMSE > 0.80).
- **Headline task metric (and direction):**
  1. Primary Metric: Test Normalized Mean-Squared Error (NMSE = MSE / Var(y)) after noise exposure, lower is better.
  2. Secondary Metric: Pre-drift convergence NMSE, lower is better.
- **Falsifier (result that would refute the claim):** Mean post-drift test NMSE >= 0.35 across 20 held-out test seeds, or failure to significantly outperform decoupled and shuffled null controls.

## Frozen protocol
- Data / environment / constants:
  - Input dimension D_in = 8, hidden dimension D_hid = 16, output dimension D_out = 4.
  - True non-linear data-generating process: h_true = tanh(W_1^* x), y = W_2^* h_true + 0.15 * sin(2.0 * W_2^* h_true).
  - Training dataset size N_train = 200 samples, test dataset size N_test = 100 samples (strictly disjoint).
  - Additive sensor noise sigma_noise = 0.05.
- Inputs / sensors / features (and which are oracles):
  - Inputs x generated from standard normal N(0, I).
  - Targets y provided only during learning episodes as supervisory drive.
  - ZERO oracles: No backward automatic differentiation tape, no global backpropagation Jacobians, no shared weight transpose W^T.
- Reward / loss / objective:
  - Objective: Minimize output prediction error epsilon_y = y - y_hat.
  - Three-factor synaptic update: Delta W = eta * Pre * Local_Error * M(t).
  - Neuromodulator surprise gate: M(t) = tanh(||epsilon_y||^2 / 0.5) during salient learning, clamped to 0.05 during non-salient noise drift.
- Training budget:
  - Phase 1: 50 training epochs on true task, learning rate eta = 0.03.
  - Phase 2: 500 drift steps under uninformative zero-mean noise fluctuations.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [660..679].
  - Success criterion: Mean post-drift test NMSE < 0.25.
- Seeds: validation=[991], test=[660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Neuromodulator): M(t) = 1.0 constant throughout both learning and drift, unmodulated plasticity.
- Null Control 2 (Shuffled Error Feedback): Error signals permuted across neurons, destroying local credit assignment.
- Baseline (Standard Backpropagation): Multi-layer perceptron trained with gradient descent on identical architecture and data.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 991.
- Log file: results/capn_m4_three_factor_plasticity/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m4_three_factor_plasticity/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-41
