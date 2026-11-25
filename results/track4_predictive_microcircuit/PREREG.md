# Pre-registration: track4_predictive_microcircuit

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A canonical 3-layer cortical predictive microcircuit (Layer 4 sensory relay, Layer 2/3 prediction error comparator, Layer 5 continuous Legendre state-space generative predictor) autonomously suppresses predictable sensory prediction errors by >= 10x (Prediction Error Suppression Ratio PESR <= 0.10) via local predictive coding plasticity (dw = eta * eps * m) without backpropagation across 20 held-out seeds, while generating sharp error spikes upon unexpected sensory anomalies.
- **Headline task metric (and direction):**
  1. Primary Metric: Prediction Error Suppression Ratio (PESR = MSE_predictable / MSE_surprise in [0, 1]), lower is better. (Untrained and feedback-ablated controls achieve PESR >= 0.90).
  2. Secondary Metric: Predictable Regime Mean Squared Error (MSE_pred), lower is better.
  3. Surprise Regime Mean Squared Error (MSE_surprise), verifying sensory surprise response.
- **Falsifier (result that would refute the claim):** Mean PESR >= 0.15 across 20 held-out test seeds, or failure to significantly outperform the decoupled feedback control and classical AR(8) baseline.

## Frozen protocol
- Data / environment / constants: Continuous microcircuit simulation with dt = 0.5 ms, N = 600 steps. Layer 5 order d = 8, memory window theta = 40.0 ms. Local predictive coding learning rate eta = 0.05.
- Inputs / sensors / features (and which are oracles): Continuous analog stream y(t). Steps 0..400: predictable multi-harmonic wave. Step 400: unexpected sensory anomaly (abrupt frequency multiplication and phase jump). ZERO oracles: no transition cue, no future knowledge, no global error signals.
- Reward / loss / objective: Local prediction error epsilon(t) = y(t) - y_hat(t) minimized via local gradient-free dendritic shunting and predictive state alignment dw/dt = eta * epsilon(t) * m(t).
- Training budget: Online adaptation across 400 continuous time steps (200 ms).
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [400..419].
  - Predictable MSE evaluated on steps [200..400]; Surprise MSE evaluated on steps [400..450].
  - Success criteria: Mean PESR <= 0.10, Mean MSE_pred < 0.02, and PESR significantly lower than AR(8) baseline.
- Seeds: validation=[996], test=[400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Top-Down Feedback): L5 -> L2/3 feedback disconnected (y_hat = 0). Must yield PESR >= 0.90.
- Null Control 2 (Shuffled Temporal Context): L5 state scrambled temporally. Must yield PESR >= 1.0.
- Baseline 1 (Static Untrained Column): Fixed w = 0 (no prediction).
- Baseline 2 (Classical AR(8) Linear Filter): 8-tap autoregressive filter with online normalized LMS update.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 996.
- Log file: results/track4_predictive_microcircuit/tuning_log.json

## Reporting
- Results file with run metadata: results/track4_predictive_microcircuit/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-36
