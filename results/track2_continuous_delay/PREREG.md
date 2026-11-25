# Pre-registration: track2_continuous_delay

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A single continuous-time state-space Legendre memory engine (order d=8, memory window theta=50.0 ms) reconstructs continuous past temporal signals u(t - tau) across unseen out-of-distribution stochastic and deterministic waveforms with mean NMSE < 0.20, achieving 12.5x state compression relative to a discrete tapped buffer.
- **Headline task metric (and direction):** Normalized Mean Squared Error (NMSE = MSE / Var(target)), lower is better. (Persistence baseline achieves NMSE ~ 1.9; trivial zero baseline achieves NMSE = 1.0).
- **Falsifier (result that would refute the claim):** Mean NMSE >= 0.25 on held-out continuous Ornstein-Uhlenbeck stochastic noise across 20 held-out test seeds, or any failure to significantly outperform the persistence baseline.

## Frozen protocol
- Data / environment / constants: Continuous state-space integration with dt = 0.5 ms, delay horizon tau = 15.0 ms, memory window theta = 50.0 ms, state dimension d = 8. Analytical state-space discretization via matrix exponential exp(A * dt).
- Inputs / sensors / features (and which are oracles): Continuous analog signal u(t) delivered sequentially at time t. ZERO oracles: no future values, no target labels, and no external clock ticks are accessible during state updates.
- Reward / loss / objective: Mean squared reconstruction error on u(t - tau) evaluated over post-warmup window [tau, T].
- Training budget: Zero-shot analytical shifted Legendre projection (0 training epochs, 0 free parameters tuned on test data).
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds.
  - Success criterion: Mean NMSE < 0.20 across the 20 seeds on stochastic continuous Ornstein-Uhlenbeck noise, and NMSE < 0.10 on deterministic broadband waveforms (PhaseShift, NovelFreqs, Chirp).
- Seeds: validation=[998], test=[200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Input): Input coupling vector B set to 0. Must yield NMSE >= 1.0 (verifying output requires sensory drive).
- Null Control 2 (Shuffled Input): Input temporal order randomly permuted. Must yield NMSE >= 1.0 (verifying genuine temporal causality).
- Baseline 1 (Persistence): Predicts current value u(t) (zero lag predictor). Expected NMSE ~ 1.5 - 2.5 on OU noise.
- Baseline 2 (Zero/Mean Baseline): Predicts signal mean. Expected NMSE = 1.0.
- Baseline 3 (Exact Tapped Delay Line): Buffer of 100 elements (12.5x larger memory), exact tap at index tau/dt. Known-answer reference: NMSE = 0.0.
- Baseline 4 (Parameter-Matched Tapped Delay Line): Buffer of 8 elements (matched to d=8 states).
- Baseline 5 (Classical AR(8) Linear Filter): 8-tap autoregressive filter.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory configurations on validation seed 998.
- Log file: results/track2_continuous_delay/tuning_log.json

## Reporting
- Results file with run metadata: results/track2_continuous_delay/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-34
