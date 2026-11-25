# Pre-registration: crfn_b48_causal_generalization

Written BEFORE any run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** Continuous Resonant Field Column demonstrates genuine causal understanding and generalization across 3 critical empirical tests: (1) zero-shot accuracy on the disjoint compositional complement where monolithic statistical memory fails; (2) counterfactual trap avoidance via forward latent rollouts where greedy surface statistics fail; and (3) rapid online impedance re-matching (k <= 15 steps) under mid-stream causal physics inversion.
- **Headline task metrics (and direction):**
  1. Disjoint Compositional Complement Accuracy (%) under noise sigma=0.25 (higher is better).
  2. Counterfactual Deceptive Trap Avoidance Rate (%) across 20 held-out seeds (higher is better).
  3. Causal Inversion Adaptation Latency (steps to restore power transmission T >= 0.85) (lower is better).
- **Falsifier (result that would refute the claim):**
  If Disjoint Compositional Complement Accuracy < 90%, Counterfactual Trap Avoidance < 85%, or Causal Adaptation Latency > 20 steps across 20 held-out seeds, or if any null control matches the system, the claim is refuted.

## Frozen protocol
- Data / environment / constants:
  - Vector dimension D = 256.
  - Task 1: 16 entities, 8 relations. Total 128 pairs. Disjoint split: 90 training pairs (70%), 38 held-out complement pairs (30%). Test strictly on the 38 unseen complement pairs under phase noise sigma = 0.25.
  - Task 2: Continuous 2D arena with deceptive lure: immediate cue suggests lure direction (+1.0 reward), but leads to collision trap in 3 steps; bypass direction (+0.3 reward) avoids collision. Evaluated over 20 trials per seed.
  - Task 3: Continuous transmission line with resonant frequency omega = 1.0. At t=100, environment impedance inverts (Z_env -> Z_env * exp(i*pi)). Local adaptation rate eta = 0.15, alpha = 0.02.
- Inputs / sensors / features (and which are oracles):
  - Continuous complex waveforms in C^D. Zero label oracles.
- Reward / loss / objective:
  - Gradient-free forward wave simulation, Lyapunov limit-cycle clean-up, and local impedance matching.
- Training budget:
  - Online sample-by-sample local adaptation. Zero backpropagation.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 independent held-out seeds [940..959].
  - Disjoint complement accuracy >= 95.0% across all 20 seeds.
  - Counterfactual trap avoidance rate >= 90.0% across all 20 seeds.
  - Causal inversion recovery latency <= 15 steps across all 20 seeds.
- Seeds: validation=[870..874]  test=[940..959] (n=20 independent held-out seeds).

## Controls and baselines (run BEFORE the headline)
- Task 1 Baseline: Monolithic Flat Memory Baseline (memorizes training pairs as concatenated keys).
- Task 1 Controls: Decoupled transmission, Shuffled phase, Blind.
- Task 2 Baseline: Greedy Surface-Statistical Agent (maximizes immediate sensory correlation).
- Task 2 Controls: Scrambled forward dynamics (destroys rollout temporal causality), Blind.
- Task 3 Baseline: Static Connectionist Baseline (fixed non-adaptive weights, eta = 0).
- Task 3 Controls: Decoupled plasticity.

## Tuning budget
- Configurations allowed on validation data: 0 (parameters frozen).
- Log file: results/crfn_b48_causal_generalization/tuning_log.json

## Reporting
- Results file with run metadata: results/crfn_b48_causal_generalization/results.json (use `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-48
