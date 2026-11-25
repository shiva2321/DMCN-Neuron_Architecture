# Pre-registration: capn_m3_latent_world_model

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A factorized latent world model z = [z_map, z_entity, z_dyn, u] running forward mental rollouts with sensory gating internally evaluates and prunes deceptive spatial paths prior to physical action, achieving >= 95.0% zero-shot trap avoidance in continuous deceptive environments across 20 held-out test seeds, whereas reactive agents without mental rollout fail (avoidance < 15.0%).
- **Headline task metric (and direction):**
  1. Primary Metric: Deceptive Trap Avoidance Rate (%) in [0, 100%], higher is better.
  2. Secondary Metric: Internal Counterfactual Pruning Margin (difference in evaluated latent energy between winning detour and pruned trap), higher is better.
- **Falsifier (result that would refute the claim):** Mean trap avoidance < 85.0% across 20 held-out test seeds, or failure to significantly outperform the reactive agent baseline.

## Frozen protocol
- Data / environment / constants:
  - Continuous 2D arena [0, 10] x [0, 10].
  - Deceptive barrier: A concave barrier positioned directly between starting location s_0 = (2.0, 5.0) and goal g = (8.0, 5.0).
  - Direct straight-line motion enters a deep hazard trap (penalty E_hazard = 100.0).
  - Upper detour (north corridor) and lower detour (south corridor) provide clear collision-free routes to the goal.
  - Dimension: D_map = 32 (radial basis / place field features), D_entity = 16, D_dyn = 8, D_unc = 8.
  - Internal rollout horizon K_rollout = 20 steps, integration dt = 0.5.
- Inputs / sensors / features (and which are oracles):
  - Agent observes current location and map features at t = 0.
  - Sensory gating closes: The agent runs 3 candidate action intents (Direct, North Detour, South Detour) in latent space.
  - Internal rollout evolves z_hat_{k+1} = z_hat_k + dt * F(z_hat_k, a_hyp).
  - ZERO oracles: No path teacher, no waypoints, no external collision feedback during rollout.
- Reward / loss / objective:
  - Selection of candidate action trajectory with minimal cumulative simulated latent energy C(a) = sum_k E(z_hat_k).
- Training budget:
  - Forward transition operator F learned from local exploratory state transitions.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [640..659] with varying barrier geometries and goal jitters.
  - Success criterion: Mean trap avoidance >= 95.0%.
- Seeds: validation=[992], test=[640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Reactive Agent): Greedy gradient policy moving directly toward the goal beacon without forward mental rollout.
- Null Control 2 (Scrambled World Model): Latent transition dynamics F shuffled, destroying causal predictive structure.
- Baseline 1 (Direct Model-Free Policy).

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 992.
- Log file: results/capn_m3_latent_world_model/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m3_latent_world_model/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-40
