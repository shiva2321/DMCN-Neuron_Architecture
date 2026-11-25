# Pre-registration: capn_m6_grand_benchmark

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** The integrated CAPN column architecture with universal continuous modality encoders (temporal Legendre state-space, 3D spherical receptive fields, and motor CPGs) achieves robust closed-loop continuous sensorimotor navigation in deceptive 3D obstacle arenas without backpropagation, goal beacon oracles, or wall-sliding physics, reaching >= 90% goal success across 20 held-out test seeds, whereas blind agents (0%) and reactive greedy controllers (0%) fail.
- **Headline task metric (and direction):**
  1. Primary Metric: Goal Success Rate (% of episodes successfully reaching goal without collision), higher is better.
  2. Secondary Metric: Path Efficiency Ratio (shortest distance / actual trajectory length), higher is better.
- **Falsifier (result that would refute the claim):** Goal success rate < 80% across 20 held-out test seeds, or blind control achieving success within 50% of the CAPN agent.

## Frozen protocol
- Data / environment / constants:
  - Continuous 3D arena (10.0 x 10.0 x 10.0) with deceptive 3D hazard spheres blocking direct start-to-goal trajectory.
  - True physics: Realistic stop-and-rebound collision dynamics (zero wall-sliding cheat: forward speed is zeroed upon obstacle contact).
  - Episode horizon: T_max = 180 steps, dt = 0.1s.
  - Start position: (1.5, 1.5, 1.5). Goal region: radius 1.0 around (8.5, 8.5, 8.5).
- Inputs / sensors / features (and which are oracles):
  - Modality 1: 14 continuous range-finder proximity rays spanning 3D unit sphere.
  - Modality 2: Continuous 3D ego-motion velocity (vx, vy, vz) projected into orthogonal Legendre temporal basis.
  - Modality 3: Goal relative direction in ego-coordinates (unit vector dir_g, distance d_g) - ONLY available at start or as coarse gradient, never exact waypoints.
  - Modality 4: Motor Central Pattern Generator (CPG) mapping continuous cortical 3D velocity intentions to smooth physical commands.
  - ZERO oracles: No goal beacon on proximity sensors, no waypoints, no wall-sliding physics, no label conditioning.
- Reward / loss / objective:
  - Task objective: Reach goal while maintaining safety clearance from 3D hazard spheres.
  - Cortical objective: CAPN latent 3D world model rolls out prospective intentions and prunes branches leading to predicted collisions.
- Training budget:
  - Online closed-loop execution across 20 independent test seeds.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [700..719].
  - Success criterion: CAPN Goal Success Rate >= 90.0% across 20 test seeds.
- Seeds: validation=[995], test=[700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Blind Agent Control): Intact CAPN agent with range-finder sensors zeroed (proves the agent relies on sensory inputs).
- Null Control 2 (Reactive Greedy Control): Agent moving directly toward the goal without counterfactual lookahead (proves the deceptive arena requires predictive imagination).
- Baseline (Model-Free Random Walk Baseline): Random exploration in continuous velocity space.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 995.
- Log file: results/capn_m6_grand_benchmark/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m6_grand_benchmark/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-43
