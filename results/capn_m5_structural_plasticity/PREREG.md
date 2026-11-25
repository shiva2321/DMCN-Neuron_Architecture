# Pre-registration: capn_m5_structural_plasticity

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** Information-theoretic structural plasticity governed by residual error coherence and variance expands network capacity proportionally under systematic distribution shifts (spawning units for new distribution modes) while maintaining strictly zero false-positive growth ($N_{\text{spawn}} = 0$) on stationary and irreducible noise streams across 20 held-out test seeds, whereas legacy timer-based baselines spuriously grow regardless of data.
- **Headline task metric (and direction):**
  1. Primary Metric: False-positive neurogenesis on stationary/noise data ($N_{\text{spawn}}$), target = 0, lower is better.
  2. Secondary Metric: Shift adaptation growth rate (% of new distribution modes allocated), higher is better.
- **Falsifier (result that would refute the claim):** Any false-positive neuron spawn on stationary or white noise streams ($N_{\text{spawn}} > 0$) across 20 held-out test seeds, or failure to spawn new units under systematic distribution shifts.

## Frozen protocol
- Data / environment / constants:
  - Input dimension D_in = 8, output dimension D_out = 4.
  - Stream 1 (Stationary): 300 samples from fixed stationary Gaussian mixture distribution.
  - Stream 2 (White Noise): 300 samples of irreducible white Gaussian noise.
  - Stream 3 (Distribution Shift): 600 samples comprising 3 sequential distinct generative clusters (introducing new concepts at t = 200 and t = 400).
- Inputs / sensors / features (and which are oracles):
  - Online continuous stream processed one sample at a time.
  - ZERO oracles: No task timers, no step counters, no epoch indicators, no change-point labels.
- Reward / loss / objective:
  - Residual error variance: sigma_eps^2 = MovingAverage(||eps||^2, window=25).
  - Residual error directional coherence: rho = <eps(t), eps(t-1)> / (||eps(t)|| * ||eps(t-1)|| + 1e-8).
  - Neurogenesis trigger: Spawn iff (sigma_eps^2 > theta_growth=0.30) AND (rho > theta_rho=0.28) consecutively for 7 steps, with biological refractory period = 40 steps.
  - Metabolic synaptic pruning: Synaptic weights with |w_ij| < 0.005 and |grad| < 0.001 decay to zero and are pruned.
- Training budget:
  - Online learning, eta = 0.03.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [680..699].
  - Success criterion: Stationary N_spawn == 0, Noise N_spawn == 0, Shift N_spawn >= 1 (mean in [2, 5]) across 20 test seeds.
- Seeds: validation=[990], test=[680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Stationary Stream Control): Validates zero false-positive growth under stationary distribution.
- Null Control 2 (Shuffled Noise Control): Validates zero false-positive growth under irreducible high-variance noise.
- Baseline (Legacy Timer-Based Neurogenesis): Periodic timer spawning a neuron every 50 steps regardless of data.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 990.
- Log file: results/capn_m5_structural_plasticity/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m5_structural_plasticity/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-42
