# Pre-registration: capn_universal_domain_agnostic

Written BEFORE any run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** Universal Domain-Agnostic CAPN Column operates identically across 3 disjoint heterogeneous modalities (Continuous 3D Sensorimotor Navigation, Chaotic Non-Linear Lorenz Attractor Tracking, and Streaming Multi-Channel Temporal Shift Detection) without backpropagation, domain-specific kinematic equations, or label leakage.
- **Headline task metric (and direction):** 
  1. Domain 1: 3D Sensorimotor Goal Success Rate (%) (higher is better).
  2. Domain 2: Chaotic Lorenz Non-linear Tracking NMSE (lower is better).
  3. Domain 3: False-Positive Spawns on Stationary & White Noise (strictly 0.0).
- **Falsifier (result that would refute the claim):**
  If Domain 1 Goal Success < 90%, Domain 2 Tracking NMSE >= 0.50, or Domain 3 produces >0 false spawns on white noise or stationary data, or any null control matches the system, the claim is refuted.

## Frozen protocol
- Data / environment / constants:
  - Domain 1: Continuous 3D Arena [0, 10]^3 with deceptive hazard sphere clusters, 14 rangefinder rays, stop-and-rebound physics, max_steps = 220.
  - Domain 2: Continuous chaotic Lorenz attractor (sigma=10, rho=28, beta=8/3, dt=0.01), 600 time steps.
  - Domain 3: 8-channel streaming continuous signal (300 steps stationary, 300 steps white noise, 600 steps systematic distribution shifts).
- Inputs / sensors / features (and which are oracles):
  - Domain 1: 14 rangefinder distances + 3D goal vector. Zero obstacle coordinate oracles.
  - Domain 2: Normalized 3D state vector (x, y, z) of the Lorenz system.
  - Domain 3: 8 continuous real-valued feature channels.
- Reward / loss / objective:
  - Domain 1: Reaching goal region (<1.0) with zero collisions.
  - Domain 2: Online one-step forward predictive error norm.
  - Domain 3: Information-theoretic residual error coherence rho_t.
- Training budget: Online sample-by-sample three-factor learning. Zero offline epochs.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 independent held-out seeds [800..819].
  - Domain 1: >= 95% collision-free goal reach.
  - Domain 2: NMSE < 0.20 and beats persistence baseline by at least 3x.
  - Domain 3: Exactly 0 false spawns on stationary and noise; >= 2 spawns on distribution shifts; STC protects critical connections.
- Seeds: test=[800..819] (n=20 independent seeds).

## Controls and baselines (run BEFORE the headline)
- Domain 1 Controls: Blind agent control (distances zeroed), reactive greedy control (no forward rollout).
- Domain 2 Controls: Decoupled surprise control (M(t)=1 constant), classical persistence baseline (predict last state).
- Domain 3 Baseline: Legacy periodic timer baseline (spawns every 50 steps regardless of data).

## Tuning budget
- Configurations allowed on validation data: 0 (parameters frozen).
- Log file: results/capn_universal_domain_agnostic/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_universal_domain_agnostic/results.json (use `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-44
