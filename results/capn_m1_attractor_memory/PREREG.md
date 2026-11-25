# Pre-registration: capn_m1_attractor_memory

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A continuous non-linear attractor network with compartmental dendritic saturation and Tsodyks-Markram short-term plasticity (STP) maintains stable multi-item working memory under continuous sensory noise (sigma = 0.40), achieving retrieval accuracy >= 95.0% and post-relaxation contrast margin SNR >= 8.0 dB across 20 held-out test seeds without requiring external codebook lookups, whereas linear recurrent networks collapse (accuracy < 70.0%, SNR < 2.0 dB) and STP-ablated networks degrade under continuous noise.
- **Headline task metric (and direction):**
  1. Primary Metric: Working Memory Pattern Retrieval Accuracy after stimulus offset (%) in [0, 100%], higher is better.
  2. Secondary Metric: Post-Relaxation Contrast Margin SNR (dB) after sensory noise injection, higher is better.
- **Falsifier (result that would refute the claim):** Mean retrieval accuracy < 85.0% or post-relaxation Contrast Margin SNR < 8.0 dB across 20 held-out test seeds, or failure to significantly outperform linear and STP-ablated controls.

## Frozen protocol
- Data / environment / constants:
  - Network state dimension D = 128.
  - Number of distinct canonical attractor patterns K = 8.
  - Continuous simulation time T = 300.0 ms, integration time-step dt = 0.5 ms (600 time-steps).
  - Membrane time constant tau_m = 20.0 ms.
  - Tsodyks-Markram STP parameters: baseline release probability U0 = 0.20, facilitation time constant tau_f = 1500.0 ms, depression time constant tau_d = 50.0 ms.
  - Non-linear dendritic transfer function: phi(x) = tanh(x * 1.8).
- Inputs / sensors / features (and which are oracles):
  - Presentation Phase (t in [0, 50] ms): Sensory cue for pattern k injected with input gain g_in = 2.0.
  - Delay & Noise Phase (t in [50, 300] ms): Sensory cue turned OFF; continuous Gaussian sensory channel noise xi(t) ~ N(0, sigma^2) with sigma = 0.40 injected at every time-step.
  - ZERO oracles: No teacher signal during the delay, no label cues, no external dictionary lookup.
- Reward / loss / objective:
  - Dynamical energy relaxation down the intrinsic recurrent attractor gradient without external intervention.
- Training budget:
  - Structural attractor weights W_rec initialized via outer-product attractor matrix: W_rec = (g_rec / D) * (P^T @ P) with g_rec = 3.5.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [600..619].
  - Tested across all K=8 attractor patterns per seed (160 total working memory evaluations).
  - Success criterion: Mean pattern retrieval accuracy >= 95.0% and Mean Contrast Margin SNR >= 8.0 dB.
- Seeds: validation=[994], test=[600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Linear Recurrent Network): phi(x) = x (linear activation; tests necessity of non-linear dendritic saturation).
- Null Control 2 (STP-Ablated Attractor): Facilitation/depression disabled (u = U0, x = 1 frozen; tests necessity of fast local dynamic memory).
- Baseline 1 (Classical Continuous Hopfield Network): Standard continuous Hopfield without short-term synaptic dynamics.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 994.
- Log file: results/capn_m1_attractor_memory/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_m1_attractor_memory/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-38
