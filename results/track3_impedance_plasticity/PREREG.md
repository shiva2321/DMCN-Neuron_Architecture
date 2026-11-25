# Pre-registration: track3_impedance_plasticity

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A gradient-free transmission-line synaptic array (K=8 synapses) coupled to a complex reactive dendritic membrane (Z0 = 1.0 - 0.5j) autonomously achieves theoretical conjugate impedance matching (Z0* = 1.0 + 0.5j) with Euclidean distance < 0.10 on coherent signal channels and achieves selective transmission contrast Delta_T = T_signal - T_noise >= 0.70 strictly via local wave resonance without backpropagation or external supervision across 20 held-out seeds.
- **Headline task metric (and direction):**
  1. Primary Metric: Selective Power Transmission Contrast Delta_T = mean(T_signal) - mean(T_noise) in [0, 1], higher is better. (Untrained and null controls achieve Delta_T <= 0.05).
  2. Secondary Metric: Somatic Signal-to-Noise Ratio (SNR = Power_soma(Signal) / Power_soma(Noise)), higher is better.
  3. Convergence Metric: Mean Euclidean distance to theoretical conjugate match ||Z_signal - Z0*||_2, lower is better (target < 0.10).
- **Falsifier (result that would refute the claim):** Mean Delta_T < 0.60 across 20 held-out test seeds, mean distance to theoretical conjugate match >= 0.15, or failure to outperform decoupled and shuffled feedback controls.

## Frozen protocol
- Data / environment / constants: Membrane characteristic impedance Z0 = 1.0 - 0.5j, conjugate target Z_match = 1.0 + 0.5j, shunt target Z_shunt = 0.02 + 0j. K = 8 synapses (4 coherent signal channels, 4 independent distractor noise channels). dt = 0.5 ms, n_steps = 300 per episode.
- Inputs / sensors / features (and which are oracles): Complex wave inputs u_j(t). ZERO oracles: no label indicating signal vs noise channels, no external error signals, no global backpropagation gradients.
- Reward / loss / objective: Unsupervised local coincidence between input wave u_j and backpropagating somatic wave V_soma.
- Training budget: 50 continuous unlabelled episodes per seed. Learning rate eta = 0.08.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [300..319].
  - Success criteria: Mean Delta_T >= 0.70, Signal distance to theoretical match < 0.10, and post-learning SNR > pre-learning SNR across all seeds.
- Seeds: validation=[997], test=[300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Somatic Feedback): Backpropagating wave V_soma set to 0. Must yield Delta_T <= 0.05.
- Null Control 2 (Shuffled Somatic Feedback): Somatic wave temporally permuted before coincidence calculation. Must yield Delta_T <= 0.05.
- Baseline 1 (Untrained / Initial Network): Initial randomly mismatched impedances. Expected Delta_T ~ 0.0.
- Baseline 2 (Classical Oja Hebbian Synapses): Standard scalar Hebbian plasticity rule with identical training data and budget.
- Baseline 3 (Uniform All-Pass): Fixed T = 1.0 for all channels (unselective transmission).

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory parameter runs on validation seed 997.
- Log file: results/track3_impedance_plasticity/tuning_log.json

## Reporting
- Results file with run metadata: results/track3_impedance_plasticity/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-35
