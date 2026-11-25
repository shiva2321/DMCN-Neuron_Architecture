# Pre-registration: crfn_b47_weight_free_symbolic

Written BEFORE any run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A Weight-Free Continuous Resonant Field Neuron (CRFN) column achieves genuine neural symbolic reasoning (relational analogy mapping, multi-hop transitive deduction, and continuous reversible arithmetic) using only continuous phasor wave interference, transmission-line impedance matching, autonomous Stuart-Landau limit-cycle noise dissipation, and direct phase demodulation—without weight matrices (W in R^{DxD}), external dictionary lookups (ItemTable), or softmax attention.
- **Headline task metrics (and direction):**
  1. Relational Analogy Phase Coherence / Accuracy (%) across 20 held-out seeds under sensory noise sigma=0.25 (higher is better).
  2. Multi-Hop Transitive Deduction Accuracy (%) across 20 held-out seeds under sensory noise sigma=0.25 (higher is better).
  3. Reversible Wavefield Arithmetic Accuracy (%) via continuous phase demodulation under noise sigma=0.15 (higher is better).
- **Falsifier (result that would refute the claim):**
  If Relational Analogy Accuracy < 90%, Transitive Deduction Accuracy < 85%, or Continuous Arithmetic Accuracy < 90% across 20 held-out seeds, or if any null control (decoupled, shuffled, blind) matches the system, the claim is refuted.

## Frozen protocol
- Data / environment / constants:
  - Vector dimension D = 256.
  - Stuart-Landau soma: mu = 1.0, dt = 0.02, steps = 50 (Lyapunov radial damping rate lambda = -2.0).
  - Impedance transmission line: Z0 = 1.0, reflection coefficient Gamma = (Z - Z0) / (Z + Z0), power transmission T = 1 - |Gamma|^2.
  - Relational Analogy: 16 distributed phase entities, 8 primitive relation operators. 32 test analogy queries per seed under noise sigma = 0.25.
  - Transitive Deduction: 20 test 2-hop relational chains under noise sigma = 0.25.
  - Continuous Arithmetic: 40 queries (20 additions, 20 inversions) on integers [0, 50], frequencies omega_k in [0.01, 0.10], decoded via direct linear phase demodulation without dictionary lookup. Noise sigma = 0.15.
- Inputs / sensors / features (and which are oracles):
  - Continuous complex waveforms Psi in C^D. Zero label oracles. Zero dictionary lookups.
- Reward / loss / objective:
  - Pure dynamical wave interference and non-linear limit-cycle dissipation. Zero backpropagation.
- Training budget:
  - Zero matrix training. Synaptic impedances matched via maximum power transfer.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 independent held-out seeds [920..939].
  - Analogy accuracy >= 95.0% across all 20 seeds.
  - Transitive accuracy >= 90.0% across all 20 seeds.
  - Arithmetic accuracy >= 95.0% across all 20 seeds.
- Seeds: validation=[860..864]  test=[920..939] (n=20 independent held-out seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1: Decoupled Inter-Column Consensus (transmission lines severed, T = 0).
- Null Control 2: Shuffled Phase Control (permutation destroys algebraic covariance).
- Null Control 3: Blind Control (sensory inputs zeroed).
- Null Control 4: Ablated Soma Control (Stuart-Landau non-linear limit cycle ablated).
- Baseline 1: Monolithic Flat Memory Analogy Baseline.
- Baseline 2: Uniform Random Classifier Baseline.

## Tuning budget
- Configurations allowed on validation data: 0 (parameters frozen).
- Log file: results/crfn_b47_weight_free_symbolic/tuning_log.json

## Reporting
- Results file with run metadata: results/crfn_b47_weight_free_symbolic/results.json (use `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-47
