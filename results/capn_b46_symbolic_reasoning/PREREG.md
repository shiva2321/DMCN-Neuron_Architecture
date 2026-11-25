# Pre-registration: capn_b46_symbolic_reasoning

Written BEFORE any run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** Genuine continuous symbolic and compositional reasoning (relational analogy mapping, multi-hop transitive deduction, and reversible wavefield arithmetic) emerges from gradient-free circular convolution algebra, continuous phasor wave interference, and Layer 5 continuous Lyapunov attractor relaxation across cortical columns without string parsing, answer branching, or algorithmic carry loops.
- **Headline task metrics (and direction):**
  1. Relational Analogy Accuracy (%) on held-out unseen quadruplets under sensory noise (higher is better).
  2. Multi-Hop Transitive Deduction Accuracy (%) on held-out relational chains under sensory noise (higher is better).
  3. Reversible Continuous Wavefield Arithmetic Accuracy (%) on held-out integer additions and inversions under phase noise (higher is better).
- **Falsifier (result that would refute the claim):**
  If held-out Relational Analogy Accuracy < 90%, Transitive Deduction Accuracy < 85%, or Continuous Arithmetic Accuracy < 90% across 20 held-out seeds, or if any null control (decoupled, shuffled, blind) matches the system (p >= 0.01), the claim is refuted.

## Frozen protocol
- Data / environment / constants:
  - Vector dimension D = 512.
  - Relational Analogy: 16 discrete entity prototypes, 8 primitive relation operators. 32 held-out test analogy problems A:B::C:? per seed under additive Gaussian sensory noise sigma = 0.25.
  - Transitive Deduction: 3-hop relational DAG chains (A -> B -> C -> D). Evaluation on composite multi-hop queries A o (R1 * R2) under noise sigma = 0.25.
  - Reversible Arithmetic: continuous phasor wavefield encoding over integers [0, 50] with sum <= 50. Geometric log-cochlear frequencies omega_k = omega_0 * beta^k. Evaluation on 40 held-out unseen integer pairs (20 forward additions, 20 backward inversions) under phase noise sigma = 0.20.
  - Layer 5 Attractor: Lyapunov energy relaxation with T_settle = 10 steps, eta_attr = 0.20.
- Inputs / sensors / features (and which are oracles):
  - Noisy sensory vectors x in R^D.
  - Zero label oracles. Zero string manipulation (no zfill, split, str()). Zero if-else branching on target labels.
- Reward / loss / objective:
  - Gradient-free exact energy relaxation and algebraic wave interference.
- Training budget:
  - Stored prototype attractor basins in L5 (Hopfield/Hebbian outer-product). Zero backpropagation or gradient steps.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 independent held-out seeds [900..919].
  - Analogy accuracy >= 95.0% across all 20 seeds.
  - Transitive deduction accuracy >= 90.0% across all 20 seeds.
  - Arithmetic addition & inversion accuracy >= 95.0% across all 20 seeds.
- Seeds: validation=[850..854]  test=[900..919] (n=20 independent held-out seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1: Decoupled Inter-Column Consensus (L2/3 horizontal projections severed).
- Null Control 2: Shuffled / Scrambled Phase Control (algebraic binding covariance destroyed via random permutation).
- Null Control 3: Blind Control (sensory inputs zeroed).
- Null Control 4: Unsettled Feedforward Control (L5 Lyapunov attractor relaxation ablated; raw noisy unbinding used directly).
- Baseline 1: Monolithic Flat Memory Baseline (memorizes training instances as concatenated flat keys [A; R]).
- Baseline 2: Uniform Random Classifier Baseline (chance rate = 1/N).

## Tuning budget
- Configurations allowed on validation data: 0 (parameters frozen).
- Log file: results/capn_b46_symbolic_reasoning/tuning_log.json

## Reporting
- Results file with run metadata: results/capn_b46_symbolic_reasoning/results.json (use `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-46
