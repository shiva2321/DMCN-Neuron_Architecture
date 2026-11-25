# Pre-registration: track5_compositional_generalization

Written BEFORE any test run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A multi-column cortical network factorizing entity and relational representations across distinct cortical columns and binding them dynamically via Layer 2/3 horizontal consensus projections achieves >= 95% zero-shot retrieval accuracy on 32 held-out, never-co-occurring compositional pairs under input noise (sigma = 0.50) across 20 held-out test seeds, whereas un-factorized monolithic memory baselines achieve 0.0% accuracy on held-out compositions.
- **Headline task metric (and direction):**
  1. Primary Metric: Zero-Shot Compositional Generalization Accuracy (Held-out Test Set) in [0, 100%], higher is better.
  2. Secondary Metric: Noise Robustness (Cosine Similarity under sigma = 0.50 input noise) in [-1, 1], higher is better.
- **Falsifier (result that would refute the claim):** Mean zero-shot compositional test accuracy < 85.0% across 20 held-out seeds, or failure to significantly outperform decoupled horizontal control and monolithic memory baseline.

## Frozen protocol
- Data / environment / constants:
  - 16 entities ($E \in \mathbb{R}^{16 \times D}$) and 8 relations ($R \in \mathbb{R}^{8 \times D}$), dimension $D = 256$, with orthogonal/quasi-orthogonal random bipolar representations.
  - Total compositional universe: $16 \times 8 = 128$ distinct pairs $(E_i, R_j)$.
  - Split: 96 pairs (75%) training, 32 pairs (25%) held-out test. All entities and relations appear in training, but the 32 test combinations are strictly disjoint and never co-occur in training.
  - Evaluation noise: Gaussian noise $\mathcal{N}(0, \sigma^2)$ with $\sigma = 0.50$ added to test query inputs.
- Inputs / sensors / features (and which are oracles):
  - Sensory input to Column 1: noisy entity vector $E_i + \xi$.
  - Sensory input to Column 2: noisy relation vector $R_j + \xi$.
  - ZERO oracles: no label leakage, no joint key provided to the network, no test instances seen during training.
- Reward / loss / objective:
  - Local column auto-associative / predictive memory in Layer 5.
  - Layer 2/3 horizontal reciprocal consensus binding between Column 1 (Entity) and Column 2 (Relation) to synthesize target compositional feature representation $Y_{i,j} = \text{bind}(E_i, R_j)$ via circular convolution / HRR in the active data path.
- Training budget:
  - Exposure to the 96 training pairs to form internal column priors and synaptic weights.
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
  - 20 disjoint held-out test seeds [500..519].
  - Evaluated on all 32 held-out combinations per seed.
  - Success criterion: Mean held-out zero-shot compositional accuracy >= 95.0%.
- Seeds: validation=[995], test=[500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519] (20 disjoint seeds).

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Decoupled Horizontal Consensus): Cross-column Layer 2/3 communication severed. Must yield near 0.0% accuracy on compositional retrieval.
- Null Control 2 (Shuffled Inter-Column Projections): Layer 2/3 horizontal consensus projections randomly permuted. Must yield near 0.0% accuracy.
- Baseline 1 (Monolithic Flat Memory): Standard flat associative memory of equivalent capacity trained on the same 96 training pairs. Lacks modular factorization; when queried with held-out unseen combination $(E_i, R_j)$, achieves 0.0% accuracy.
- Baseline 2 (Nearest-Neighbor Flat Memory): Flat exemplar lookup directly on raw inputs; scores 0.0% on unseen combinations.

## Tuning budget
- Configurations allowed on validation data: <= 5 exploratory runs on validation seed 995.
- Log file: results/track5_compositional_generalization/tuning_log.json

## Reporting
- Results file with run metadata: results/track5_compositional_generalization/results.json (written via `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> B-37
