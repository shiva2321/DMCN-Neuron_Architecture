# Pre-registration: track1_dcaap_xor

Written BEFORE running final frozen evaluations. Changing anything below after seeing test results constitutes a NEW experiment ID (GEMINI.md §2.2).

- **Claim (one sentence):** A single biophysical two-compartment neuron with voltage-gated calcium excitation and delayed-rectifier potassium shunting in the apical dendrite (Gidon et al., Science 2020) solves continuous-time non-linear XOR without label conditioning, and this capability strictly collapses under potassium-shunting ablation, passive-cable ablation, or linear reduction.
- **Headline task metric (and direction):** XOR classification accuracy (%) over continuous analog input currents across 20 held-out random test seeds (higher is better; target 100.0%).
- **Falsifier (result that would refute the claim):** Any of the following:
  1. The intact biophysical dCaAP neuron achieves < 95.0% accuracy on held-out test seeds with input amplitude noise (+/-15%) and arrival timing jitter (+/-5 ms).
  2. The ablated null control without potassium shunting ($g_K = 0$) achieves >= 90.0% accuracy (which would prove the non-monotonic shunting mechanism is decoration rather than causal).
  3. The passive dendrite null control ($g_{Ca} = g_K = 0$) achieves >= 90.0% accuracy.
  4. The standard tanh MLP baseline fails to achieve 100.0% on clean XOR (which would indicate a straw-man baseline).

## Frozen protocol
- **Data / environment / constants:**
  - 4 canonical XOR patterns: `(0, 0)`, `(1, 0)`, `(0, 1)`, `(1, 1)`.
  - Continuous time simulation duration: $T = 100.0$ ms, time step $dt = 0.05$ ms (2000 steps).
  - Synaptic inputs delivered as alpha-function conductance pulses: $I_{syn}(t) = \sum_i w_i \cdot \frac{t - t_i}{\tau_s} \exp\left(1 - \frac{t - t_i}{\tau_s}\right) \Theta(t - t_i)$.
  - Base synaptic amplitude $I_0 = 1.0$ nA, nominal arrival time $t_0 = 15.0$ ms, $\tau_s = 2.0$ ms.
- **Inputs / sensors / features (and which are oracles):**
  - Inputs are continuous analog currents injected into the dendritic compartment.
  - Oracles: NONE. Stimulus generation code paths are identical for all patterns. ZERO branching on label.
- **Reward / loss / objective:**
  - Somatic spike count $N_{spike}$ within $[0, T]$: Binary classification readout $\hat{y} = 1$ if $N_{spike} \ge 1$ else $0$.
  - Binary Cross-Entropy / Accuracy compared against XOR ground truth $y \in \{0, 1\}$.
- **Training budget:**
  - Fixed biophysical conductances (zero backpropagation needed for the core Gidon mechanism).
  - Validation tuning budget: Maximum 5 parameter configurations logged in `tuning_log.json` on validation seed 999.
- **Evaluation size and success criterion:**
  - 20 held-out test seeds: `[100, 101, 102, ..., 119]`.
  - Each seed tests all 4 patterns under independent analog current amplitude noise ($\mathcal{U}(-5.0\%, +5.0\%)$) and independent arrival timing jitter ($\mathcal{U}(-1.5\text{ ms}, +1.5\text{ ms})$).
  - Absolute pass criterion: Intact model mean accuracy $\ge 95.0\%$ across all 20 seeds with 95% Wilson/Student CI.
- **Seeds:**
  - Validation seed: `[999]`
  - Held-out test seeds: `[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]` (20 seeds, strictly disjoint).

## Controls and baselines (run BEFORE the headline)
- **Null Control 1 (Kv Outward Shunt Ablation):** $g_{K,d} = 0.0$. Supra-threshold input cannot be shunted; must fire on (1,1) and fail XOR ($\le 75.0\%$).
- **Null Control 2 (Passive Dendrite):** $g_{Ca,d} = 0.0, g_{K,d} = 0.0$. Linear cable equation; cannot solve non-linearly separable problem ($\le 75.0\%$).
- **Null Control 3 (Single-Layer Linear Perceptron):** Linear threshold unit on input currents. Proven theoretically to achieve at most 75.0% on XOR.
- **Baseline 4 (Tuned 2-Layer Tanh MLP):** Architecture: 2 inputs -> 4 hidden units (tanh) -> 1 output unit (sigmoid). Trained with Adam on XOR. Must reach 100.0% across test seeds to confirm baseline validity.

## Tuning budget
- Configurations allowed on validation data: 5
- Log file: `results/track1_dcaap_xor/tuning_log.json`

## Reporting
- Results file with run metadata: `results/track1_dcaap_xor/results.json` (via `.integrity/results_meta.py`)
- Ledger row to update: `docs/CLAIMS.md` -> `B-33`
