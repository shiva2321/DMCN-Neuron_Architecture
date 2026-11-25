# Pre-registration: dmcn_master_eval_v1

Written BEFORE running any validation or frozen test suite (GEMINI.md §4.1). Changing any protocol parameter after seeing held-out test results constitutes a NEW experiment ID (GEMINI.md §2.2).

- **Claim (one sentence):** A single unified Dendritic Multiplexed Cascade Neuron (`DMCNLayer` / `DMCNNetwork` / `DMCNTorchNetwork`, dual NumPy + PyTorch backend, zero backpropagation, zero external codebooks, zero episode replay lists) learns single-cell causal nonlinear/temporal logic, remembers and overwrites 1-shot associations, generalizes on held-out real handwritten digits (`sklearn` 8x8 digits), retains knowledge across task-free continual streams without task-ID oracles, self-stabilizes under 10x gain shocks with irregular `dt`, and learns delayed-reward memory RL across 20 disjoint held-out test seeds `[2000..2019]`, while every ablated pillar degrades its target capability and all null controls fail to match it.

## Headline Task Metrics & Pre-Registered Falsifiers

| Sub-Experiment | Headline Metric (Direction) | Pre-Registered Falsifier (Claim Refuted If) |
| :--- | :--- | :--- |
| `e1a_causal_xor` | Held-out noisy XOR accuracy (higher is better) | Mean < 90.0%, OR fails to beat Point Logistic & `ablate_nonmonotonic` by >= 15 pp |
| `e1b_delayed_xor` | Held-out delayed temporal XOR accuracy (higher is better) | Mean < 85.0%, OR fails to beat `ablate_spine_traces` by >= 20 pp |
| `e2_associative_recall` | 1-shot recall & overwrite Top-1 accuracy (higher is better) | Overall mean < 90.0%, OR overwrite accuracy < 80.0%, OR fails to beat Additive Hebbian on overwrite by >= 20 pp |
| `e3_continual_regimes` | Task-free 4-regime final held-out avg accuracy & BWT (higher is better) | Final avg accuracy < 70.0%, OR fails to beat Online Backprop MLP (SGD) by >= 15 pp on final avg accuracy |
| `e4_nonstationary_chaos` | Post-10x-gain-shock causal Lorenz NMSE (lower is better) | Mean NMSE >= 0.50 (persistence ~ 1.0), OR fails to beat `ablate_homeostasis` |
| `e5_delayed_cue_tmaze` | Held-out delayed-cue T-maze accuracy (higher is better) | Mean < 80.0%, OR fails to beat every memoryless/constant/blind control (capped at ~50%) by >= 25 pp |
| `e6a_digits_recognition` | Held-out unseen handwritten digit Top-1 accuracy (higher is better) | Mean < 80.0% on held-out 10-class digits, OR fails to beat shuffled-label/blind/untrained controls |
| `e6b_digits_continual` | Task-free 5-pair continual digit final avg accuracy (higher is better) | Reported honestly alongside Online Backprop MLP (SGD/Adam) and `n_branches=1` ablation (exploratory stress test; falsifier if <= majority class 10%) |

## Frozen Protocol
- **Disjoint Seeds:**
  - Validation tuning seed (max 5 exploratory runs logged in `tuning_log.json`): `[999]`
  - Held-out test seeds (20 independent seeds, touched ONCE after freezing): `[2000, 2001, ..., 2019]`
- **Zero Oracles & Zero Cheats:**
  - Strictly causal step-by-step execution (`step(x_t, dt)`), zero future array peeking, zero offline trajectory normalization, zero external `ItemTable`, zero episode replay lists, zero task-ID or boundary labels in continual tasks, zero goal-beacon or physics oracles in RL.
- **Computational Cost Profiling:**
  - Exact analytical MACs/FLOPs, parameter counts, FP32/FP64 memory bytes (`count_layer_ops`), and real wall-clock latency (microseconds/step on CPU NumPy & GPU CUDA PyTorch) measured in the same script against matched-parameter Backprop MLP baselines.
- **Outputs:**
  - `results/dmcn_master_eval_v1/tuning_log.json`
  - `results/dmcn_master_eval_v1/results.json` (written via `.integrity/results_meta.py` with full `_meta` block).
