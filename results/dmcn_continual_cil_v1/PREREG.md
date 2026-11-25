# Pre-registration: dmcn_continual_cil_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A 2-layer `DMCN` combining a nonlinear dendritic expansion layer (`64 -> 256`, zero backprop) with local Sherman-Morrison synaptic precision metaplasticity (`use_precision_plasticity=True`, zero replay buffer, zero task IDs, zero task boundaries) solves strictly sequential 5-pair Class-Incremental Learning on real handwritten digits (`{0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}`) with >= 90.0% final held-out 10-class accuracy and near-zero forgetting (Pair-0 retention >= 88.0%), beating `ablate_precision`, Online Backprop MLP (SGD & Adam), and Raw-Pixel Linear RLS across 20 disjoint held-out test seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric:** Final 10-class held-out test accuracy after completing all 5 sequential digit pairs (higher is better).
- **Secondary Metrics:** Pair-0 (`{0,1}`) final retention after learning Pairs 1–4, Pair-0 Backward Transfer (BWT), gap vs. joint offline upper bound, and exact per-step FLOPs / FP32 RAM / CPU latency.
- **Falsifiers (Claim Refuted If):**
  1. Final 10-class held-out accuracy `< 90.0%` over 20 test seeds, OR
  2. Pair-0 (`{0,1}`) retention `< 88.0%`, OR
  3. Fails to beat `ablate_precision`, Online Backprop MLP (SGD & Adam) by `>= 50 pp`, OR
  4. Fails to beat Raw-Pixel Linear RLS (`64 -> 10`, no nonlinear dendritic expansion) by `>= 5 pp`, OR
  5. Any null control (`Untrained`, `Blind`, `Shuffled Labels`, `Majority`) exceeds `20%`.

## Frozen Protocol
- Dataset: `sklearn.datasets.load_digits` (`1,797` real 8x8 handwritten digit images, 10 classes). Each seed creates a disjoint 80% train / 20% held-out test split.
- Strictly sequential 1-pass-per-pair stream: `{0,1}`, then `{2,3}`, then `{4,5}`, then `{6,7}`, then `{8,9}`. Zero replay buffers, zero stored exemplars, zero task IDs or boundary notifications.
- Validation tuning seed: `[999]` (`results/dmcn_continual_cil_v1/tuning_log.json`, disjoint validation pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_continual_cil_v1/results.json` via `.integrity/results_meta.py`).
