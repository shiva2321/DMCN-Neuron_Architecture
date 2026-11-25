# Pre-registration: dmcn_split_mnist_cuda_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A native batched CUDA 2-layer `DMCNTorchLayer` (`784 -> 2048` nonlinear dendritic expansion + `2049 -> 10` exact batched Woodbury local synaptic precision plasticity, zero backpropagation, zero replay buffer, zero task IDs, zero task boundaries) solves strictly sequential 5-pair Split-MNIST (`60,000` train / `10,000` held-out test, `{0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}`) with >= 94.0% final 10-class accuracy and >= 95.0% Pair-0 retention, beating Raw-Pixel Linear RLS (`784 -> 10`) by >= 8.0 pp and PyTorch Backprop MLPs (SGD & Adam) and `ablate_precision` by >= 60.0 pp while matching the FP64 joint offline ridge upper bound within 0.25 pp across 20 independent test seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric:** Final 10-class accuracy on the official `10,000`-image held-out MNIST test set after completing all 5 sequential digit pairs (higher is better).
- **Secondary Metrics:** Pair-0 (`{0,1}`) final retention and Backward Transfer (BWT), gap vs. FP64 Joint Offline Ridge Upper Bound, and measured CUDA GPU throughput (`samples/sec` and `us/sample`) & VRAM footprint vs. PyTorch Backprop MLP (SGD & Adam).
- **Falsifiers (Claim Refuted If):**
  1. Final 10-class held-out MNIST accuracy `< 94.0%` over 20 test seeds, OR
  2. Pair-0 (`{0,1}`) final retention `< 95.0%`, OR
  3. Fails to beat `ablate_precision`, PyTorch Backprop MLP (SGD), and PyTorch Backprop MLP (Adam) by `>= 60.0 pp`, OR
  4. Fails to beat Raw-Pixel Linear RLS (`784 -> 10`, no nonlinear dendritic expansion) by `>= 8.0 pp`, OR
  5. Mean absolute accuracy gap vs. FP64 Joint Offline Ridge Upper Bound exceeds `0.25 pp`, OR
  6. Any null control (`Untrained`, `Blind`, `Shuffled Labels`, `Majority`) exceeds `20.0%`.

## Frozen Protocol
- Dataset: Official MNIST (`torchvision.datasets.MNIST`, `60,000` train / `10,000` test, 28x28 = `784` pixels scaled to `[0, 1]`, 10 classes).
- Strictly sequential 1-pass-per-pair stream at `batch_size = 256` on CUDA GPU: `{0,1}`, then `{2,3}`, then `{4,5}`, then `{6,7}`, then `{8,9}`. Zero replay buffers, zero exemplars, zero task IDs or task boundary signals.
- Independent per-seed randomness: dendritic receptive-field initialization, baseline MLP weights, within-pair presentation shuffle, and shuffled-label control permutation.
- Validation tuning seed: `[999]` (`results/dmcn_split_mnist_cuda_v1/tuning_log.json`, disjoint validation pool `[980..999]`, max 5 tuning runs).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_split_mnist_cuda_v1/results.json` via `.integrity/results_meta.py`).
