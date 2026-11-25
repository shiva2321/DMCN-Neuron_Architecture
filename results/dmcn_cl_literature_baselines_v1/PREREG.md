# Pre-registration: dmcn_cl_literature_baselines_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).
(Dataset note: switched from CIFAR-10 to Fashion-MNIST BEFORE any run because the CIFAR-10 mirror stalled at 85 kB/s; Fashion-MNIST also tests raw 784-pixel generalization without any pretrained backbone.)

- **Claim (one sentence):** On both 28x28 Split-MNIST and 28x28 Split-FashionMNIST (raw 784 pixels, 60,000 train / 10,000 test each), a 2-layer batched CUDA `DMCN` (`784 -> 2048 -> 10`, nonlinear dendritic expansion + exact Woodbury local precision plasticity, zero replay, zero backprop, zero task IDs) beats standard Continual Learning literature baselines — Elastic Weight Consolidation (`MLP+EWC`, Kirkpatrick et al. 2017), Experience Replay (`MLP+ER-200` and `MLP+ER-500`, Chaudhry et al. 2019), Vanilla Adam MLP, and `ACIL / Linear-RLS (784 -> 10)` — while matching the FP64 joint offline ridge upper bound within 0.25 pp across 20 independent seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metrics:** Final 10-class held-out test accuracy on Split-MNIST (`10,000` test) and Split-FashionMNIST (`10,000` test) after strictly sequential 5-pair training `{0,1} -> {2,3} -> {4,5} -> {6,7} -> {8,9}` (higher is better).
- **Secondary Metrics:** Pair-0 retention, 4-branch Block-Diagonal DMCN (`4x` less precision VRAM) accuracy, exact gap vs. FP64 joint offline ridge, and gain over `ACIL / Linear-RLS (784 -> 10)`.
- **Falsifiers (Claim Refuted If):**
  1. On Split-MNIST, `DMCN` `< 94.0%`, OR fails to beat `MLP+EWC`, `MLP+ER-200`, `MLP+ER-500`, `MLP+Adam`, and `ACIL/Linear-RLS` (by `>= 8.0 pp`), OR
  2. On Split-FashionMNIST, `DMCN` `< 83.0%`, OR fails to beat `MLP+Adam`, `MLP+EWC`, `MLP+ER-200`, `MLP+ER-500`, and `ACIL/Linear-RLS` (by `>= 3.0 pp`), OR
  3. Mean absolute gap vs. FP64 Joint Offline Ridge exceeds `0.25 pp` on either dataset, OR
  4. Any null control (`Untrained`, `Blind`, `Shuffled Labels`, `Majority`) exceeds `20.0%` on either dataset.

## Frozen Protocol
- Split-MNIST & Split-FashionMNIST: official `60,000` train / `10,000` test (`784` raw pixels in `[0,1]`).
- Strictly sequential 1-pass-per-pair stream at `batch_size = 256` on CUDA GPU. Note: `MLP+EWC` receives task boundary notifications and `MLP+ER-200 / ER-500` store 200 / 500 raw image exemplars in a reservoir replay buffer; DMCN uses zero task boundaries and zero stored exemplars.
- Validation tuning seed: `[999]` (`results/dmcn_cl_literature_baselines_v1/tuning_log.json`, disjoint pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_cl_literature_baselines_v1/results.json`).
