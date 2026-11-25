# Pre-registration: dmcn_pdnc_general_v1

Written BEFORE running validation or frozen 20-seed test evaluation (`GEMINI.md` §4.1).

- **Claim (one sentence):** **Prospective Dendritic Natural-Credit (PDNC)** — unifying local apical prospective settlement, symmetric Kolen-Pollack reciprocal alignment (zero weight transport), diagonal BONG-momentum conditioning, online Woodbury null-space subspace protection, and buffer-free episode-consolidated TD($\lambda$) 3-factor eligibility — enables gradient-free DMCNs to learn deep hidden features reliably across **Raw `16x16` Pixel Pong RL**, **noisy 24D nonlinear XOR discovery**, and **boundary-free continual learning**, closing the backprop gap on Raw-Pixel RL (`>= 70%`) while preserving prior task subspaces without replay or task boundaries across `20` held-out test seeds `[2000..2019]`.
- **Headline task metric (and direction):**
  - **Primary:** `t2_pdnc_pixel_hit_rate` (%) on Raw `16x16` Pixel Pong (higher is better).
  - **Co-primary generality metrics:** `xor_pdnc_acc` (24D noisy XOR test accuracy %, higher is better), `cl3_subspace_mean_acc` & `cl3_subspace_t0_ret` (3-task continual mean accuracy & Task-0 retention %, higher is better), plus `fmnist_subspace_acc` vs `fmnist_diagonal_acc` on replay-free Split-FashionMNIST (`d_hid=256`, higher is better).
- **Falsifiers (Claim Refuted If Any Trigger):**
  1. `t2_pdnc_pixel_hit_rate < 65.0%` OR `t2_pdnc_pixel_hit_rate < t2_ablate_l1_learning_hit_rate + 12.0 pp` over 20 seeds `[2000..2019]`.
  2. `t2_pdnc_pixel_hit_rate` trails `t2_bp_a2c_pixel_hit_rate` by more than `8.0 pp` (we require PDNC to match or rival end-to-end Backprop A2C on identical 2,000 serves).
  3. Any Suite A null control (`Untrained`, `Blind`, `Shuffled Reward`, `Random`) exceeds `38.0%`.
  4. In Suite B (24D noisy XOR), `xor_pdnc_acc < 85.0%` OR `xor_pdnc_acc < xor_ablate_l1_acc + 20.0 pp`.
  5. In Suite C (boundary-free 3-task continual stream), `cl3_subspace_t0_ret < 80.0%` OR `cl3_subspace_mean_acc < cl3_diagonal_mean_acc + 15.0 pp`, OR on `Split-FashionMNIST` (`d_hid=256`), `fmnist_subspace_acc` fails to beat `fmnist_diagonal_acc` by `>= 25.0 pp` (disclosing honestly alongside `fmnist_frozen_woodbury_acc`).

## Frozen protocol
- **Data / environment / constants:**
  - **Suite A (Pong):** Deterministic 2D Pong (`PongEnv` from `dmcn_pong_rl_benchmark`), `obs_pixels(16)` (256 grayscale pixels ONLY, zero coordinates), `2,000` training serves (`seed + 2000`), `100` frozen-weight ID test serves (`seed + 7000`), `100` OOD faster-ball/narrow-paddle serves (`seed + 8000`).
  - **Suite B (24D Noisy XOR):** `n_train = 1600`, `n_test = 400`, `X ~ N(0, I_24)`, label `y = sign(x_0 * x_1)` (22 pure noise dims), batch size `32`, `15` epochs.
  - **Suite C1 (3-Task Continual Stream):** Three mutually orthogonal rotated `8D` subspaces `Q_0, Q_1, Q_2` (`Q ~ Haar(O(24))`) inside `R^24`, `256` samples/task, `25` epochs/task, batch `32`, sequential `T0 -> T1 -> T2` with **0 task boundaries, 0 SVD, 0 replay** (every coordinate is nonzero across all tasks, eliminating axis-aligned gradient-zero shortcuts).
  - **Suite C2 (Split-FashionMNIST):** `28x28` pixels (`784-D`) -> `256` hidden -> `10` classes, strictly sequential 5 pairs `{0,1}..{8,9}`, `0` replay, `0` task IDs, evaluated on full `10,000`-image FashionMNIST test set.
- **Inputs / sensors / features (and which are oracles):** Zero coordinate oracles in Raw Pixel Pong; zero task-ID oracles in continual learning; random disjoint RNG streams for `W` and `B_fb` (`seed` vs `seed + 7919`) verifying zero weight transport.
- **Reward / loss / objective:** Suite A: strictly sparse `+1` hit / `-1` miss / `0` everywhere else (no reward shaping); Suites B & C: MSE / one-hot error without autograd.
- **Seeds:** Validation = `[999]` (from disjoint pool `[980..999]`, logged in `results/dmcn_pdnc_general_v1/tuning_log.json`). Frozen Test = `[2000..2019]` (`n = 20`, run ONCE after freezing validation config).

## Controls and baselines (run BEFORE/ALONGSIDE the headline)
- **Null controls:** Untrained, Blind (`obs = 0`), Shuffled Reward / Shuffled Labels, Scrambled Apical Feedback (`shuffle_apical=True`), Random policy.
- **Component ablations:** `ablate_l1_learning`, `ablate_kp_alignment` (`freeze_random_feedback=True`), `ablate_eligibility`, `ablate_prospective_settle`, `ablate_subspace` (`diagonal`).
- **Baselines:** Fair PyTorch Backprop A2C (`TorchA2CMLP`) on Suite A; matched PyTorch Backprop Adam MLP on Suites B & C1; `Frozen 256 + Woodbury` disclosure baseline on Suite C2.

## Tuning budget
- Maximum **3** validation configurations on seed `999` before freezing `dmcn_pdnc_general_benchmark.py`.
- Log file: `results/dmcn_pdnc_general_v1/tuning_log.json`.

## Reporting
- Results file with run metadata: `results/dmcn_pdnc_general_v1/results.json` (via `.integrity/results_meta.py`).
- Human approval required before modifying `docs/CLAIMS.md`.
