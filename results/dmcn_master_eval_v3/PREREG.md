# Pre-registration: dmcn_master_eval_v3

Written BEFORE running validation or held-out test seeds for v3 (GEMINI.md §2.2, §4.1). `v1` and `v2` results remain untouched on record.

- **Claim (one sentence):** Adding (1) `center_homeostasis=True` for DC+gain-shock invariance in `E4`, (2) `gamma_forget=0.0` in episodic `E5` RL, (3) a sparse `k-WTA` expansion + `boids_repulsion` dendritic routing layer for task-free continual handwritten digits (`E6b`), and (4) decentralized `SwarmBoidNetwork` collective intelligence (`E8`) fixes the v2 `E4`/`E5` issues, beats Backprop MLP & single-branch DMCN on `E6b`, and achieves collective noise-filtering & moving-target tracking in `E8` that significantly outperforms decoupled, scrambled, blind, and single-agent controls across 20 held-out seeds `[2000..2019]`.

## Pre-Registered Falsifiers (v3 Specific)
- **`E4` Chaos:** Post-10x-gain+offset-shock Lorenz NMSE `< 0.30` AND lower than `ablate_homeostasis`, `Persistence`, and `NLMS`.
- **`E5` RL:** Mean accuracy `>= 90.0%` while memoryless/blind/shuffled controls remain near `50%`.
- **`E6b` Task-Free Continual Digits:** Final avg accuracy `>= 40.0%` (up from `22.36%` in v2) AND beats Backprop MLP-SGD/Adam and `n_branches=1` by `>= 15 pp`.
- **`E8` Collective Swarm Boids + DMCN (`sigma_noise = 2.0`):**
  - Final tracking error `< 2.5` AND lower than `decouple_neighbors`, `scramble_neighbors`, `blind`, and `single_agent` by `>= 30%`.
  - Flock polarization `>= 0.75` AND higher than `decouple_neighbors` and `scramble_neighbors` by `>= 0.30`.
- **No Regression:** `E1a >= 95%`, `E1b >= 88%`, `E2 >= 95%`, `E3 >= 75%`, `E6a >= 91%`.

## Frozen Protocol
- Validation seed: `[999]` (`results/dmcn_master_eval_v3/tuning_log.json`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_master_eval_v3/results.json`).
