# Pre-registration: dmcn_pong_rl_v2

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).
(New experiment ID following dmcn_pong_rl_v1, which stays untouched on record.)

- **Claim (one sentence):** Upgrading DMCN's 3-factor RL to local **TD(lambda) Actor-Critic** (1-neuron local Critic + per-step TD error + local entropy bonus, zero backprop, zero reward shaping) paired with **causal transient (derivative) spine channels** (`x - s`, zero velocity oracle) raises Position-Only Pong held-out return rate from `64.0%` (`v1`) to `>= 85.0%` (with OOD generalization `>= 70.0%`, directional alignment `>= 70.0%`, and positive velocity sensitivity), while local retinotopic `5x5` dendritic receptive fields + pixel transients raise Raw `16x16` Pixel Pong (`256` pixels, zero coordinates) from `31.4%` (`v1`) to `>= 65.0%` across 20 held-out seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metrics:**
  - Tier 1 (Position-Only Pong): Held-out frozen-weight return rate (%) over `100` unseen serves after `2,000` training serves.
  - Tier 2 (Raw `16x16` Pixel Pong): Held-out frozen-weight return rate (%) over `100` unseen serves after `2,000` training serves.
- **Secondary Metrics:** Tier 1 OOD return rate (faster ball + narrower paddle), directional tracking alignment, velocity-sensitivity delta (`P(Up | vy > 0) - P(Up | vy < 0)`), and head-to-head comparison vs. PyTorch Backprop Actor-Critic (`A2C`).
- **Falsifiers (Claim Refuted If):**
  1. Tier 1 DMCN held-out return rate `< 85.0%` OR OOD return rate `< 70.0%` over 20 seeds `[2000..2019]`, OR
  2. Tier 1 directional tracking alignment `< 70.0%` OR velocity sensitivity `<= 0.0 pp`, OR
  3. Tier 2 Raw-Pixel DMCN held-out return rate `< 65.0%` OR fails to beat Blind pixel control by `>= 35.0 pp`, OR
  4. Any null control (`Untrained`, `Blind`, `Shuffled Reward`, `Random`, `Constant-Stay`, `Constant-Up`) exceeds `35.0%`.

## Frozen Protocol
- Identical deterministic 2D Pong physics as `v1` (`x, y in [0, 1]`, sparse `+1` hit / `-1` miss / `0` everywhere else; ZERO reward shaping).
- Training budget: `2,000` serves per seed for BOTH DMCN and PyTorch Backprop A2C baseline (Tier 1 & Tier 2).
- Held-out evaluation (frozen weights, zero learning): `100` in-distribution unseen serves + `100` OOD serves (`vx in [0.075, 0.095]`, `paddle_half_h = 0.09`).
- Validation tuning seed: `[999]` (`results/dmcn_pong_rl_v2/tuning_log.json`, disjoint pool `[980..999]`, max 5 runs).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_pong_rl_v2/results.json`).
