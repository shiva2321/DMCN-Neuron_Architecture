# Pre-registration: dmcn_pong_rl_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** Using strictly sparse terminal game rewards (`+1` paddle hit, `-1` miss, `0` every other frame; zero distance reward shaping, zero hardcoded tracking policy, zero backpropagation), a 2-layer `DMCN` (Pillar 2 spine ODEs + nonlinear dendritic layer + Pillar 4 3-factor eligibility traces) learns to play Position-Only Partially-Observable Pong (`[ball_x, ball_y, paddle_y]` with NO velocity given) from random weights to >= 75.0% held-out return rate and >= 60.0% out-of-distribution (faster ball + narrower paddle) return rate across 20 seeds `[2000..2019]`, beating `ablate_eligibility`, `Untrained`, `Blind`, `Shuffled Reward`, `Random`, and `Constant` controls — and we simultaneously test Raw-Pixel (`16x16`) Pong honestly with the same zero-backprop architecture.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric (Tier 1 — Position-Only Pong):** Held-out frozen-weight ball return rate (%) over `100` unseen serves after `600` training serves (higher is better).
- **Secondary Metrics:**
  - Out-of-distribution (OOD) generalization return rate (`100` serves with unseen faster `vx` and narrower paddle),
  - Tier 2 Raw-Pixel (`16x16 = 256` pixels, zero coordinates) held-out return rate (reported honestly regardless of outcome),
  - Mechanistic analysis of what DMCN learned: directional policy alignment (`P(Up | ball_y > paddle_y)` and `P(Down | ball_y < paddle_y)`) and velocity-sensitivity delta (`P(Up | vy > 0) - P(Up | vy < 0)` at matched position).
- **Falsifiers (Tier 1 Claim Refuted If):**
  1. Tier 1 DMCN held-out return rate `< 75.0%` over 20 seeds `[2000..2019]`, OR
  2. Tier 1 DMCN OOD generalization return rate `< 60.0%`, OR
  3. `ablate_eligibility` matches or beats intact DMCN, OR
  4. Any null control (`Untrained`, `Blind`, `Shuffled Reward`, `Random`, `Constant-Stay`, `Constant-Up`) exceeds `40.0%` or comes within `35.0 pp` of intact DMCN, OR
  5. Directional policy alignment `P(correct direction)` is `<= 60%` (meaning the paddle is not genuinely tracking the ball).

## Frozen Protocol
- Custom deterministic 2D Pong physics (`x, y in [0, 1]`, left wall & top/bottom elastic bounces, right paddle at `x = 0.96`).
- Training: `600` serves per seed (`paddle_half_h = 0.12`, `vx in [0.055, 0.075]`, `vy in [-0.06, 0.06]`, `paddle_speed = 0.08`). Actions: `{-1 (Down), 0 (Stay), +1 (Up)}`.
- Reward: strictly `+1.0` on paddle reflection, `-1.0` on miss (`x >= 1.0`), `0.0` on all intermediate frames. ZERO reward shaping.
- Held-out evaluation (weights frozen, zero learning): `100` in-distribution unseen serves + `100` OOD serves (`vx in [0.075, 0.095]`, `paddle_half_h = 0.09`).
- Tier 2 (Raw-Pixel Pong): exact same physics rendered as `16x16` grayscale frames (`256` dims, zero coordinates), `600` training serves, `100` held-out test serves.
- Validation tuning seed: `[999]` (`results/dmcn_pong_rl_v1/tuning_log.json`, disjoint pool `[980..999]`, max 5 runs).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_pong_rl_v1/results.json`).
