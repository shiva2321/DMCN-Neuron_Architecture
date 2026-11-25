# Pre-registration: dmcn_self_gating_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A single frozen `SelfGatingDMCNLayer` configuration (ZERO per-task pillar switching, ZERO task IDs, ZERO backprop) automatically shifts its local causal Hedge gate weights to select the winning pathway across 3 mutually conflicting tasks — Unseen Lorenz Chaos (`raw`/`transient` wins), `100x` Sensor Gain-Shock Nonlinear Regression (`homeo` wins), and 4-Step Delayed-Cue Classification (`spine` wins) — achieving strong performance on ALL 3 tasks where every fixed single-pathway model fails at least one task across 20 seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metrics:**
  - Task A (Lorenz Chaos): Unseen trajectory NMSE (lower is better; `<= 0.15`) + gate shift `(g_raw + g_transient) > g_homeo`.
  - Task B (`100x` Gain-Shock Nonlinear Regression): Unseen gain `[0.2, 20.0]` NMSE (`<= 0.25`) + gate shift `(g_homeo + g_spine) > g_raw`.
  - Task C (4-Step Delayed-Cue Classification): Held-out accuracy (`>= 90.0%`) + gate shift `g_spine > max(g_raw, g_homeo)`.
- **Falsifiers (Claim Refuted If):**
  1. `SelfGating-DMCN` fails any of the 3 task thresholds (`Lorenz NMSE > 0.15`, `Gain-Shock NMSE > 0.25`, `Delayed-Cue Acc < 90.0%`), OR
  2. Its learned causal Hedge gates fail to select the mechanistically appropriate pathway on any of the 3 tasks, OR
  3. `ablate_self_gating` (uniform 0.25 blend) matches or beats `SelfGating-DMCN` on average across the 3 tasks, OR
  4. Any null control (`Untrained`, `Blind`, `Shuffled`) matches `SelfGating-DMCN`.

## Frozen Protocol
- Identical `SelfGatingConfig(d_hid=32, beta_gate=5.0, alpha_loss=0.06, use_precision=True, precision_ridge=2.0)` across all 3 tasks.
- Validation seed: `[999]` (`results/dmcn_self_gating_v1/tuning_log.json`, disjoint pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_self_gating_v1/results.json`).
