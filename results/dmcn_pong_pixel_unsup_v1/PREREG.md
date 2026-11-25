# Pre-registration: dmcn_pong_pixel_unsup_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** Adding `UnsupOjaDendriticLayer` (motion-driven k-WTA competitive Oja feature discovery on Layer 1, zero backprop, zero coordinates, zero reward shaping) in front of DMCN's Local TD(lambda) Actor-Critic improves Raw `16x16` Pixel Pong held-out return rate significantly over fixed random Layer-1 DMCN (`47.65%` in `dmcn_pong_rl_v2`) and `ablate_unsup`, and we measure honestly how close it gets to end-to-end PyTorch Backprop A2C (`72.30%`) across 20 held-out test seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric:** Held-out frozen-weight return rate (%) over `100` unseen serves on Raw `16x16` Pixel Pong (`256` grayscale pixels, ZERO coordinates) after `2,000` training serves.
- **Secondary Metrics:** Directional tracking alignment (%), receptive-field active-pixel concentration (proving Layer 1 genuinely discovered moving objects), and exact gap vs. PyTorch Backprop A2C.
- **Falsifiers (Claim Refuted If):**
  1. `UnsupOja-DMCN` fails to beat `ablate_unsup` (identical architecture with frozen Layer 1) by `>= 5.0 pp` over 20 seeds `[2000..2019]`, OR
  2. `UnsupOja-DMCN` held-out return rate `< 55.0%`, OR
  3. Any null control (`Untrained`, `Blind`, `Shuffled Reward`, `Random`) exceeds `35.0%`.

## Frozen Protocol
- Exact same deterministic 2D Pong physics and `16x16` grayscale renderer as `dmcn_pong_rl_v2` (sparse `+1` hit / `-1` miss / `0` everywhere else; `2,000` training serves; `100` frozen-weight held-out test serves).
- Layer-1 unsupervised Oja plasticity runs strictly online during normal gameplay (zero separate oracle pretraining).
- Validation seed: `[999]` (`results/dmcn_pong_pixel_unsup_v1/tuning_log.json`, disjoint pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_pong_pixel_unsup_v1/results.json`).
