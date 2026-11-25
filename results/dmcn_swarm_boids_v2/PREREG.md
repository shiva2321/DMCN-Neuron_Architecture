# Pre-registration: dmcn_swarm_boids_v2

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A decentralized Boids swarm equipped with per-boid local nonlinear `DMCN` proprioceptive wind adaptation (`[pos/10, vel, 1] -> 24 tanh -> 2` online local plasticity, zero oracle wind frequency, zero centralized controller, zero backprop) matches or beats a classical Integral/PID disturbance observer on uniform crosswind (Regime 1), beats BOTH `Pure Boids` AND the classical `PID-Observer` on a spatially-varying nonlinear wind shear field (Regime 2), and rescues 31% sensor-failed blind boids via emergent neighbor consensus under gusting wind (Regime 3) across 20 held-out test seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metric:** Mean post-adaptation centroid tracking error (`t >= 90..179`, lower is better) across Regimes 1, 2, and 3; plus blind-subset tracking error in Regime 3.
- **Secondary Metrics:** Mean neighbor collision count, polarization, and per-boid FLOPs / FP32 RAM.
- **Falsifiers (Claim Refuted If):**
  1. In Regime 1 (Uniform Crosswind), `Swarm+DMCN` fails to beat `Pure Boids` by `>= 3.0x` (note: matching `PID-Observer` on uniform wind is expected and reported honestly), OR
  2. In Regime 2 (Spatially-Varying Wind Shear Field), `Swarm+DMCN` fails to beat BOTH `Pure Boids` (by `>= 2.0x`) AND the tuned classical `PID-Observer` (by `>= 15%`), OR
  3. In Regime 3 (31% Blind Boids + Gusting Wind), `Swarm+DMCN` blind-boid error fails to beat `Decoupled` blind-boid error by `>= 3.0x` and `Pure Boids` by `>= 2.0x`, OR
  4. Any null control (`Decoupled`, `Scrambled`, `Blind`, `ablate_learning`) matches `Swarm+DMCN` in any regime.

## Frozen Protocol
- `N = 16` boids in 2D open space, `T = 180` steps, `dt = 0.2`, cruising speed `v_cruise = 1.5`, moving target on figure-8 / sinusoidal patrol trajectory so boids traverse multiple spatial wind zones.
- Noisy target bearing sensor (`sigma = 1.5`) for sighted boids; `0` bearing for the 5 blind boids (`31.25%`) in Regime 3.
- Proprioceptive slip signal `slip = actual_vel - commanded_vel` is strictly local to each boid (identical signal available to both `Swarm+DMCN` and `Boids + PID-Observer`).
- Validation tuning seed: `[999]` (`results/dmcn_swarm_boids_v2/tuning_log.json`, disjoint pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_swarm_boids_v2/results.json`).
