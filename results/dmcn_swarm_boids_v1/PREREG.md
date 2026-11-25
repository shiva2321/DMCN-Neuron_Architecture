# Pre-registration: dmcn_swarm_boids_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** A decentralized self-propelled Boids flock where each agent carries a local `DMCNLayer` (`SwarmBoidNetwork`, zero central coordinator, zero global oracles) achieves low-collision migrating-target tracking under extreme per-agent sensor noise (`sigma = 2.0`) AND adapts online via local proprioceptive velocity-slip plasticity to cancel unseen mid-flight crosswind disturbances (`w_wind = [0.0, -1.1]` at `t >= 50`), significantly outperforming decoupled, scrambled, blind, single-agent, `ablate_learning`, and classic `Pure Boids` / `EMA-Boids` baselines across 20 disjoint held-out test seeds `[2000..2019]`.

## Headline Metrics & Pre-Registered Falsifiers

| Regime | Headline Metrics (Direction) | Pre-Registered Falsifier (Claim Refuted If) |
| :--- | :--- | :--- |
| **Regime A (Noisy Migrating Target, `sigma=2.0`)** | Mean centroid tracking error over final 50 steps (↓), polarization (↑), total collisions (↓) | `Swarm+DMCN` error `>= 3.0`, OR fails to beat `Decoupled`, `Scrambled`, `Single Agent`, and `Blind` by `>= 25%`, OR collisions worse than `Pure Boids` |
| **Regime B (Noisy Target + Unseen Crosswind `w=[0,-1.1]` at `t>=50`)** | Post-wind centroid tracking error over final 50 steps (↓), collisions (↓) | `Swarm+DMCN` post-wind error `>= 3.0`, OR fails to beat `Pure Boids`, `EMA-Boids`, `ablate_learning`, `Decoupled`, `Scrambled`, and `Blind` |

## Frozen Protocol
- `n_boids = 16`, `n_steps = 150`, `dt = 0.2`, `noise_sigma = 2.0`, migrating target `p*(t) = [0.22 * t, 3.5 * sin(0.05 * t)]`.
- Identical target trajectory and identical per-agent Gaussian noise array `sensor_noise[t, i]` shared across all models and controls on each seed.
- Validation tuning seed: `[999]` (logged to `results/dmcn_swarm_boids_v1/tuning_log.json`, disjoint validation seed pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_swarm_boids_v1/results.json` via `.integrity/results_meta.py`).
