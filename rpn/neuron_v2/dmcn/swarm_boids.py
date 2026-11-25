"""Boids-DMCN Collective Swarm Intelligence Engine (v2 — Self-Propelled Cruising + Local Slip Adaptation).

Combines Craig Reynolds' three local flocking rules (Separation, Alignment, Cohesion)
with self-propelled cruising-speed physics and per-agent DMCN cellular memory:
- Multi-timescale DMCN spine ODE traces filter extreme per-agent sensor noise (sigma=2.0)
  across time without central coordination.
- Strictly local proprioceptive velocity-slip plasticity (desired_vel - achieved_vel)
  enables online cancellation of unseen mid-flight crosswind disturbances without any
  global oracle or backpropagation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from .config import AblationConfig, DMCNConfig
from .numpy_core import DMCNLayer


@dataclass
class SwarmConfig:
    n_boids: int = 16
    perception_radius: float = 5.0
    separation_radius: float = 0.9
    w_sep: float = 1.6
    w_align: float = 1.4
    w_coh: float = 0.5
    w_goal: float = 1.6
    cruise_speed: float = 1.4
    min_speed: float = 0.5
    max_speed: float = 2.6
    speed_relax: float = 0.25
    use_dmcn_memory: bool = True
    use_ema_filter: bool = False
    ema_alpha: float = 0.25
    ablate_spine: bool = False
    ablate_learning: bool = False
    seed: int = 42


class SwarmBoidNetwork:
    """Decentralized self-propelled flock of DMCN-powered Boids with built-in null controls."""

    def __init__(self, cfg: SwarmConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.n = int(cfg.n_boids)

        # 2D positions & self-propelled initial velocities at cruise_speed
        self.pos = self.rng.randn(self.n, 2) * 1.2
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        self.vel = np.column_stack([np.cos(angles), np.sin(angles)]) * cfg.cruise_speed

        # Per-boid classical EMA bearing filter state (for strong classical baseline comparison)
        self.ema_bearing = np.zeros((self.n, 2), dtype=np.float64)

        # Last commanded steering velocity (for strictly local proprioceptive wind-slip error)
        self.last_cmd_vel = self.vel.copy()
        self.last_pos = self.pos.copy()

        # Two lightweight cellular DMCN controllers per boid:
        # 1) Bearing-denoising DMCN (4D -> 2D with 4-scale spine ODE bank)
        # 2) Proprioceptive disturbance-cancellation DMCN (3D -> 2D, learns counter-wind from local slip)
        self.brains = [
            DMCNLayer(
                DMCNConfig(
                    d_in=4,
                    d_out=2,
                    n_branches=1,
                    n_spine_scales=4,
                    tau_min=0.4,
                    tau_max=3.0,
                    soma_activation="linear",
                    use_nonmonotonic=False,
                    eta_fast=0.15,
                    gamma_forget=0.005,
                    eta_consol=0.03,
                    weight_init_scale=0.02,
                    seed=cfg.seed + i,
                    ablation=AblationConfig(
                        ablate_homeostasis=True,
                        ablate_spine_traces=cfg.ablate_spine,
                    ),
                )
            )
            for i in range(self.n)
        ]
        self.wind_adapters = [
            DMCNLayer(
                DMCNConfig(
                    d_in=3,
                    d_out=2,
                    n_branches=1,
                    n_spine_scales=2,
                    tau_min=0.4,
                    tau_max=2.0,
                    soma_activation="linear",
                    use_nonmonotonic=False,
                    eta_fast=0.35,
                    gamma_forget=0.0,
                    eta_consol=0.05,
                    weight_init_scale=0.0,
                    seed=cfg.seed + 100 + i,
                    ablation=AblationConfig(
                        ablate_homeostasis=True,
                        ablate_spine_traces=cfg.ablate_spine,
                    ),
                )
            )
            for i in range(self.n)
        ]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.pos = self.rng.randn(self.n, 2) * 1.2
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        self.vel = np.column_stack([np.cos(angles), np.sin(angles)]) * self.cfg.cruise_speed
        self.ema_bearing.fill(0.0)
        self.last_cmd_vel = self.vel.copy()
        self.last_pos = self.pos.copy()
        for b, w_ad in zip(self.brains, self.wind_adapters):
            b.reset_temporal_state()
            b.reset_fast_memory()
            w_ad.reset_temporal_state()
            w_ad.reset_fast_memory()

    def step(
        self,
        noisy_bearings: np.ndarray,
        dt: float = 0.2,
        external_wind: Optional[np.ndarray] = None,
        decouple_neighbors: bool = False,
        scramble_neighbors: bool = False,
        blind: bool = False,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """Advance decentralized swarm by dt using local Boids rules + DMCN adaptation."""
        if blind:
            bearings = np.zeros((self.n, 2), dtype=np.float64)
        else:
            bearings = np.asarray(noisy_bearings, dtype=np.float64).reshape(self.n, 2)

        wind_vec = np.zeros(2, dtype=np.float64) if external_wind is None else np.asarray(external_wind, dtype=np.float64).reshape(2)

        pos_ref = self.pos
        vel_ref = self.vel
        if scramble_neighbors:
            perm = self.rng.permutation(self.n)
            pos_ref = self.pos[perm]
            # Preserve neighbor speed magnitudes while scrambling neighbor heading directions
            rand_ang = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n)
            spds = np.linalg.norm(self.vel, axis=1)
            vel_ref = np.column_stack([np.cos(rand_ang) * spds, np.sin(rand_ang) * spds])

        # Strictly local proprioceptive velocity measurement from previous step: achieved_vel = (pos - last_pos) / dt
        achieved_vel = (self.pos - self.last_pos) / max(dt, 1e-6)
        slip_err = self.last_cmd_vel - achieved_vel  # equals -wind_vec locally!

        new_vel = np.zeros_like(self.vel)
        cmd_vel = np.zeros_like(self.vel)
        min_pair_dist = float("inf")
        n_collisions = 0

        for i in range(self.n):
            f_sep = np.zeros(2, dtype=np.float64)
            f_align = np.zeros(2, dtype=np.float64)
            f_coh = np.zeros(2, dtype=np.float64)
            n_neigh = 0

            if not decouple_neighbors:
                diffs = pos_ref - self.pos[i]
                dists = np.linalg.norm(diffs, axis=1)
                for j in range(self.n):
                    if i == j:
                        continue
                    d = float(dists[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
                    if d < 0.30:
                        n_collisions += 1
                    if d < self.cfg.perception_radius:
                        n_neigh += 1
                        f_align += vel_ref[j]
                        f_coh += pos_ref[j]
                        if d < self.cfg.separation_radius:
                            f_sep -= diffs[j] / (d * d + 0.08)

            if n_neigh > 0:
                f_align = (f_align / n_neigh) - self.vel[i]
                f_coh = (f_coh / n_neigh) - self.pos[i]
                coh_n = float(np.linalg.norm(f_coh))
                if coh_n > 1e-8:
                    f_coh /= coh_n

            boids_drive = (
                self.cfg.w_sep * f_sep
                + self.cfg.w_align * f_align
                + self.cfg.w_coh * f_coh
            )

            u_wind_comp = np.zeros(2, dtype=np.float64)
            # Goal/adaptation drive: DMCN cellular filter+adapter vs EMA baseline vs raw Reynolds Boids
            if self.cfg.use_dmcn_memory:
                obs_i = np.array([bearings[i, 0], bearings[i, 1], f_align[0], f_align[1]], dtype=np.float64)
                dmcn_out = np.clip(self.brains[i].step(obs_i, dt=dt), -2.5, 2.5)

                # Local proprioceptive wind-slip cancellation controller
                slip_ctx = np.array([1.0, self.vel[i, 0] * 0.2, self.vel[i, 1] * 0.2], dtype=np.float64)
                u_wind_comp = np.clip(self.wind_adapters[i].step(slip_ctx, dt=dt), -2.5, 2.5)

                if learn and not self.cfg.ablate_learning:
                    target_bearing = np.clip(bearings[i], -2.0, 2.0)
                    self.brains[i].learn_local(target_bearing - dmcn_out)
                    # Learn to match observed proprioceptive slip (-wind_vec) directly!
                    self.wind_adapters[i].learn_local(np.clip(slip_err[i], -2.5, 2.5))

                goal_drive = self.cfg.w_goal * dmcn_out
            elif self.cfg.use_ema_filter:
                a_e = self.cfg.ema_alpha
                self.ema_bearing[i] = (1.0 - a_e) * self.ema_bearing[i] + a_e * bearings[i]
                goal_drive = self.cfg.w_goal * self.ema_bearing[i]
            else:
                goal_drive = self.cfg.w_goal * bearings[i]

            steer = goal_drive + boids_drive
            v_i = self.vel[i] + dt * steer

            # Self-propelled Reynolds/Vicsek cruising-speed relaxation on intrinsic flock velocity
            spd = float(np.linalg.norm(v_i))
            if spd > 1e-8:
                target_spd = (1.0 - self.cfg.speed_relax) * spd + self.cfg.speed_relax * self.cfg.cruise_speed
                target_spd = float(np.clip(target_spd, self.cfg.min_speed, self.cfg.max_speed))
                v_i = (v_i / spd) * target_spd
            new_vel[i] = v_i
            # Commanded physical velocity adds learned local counter-wind thrust without starving cruise speed
            cmd_vel[i] = v_i + u_wind_comp

        self.last_pos = self.pos.copy()
        self.last_cmd_vel = new_vel.copy()
        self.vel = new_vel
        # Physical position update: commanded velocity (with counter-wind) + unseen external crosswind
        self.pos = self.pos + dt * (cmd_vel + wind_vec[None, :])

        v_norms = np.linalg.norm(self.vel, axis=1, keepdims=True) + 1e-9
        unit_v = self.vel / v_norms
        polarization = float(np.linalg.norm(np.mean(unit_v, axis=0)))

        return {
            "centroid": np.mean(self.pos, axis=0).copy(),
            "polarization": polarization,
            "n_collisions": n_collisions // 2,
            "min_pair_dist": 0.0 if min_pair_dist == float("inf") else min_pair_dist,
        }
