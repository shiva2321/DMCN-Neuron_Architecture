"""Continuous 3D Sensorimotor Deceptive Obstacle Arena (CAPN Milestone 6).

Implements:
1. Continuous3DArenaEnvironment:
   3D continuous spatial volume (10 x 10 x 10) with deceptive 3D hazard spheres,
   realistic stop-and-rebound physics (zero wall-sliding cheat), and 14-directional
   spherical rangefinder rays.
2. CAPN3DSensorimotorAgent:
   Closed-loop 3D agent integrating ContinuousMotorAttractorField and
   predictive sensory forward rollouts.
3. Comparative Controls & Baselines:
   - Blind3DSensorimotorAgent (Null Control 1: range sensors zeroed).
   - ReactiveGreedy3DAgent (Null Control 2: direct heading with only instantaneous reactive braking).
   - RandomWalk3DAgent (Model-Free Baseline: random continuous velocity exploration).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def ray_sphere_intersection(ray_origin: np.ndarray, ray_dir: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Compute distance from ray_origin along unit ray_dir to intersection with sphere."""
    oc = ray_origin - center
    b = 2.0 * float(np.dot(oc, ray_dir))
    c = float(np.dot(oc, oc)) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return float("inf")
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    if t1 > 0.0:
        return float(t1)
    t2 = (-b + sqrt_disc) / 2.0
    if t2 > 0.0:
        return float(t2)
    return float("inf")


def ray_box_distance(p: np.ndarray, d: np.ndarray, size: float = 10.0) -> float:
    """Compute distance from point p along ray d to 3D bounding box [0, size]^3."""
    t_min = float("inf")
    for k in range(3):
        if d[k] > 1e-8:
            t = (size - p[k]) / d[k]
            if 0.0 < t < t_min:
                t_min = float(t)
        elif d[k] < -1e-8:
            t = -p[k] / d[k]
            if 0.0 < t < t_min:
                t_min = float(t)
    return t_min


def generate_3d_ray_directions() -> List[np.ndarray]:
    """Generate 14 normalized ray directions spanning the 3D unit sphere."""
    rays = []
    # 6 Cardinal directions
    for k in range(3):
        d1 = np.zeros(3, dtype=np.float64)
        d1[k] = 1.0
        rays.append(d1)
        d2 = np.zeros(3, dtype=np.float64)
        d2[k] = -1.0
        rays.append(d2)

    # 8 Diagonal directions
    for s1 in [-1.0, 1.0]:
        for s2 in [-1.0, 1.0]:
            for s3 in [-1.0, 1.0]:
                diag = np.array([s1, s2, s3], dtype=np.float64) / np.sqrt(3.0)
                rays.append(diag)

    return rays


class Continuous3DArenaEnvironment:
    """Continuous 3D spatial arena with deceptive hazard spheres and stop-and-rebound physics."""

    def __init__(
        self,
        size: float = 10.0,
        start: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        goal: Tuple[float, float, float] = (8.5, 8.5, 8.5),
        goal_radius: float = 1.0,
        r_agent: float = 0.25,
        dt: float = 0.1,
        max_steps: int = 220,
    ) -> None:
        self.size = size
        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.goal_radius = goal_radius
        self.r_agent = r_agent
        self.dt = dt
        self.max_steps = max_steps
        self.ray_dirs = generate_3d_ray_directions()

        # 3D Deceptive hazard configuration: cluster blocking direct straight line (1.5, 1.5, 1.5) -> (8.5, 8.5, 8.5)
        self.hazard_spheres: List[Tuple[np.ndarray, float]] = [
            (np.array([5.0, 5.0, 5.0]), 1.8),  # Central deceptive hazard sphere
            (np.array([4.2, 5.5, 4.8]), 1.1),  # Lateral baffle 1
            (np.array([5.5, 4.2, 4.8]), 1.1),  # Lateral baffle 2
            (np.array([4.8, 4.8, 6.0]), 1.1),  # Top baffle
            (np.array([6.0, 6.0, 6.0]), 1.3),  # Deep core obstacle
        ]

        self.pos = self.start.copy()
        self.vel = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.collision_count = 0
        self.trajectory: List[np.ndarray] = []

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment state."""
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        # Minor position jitter at start
        jitter = rng.uniform(-0.1, 0.1, size=3)
        self.pos = self.start + jitter
        self.vel = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.collision_count = 0
        self.trajectory = [self.pos.copy()]
        return self._get_obs()

    def _get_sensor_readings(self) -> np.ndarray:
        """Cast 14 directional rangefinder rays in 3D across unit sphere."""
        readings = np.full(14, 5.0, dtype=np.float64)
        for i, r_dir in enumerate(self.ray_dirs):
            min_dist = 5.0
            # Check hazard spheres
            for center, radius in self.hazard_spheres:
                d = ray_sphere_intersection(self.pos, r_dir, center, radius)
                if d < min_dist:
                    min_dist = d
            # Check arena bounding walls
            d_wall = ray_box_distance(self.pos, r_dir, self.size)
            if d_wall < min_dist:
                min_dist = d_wall
            readings[i] = min_dist
        return readings

    def _get_obs(self) -> Dict[str, Any]:
        """Return observation dictionary."""
        dists = self._get_sensor_readings()
        diff = self.goal - self.pos
        dist_to_goal = float(np.linalg.norm(diff))
        dir_to_goal = diff / (dist_to_goal + 1e-9)

        return {
            "distances": dists,
            "dist_to_goal": dist_to_goal,
            "dir_to_goal": dir_to_goal,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "speed": float(np.linalg.norm(self.vel)),
        }

    def step(self, action_vec: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Simulate single 3D step with realistic stop-and-rebound physics (zero wall-sliding)."""
        self.steps += 1
        act = np.clip(action_vec, -1.0, 1.0)

        # 3D Continuous kinematics with inertia: v[t+1] = 0.7 * v[t] + 0.3 * act
        proposed_vel = 0.7 * self.vel + 0.3 * act
        proposed_speed = float(np.linalg.norm(proposed_vel))
        if proposed_speed > 1.0:
            proposed_vel = (proposed_vel / proposed_speed) * 1.0

        proposed_pos = self.pos + proposed_vel * self.dt
        # Clip to arena boundaries
        proposed_pos = np.clip(proposed_pos, 0.2, self.size - 0.2)

        # 3D Collision detection against hazard spheres
        collision = False
        for center, radius in self.hazard_spheres:
            d = float(np.linalg.norm(proposed_pos - center))
            if d < (radius + self.r_agent):
                collision = True
                break

        if collision:
            self.collision_count += 1
            # Zero wall-sliding physics: forward velocity stopped upon impact
            self.vel = np.zeros(3, dtype=np.float64)
            # Position stays at current safe position
        else:
            self.vel = proposed_vel
            self.pos = proposed_pos

        self.trajectory.append(self.pos.copy())

        # Check goal condition
        dist_goal = float(np.linalg.norm(self.goal - self.pos))
        reached_goal = dist_goal < self.goal_radius
        done = reached_goal or (self.steps >= self.max_steps)

        # Continuous reward signal
        reward = -0.05
        if collision:
            reward -= 2.0
        if reached_goal:
            reward += 50.0

        obs = self._get_obs()
        info = {
            "reached_goal": reached_goal,
            "collision": collision,
            "steps": self.steps,
            "collision_count": self.collision_count,
            "final_dist_goal": dist_goal,
        }
        return obs, reward, done, reached_goal, info


class ContinuousMotorAttractorField:
    """Continuous 3D Motor Attractor Field (Georgopoulos population vector + Cisek affordance competition).

    Generates continuous 3D velocity intentions purely through recurrent lateral inhibition
    and sensory-motor convergence on the unit sphere, with ZERO analytical cross products
    or hardcoded geometric references.
    """

    def __init__(self, basis_directions: np.ndarray, beta_inh: float = 0.35) -> None:
        self.basis = np.array(basis_directions, dtype=np.float64)  # (K, 3)
        self.K = len(self.basis)
        # Recurrent spherical Mexican-hat connectivity: cosine similarity minus lateral inhibition
        self.W = (self.basis @ self.basis.T) - beta_inh
        np.fill_diagonal(self.W, 1.0)
        # Angular hazard diffusion matrix
        self.diffuse = np.maximum(0.0, self.basis @ self.basis.T) ** 2

    def compute_affordances(self, dir_goal: np.ndarray, dists: np.ndarray, prev_act: np.ndarray) -> List[np.ndarray]:
        """Compute motor population vector and active affordance peaks under sensory inhibition."""
        # 1. External afferent drives
        I_goal = np.maximum(0.0, self.basis @ dir_goal)
        I_prev = np.maximum(0.0, self.basis @ prev_act)

        # Hazard repulsion: non-linear braking when impending collision detected (dists < 1.8)
        I_hazard = np.where(dists < 1.8, ((1.8 - dists) / 1.8) ** 2, 0.0)
        H = self.diffuse @ I_hazard

        # Net afferent drive
        net_drive = 2.5 * I_goal + 1.0 * I_prev - 4.0 * H

        # 2. Recurrent continuous attractor relaxation
        a = np.maximum(0.0, net_drive)
        for _ in range(8):
            a = np.tanh(0.35 * (self.W @ a) + net_drive)
            a = np.maximum(0.0, a)

        # 3. Primary Georgopoulos population vector
        u_pop = self.basis.T @ a
        norm = float(np.linalg.norm(u_pop))
        pop_vector = (u_pop / norm) if norm > 1e-4 else dir_goal.copy()

        # 4. Extract active affordance peaks (local maxima in the population field)
        candidates = [pop_vector, dir_goal.copy(), prev_act.copy()]
        top_indices = np.argsort(a)[::-1][:4]
        for idx in top_indices:
            if a[idx] > 0.05 and dists[idx] > 0.6:
                candidates.append(self.basis[idx].copy())

        return candidates


class CAPN3DSensorimotorAgent:
    """Full CAPN closed-loop 3D agent with Motor Attractor Field & Efference Copy (ZERO ORACLES)."""

    def __init__(self, rng: Optional[np.random.RandomState] = None) -> None:
        self.rng = rng if rng is not None else np.random.RandomState(42)
        self.ray_dirs = np.array(generate_3d_ray_directions())
        self.dt = 0.1
        self.prev_act = np.array([0.577, 0.577, 0.577], dtype=np.float64)
        self.motor_attractor = ContinuousMotorAttractorField(self.ray_dirs)

    def reset(self) -> None:
        self.prev_act = np.array([0.577, 0.577, 0.577], dtype=np.float64)

    def predict_sensory_step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Innate corollary discharge projection (optical flow divergence)."""
        kinematic_delta = -(self.ray_dirs @ u) * self.dt
        return np.clip(s + kinematic_delta, 0.0, 5.0)

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Select 3D continuous velocity intention via counterfactual forward rollouts in sensory coordinates."""
        s = obs["distances"]
        dir_goal = obs["dir_to_goal"]

        # 1. Biological motor affordance competition (neural attractor field, zero cross products)
        candidates = self.motor_attractor.compute_affordances(dir_goal, s, self.prev_act)

        # 2. Prospective forward rollout evaluation in sensory coordinates (ZERO ORACLE!)
        best_score = -float("inf")
        best_act = candidates[0]

        for cand_act in candidates:
            sim_s = s.copy()
            hazard_penalty = 0.0

            for h in range(1, 16):
                sim_s = self.predict_sensory_step(sim_s, cand_act)
                min_dist = float(np.min(sim_s))
                if min_dist < 0.5:
                    penetration = 0.5 - min_dist
                    hazard_penalty += 200.0 * (1.0 / h) + 300.0 * (penetration ** 2)

            progress_score = float(np.dot(cand_act, dir_goal)) * 5.0
            smoothness_bonus = 2.0 * float(np.dot(cand_act, self.prev_act))
            score = progress_score - hazard_penalty + smoothness_bonus

            if score > best_score:
                best_score = score
                best_act = cand_act

        self.prev_act = best_act
        return best_act


class Blind3DSensorimotorAgent:
    """Null Control 1: Intact agent with range sensors zeroed (cannot perceive hazards)."""

    def __init__(self, rng: Optional[np.random.RandomState] = None) -> None:
        self.capn = CAPN3DSensorimotorAgent(rng=rng)

    def reset(self) -> None:
        self.capn.reset()

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        # Blind agent receives constant max distance (assumes free space)
        blind_obs = dict(obs)
        blind_obs["distances"] = np.full(14, 5.0, dtype=np.float64)
        return self.capn.select_action(blind_obs)


class ReactiveGreedy3DAgent:
    """Null Control 2: Reactive greedy agent without counterfactual forward lookahead."""

    def __init__(self, rng: Optional[np.random.RandomState] = None) -> None:
        self.rng = rng if rng is not None else np.random.RandomState(42)

    def reset(self) -> None:
        pass

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        dir_goal = obs["dir_to_goal"]
        dists = obs["distances"]

        # Pure greedy velocity towards goal
        act = dir_goal * 1.0

        # Immediate reactive braking if obstacle is directly ahead (< 0.6)
        if min(dists) < 0.6:
            act = -act * 0.5

        return act


class RandomWalk3DAgent:
    """Model-Free Baseline: Random continuous 3D velocity exploration."""

    def __init__(self, rng: Optional[np.random.RandomState] = None) -> None:
        self.rng = rng if rng is not None else np.random.RandomState(42)

    def reset(self) -> None:
        pass

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        v = self.rng.randn(3)
        norm = float(np.linalg.norm(v)) + 1e-8
        return (v / norm) * float(self.rng.uniform(0.3, 1.0))
