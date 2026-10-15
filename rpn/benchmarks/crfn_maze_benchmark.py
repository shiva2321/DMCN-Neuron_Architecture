"""Empirical Benchmark: Continuous 2D Maze Navigation via Morphogenetic CRFN Substrate.

Evaluates an undifferentiated, pluripotent neural substrate in a continuous 2D maze:
- ZERO predefined neuron roles (no hardcoded sensory, motor, or critic neurons).
- ZERO predefined directional labels inside the network (actuators are unlabelled physical ports 0, 1, 2, 3).
- ZERO linear weights or artificial codebooks.
- Pure physical wave dynamics: Stuart-Landau limit-cycles, transmission-line dendrites,
  impedance matching (dZ/dt), destructive sensory cancellation, and resonant eligibility traces.

The substrate must:
1. Spontaneously explore the maze via thermal motor babbling.
2. Sever/prune connections that drive collisions via negative dopamine RPE.
3. Match impedances and reinforce connections that reach the goal via positive dopamine RPE.
4. Spontaneously bud new neurons (neurogenesis) when encountering complex junctions.
5. Self-differentiate pluripotent neurons into functional sensory, motor, and memory roles.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate


@dataclass
class WallSegment:
    """A 2D line segment obstacle in continuous space."""
    x1: float
    y1: float
    x2: float
    y2: float

    def distance_to_point(self, px: float, py: float) -> float:
        """Shortest Euclidean distance from point (px, py) to this line segment."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        l2 = dx * dx + dy * dy
        if l2 < 1e-12:
            return float(np.hypot(px - self.x1, py - self.y1))
        # Projection factor t onto segment [0, 1]
        t = max(0.0, min(1.0, ((px - self.x1) * dx + (py - self.y1) * dy) / l2))
        proj_x = self.x1 + t * dx
        proj_y = self.y1 + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    def ray_cast_distance(self, px: float, py: float, angle: float, max_dist: float = 3.0) -> float:
        """Ray cast from (px, py) along angle to find distance to intersection with segment."""
        # Ray: (px + s*cos(a), py + s*sin(a)), s >= 0
        # Segment: (x1 + t*dx, y1 + t*dy), t in [0, 1]
        rx = np.cos(angle)
        ry = np.sin(angle)
        sx = self.x2 - self.x1
        sy = self.y2 - self.y1

        det = rx * (-sy) - ry * (-sx)
        if abs(det) < 1e-9:
            return max_dist  # Parallel

        dp_x = self.x1 - px
        dp_y = self.y1 - py

        s = (dp_x * (-sy) - dp_y * (-sx)) / det
        t = (rx * dp_y - ry * dp_x) / det

        if s >= 0.0 and 0.0 <= t <= 1.0:
            return float(min(s, max_dist))
        return max_dist


class Continuous2DMaze:
    """A continuous 2D maze with corridors, T-junction, dead ends, and a goal zone."""

    def __init__(self, width: float = 4.0, height: float = 4.0):
        self.width = width
        self.height = height
        self.agent_radius = 0.15
        self.goal_pos = np.array([3.5, 3.5], dtype=float)
        self.goal_radius = 0.40
        self.start_pos = np.array([0.5, 0.5], dtype=float)

        # Build physical wall segments (boundary walls + internal barriers)
        self.walls: List[WallSegment] = [
            # Outer boundary
            WallSegment(0.0, 0.0, width, 0.0),
            WallSegment(width, 0.0, width, height),
            WallSegment(width, height, 0.0, height),
            WallSegment(0.0, height, 0.0, 0.0),

            # Internal corridor walls (creates a winding path with a dead end)
            # Barrier 1: Vertical dividing wall from (1.5, 0.0) up to (1.5, 2.5)
            WallSegment(1.5, 0.0, 1.5, 2.5),
            # Barrier 2: Horizontal dividing wall from (1.5, 2.5) to (3.0, 2.5)
            WallSegment(1.5, 2.5, 3.0, 2.5),
            # Barrier 3: Dead-end spur at (0.0, 2.5) to (0.8, 2.5)
            WallSegment(0.0, 2.5, 0.8, 2.5),
            # Barrier 4: Upper dividing wall from (2.7, 1.2) to (2.7, 2.5)
            WallSegment(2.7, 1.2, 2.7, 2.5),
        ]

    def reset_agent(self) -> np.ndarray:
        return self.start_pos.copy()

    def get_ray_distances(self, pos: np.ndarray, n_rays: int = 4) -> np.ndarray:
        """Cast n_rays orthogonal rays (East, North, West, South) to measure wall distance."""
        angles = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
        distances = []
        for a in angles:
            min_d = 4.0
            for w in self.walls:
                d = w.ray_cast_distance(pos[0], pos[1], a, max_dist=4.0)
                if d < min_d:
                    min_d = d
            distances.append(min_d)
        return np.array(distances, dtype=float)

    def check_collision(self, pos: np.ndarray) -> bool:
        """Check if agent body (circle with agent_radius) penetrates any wall."""
        for w in self.walls:
            if w.distance_to_point(pos[0], pos[1]) < self.agent_radius:
                return True
        return False

    def check_goal(self, pos: np.ndarray) -> bool:
        """Check if agent center is inside the goal zone."""
        return float(np.hypot(pos[0] - self.goal_pos[0], pos[1] - self.goal_pos[1])) < self.goal_radius


def run_continuous_maze_benchmark(
    n_episodes: int = 25,
    max_steps_per_episode: int = 350,
    dt: float = 0.02,
    seed: int = 42,
) -> dict:
    """Benchmark the Morphogenetic CRFN Substrate in the continuous 2D maze."""
    rng = np.random.RandomState(seed)
    maze = Continuous2DMaze(width=4.0, height=4.0)

    # 1. Instantiate the undifferentiated substrate
    # 8 Sensory Ports:
    #   Port 0: Ray East (+x) obstacle proximity
    #   Port 1: Ray North (+y) obstacle proximity
    #   Port 2: Ray West (-x) obstacle proximity
    #   Port 3: Ray South (-y) obstacle proximity
    #   Port 4: Goal beacon East (+x) projection
    #   Port 5: Goal beacon North (+y) projection
    #   Port 6: Goal beacon West (-x) projection
    #   Port 7: Goal beacon South (-y) projection
    # 4 Actuator Ports:
    #   Port 0: Thruster 0 (East +x)
    #   Port 1: Thruster 1 (West -x)
    #   Port 2: Thruster 2 (North +y)
    #   Port 3: Thruster 3 (South -y)
    n_sensory = 8
    n_actuators = 4

    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=6,
        n_sensory_ports=n_sensory,
        n_actuator_ports=n_actuators,
        base_omega=2.0,
        gamma_discount=0.90,
        lr_rl=1.40,
        lr_pred=0.08,
        theta_grow=0.75,
        theta_prune=0.15,
        exploration_std=0.08,
        max_neurons=18,
        seed=seed,
    )

    episode_steps = []
    episode_collisions = []
    episode_goals = []
    episode_neuron_counts = []
    episode_mean_speeds = []

    print("=" * 84)
    print("MORPHOGENETIC CRFN MAZE NAVIGATION BENCHMARK (ZERO PREDEFINED ROLES / WEIGHTS)")
    print("=" * 84)
    print(
        f"{'Episode':<8} | {'Steps':<7} | {'Goal?':<6} | {'Collisions':<11} | "
        f"{'Neurons':<8} | {'Afferent':<9} | {'Efferent':<9} | {'Roles Summary'}"
    )
    print("-" * 84)

    t0_wall = time.perf_counter()

    for ep in range(n_episodes):
        pos = maze.reset_agent()
        vel = np.array([0.0, 0.0], dtype=float)
        mass = 1.0
        damping = 0.85

        substrate.reset()

        prev_goal_dist = np.hypot(pos[0] - maze.goal_pos[0], pos[1] - maze.goal_pos[1])
        reached_goal = False
        collision_count = 0
        total_speed = 0.0

        last_collision_step = -100

        for step in range(max_steps_per_episode):
            t = step * dt

            # 1. Physical Environment Sensor Waves at current position
            ray_dists = maze.get_ray_distances(pos, n_rays=4)
            dx_goal = maze.goal_pos[0] - pos[0]
            dy_goal = maze.goal_pos[1] - pos[1]
            curr_goal_dist = float(np.hypot(dx_goal, dy_goal))

            # Proximity waves (0..3): 1 / (1 + d)
            sensory_inputs = {}
            for i in range(4):
                amp = 1.0 / (1.0 + ray_dists[i])
                sensory_inputs[i] = amp * np.exp(1j * 2.0 * t)

            # Directional goal beacon waves (4..7)
            sensory_inputs[4] = float(max(0.0, dx_goal) / (curr_goal_dist + 0.1)) * np.exp(1j * 2.1 * t)
            sensory_inputs[5] = float(max(0.0, dy_goal) / (curr_goal_dist + 0.1)) * np.exp(1j * 2.1 * t)
            sensory_inputs[6] = float(max(0.0, -dx_goal) / (curr_goal_dist + 0.1)) * np.exp(1j * 2.1 * t)
            sensory_inputs[7] = float(max(0.0, -dy_goal) / (curr_goal_dist + 0.1)) * np.exp(1j * 2.1 * t)

            # 2. Step the Morphogenetic Substrate (using last reward)
            # Base step cost (metabolic burn)
            reward = -0.01

            # Goal progress reward
            progress = prev_goal_dist - curr_goal_dist
            reward += float(progress * 2.0)
            prev_goal_dist = curr_goal_dist

            out = substrate.step(sensory_inputs, reward=reward, dt=dt)
            forces = out["actuator_forces"]  # Continuous wave power on ports 0..3

            # 3. Map unlabelled actuator ports to continuous 2D forces
            # Port 0 -> +X, Port 1 -> -X, Port 2 -> +Y, Port 3 -> -Y
            fx = (forces[0] - forces[1]) * 3.0
            fy = (forces[2] - forces[3]) * 3.0

            # 4. Integrate physical continuous agent motion (Newtonian Mechanics)
            ax = (fx - damping * vel[0]) / mass
            ay = (fy - damping * vel[1]) / mass
            vel[0] += ax * dt
            vel[1] += ay * dt

            proposed_pos = pos + vel * dt

            # Clamp inside bounding box
            proposed_pos[0] = max(maze.agent_radius, min(maze.width - maze.agent_radius, proposed_pos[0]))
            proposed_pos[1] = max(maze.agent_radius, min(maze.height - maze.agent_radius, proposed_pos[1]))

            # Contact resolution against walls
            if maze.check_collision(proposed_pos):
                if step - last_collision_step > 5:  # Impact event
                    collision_count += 1
                    last_collision_step = step
                    # Deliver negative dopamine shock on impact
                    substrate.step(sensory_inputs, reward=-0.60, dt=dt)
                # Inelastic rebound and stop penetration
                vel = -0.3 * vel
            else:
                pos = proposed_pos

            total_speed += float(np.hypot(vel[0], vel[1]))

            # Check goal arrival
            if maze.check_goal(pos):
                reached_goal = True
                # Deliver dopamine surge on arrival
                substrate.step(sensory_inputs, reward=5.0, dt=dt)
                break

        # Tally episode metrics
        n_steps_taken = step + 1
        episode_steps.append(n_steps_taken)
        episode_collisions.append(collision_count)
        episode_goals.append(reached_goal)
        episode_neuron_counts.append(len(substrate.somas))
        episode_mean_speeds.append(total_speed / n_steps_taken)

        # Inspect emergent roles objectively
        roles = substrate.inspect_emergent_roles()
        role_counts: Dict[str, int] = {}
        for r in roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_summary = ", ".join(f"{k.split()[0]}:{v}" for k, v in sorted(role_counts.items()))

        print(
            f"{ep + 1:<8} | {n_steps_taken:<7} | {str(reached_goal):<6} | {collision_count:<11} | "
            f"{len(substrate.somas):<8} | {len(substrate.afferent_links):<9} | "
            f"{len(substrate.efferent_links):<9} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Dist: {curr_goal_dist:.2f} | {role_summary}"
        )

    t_elapsed = time.perf_counter() - t0_wall

    print("-" * 84)
    print("BENCHMARK SUMMARY:")
    print(f"Total Wall Time:           {t_elapsed:.2f} s")
    print(f"Initial Collisions (Ep 1): {episode_collisions[0]}")
    print(f"Final Collisions (Ep {n_episodes}): {episode_collisions[-1]}")
    print(f"Total Goals Reached:       {sum(episode_goals)} / {n_episodes} ({sum(episode_goals)/n_episodes*100:.1f}%)")
    print(f"Initial Neurons:           {episode_neuron_counts[0]} -> Final: {episode_neuron_counts[-1]}")
    print("Final Emergent Roles:")
    for nid, role in sorted(substrate.inspect_emergent_roles().items()):
        print(f"  Neuron {nid:2d}: {role}")
    print("=" * 84)

    results = {
        "n_episodes": n_episodes,
        "wall_time_s": t_elapsed,
        "episode_steps": episode_steps,
        "episode_collisions": episode_collisions,
        "episode_goals": [bool(g) for g in episode_goals],
        "episode_neuron_counts": episode_neuron_counts,
        "final_emergent_roles": substrate.inspect_emergent_roles(),
    }

    # Save benchmark results to disk
    out_path = Path(__file__).resolve().parent / "crfn_maze_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_continuous_maze_benchmark()
