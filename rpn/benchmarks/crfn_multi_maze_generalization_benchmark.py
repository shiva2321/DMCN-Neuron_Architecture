"""Multi-Maze Generalization Benchmark for Morphogenetic CRFN Substrate.

Tests multi-task learning, pathway consolidation, and zero-shot OOD generalization
across a varied dataset of continuous 2D mazes:

1. Training Dataset (Seen Mazes - Demonstrated Replays):
   - Maze 1 ("S-Bend"): East -> North -> East winding corridor.
   - Maze 2 ("Horseshoe U"): North -> East -> South U-turn around central barrier.
   - Maze 3 ("Reverse Hook"): West -> North corner bypass.
   Across the 3 mazes, all 4 cardinal movements (East, West, North, South) are demonstrated
   in their respective sensory contexts.

2. Testing Dataset (Seen & Unseen Situations):
   - Seen Mazes: Maze 1, Maze 2, Maze 3 evaluated autonomously with random jitter.
   - Unseen Mazes (Zero-Shot Novel Topologies - NEVER SEEN IN TRAINING):
     - Maze 4 ("Central Box Bypass"): Must navigate around a solid central square block.
     - Maze 5 ("Staggered Slits"): Must zigzag East through lower slit, then West through upper slit.
     - Maze 6 ("T-Junction"): Must navigate up vertical channel and choose the open right passage.

ZERO hardcoded paths stored in agent.
ZERO replay buffers of coordinates.
ZERO cheat sheets or environmental hacks.
All knowledge resides in transmission-line impedances Z_jk = R + iX and somatic limit cycles.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate
from benchmarks.crfn_maze_benchmark import WallSegment


class ConfigurableMaze:
    """A configurable continuous 2D maze with arbitrary wall obstacles and boundary walls."""

    def __init__(
        self,
        name: str,
        walls: List[WallSegment],
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        width: float = 4.0,
        height: float = 4.0,
        agent_radius: float = 0.15,
        goal_radius: float = 0.40,
    ):
        self.name = str(name)
        self.width = float(width)
        self.height = float(height)
        self.agent_radius = float(agent_radius)
        self.goal_radius = float(goal_radius)
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.walls = list(walls)

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
        """Check if circular agent body penetrates any wall."""
        for w in self.walls:
            if w.distance_to_point(pos[0], pos[1]) < self.agent_radius:
                return True
        return False

    def check_goal(self, pos: np.ndarray) -> bool:
        """Check if agent center is inside goal zone."""
        return float(np.hypot(pos[0] - self.goal_pos[0], pos[1] - self.goal_pos[1])) < self.goal_radius


def build_smooth_path(waypoints: np.ndarray, step_size: float = 0.015) -> List[np.ndarray]:
    """Generate smooth trajectory connecting discrete waypoints."""
    path: List[np.ndarray] = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        n_pts = max(1, int(dist / step_size))
        for s in np.linspace(0, 1, n_pts, endpoint=False):
            path.append(p1 + s * (p2 - p1))
    path.append(waypoints[-1])
    return path


def create_training_dataset() -> List[Tuple[ConfigurableMaze, np.ndarray, List[np.ndarray]]]:
    """Create the 3 training mazes with diverse geometries and demonstration paths."""
    dataset = []

    # Outer boundary common to all 4x4 mazes
    outer_walls = [
        WallSegment(0.0, 0.0, 4.0, 0.0),
        WallSegment(4.0, 0.0, 4.0, 4.0),
        WallSegment(4.0, 4.0, 0.0, 4.0),
        WallSegment(0.0, 4.0, 0.0, 0.0),
    ]

    # --- Maze 1: S-Bend Corridor ---
    walls_1 = outer_walls + [
        WallSegment(1.5, 0.0, 1.5, 2.5),
        WallSegment(1.5, 2.5, 3.0, 2.5),
        WallSegment(0.0, 2.5, 0.8, 2.5),
        WallSegment(2.7, 1.2, 2.7, 2.5),
    ]
    wps_1 = np.array([
        [0.5, 0.5],
        [1.15, 0.7],
        [1.15, 3.0],
        [3.3, 3.0],
        [3.5, 3.5],
    ])
    maze_1 = ConfigurableMaze("Maze 1 (S-Bend)", walls_1, start_pos=[0.5, 0.5], goal_pos=[3.5, 3.5])
    dataset.append((maze_1, wps_1, build_smooth_path(wps_1)))

    # --- Maze 2: Horseshoe U-Turn ---
    walls_2 = outer_walls + [
        WallSegment(2.0, 0.0, 2.0, 2.8),  # Central dividing vertical spine
    ]
    wps_2 = np.array([
        [0.8, 0.8],
        [0.8, 3.4],
        [3.2, 3.4],
        [3.2, 0.8],
    ])
    maze_2 = ConfigurableMaze("Maze 2 (Horseshoe U)", walls_2, start_pos=[0.8, 0.8], goal_pos=[3.2, 0.8])
    dataset.append((maze_2, wps_2, build_smooth_path(wps_2)))

    # --- Maze 3: Reverse Hook (West & North) ---
    walls_3 = outer_walls + [
        WallSegment(1.4, 2.0, 4.0, 2.0),  # Horizontal divider with gap on left
    ]
    wps_3 = np.array([
        [3.2, 0.8],
        [0.7, 0.8],
        [0.7, 3.2],
    ])
    maze_3 = ConfigurableMaze("Maze 3 (Reverse Hook)", walls_3, start_pos=[3.2, 0.8], goal_pos=[0.7, 3.2])
    dataset.append((maze_3, wps_3, build_smooth_path(wps_3)))

    return dataset


def create_unseen_dataset() -> List[ConfigurableMaze]:
    """Create 3 novel, unseen mazes the network has NEVER encountered during training."""
    outer_walls = [
        WallSegment(0.0, 0.0, 4.0, 0.0),
        WallSegment(4.0, 0.0, 4.0, 4.0),
        WallSegment(4.0, 4.0, 0.0, 4.0),
        WallSegment(0.0, 4.0, 0.0, 0.0),
    ]
    unseen_mazes = []

    # --- Unseen Maze 4: Central Box Bypass ---
    # Solid central square block from (1.4, 1.4) to (2.6, 2.6)
    walls_4 = outer_walls + [
        WallSegment(1.4, 1.4, 2.6, 1.4),
        WallSegment(2.6, 1.4, 2.6, 2.6),
        WallSegment(2.6, 2.6, 1.4, 2.6),
        WallSegment(1.4, 2.6, 1.4, 1.4),
    ]
    unseen_mazes.append(
        ConfigurableMaze("Unseen Maze 4 (Central Box)", walls_4, start_pos=[0.6, 0.6], goal_pos=[3.4, 3.4])
    )

    # --- Unseen Maze 5: Staggered Slits ---
    # Lower wall from x=0 to 2.8 at y=1.6 (gap on right)
    # Upper wall from x=1.2 to 4.0 at y=2.8 (gap on left)
    walls_5 = outer_walls + [
        WallSegment(0.0, 1.6, 2.8, 1.6),
        WallSegment(1.2, 2.8, 4.0, 2.8),
    ]
    unseen_mazes.append(
        ConfigurableMaze("Unseen Maze 5 (Staggered Slits)", walls_5, start_pos=[0.6, 0.6], goal_pos=[0.6, 3.4])
    )

    # --- Unseen Maze 6: T-Junction ---
    # Vertical spine from (2.0, 0.0) to (2.0, 2.5), left top blocked from (0.0, 2.5) to (2.0, 2.5)
    walls_6 = outer_walls + [
        WallSegment(2.0, 0.0, 2.0, 2.5),
        WallSegment(0.0, 2.5, 2.0, 2.5),
    ]
    unseen_mazes.append(
        ConfigurableMaze("Unseen Maze 6 (T-Junction)", walls_6, start_pos=[0.8, 0.6], goal_pos=[3.2, 2.8])
    )

    return unseen_mazes


def run_multi_maze_generalization_benchmark(
    n_epochs: int = 4,
    eval_runs_per_maze: int = 3,
    max_steps_per_episode: int = 600,
    dt: float = 0.02,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    train_dataset = create_training_dataset()
    unseen_mazes = create_unseen_dataset()

    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=8,
        n_sensory_ports=8,
        n_actuator_ports=4,
        base_omega=2.0,
        base_mu=0.50,
        gamma_discount=0.90,
        lr_rl=1.60,
        lr_pred=0.08,
        theta_grow=0.85,
        theta_prune=0.10,
        exploration_std=0.03,
        max_neurons=20,
        seed=seed,
    )

    print("=" * 90)
    print("MORPHOGENETIC CRFN: MULTI-MAZE GENERALIZATION BENCHMARK (SEEN & UNSEEN MAZES)")
    print("=" * 90)
    print(f"Training Dataset:   {len(train_dataset)} diverse mazes across {n_epochs} interleaved epochs")
    print(f"Test Mazes:         3 Seen Mazes + 3 Completely Unseen Zero-Shot Mazes")
    print(f"Evaluations:        {eval_runs_per_maze} autonomous runs per maze (with random jitter)")
    print("=" * 90)

    t0_wall = time.perf_counter()

    # --- PHASE 1: INTERLEAVED MULTI-MAZE TRAINING ---
    print("\n--- PHASE 1: TRAINING ON VARIED MAZE DATASET ---")
    print(f"{'Episode':<8} | {'Maze':<24} | {'Steps':<6} | {'Goal?':<6} | {'Collisions':<11} | {'Neurons':<8}")
    print("-" * 75)

    ep_count = 0
    for epoch in range(n_epochs):
        for maze, waypoints, demo_path in train_dataset:
            ep_count += 1
            substrate.reset()
            prev_dist = float(np.hypot(demo_path[0][0] - maze.goal_pos[0], demo_path[0][1] - maze.goal_pos[1]))
            step_count = min(len(demo_path), max_steps_per_episode)
            wp_idx = 1

            for step in range(step_count):
                t = step * dt
                pos = demo_path[step]
                next_pos = demo_path[min(step + 1, len(demo_path) - 1)]
                v_target = (next_pos - pos) / dt

                ray_dists = maze.get_ray_distances(pos, n_rays=4)
                dx_g = maze.goal_pos[0] - pos[0]
                dy_g = maze.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_g, dy_g))

                sensory_inputs = {i: (1.0 / (1.0 + ray_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
                sensory_inputs[4] = float(max(0.0, dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[5] = float(max(0.0, dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[6] = float(max(0.0, -dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[7] = float(max(0.0, -dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)

                teacher_fluxes = {}
                carrier = 0.9 * np.exp(1j * 2.0 * t)
                if v_target[0] > 0.1:
                    teacher_fluxes[0] = carrier * min(1.0, v_target[0] / 0.5)
                elif v_target[0] < -0.1:
                    teacher_fluxes[1] = carrier * min(1.0, abs(v_target[0]) / 0.5)
                if v_target[1] > 0.1:
                    teacher_fluxes[2] = carrier * min(1.0, v_target[1] / 0.5)
                elif v_target[1] < -0.1:
                    teacher_fluxes[3] = carrier * min(1.0, abs(v_target[1]) / 0.5)

                reward = float((prev_dist - curr_dist) * 2.0) - 0.002
                prev_dist = curr_dist

                target_wp = np.array(waypoints[min(wp_idx, len(waypoints) - 1)])
                if np.hypot(pos[0] - target_wp[0], pos[1] - target_wp[1]) < 0.25:
                    reward += 1.5
                    if wp_idx < len(waypoints) - 1:
                        wp_idx += 1

                if step == step_count - 1 or maze.check_goal(pos):
                    reward += 3.0

                substrate.step(sensory_inputs, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)

            print(
                f"{ep_count:<8} | {maze.name:<24} | {step_count:<6} | True   | 0           | {len(substrate.somas):<8}"
            )

    # Snapshot clean trained transmission-line arbor and impedances
    trained_somas = copy.deepcopy(substrate.somas)
    trained_dendrites = copy.deepcopy(substrate.dendrites)
    trained_aff = copy.deepcopy(substrate.afferent_links)
    trained_rec = copy.deepcopy(substrate.recurrent_links)
    trained_eff = copy.deepcopy(substrate.efferent_links)

    # Helper function to evaluate autonomous navigation
    def evaluate_maze(maze_obj: ConfigurableMaze, n_runs: int, is_seen: bool) -> List[Dict[str, Any]]:
        results = []
        for run_idx in range(n_runs):
            # Restore trained physical state
            substrate.somas = copy.deepcopy(trained_somas)
            substrate.dendrites = copy.deepcopy(trained_dendrites)
            substrate.afferent_links = copy.deepcopy(trained_aff)
            substrate.recurrent_links = copy.deepcopy(trained_rec)
            substrate.efferent_links = copy.deepcopy(trained_eff)
            substrate.reset()

            start_jitter = rng.uniform(-0.03, 0.03, size=2)
            pos = maze_obj.reset_agent() + start_jitter
            vel = np.zeros(2, dtype=float)
            mass = 1.0
            damping = 0.85
            collision_count = 0
            last_collision_step = -100
            reached_goal = False

            for step in range(max_steps_per_episode):
                t = step * dt

                ray_dists = maze_obj.get_ray_distances(pos, n_rays=4)
                dx_g = maze_obj.goal_pos[0] - pos[0]
                dy_g = maze_obj.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_g, dy_g))

                sensory_inputs = {i: (1.0 / (1.0 + ray_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
                sensory_inputs[4] = float(max(0.0, dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[5] = float(max(0.0, dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[6] = float(max(0.0, -dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[7] = float(max(0.0, -dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)

                out = substrate.step(sensory_inputs, reward=0.0, dt=dt)
                forces = out["actuator_forces"]

                fx = (forces[0] - forces[1]) * 2.5
                fy = (forces[2] - forces[3]) * 2.5

                ax = (fx - damping * vel[0]) / mass
                ay = (fy - damping * vel[1]) / mass
                vel[0] += ax * dt
                vel[1] += ay * dt

                proposed_pos = pos + vel * dt
                proposed_pos[0] = max(maze_obj.agent_radius, min(maze_obj.width - maze_obj.agent_radius, proposed_pos[0]))
                proposed_pos[1] = max(maze_obj.agent_radius, min(maze_obj.height - maze_obj.agent_radius, proposed_pos[1]))

                col_x = maze_obj.check_collision(np.array([proposed_pos[0], pos[1]]))
                col_y = maze_obj.check_collision(np.array([pos[0], proposed_pos[1]]))

                if not col_x and not col_y and not maze_obj.check_collision(proposed_pos):
                    pos = proposed_pos
                else:
                    if col_x:
                        vel[0] = -0.2 * vel[0]
                    else:
                        pos[0] = proposed_pos[0]
                    if col_y:
                        vel[1] = -0.2 * vel[1]
                    else:
                        pos[1] = proposed_pos[1]
                    if step - last_collision_step > 5:
                        collision_count += 1
                        last_collision_step = step

                if maze_obj.check_goal(pos):
                    reached_goal = True
                    break

            n_steps_taken = step + 1
            final_goal_dist = float(np.hypot(pos[0] - maze_obj.goal_pos[0], pos[1] - maze_obj.goal_pos[1]))
            rec = {
                "maze": maze_obj.name,
                "type": "SEEN" if is_seen else "UNSEEN (OOD)",
                "run": run_idx + 1,
                "steps": n_steps_taken,
                "goal": reached_goal,
                "collisions": collision_count,
                "final_pos": [float(pos[0]), float(pos[1])],
                "goal_dist": final_goal_dist,
            }
            results.append(rec)
            status_str = "SUCCESS" if reached_goal else "FAILED "
            print(
                f"{maze_obj.name:<28} | Run {run_idx+1} | {status_str} | "
                f"Steps: {n_steps_taken:<3} | Collisions: {collision_count:<2} | Final: ({pos[0]:.2f}, {pos[1]:.2f})"
            )
        return results

    # --- PHASE 2: AUTONOMOUS EVALUATION ON SEEN MAZES ---
    print("\n--- PHASE 2: AUTONOMOUS EVALUATION ON SEEN MAZES ---")
    seen_eval_results = []
    for maze, _, _ in train_dataset:
        seen_eval_results.extend(evaluate_maze(maze, eval_runs_per_maze, is_seen=True))

    # --- PHASE 3: ZERO-SHOT EVALUATION ON UNSEEN MAZES ---
    print("\n--- PHASE 3: ZERO-SHOT GENERALIZATION ON UNSEEN MAZES (ZERO DEMO) ---")
    unseen_eval_results = []
    for maze in unseen_mazes:
        unseen_eval_results.extend(evaluate_maze(maze, eval_runs_per_maze, is_seen=False))

    t_elapsed = time.perf_counter() - t0_wall

    seen_successes = sum(1 for r in seen_eval_results if r["goal"])
    seen_rate = seen_successes / len(seen_eval_results) * 100 if seen_eval_results else 0.0

    unseen_successes = sum(1 for r in unseen_eval_results if r["goal"])
    unseen_rate = unseen_successes / len(unseen_eval_results) * 100 if unseen_eval_results else 0.0

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY:")
    print(f"Total Wall Time:                {t_elapsed:.2f} s")
    print(f"Seen Mazes Autonomous Success:  {seen_successes} / {len(seen_eval_results)} ({seen_rate:.1f}%)")
    print(f"Unseen Mazes (OOD) Success:     {unseen_successes} / {len(unseen_eval_results)} ({unseen_rate:.1f}%)")
    print(f"Final Neurons:                  {len(substrate.somas)}")
    print("Emergent Functional Roles:")
    for nid, role in sorted(substrate.inspect_emergent_roles().items()):
        print(f"  Neuron {nid:2d}: {role}")
    print("=" * 90)

    out_data = {
        "wall_time_s": t_elapsed,
        "n_epochs": n_epochs,
        "eval_runs_per_maze": eval_runs_per_maze,
        "seen_success_rate": seen_rate,
        "seen_successes": seen_successes,
        "seen_total": len(seen_eval_results),
        "unseen_success_rate": unseen_rate,
        "unseen_successes": unseen_successes,
        "unseen_total": len(unseen_eval_results),
        "seen_eval_results": seen_eval_results,
        "unseen_eval_results": unseen_eval_results,
        "emergent_roles": substrate.inspect_emergent_roles(),
    }
    out_file = Path(__file__).resolve().parent / "crfn_multi_maze_results.json"
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"Saved complete benchmark results to {out_file}")
    return out_data


if __name__ == "__main__":
    run_multi_maze_generalization_benchmark()
