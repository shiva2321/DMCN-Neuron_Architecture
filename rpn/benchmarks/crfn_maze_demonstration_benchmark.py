"""Demonstration-Guided Learning Benchmark in Continuous 2D Maze.

Tests observational/imitation learning in the Morphogenetic CRFN Substrate:
- Phase 1 (Demonstration): Guided episodes where a teacher leads the agent along the collision-free corridor
  to reach the goal. Sensory waves and motor waves co-occur, ringing eligibility traces in real time.
  Goal attainment triggers a dopamine surge (+5.0), matching transmission-line impedances.
- Phase 2 (Autonomous Navigation): Teacher is 100% removed (guidance force = 0, teacher flux = None).
  The agent is placed at the start (0.5, 0.5) (with slight random jitter) and must navigate purely
  using its internally learned wave transmission lines and Stuart-Landau somatic oscillations.

ZERO hardcoded paths stored in the agent.
ZERO replay buffers or coordinate tapes.
All learning resides entirely within transmission-line impedances Z_jk = R + iX.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate
from benchmarks.crfn_maze_benchmark import Continuous2DMaze


CORRIDOR_WAYPOINTS = np.array([
    [0.5, 0.5],   # Start
    [1.15, 0.7],  # Align with corridor opening
    [1.15, 3.0],  # Move North through opening at y=2.5 into upper corridor
    [3.3, 3.0],   # Move East along upper corridor
    [3.5, 3.5],   # Arrive at goal zone
])


def build_demonstration_path() -> List[np.ndarray]:
    """Generate a smooth, collision-free demonstration trajectory from start to goal."""
    waypoints = CORRIDOR_WAYPOINTS
    path: List[np.ndarray] = []
    step_size = 0.015  # ~0.75 m/s at dt=0.02
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        n_pts = max(1, int(dist / step_size))
        for s in np.linspace(0, 1, n_pts, endpoint=False):
            path.append(p1 + s * (p2 - p1))
    path.append(waypoints[-1])
    return path


def run_demonstration_maze_benchmark(
    n_demo_episodes: int = 10,
    n_autonomous_episodes: int = 10,
    max_steps_per_episode: int = 600,
    dt: float = 0.02,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    maze = Continuous2DMaze(width=4.0, height=4.0)
    demo_path = build_demonstration_path()

    # 8 Sensory Ports (4 ray proximity + 4 goal beacon directions)
    # 4 Actuator Ports (East, West, North, South thrusters)
    substrate = MorphogeneticCRFNSubstrate(
        n_initial_neurons=8,
        n_sensory_ports=8,
        n_actuator_ports=4,
        base_omega=2.0,
        base_mu=0.50,
        gamma_discount=0.90,
        lr_rl=1.80,  # Fast imprint during demonstration
        lr_pred=0.08,
        theta_grow=0.85,
        theta_prune=0.10,
        exploration_std=0.03,  # Low noise for clean imitation
        max_neurons=18,
        seed=seed,
    )

    total_episodes = n_demo_episodes + n_autonomous_episodes

    print("=" * 86)
    print("MORPHOGENETIC CRFN: DEMONSTRATION-GUIDED LEARNING & AUTONOMOUS MAZE NAVIGATION")
    print("=" * 86)
    print(
        f"{'Episode':<8} | {'Mode':<11} | {'Steps':<6} | {'Goal?':<6} | {'Collisions':<11} | "
        f"{'Final Pos':<14} | {'Goal Dist':<10} | {'Neurons':<8}"
    )
    print("-" * 86)

    t0_wall = time.perf_counter()
    demo_results = []
    auto_results = []

    for ep in range(total_episodes):
        is_demo = ep < n_demo_episodes
        mode_str = "GUIDED DEMO" if is_demo else "AUTONOMOUS"

        substrate.reset()
        collision_count = 0
        reached_goal = False

        if is_demo:
            # Demonstration Phase: Teacher leads agent smoothly along collision-free path
            prev_dist = np.hypot(demo_path[0][0] - maze.goal_pos[0], demo_path[0][1] - maze.goal_pos[1])
            step_count = min(len(demo_path), max_steps_per_episode)
            wp_idx = 1

            for step in range(step_count):
                t = step * dt
                pos = demo_path[step]
                next_pos = demo_path[min(step + 1, len(demo_path) - 1)]
                v_target = (next_pos - pos) / dt

                ray_dists = maze.get_ray_distances(pos, n_rays=4)
                dx_goal = maze.goal_pos[0] - pos[0]
                dy_goal = maze.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_goal, dy_goal))

                # Build sensory input waves
                sensory_inputs = {}
                for i in range(4):
                    amp = 1.0 / (1.0 + ray_dists[i])
                    sensory_inputs[i] = amp * np.exp(1j * 2.0 * t)
                sensory_inputs[4] = float(max(0.0, dx_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[5] = float(max(0.0, dy_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[6] = float(max(0.0, -dx_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[7] = float(max(0.0, -dy_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)

                # Teacher translates desired velocity into actuator fluxes
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

                # Progress and waypoint sub-goal rewards
                progress = prev_dist - curr_dist
                reward = float(progress * 2.0) - 0.002
                prev_dist = curr_dist

                # Check waypoint passage (reinforce intermediate turns)
                target_wp = np.array(CORRIDOR_WAYPOINTS[min(wp_idx, len(CORRIDOR_WAYPOINTS) - 1)])
                if np.hypot(pos[0] - target_wp[0], pos[1] - target_wp[1]) < 0.25:
                    reward += 1.5  # Sub-goal dopamine pulse for mastering this corridor segment
                    if wp_idx < len(CORRIDOR_WAYPOINTS) - 1:
                        wp_idx += 1

                # Check if arrived at final goal
                if maze.check_goal(pos) or step == step_count - 1:
                    reached_goal = True
                    reward += 3.0  # Final dopamine surge at destination

                # Step substrate with teacher motor flux
                substrate.step(sensory_inputs, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)

            n_steps_taken = step_count
            final_pos = demo_path[-1]
            curr_goal_dist = float(np.hypot(final_pos[0] - maze.goal_pos[0], final_pos[1] - maze.goal_pos[1]))

            # Snapshot the learned impedances and arbor structure for clean evaluation
            if ep == n_demo_episodes - 1:
                trained_somas = copy.deepcopy(substrate.somas)
                trained_dendrites = copy.deepcopy(substrate.dendrites)
                trained_aff = copy.deepcopy(substrate.afferent_links)
                trained_rec = copy.deepcopy(substrate.recurrent_links)
                trained_eff = copy.deepcopy(substrate.efferent_links)

        else:
            # Autonomous Evaluation Phase: ZERO Teacher, 100% on internal waves
            # Evaluate the clean demonstrated policy from initial conditions
            substrate.somas = copy.deepcopy(trained_somas)
            substrate.dendrites = copy.deepcopy(trained_dendrites)
            substrate.afferent_links = copy.deepcopy(trained_aff)
            substrate.recurrent_links = copy.deepcopy(trained_rec)
            substrate.efferent_links = copy.deepcopy(trained_eff)
            substrate.reset()

            start_jitter = rng.uniform(-0.03, 0.03, size=2)
            pos = maze.reset_agent() + start_jitter
            vel = np.zeros(2, dtype=float)
            mass = 1.0
            damping = 0.85
            prev_goal_dist = np.hypot(pos[0] - maze.goal_pos[0], pos[1] - maze.goal_pos[1])
            last_collision_step = -100

            for step in range(max_steps_per_episode):
                t = step * dt

                ray_dists = maze.get_ray_distances(pos, n_rays=4)
                dx_goal = maze.goal_pos[0] - pos[0]
                dy_goal = maze.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_goal, dy_goal))

                sensory_inputs = {}
                for i in range(4):
                    amp = 1.0 / (1.0 + ray_dists[i])
                    sensory_inputs[i] = amp * np.exp(1j * 2.0 * t)
                sensory_inputs[4] = float(max(0.0, dx_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[5] = float(max(0.0, dy_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[6] = float(max(0.0, -dx_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sensory_inputs[7] = float(max(0.0, -dy_goal) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)

                # Autonomous step: teacher_actuator_fluxes is None, reward = 0 during pure evaluation
                out = substrate.step(sensory_inputs, reward=0.0, dt=dt)
                forces = out["actuator_forces"]

                fx = (forces[0] - forces[1]) * 2.5
                fy = (forces[2] - forces[3]) * 2.5

                ax = (fx - damping * vel[0]) / mass
                ay = (fy - damping * vel[1]) / mass
                vel[0] += ax * dt
                vel[1] += ay * dt

                proposed_pos = pos + vel * dt
                proposed_pos[0] = max(maze.agent_radius, min(maze.width - maze.agent_radius, proposed_pos[0]))
                proposed_pos[1] = max(maze.agent_radius, min(maze.height - maze.agent_radius, proposed_pos[1]))

                col_x = maze.check_collision(np.array([proposed_pos[0], pos[1]]))
                col_y = maze.check_collision(np.array([pos[0], proposed_pos[1]]))

                if not col_x and not col_y and not maze.check_collision(proposed_pos):
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

                if maze.check_goal(pos):
                    reached_goal = True
                    break

            n_steps_taken = step + 1
            final_pos = pos
            curr_goal_dist = float(np.hypot(final_pos[0] - maze.goal_pos[0], final_pos[1] - maze.goal_pos[1]))

        record = {
            "episode": ep + 1,
            "mode": mode_str,
            "steps": n_steps_taken,
            "goal": reached_goal,
            "collisions": collision_count,
            "final_pos": [float(final_pos[0]), float(final_pos[1])],
            "goal_dist": float(curr_goal_dist),
            "neurons": len(substrate.somas),
        }
        if is_demo:
            demo_results.append(record)
        else:
            auto_results.append(record)

        print(
            f"{ep + 1:<8} | {mode_str:<11} | {n_steps_taken:<6} | {str(reached_goal):<6} | "
            f"{collision_count:<11} | ({final_pos[0]:.2f}, {final_pos[1]:.2f}){'':<4} | "
            f"{curr_goal_dist:.2f}{'':<6} | {len(substrate.somas):<8}"
        )

    t_elapsed = time.perf_counter() - t0_wall
    auto_goals = [r["goal"] for r in auto_results]
    auto_success_rate = sum(auto_goals) / len(auto_goals) * 100 if auto_goals else 0.0

    print("-" * 86)
    print("BENCHMARK SUMMARY:")
    print(f"Total Wall Time:                 {t_elapsed:.2f} s")
    print(f"Guided Demonstration Episodes:   {n_demo_episodes}")
    print(f"Autonomous Evaluation Episodes:  {n_autonomous_episodes}")
    print(f"Autonomous Goal Success Rate:    {sum(auto_goals)} / {len(auto_goals)} ({auto_success_rate:.1f}%)")
    print(f"Final Neurons:                   {len(substrate.somas)}")
    print("Emergent Functional Roles:")
    for nid, role in sorted(substrate.inspect_emergent_roles().items()):
        print(f"  Neuron {nid:2d}: {role}")
    print("=" * 86)

    out_data = {
        "n_demo_episodes": n_demo_episodes,
        "n_autonomous_episodes": n_autonomous_episodes,
        "wall_time_s": t_elapsed,
        "autonomous_success_rate": auto_success_rate,
        "demo_results": demo_results,
        "autonomous_results": auto_results,
        "final_emergent_roles": substrate.inspect_emergent_roles(),
    }
    out_file = Path(__file__).resolve().parent / "crfn_maze_demonstration_results.json"
    out_file.write_text(json.dumps(out_data, indent=2))
    print(f"Saved results to {out_file}")
    return out_data


if __name__ == "__main__":
    run_demonstration_maze_benchmark()
