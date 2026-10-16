"""Generate full step-by-step telemetry for Live Maze & Neural Dynamics Visualizer.

Captures complete high-resolution state at every dt=0.02s:
- Maze geometry (walls, start, goal, radius)
- Agent kinematics (pos, vel, orientation)
- Sensory ray casts (4 distance readings + goal vector)
- Actuator forces (East, West, North, South thrusts)
- Somatic oscillators (z_j real and imag for all somas)
- Neuromodulated impedances (power transmission T across efferent links)
- Dopamine RPE pulses delta(t) and rewards
- Collisions and goal triggers
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate
from benchmarks.crfn_multi_maze_generalization_benchmark import (
    ConfigurableMaze,
    create_training_dataset,
    create_unseen_dataset,
)


def serialize_wall(w) -> Dict[str, float]:
    return {"x1": float(w.x1), "y1": float(w.y1), "x2": float(w.x2), "y2": float(w.y2)}


def generate_telemetry_dataset() -> Dict[str, Any]:
    rng = np.random.RandomState(42)
    dt = 0.02
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
        seed=42,
    )

    recorded_episodes: List[Dict[str, Any]] = []

    print("Generating telemetry for Live Visualizer...")

    # --- 1. RECORD TRAINING DEMONSTRATION EPISODES ---
    # We record Demonstration on Maze 1, Maze 2, Maze 3
    for ep_idx, (maze, waypoints, demo_path) in enumerate(train_dataset):
        substrate.reset()
        ep_name = f"Train Ep {ep_idx + 1}: Guided Demo on {maze.name}"
        frames = []
        step_count = min(len(demo_path), 500)
        prev_dist = float(np.hypot(demo_path[0][0] - maze.goal_pos[0], demo_path[0][1] - maze.goal_pos[1]))
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
            is_wp = False
            if np.hypot(pos[0] - target_wp[0], pos[1] - target_wp[1]) < 0.25:
                reward += 1.5
                is_wp = True
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1

            is_goal = False
            if step == step_count - 1 or maze.check_goal(pos):
                reward += 3.0
                is_goal = True

            out = substrate.step(sensory_inputs, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)

            # Sample every 2 steps (10 Hz) to keep payload compact while perfectly smooth
            if step % 2 == 0 or is_goal:
                forces = out["actuator_forces"]
                soma_states = [[float(s.z.real), float(s.z.imag)] for s in substrate.somas.values()]
                eff_trans = [
                    {"src": nid, "act": aid, "T": float(l.power_transmission), "Z_real": float(l.Z.real)}
                    for (nid, aid), l in list(substrate.efferent_links.items())[:12]
                ]
                frames.append({
                    "step": step,
                    "t": round(t, 3),
                    "pos": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
                    "vel": [round(float(v_target[0]), 3), round(float(v_target[1]), 3)],
                    "rays": [round(float(r), 3) for r in ray_dists],
                    "goal_vec": [round(float(dx_g), 3), round(float(dy_g), 3)],
                    "forces": [round(float(f), 4) for f in forces],
                    "dopamine_rpe": round(float(out["rpe"]), 4),
                    "reward": round(float(reward), 4),
                    "n_neurons": len(substrate.somas),
                    "somas": soma_states,
                    "eff_trans": eff_trans,
                    "collision": False,
                    "goal": is_goal,
                })

        recorded_episodes.append({
            "id": f"demo_{ep_idx + 1}",
            "title": ep_name,
            "type": "TRAINING DEMO",
            "maze": {
                "name": maze.name,
                "width": maze.width,
                "height": maze.height,
                "start": [float(maze.start_pos[0]), float(maze.start_pos[1])],
                "goal": [float(maze.goal_pos[0]), float(maze.goal_pos[1])],
                "goal_radius": maze.goal_radius,
                "walls": [serialize_wall(w) for w in maze.walls],
            },
            "frames": frames,
        })
        print(f"Recorded {ep_name} ({len(frames)} frames)")

    # Complete 3 more epochs of training to fully consolidate impedances
    for epoch in range(1, 4):
        for maze, waypoints, demo_path in train_dataset:
            substrate.reset()
            prev_dist = float(np.hypot(demo_path[0][0] - maze.goal_pos[0], demo_path[0][1] - maze.goal_pos[1]))
            step_count = min(len(demo_path), 500)
            wp_idx = 1
            for step in range(step_count):
                t = step * dt
                pos = demo_path[step]
                next_pos = demo_path[min(step + 1, len(demo_path) - 1)]
                v_target = (next_pos - pos) / dt
                ray_dists = maze.get_ray_distances(pos, n_rays=4)
                dx_g = maze.goal_pos[0] - pos[0]; dy_g = maze.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_g, dy_g))
                sens = {i: (1.0 / (1.0 + ray_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
                sens[4] = float(max(0.0, dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[5] = float(max(0.0, dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[6] = float(max(0.0, -dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[7] = float(max(0.0, -dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                teacher_fluxes = {}
                carrier = 0.9 * np.exp(1j * 2.0 * t)
                if v_target[0] > 0.1: teacher_fluxes[0] = carrier * min(1.0, v_target[0] / 0.5)
                elif v_target[0] < -0.1: teacher_fluxes[1] = carrier * min(1.0, abs(v_target[0]) / 0.5)
                if v_target[1] > 0.1: teacher_fluxes[2] = carrier * min(1.0, v_target[1] / 0.5)
                elif v_target[1] < -0.1: teacher_fluxes[3] = carrier * min(1.0, abs(v_target[1]) / 0.5)
                reward = float((prev_dist - curr_dist) * 2.0) - 0.002
                prev_dist = curr_dist
                target_wp = np.array(waypoints[min(wp_idx, len(waypoints) - 1)])
                if np.hypot(pos[0] - target_wp[0], pos[1] - target_wp[1]) < 0.25:
                    reward += 1.5
                    if wp_idx < len(waypoints) - 1: wp_idx += 1
                if step == step_count - 1 or maze.check_goal(pos): reward += 3.0
                substrate.step(sens, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)

    # Snapshot clean trained state
    trained_somas = copy.deepcopy(substrate.somas)
    trained_dendrites = copy.deepcopy(substrate.dendrites)
    trained_aff = copy.deepcopy(substrate.afferent_links)
    trained_rec = copy.deepcopy(substrate.recurrent_links)
    trained_eff = copy.deepcopy(substrate.efferent_links)

    # --- 2. RECORD AUTONOMOUS EVALUATION EPISODES ---
    # We record:
    # A) Seen Maze 1 (S-Bend) Autonomous
    # B) Seen Maze 2 (Horseshoe U) Autonomous
    # C) Unseen Maze 4 (Central Box Bypass - Zero-Shot OOD Success)
    # D) Unseen Maze 6 (T-Junction - Zero-Shot OOD)

    eval_scenarios = [
        ("auto_seen_1", "Autonomous on Seen Maze 1 (S-Bend)", train_dataset[0][0], "SEEN MAZE"),
        ("auto_seen_2", "Autonomous on Seen Maze 2 (Horseshoe U)", train_dataset[1][0], "SEEN MAZE"),
        ("auto_unseen_4", "Zero-Shot OOD on Unseen Maze 4 (Central Box Bypass)", unseen_mazes[0], "UNSEEN OOD (ZERO DEMO)"),
        ("auto_unseen_6", "Zero-Shot OOD on Unseen Maze 6 (T-Junction)", unseen_mazes[2], "UNSEEN OOD (ZERO DEMO)"),
    ]

    for ep_id, title, maze_obj, tag in eval_scenarios:
        # Restore trained state
        substrate.somas = copy.deepcopy(trained_somas)
        substrate.dendrites = copy.deepcopy(trained_dendrites)
        substrate.afferent_links = copy.deepcopy(trained_aff)
        substrate.recurrent_links = copy.deepcopy(trained_rec)
        substrate.efferent_links = copy.deepcopy(trained_eff)
        substrate.reset()

        pos = maze_obj.reset_agent() + rng.uniform(-0.02, 0.02, size=2)
        vel = np.zeros(2, dtype=float)
        mass = 1.0
        damping = 0.85
        frames = []
        last_collision_step = -100

        for step in range(450):
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
            is_collision = False

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
                    is_collision = True
                    last_collision_step = step

            is_goal = maze_obj.check_goal(pos)

            if step % 2 == 0 or is_goal or is_collision:
                soma_states = [[float(s.z.real), float(s.z.imag)] for s in substrate.somas.values()]
                eff_trans = [
                    {"src": nid, "act": aid, "T": float(l.power_transmission), "Z_real": float(l.Z.real)}
                    for (nid, aid), l in list(substrate.efferent_links.items())[:12]
                ]
                frames.append({
                    "step": step,
                    "t": round(t, 3),
                    "pos": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
                    "vel": [round(float(vel[0]), 3), round(float(vel[1]), 3)],
                    "rays": [round(float(r), 3) for r in ray_dists],
                    "goal_vec": [round(float(dx_g), 3), round(float(dy_g), 3)],
                    "forces": [round(float(f), 4) for f in forces],
                    "dopamine_rpe": round(float(out["rpe"]), 4),
                    "reward": 0.0,
                    "n_neurons": len(substrate.somas),
                    "somas": soma_states,
                    "eff_trans": eff_trans,
                    "collision": is_collision,
                    "goal": is_goal,
                })

            if is_goal:
                break

        recorded_episodes.append({
            "id": ep_id,
            "title": title,
            "type": tag,
            "maze": {
                "name": maze_obj.name,
                "width": maze_obj.width,
                "height": maze_obj.height,
                "start": [float(maze_obj.start_pos[0]), float(maze_obj.start_pos[1])],
                "goal": [float(maze_obj.goal_pos[0]), float(maze_obj.goal_pos[1])],
                "goal_radius": maze_obj.goal_radius,
                "walls": [serialize_wall(w) for w in maze_obj.walls],
            },
            "frames": frames,
        })
        print(f"Recorded {title} ({len(frames)} frames, Reached: {is_goal})")

    out_file = Path(__file__).resolve().parent / "crfn_live_telemetry.json"
    data = {"episodes": recorded_episodes}
    out_file.write_text(json.dumps(data, indent=2))
    print(f"Successfully generated telemetry data at {out_file} ({len(recorded_episodes)} episodes).")
    return data


if __name__ == "__main__":
    generate_telemetry_dataset()
