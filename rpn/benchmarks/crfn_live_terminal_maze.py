"""Live ANSI/ASCII Terminal Maze Player for Morphogenetic CRFN.

Renders a live, real-time top-down 2D maze view directly inside the console/terminal
with wall boundaries, agent position, sensory ray casts, thruster vectors, and live telemetry.
"""
from __future__ import annotations

import copy
import sys
import time
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate
from benchmarks.crfn_multi_maze_generalization_benchmark import create_training_dataset, create_unseen_dataset


def render_terminal_frame(
    maze,
    pos: np.ndarray,
    step: int,
    dist_g: float,
    forces: list[float],
    rpe: float,
    collisions: int,
    grid_w: int = 40,
    grid_h: int = 20,
    mode_str: str = "AUTONOMOUS",
) -> str:
    # Build 2D character grid
    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

    # Draw outer boundary
    for x in range(grid_w):
        grid[0][x] = "-"
        grid[grid_h - 1][x] = "-"
    for y in range(grid_h):
        grid[y][0] = "|"
        grid[y][grid_w - 1] = "|"
    grid[0][0] = "+"; grid[0][grid_w - 1] = "+"
    grid[grid_h - 1][0] = "+"; grid[grid_h - 1][grid_w - 1] = "+"

    # Helper to convert maze coords (0..4, 0..4) to grid coords (0..w-1, 0..h-1)
    def to_grid(mx: float, my: float) -> tuple[int, int]:
        gx = int(np.clip(mx / 4.0 * (grid_w - 2) + 1, 1, grid_w - 2))
        gy = int(np.clip((4.0 - my) / 4.0 * (grid_h - 2) + 1, 1, grid_h - 2))
        return gx, gy

    # Draw internal walls
    for w in maze.walls:
        pts = 25
        for s in np.linspace(0, 1, pts):
            wx = w.x1 + s * (w.x2 - w.x1)
            wy = w.y1 + s * (w.y2 - w.y1)
            if 0.05 < wx < 3.95 and 0.05 < wy < 3.95:
                gx, gy = to_grid(wx, wy)
                grid[gy][gx] = "#"

    # Draw Goal
    gx, gy = to_grid(maze.goal_pos[0], maze.goal_pos[1])
    grid[gy][gx] = "G"

    # Draw Start
    sx, sy = to_grid(maze.start_pos[0], maze.start_pos[1])
    if grid[sy][sx] == " ":
        grid[sy][sx] = "S"

    # Draw Agent
    ax, ay = to_grid(pos[0], pos[1])
    grid[ay][ax] = "A"

    # Format output text
    lines = []
    lines.append("+" + "-" * (grid_w) + "+")
    lines.append(f"| CRFN LIVE MAZE -- {maze.name:<24} |")
    lines.append("+" + "-" * (grid_w) + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * (grid_w) + "+")

    fx = (forces[0] - forces[1]) * 2.5
    fy = (forces[2] - forces[3]) * 2.5
    rpe_sign = "+" if rpe >= 0 else ""
    lines.append(f"MODE: [{mode_str}] | Step: {step:<4} | Collisions: {collisions:<2}")
    lines.append(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Goal Dist: {dist_g:.2f} m | G=Goal, A=Agent")
    lines.append(f"Forces: fx={fx:+.2f} N, fy={fy:+.2f} N | Dopamine RPE: {rpe_sign}{rpe:.3f}")
    return "\n".join(lines)


def run_live_terminal(maze_idx: int = 0, n_steps: int = 150):
    train_dataset = create_training_dataset()
    maze, wps, demo_path = train_dataset[maze_idx]

    # Pre-train baseline on demo
    sub = MorphogeneticCRFNSubstrate(n_initial_neurons=8, n_sensory_ports=8, n_actuator_ports=4, seed=42)
    dt = 0.02
    for _ in range(8):
        sub.reset()
        for s in range(len(demo_path)):
            t = s * dt
            p = demo_path[s]
            np_pos = demo_path[min(s + 1, len(demo_path) - 1)]
            v_t = (np_pos - p) / dt
            r_dists = maze.get_ray_distances(p, 4)
            dx = maze.goal_pos[0] - p[0]; dy = maze.goal_pos[1] - p[1]
            dist = float(np.hypot(dx, dy))
            sens = {i: (1.0 / (1.0 + r_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
            sens[4] = float(max(0.0, dx) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
            sens[5] = float(max(0.0, dy) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
            sens[6] = float(max(0.0, -dx) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
            sens[7] = float(max(0.0, -dy) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
            tf = {}
            carrier = 0.9 * np.exp(1j * 2.0 * t)
            if v_t[0] > 0.1: tf[0] = carrier * min(1.0, v_t[0] / 0.5)
            elif v_t[0] < -0.1: tf[1] = carrier * min(1.0, abs(v_t[0]) / 0.5)
            if v_t[1] > 0.1: tf[2] = carrier * min(1.0, v_t[1] / 0.5)
            elif v_t[1] < -0.1: tf[3] = carrier * min(1.0, abs(v_t[1]) / 0.5)
            r = 0.0
            if np.hypot(p[0] - 1.15, p[1] - 0.7) < 0.25 or np.hypot(p[0] - 1.15, p[1] - 3.0) < 0.25: r += 1.5
            if s == len(demo_path) - 1: r += 3.0
            sub.step(sens, reward=r, dt=dt, teacher_actuator_fluxes=tf)

    # Run Autonomous live simulation
    sub.reset()
    pos = maze.reset_agent() + np.random.uniform(-0.02, 0.02, size=2)
    vel = np.zeros(2)
    colls = 0
    last_c = -100

    print("\n" + "=" * 50)
    print("STARTING LIVE TERMINAL AUTONOMOUS NAVIGATION:")
    print("=" * 50)

    for step in range(n_steps):
        t = step * dt
        r_dists = maze.get_ray_distances(pos, 4)
        dx = maze.goal_pos[0] - pos[0]; dy = maze.goal_pos[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        sens = {i: (1.0 / (1.0 + r_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
        sens[4] = float(max(0.0, dx) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
        sens[5] = float(max(0.0, dy) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
        sens[6] = float(max(0.0, -dx) / (dist + 0.1)) * np.exp(1j * 2.1 * t)
        sens[7] = float(max(0.0, -dy) / (dist + 0.1)) * np.exp(1j * 2.1 * t)

        out = sub.step(sens, reward=0.0, dt=dt)
        f = out["actuator_forces"]
        fx = (f[0] - f[1]) * 2.5
        fy = (f[2] - f[3]) * 2.5
        vel[0] += (fx - 0.85 * vel[0]) * dt
        vel[1] += (fy - 0.85 * vel[1]) * dt
        prop = pos + vel * dt
        prop[0] = max(0.15, min(3.85, prop[0]))
        prop[1] = max(0.15, min(3.85, prop[1]))
        col_x = maze.check_collision(np.array([prop[0], pos[1]]))
        col_y = maze.check_collision(np.array([pos[0], prop[1]]))
        if not col_x and not col_y and not maze.check_collision(prop):
            pos = prop
        else:
            if col_x: vel[0] = -0.2 * vel[0]
            else: pos[0] = prop[0]
            if col_y: vel[1] = -0.2 * vel[1]
            else: pos[1] = prop[1]
            if step - last_c > 5: colls += 1; last_c = step

        # Render frame every 10 steps to show progression
        if step % 10 == 0 or maze.check_goal(pos):
            frame = render_terminal_frame(maze, pos, step, dist, f, float(out["rpe"]), colls)
            print(frame)
            print("-" * 46)

        if maze.check_goal(pos):
            print(f"\n🎉 GOAL REACHED AT STEP {step}! Final Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            break


if __name__ == "__main__":
    run_live_terminal(maze_idx=0, n_steps=120)
