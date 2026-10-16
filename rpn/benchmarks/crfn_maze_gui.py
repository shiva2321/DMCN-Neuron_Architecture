"""Local Native Desktop GUI for Morphogenetic CRFN Maze Navigation.

Provides a standalone, lightweight, real-time desktop window (Tkinter):
- Left Panel: Live 2D Continuous Maze with Wall Obstacles, Agent Locomotion,
  4 Ray-Cast Proximity Sensors, Goal Beacon, and Thruster Vectors.
- Right Panel: Real-Time Telemetry & Neural Substrate Monitoring:
  - Live somatic oscillation power P = |z|^2
  - Live dopamine RPE pulses delta(t)
  - Efferent thruster forces (East, West, North, South)
  - Neurogenesis & active neuron counter
  - Real-time collision and goal indicators
- Live Physics Modes:
  - [ Run Live Autonomous ]: Watch the trained neural wavefield drive the agent autonomously.
  - [ Run Live Training Demo ]: Watch the teacher guide the agent while dopamine welds impedances.
  - [ Novel Start Test ]: Test from an unseen starting position.
  - Maze selector: Switch between Maze 1 (S-Bend), Maze 2 (Horseshoe), and Unseen Maze 4 (Central Box).

Zero browser dependencies. Pure native desktop window.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate
from benchmarks.crfn_maze_benchmark import WallSegment
from benchmarks.crfn_multi_maze_generalization_benchmark import (
    ConfigurableMaze,
    create_training_dataset,
    create_unseen_dataset,
)


class CRFNLiveMazeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Morphogenetic CRFN — Live Continuous Maze Substrate")
        self.root.geometry("980x640")
        self.root.configure(bg="#14171d")
        self.root.resizable(False, False)

        # 1. Initialize Datasets & Neural Substrates
        self.train_dataset = create_training_dataset()
        self.unseen_mazes = create_unseen_dataset()

        # Available Mazes dictionary
        self.mazes: Dict[str, ConfigurableMaze] = {
            "Maze 1: S-Bend Corridor (Seen)": self.train_dataset[0][0],
            "Maze 2: Horseshoe U-Turn (Seen)": self.train_dataset[1][0],
            "Maze 3: Reverse Hook (Seen)": self.train_dataset[2][0],
            "Unseen Maze 4: Central Box Bypass (OOD)": self.unseen_mazes[0],
            "Unseen Maze 6: T-Junction (OOD)": self.unseen_mazes[2],
        }

        self.current_maze_name = "Maze 1: S-Bend Corridor (Seen)"
        self.current_maze = self.mazes[self.current_maze_name]

        # Simulation physics state
        self.dt = 0.02
        self.pos = self.current_maze.reset_agent()
        self.vel = np.zeros(2, dtype=float)
        self.step_idx = 0
        self.collisions = 0
        self.last_collision_step = -100
        self.reached_goal = False
        self.trail: List[Tuple[float, float]] = []

        # Simulation control
        self.is_running = False
        self.mode = "AUTONOMOUS"  # "AUTONOMOUS" or "DEMONSTRATION"
        self.speed_mult = 1  # 1x, 2x, 5x, 10x
        self.demo_path: List[np.ndarray] = self.train_dataset[0][2]
        self.demo_wps: np.ndarray = self.train_dataset[0][1]
        self.demo_step = 0
        self.demo_wp_idx = 1
        self.prev_demo_dist = 0.0

        # Substrate
        self._init_substrate()
        self._train_demonstration_baseline()

        # UI Setup
        self._build_ui()
        self.reset_simulation()
        self._draw_maze()

        # Animation loop
        self.root.after(20, self._simulation_tick)

    def _init_substrate(self):
        """Instantiate clean morphogenetic substrate."""
        self.substrate = MorphogeneticCRFNSubstrate(
            n_initial_neurons=8,
            n_sensory_ports=8,
            n_actuator_ports=4,
            base_omega=2.0,
            base_mu=0.50,
            gamma_discount=0.90,
            lr_rl=1.80,
            lr_pred=0.08,
            theta_grow=0.85,
            theta_prune=0.10,
            exploration_std=0.03,
            max_neurons=20,
            seed=42,
        )

    def _train_demonstration_baseline(self):
        """Train baseline impedances on Maze 1 for instantaneous live autonomous running."""
        cache_path = Path(__file__).resolve().parent / "crfn_trained_cache.pkl"
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    self.trained_somas = data["somas"]
                    self.trained_dendrites = data["dendrites"]
                    self.trained_aff = data["aff"]
                    self.trained_rec = data["rec"]
                    self.trained_eff = data["eff"]
                self._restore_trained_state()
                return
            except Exception:
                pass

        maze = self.train_dataset[0][0]
        wps = self.train_dataset[0][1]
        demo_path = self.train_dataset[0][2]
        dt = 0.02

        for _ in range(10):
            self.substrate.reset()
            prev_dist = float(np.hypot(demo_path[0][0] - maze.goal_pos[0], demo_path[0][1] - maze.goal_pos[1]))
            wp_idx = 1
            for s in range(len(demo_path)):
                t = s * dt
                pos = demo_path[s]
                next_pos = demo_path[min(s + 1, len(demo_path) - 1)]
                v_target = (next_pos - pos) / dt

                ray_dists = maze.get_ray_distances(pos, n_rays=4)
                dx_g = maze.goal_pos[0] - pos[0]
                dy_g = maze.goal_pos[1] - pos[1]
                curr_dist = float(np.hypot(dx_g, dy_g))

                sens = {i: (1.0 / (1.0 + ray_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
                sens[4] = float(max(0.0, dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[5] = float(max(0.0, dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[6] = float(max(0.0, -dx_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)
                sens[7] = float(max(0.0, -dy_g) / (curr_dist + 0.1)) * np.exp(1j * 2.1 * t)

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
                target_wp = np.array(wps[min(wp_idx, len(wps) - 1)])
                if np.hypot(pos[0] - target_wp[0], pos[1] - target_wp[1]) < 0.25:
                    reward += 1.5
                    if wp_idx < len(wps) - 1:
                        wp_idx += 1
                if s == len(demo_path) - 1:
                    reward += 3.0

                self.substrate.step(sens, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)

        # Snapshot baseline trained state
        self.trained_somas = copy.deepcopy(self.substrate.somas)
        self.trained_dendrites = copy.deepcopy(self.substrate.dendrites)
        self.trained_aff = copy.deepcopy(self.substrate.afferent_links)
        self.trained_rec = copy.deepcopy(self.substrate.recurrent_links)
        self.trained_eff = copy.deepcopy(self.substrate.efferent_links)

        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "somas": self.trained_somas,
                    "dendrites": self.trained_dendrites,
                    "aff": self.trained_aff,
                    "rec": self.trained_rec,
                    "eff": self.trained_eff,
                }, f)
        except Exception:
            pass

    def _restore_trained_state(self):
        self.substrate.somas = copy.deepcopy(self.trained_somas)
        self.substrate.dendrites = copy.deepcopy(self.trained_dendrites)
        self.substrate.afferent_links = copy.deepcopy(self.trained_aff)
        self.substrate.recurrent_links = copy.deepcopy(self.trained_rec)
        self.substrate.efferent_links = copy.deepcopy(self.trained_eff)
        self.substrate.reset()

    def _build_ui(self):
        # Left: Maze Canvas Frame
        left_frame = tk.Frame(self.root, bg="#14171d")
        left_frame.pack(side=tk.LEFT, padx=16, pady=16)

        self.canvas_size = 560
        self.scale = self.canvas_size / 4.0  # 140 pixels per meter

        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#181c24",
            highlightthickness=1,
            highlightbackground="#2c3240",
        )
        self.canvas.pack()

        # Canvas bottom info strip
        self.canvas_status_var = tk.StringVar(value="Status: Ready")
        lbl_status = tk.Label(
            left_frame,
            textvariable=self.canvas_status_var,
            bg="#14171d",
            fg="#8e9aaf",
            font=("Consolas", 10),
            anchor="w",
        )
        lbl_status.pack(fill=tk.X, pady=(6, 0))

        # Right: Dashboard & Controls Frame
        right_frame = tk.Frame(self.root, bg="#14171d", width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 16), pady=16)

        # Title Card
        title_box = tk.Frame(right_frame, bg="#1e232d", padx=12, pady=10)
        title_box.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            title_box,
            text="CRFN NEURAL SUBSTRATE",
            bg="#1e232d",
            fg="#00d2ff",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Continuous Resonance • Impedance Matching",
            bg="#1e232d",
            fg="#8e9aaf",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        # Maze Selector
        sel_box = tk.Frame(right_frame, bg="#1e232d", padx=12, pady=10)
        sel_box.pack(fill=tk.X, pady=(0, 10))
        tk.Label(sel_box, text="SELECT MAZE TOPOLOGY:", bg="#1e232d", fg="#e2e8f0", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        self.maze_combo = ttk.Combobox(sel_box, values=list(self.mazes.keys()), state="readonly", font=("Segoe UI", 9))
        self.maze_combo.current(0)
        self.maze_combo.pack(fill=tk.X, pady=(0, 4))
        self.maze_combo.bind("<<ComboboxSelected>>", self._on_maze_selected)

        # Control Buttons Card
        btn_box = tk.Frame(right_frame, bg="#1e232d", padx=12, pady=10)
        btn_box.pack(fill=tk.X, pady=(0, 10))

        btn_row1 = tk.Frame(btn_box, bg="#1e232d")
        btn_row1.pack(fill=tk.X, pady=(0, 6))

        self.btn_auto = tk.Button(
            btn_row1,
            text="▶ Run Autonomous",
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5,
            command=self._start_autonomous,
        )
        self.btn_auto.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_demo = tk.Button(
            btn_row1,
            text="🎓 Guided Demo",
            bg="#334155",
            fg="#ffffff",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=10,
            pady=5,
            command=self._start_demo,
        )
        self.btn_demo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        btn_row2 = tk.Frame(btn_box, bg="#1e232d")
        btn_row2.pack(fill=tk.X)

        self.btn_pause = tk.Button(
            btn_row2,
            text="⏸ Pause",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            command=self._toggle_pause,
        )
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_reset = tk.Button(
            btn_row2,
            text="🔄 Reset",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            command=self.reset_simulation,
        )
        self.btn_reset.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Speed Controls
        speed_row = tk.Frame(btn_box, bg="#1e232d")
        speed_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(speed_row, text="Speed:", bg="#1e232d", fg="#64748b", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        for s in [1, 2, 5, 10]:
            btn_s = tk.Button(
                speed_row,
                text=f"{s}x",
                bg="#0f172a",
                fg="#94a3b8",
                font=("Consolas", 8),
                relief=tk.FLAT,
                command=lambda val=s: self._set_speed(val),
                padx=6,
            )
            btn_s.pack(side=tk.LEFT, padx=3)

        # Telemetry Card
        telem_box = tk.Frame(right_frame, bg="#1e232d", padx=12, pady=10)
        telem_box.pack(fill=tk.BOTH, expand=True)

        tk.Label(telem_box, text="LIVE SENSORIMOTOR TELEMETRY:", bg="#1e232d", fg="#e2e8f0", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))

        self.lbl_mode = tk.Label(telem_box, text="MODE: STANDBY", bg="#1e232d", fg="#38bdf8", font=("Consolas", 10, "bold"), anchor="w")
        self.lbl_mode.pack(fill=tk.X)

        self.lbl_pos = tk.Label(telem_box, text="Pos: (0.50, 0.50)  |  Goal Dist: 4.24 m", bg="#1e232d", fg="#cbd5e1", font=("Consolas", 9), anchor="w")
        self.lbl_pos.pack(fill=tk.X)

        self.lbl_step = tk.Label(telem_box, text="Step: 0  |  Collisions: 0", bg="#1e232d", fg="#cbd5e1", font=("Consolas", 9), anchor="w")
        self.lbl_step.pack(fill=tk.X)

        self.lbl_forces = tk.Label(telem_box, text="Forces: fx = +0.00 N, fy = +0.00 N", bg="#1e232d", fg="#34d399", font=("Consolas", 9), anchor="w")
        self.lbl_forces.pack(fill=tk.X)

        self.lbl_neurons = tk.Label(telem_box, text="Neurons: 18  |  Somatic Power: 0.00 W", bg="#1e232d", fg="#cbd5e1", font=("Consolas", 9), anchor="w")
        self.lbl_neurons.pack(fill=tk.X)

        self.lbl_rpe = tk.Label(telem_box, text="Dopamine RPE (delta): +0.000", bg="#1e232d", fg="#fbbf24", font=("Consolas", 9, "bold"), anchor="w")
        self.lbl_rpe.pack(fill=tk.X)

        # Thruster Channel Gauges
        tk.Label(telem_box, text="Actuator Power Transmission:", bg="#1e232d", fg="#64748b", font=("Segoe UI", 8), anchor="w").pack(fill=tk.X, pady=(6, 2))
        self.lbl_thrusters = tk.Label(
            telem_box,
            text="[E: 99%] [W: 55%] [N: 100%] [S: 55%]",
            bg="#0f172a",
            fg="#38bdf8",
            font=("Consolas", 9, "bold"),
            padx=6,
            pady=4,
            relief=tk.FLAT,
        )
        self.lbl_thrusters.pack(fill=tk.X, pady=(0, 6))

        # Goal celebration label
        self.lbl_result = tk.Label(telem_box, text="", bg="#1e232d", fg="#22c55e", font=("Segoe UI", 11, "bold"))
        self.lbl_result.pack(fill=tk.X, pady=(4, 0))

    def _to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert maze coords (0 to 4) to canvas pixel coords (inverted y)."""
        cx = x * self.scale
        cy = self.canvas_size - (y * self.scale)
        return cx, cy

    def _draw_maze(self):
        """Render static walls and goal zone on canvas."""
        self.canvas.delete("all")

        # Subtle background grid
        for i in range(1, 4):
            x = i * self.scale
            self.canvas.create_line(x, 0, x, self.canvas_size, fill="#1e2430", dash=(2, 4))
            self.canvas.create_line(0, x, self.canvas_size, x, fill="#1e2430", dash=(2, 4))

        # Draw Goal Zone (Golden glowing circle)
        gx, gy = self._to_canvas(self.current_maze.goal_pos[0], self.current_maze.goal_pos[1])
        gr = self.current_maze.goal_radius * self.scale
        self.canvas.create_oval(gx - gr * 1.3, gy - gr * 1.3, gx + gr * 1.3, gy + gr * 1.3, fill="", outline="#453805", width=2)
        self.canvas.create_oval(gx - gr, gy - gr, gx + gr, gy + gr, fill="#3b2d04", outline="#eab308", width=2)
        self.canvas.create_text(gx, gy, text="GOAL", fill="#fde047", font=("Segoe UI", 9, "bold"))

        # Draw Start Zone
        sx, sy = self._to_canvas(self.current_maze.start_pos[0], self.current_maze.start_pos[1])
        sr = 0.22 * self.scale
        self.canvas.create_oval(sx - sr, sy - sr, sx + sr, sy + sr, fill="", outline="#0369a1", dash=(3, 3), width=1)
        self.canvas.create_text(sx, sy, text="START", fill="#38bdf8", font=("Segoe UI", 7))

        # Draw Walls
        for w in self.current_maze.walls:
            x1, y1 = self._to_canvas(w.x1, w.y1)
            x2, y2 = self._to_canvas(w.x2, w.y2)
            self.canvas.create_line(x1, y1, x2, y2, fill="#94a3b8", width=5, capstyle=tk.ROUND)

    def _draw_dynamic_state(self, ray_dists: np.ndarray, forces: List[float]):
        """Render agent body, breadcrumb trail, sensory rays, and thrusters."""
        self.canvas.delete("dynamic")

        # 1. Draw Breadcrumb Trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                x1, y1 = self._to_canvas(self.trail[i - 1][0], self.trail[i - 1][1])
                x2, y2 = self._to_canvas(self.trail[i][0], self.trail[i][1])
                alpha_col = "#0284c7" if self.mode == "AUTONOMOUS" else "#a855f7"
                self.canvas.create_line(x1, y1, x2, y2, fill=alpha_col, width=2, tags="dynamic")

        ax, ay = self._to_canvas(self.pos[0], self.pos[1])
        ar = self.current_maze.agent_radius * self.scale

        # 2. Draw 4 Sensory Ray Proximity Lasers (East, North, West, South)
        angles = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
        for idx, (dist, ang) in enumerate(zip(ray_dists, angles)):
            r_len = dist * self.scale
            end_x = ax + r_len * np.cos(ang)
            end_y = ay - r_len * np.sin(ang)  # Invert for canvas

            # Proximity color: red when close (<0.35m), yellow medium, green when clear
            if dist < 0.35:
                ray_col = "#ef4444"
                lw = 2
            elif dist < 0.8:
                ray_col = "#eab308"
                lw = 1
            else:
                ray_col = "#22c55e"
                lw = 1

            self.canvas.create_line(ax, ay, end_x, end_y, fill=ray_col, width=lw, dash=(2, 3), tags="dynamic")
            self.canvas.create_oval(end_x - 2, end_y - 2, end_x + 2, end_y + 2, fill=ray_col, outline="", tags="dynamic")

        # 3. Draw Goal Beacon Line (Dashed subtle line to goal)
        gx, gy = self._to_canvas(self.current_maze.goal_pos[0], self.current_maze.goal_pos[1])
        self.canvas.create_line(ax, ay, gx, gy, fill="#ca8a04", width=1, dash=(3, 5), tags="dynamic")

        # 4. Draw Thruster Force Arrow (Red/Orange vector)
        fx = forces[0] - forces[1]
        fy = forces[2] - forces[3]
        f_len = math.hypot(fx, fy) * 35.0
        if f_len > 2.0:
            f_ang = math.atan2(fy, fx)
            end_fx = ax + f_len * math.cos(f_ang)
            end_fy = ay - f_len * math.sin(f_ang)
            self.canvas.create_line(ax, ay, end_fx, end_fy, fill="#f97316", width=3, arrow=tk.LAST, tags="dynamic")

        # 5. Draw Agent Body (Bright cyan circle)
        agent_fill = "#38bdf8" if self.mode == "AUTONOMOUS" else "#c084fc"
        self.canvas.create_oval(ax - ar, ay - ar, ax + ar, ay + ar, fill=agent_fill, outline="#ffffff", width=2, tags="dynamic")
        # Center core
        self.canvas.create_oval(ax - 3, ay - 3, ax + 3, ay + 3, fill="#0f172a", outline="", tags="dynamic")

    def _on_maze_selected(self, event=None):
        name = self.maze_combo.get()
        if name in self.mazes:
            self.current_maze_name = name
            self.current_maze = self.mazes[name]
            # If Maze 1, 2, 3 find demonstration path
            for m, wps, p in self.train_dataset:
                if m.name in name:
                    self.demo_path = p
                    self.demo_wps = wps
                    break
            self.reset_simulation()
            self._draw_maze()

    def _start_autonomous(self):
        self._restore_trained_state()
        self.mode = "AUTONOMOUS"
        self.pos = self.current_maze.reset_agent() + np.random.uniform(-0.02, 0.02, size=2)
        self.vel = np.zeros(2, dtype=float)
        self.step_idx = 0
        self.collisions = 0
        self.trail = [(float(self.pos[0]), float(self.pos[1]))]
        self.reached_goal = False
        self.is_running = True
        self.lbl_mode.config(text="MODE: RUNNING AUTONOMOUS", fg="#38bdf8")
        self.lbl_result.config(text="")
        self.canvas_status_var.set(f"Running autonomously on {self.current_maze.name} (Zero Teacher Guidance)")

    def _start_demo(self):
        self.mode = "DEMONSTRATION"
        self.substrate.reset()
        self.pos = self.demo_path[0].copy()
        self.vel = np.zeros(2, dtype=float)
        self.step_idx = 0
        self.demo_step = 0
        self.demo_wp_idx = 1
        self.prev_demo_dist = float(np.hypot(self.pos[0] - self.current_maze.goal_pos[0], self.pos[1] - self.current_maze.goal_pos[1]))
        self.collisions = 0
        self.trail = [(float(self.pos[0]), float(self.pos[1]))]
        self.reached_goal = False
        self.is_running = True
        self.lbl_mode.config(text="MODE: GUIDED DEMONSTRATION", fg="#c084fc")
        self.lbl_result.config(text="")
        self.canvas_status_var.set(f"Teacher demonstration on {self.current_maze.name} (Dopamine Welding Active)")

    def _toggle_pause(self):
        self.is_running = not self.is_running
        txt = "▶ Resume" if not self.is_running else "⏸ Pause"
        self.btn_pause.config(text=txt)

    def _set_speed(self, speed: int):
        self.speed_mult = speed

    def reset_simulation(self):
        self.is_running = False
        self.step_idx = 0
        self.collisions = 0
        self.reached_goal = False
        self.pos = self.current_maze.reset_agent()
        self.vel = np.zeros(2, dtype=float)
        self.trail = [(float(self.pos[0]), float(self.pos[1]))]
        self.lbl_result.config(text="")
        self.lbl_mode.config(text="MODE: STANDBY", fg="#94a3b8")
        self.btn_pause.config(text="⏸ Pause")
        self._draw_maze()
        self._draw_dynamic_state(np.array([1.5, 1.5, 0.5, 0.5]), [0.0, 0.0, 0.0, 0.0])

    def _simulation_tick(self):
        """Execute one simulation frame at target FPS."""
        if self.is_running and not self.reached_goal:
            for _ in range(self.speed_mult):
                self._step_physics()
                if self.reached_goal:
                    break

        self.root.after(20, self._simulation_tick)

    def _step_physics(self):
        dt = self.dt
        t = self.step_idx * dt
        self.step_idx += 1

        ray_dists = self.current_maze.get_ray_distances(self.pos, n_rays=4)
        dx_g = self.current_maze.goal_pos[0] - self.pos[0]
        dy_g = self.current_maze.goal_pos[1] - self.pos[1]
        dist_g = float(np.hypot(dx_g, dy_g))

        sensory_inputs = {i: (1.0 / (1.0 + ray_dists[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
        sensory_inputs[4] = float(max(0.0, dx_g) / (dist_g + 0.1)) * np.exp(1j * 2.1 * t)
        sensory_inputs[5] = float(max(0.0, dy_g) / (dist_g + 0.1)) * np.exp(1j * 2.1 * t)
        sensory_inputs[6] = float(max(0.0, -dx_g) / (dist_g + 0.1)) * np.exp(1j * 2.1 * t)
        sensory_inputs[7] = float(max(0.0, -dy_g) / (dist_g + 0.1)) * np.exp(1j * 2.1 * t)

        reward = 0.0
        teacher_fluxes = None

        if self.mode == "DEMONSTRATION":
            # Teacher guides agent along demo path
            if self.demo_step < len(self.demo_path):
                self.pos = self.demo_path[self.demo_step].copy()
                next_pos = self.demo_path[min(self.demo_step + 1, len(self.demo_path) - 1)]
                v_target = (next_pos - self.pos) / dt
                self.demo_step += 1

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

                reward = float((self.prev_demo_dist - dist_g) * 2.0) - 0.002
                self.prev_demo_dist = dist_g

                target_wp = np.array(self.demo_wps[min(self.demo_wp_idx, len(self.demo_wps) - 1)])
                if np.hypot(self.pos[0] - target_wp[0], self.pos[1] - target_wp[1]) < 0.25:
                    reward += 1.5
                    if self.demo_wp_idx < len(self.demo_wps) - 1:
                        self.demo_wp_idx += 1

                if self.demo_step >= len(self.demo_path) or self.current_maze.check_goal(self.pos):
                    reward += 3.0
                    self.reached_goal = True
            else:
                self.reached_goal = True

        out = self.substrate.step(sensory_inputs, reward=reward, dt=dt, teacher_actuator_fluxes=teacher_fluxes)
        forces = out["actuator_forces"]

        if self.mode == "AUTONOMOUS":
            fx = (forces[0] - forces[1]) * 2.5
            fy = (forces[2] - forces[3]) * 2.5
            damping = 0.85
            mass = 1.0

            ax = (fx - damping * self.vel[0]) / mass
            ay = (fy - damping * self.vel[1]) / mass
            self.vel[0] += ax * dt
            self.vel[1] += ay * dt

            proposed_pos = self.pos + self.vel * dt
            proposed_pos[0] = max(self.current_maze.agent_radius, min(self.current_maze.width - self.current_maze.agent_radius, proposed_pos[0]))
            proposed_pos[1] = max(self.current_maze.agent_radius, min(self.current_maze.height - self.current_maze.agent_radius, proposed_pos[1]))

            col_x = self.current_maze.check_collision(np.array([proposed_pos[0], self.pos[1]]))
            col_y = self.current_maze.check_collision(np.array([self.pos[0], proposed_pos[1]]))

            if not col_x and not col_y and not self.current_maze.check_collision(proposed_pos):
                self.pos = proposed_pos
            else:
                if col_x:
                    self.vel[0] = -0.2 * self.vel[0]
                else:
                    self.pos[0] = proposed_pos[0]
                if col_y:
                    self.vel[1] = -0.2 * self.vel[1]
                else:
                    self.pos[1] = proposed_pos[1]
                if self.step_idx - self.last_collision_step > 5:
                    self.collisions += 1
                    self.last_collision_step = self.step_idx

            if self.current_maze.check_goal(self.pos):
                self.reached_goal = True

        # Append trail
        self.trail.append((float(self.pos[0]), float(self.pos[1])))
        if len(self.trail) > 120:
            self.trail.pop(0)

        # Update UI Telemetry
        self._update_telemetry(dist_g, forces, out)
        self._draw_dynamic_state(ray_dists, forces)

        if self.reached_goal:
            self.lbl_result.config(text=f"🎯 GOAL REACHED! (Steps: {self.step_idx}, Collisions: {self.collisions})")
            self.canvas_status_var.set(f"Completed in {self.step_idx} steps with {self.collisions} wall scrapes!")

    def _update_telemetry(self, dist_g: float, forces: List[float], out: Dict[str, Any]):
        fx = (forces[0] - forces[1]) * 2.5
        fy = (forces[2] - forces[3]) * 2.5

        self.lbl_pos.config(text=f"Pos: ({self.pos[0]:.2f}, {self.pos[1]:.2f})  |  Goal Dist: {dist_g:.2f} m")
        self.lbl_step.config(text=f"Step: {self.step_idx:<4} |  Collisions: {self.collisions:<2}")
        self.lbl_forces.config(text=f"Forces: fx = {fx:+.2f} N, fy = {fy:+.2f} N")

        soma_p = sum(s.amplitude ** 2 for s in self.substrate.somas.values())
        self.lbl_neurons.config(text=f"Neurons: {len(self.substrate.somas):<2} |  Somatic Power: {soma_p:.2f} W")

        rpe = float(out["rpe"])
        rpe_col = "#22c55e" if rpe > 0.05 else ("#ef4444" if rpe < -0.05 else "#94a3b8")
        self.lbl_rpe.config(text=f"Dopamine RPE (delta): {rpe:+.3f}", fg=rpe_col)

        # Efferent Transmissions across 4 channels
        T_channels = []
        for aid in range(4):
            ch_T = [l.power_transmission for (n, a), l in self.substrate.efferent_links.items() if a == aid]
            mean_T = float(np.mean(ch_T)) * 100 if ch_T else 0.0
            T_channels.append(mean_T)
        self.lbl_thrusters.config(
            text=f"[E: {T_channels[0]:.0f}%]  [W: {T_channels[1]:.0f}%]  [N: {T_channels[2]:.0f}%]  [S: {T_channels[3]:.0f}%]"
        )


def launch_gui():
    root = tk.Tk()
    app = CRFNLiveMazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
