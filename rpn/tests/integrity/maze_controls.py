"""Gate-owned reference harness for the CRFN maze claims (B-30, B-31, B-32).

This is an INDEPENDENT re-implementation of the evaluation loop in
``benchmarks/crfn_maze_demonstration_benchmark.py`` (commit a877a3b) so the controls do not depend on the code
under test. It imports only the environment geometry, the demonstration path and the substrate itself.

Frozen reference protocol (do not "tune" these to make a claim pass - GEMINI.md section 2.2):
  dt=0.02, gain=2.5, damping=0.85, mass=1, jitter=+-0.03, max_steps=600, 10 demonstrations,
  eval physics = axis-separated wall sliding ("slide"); original physics = stop-and-rebound ("bounce").
All randomness is seeded per episode so results are deterministic.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from benchmarks.crfn_maze_benchmark import Continuous2DMaze, WallSegment
from benchmarks.crfn_maze_demonstration_benchmark import CORRIDOR_WAYPOINTS, build_demonstration_path
from benchmarks.crfn_multi_maze_generalization_benchmark import ConfigurableMaze
from neuron_v2.crfn.crfn_morphogenetic import MorphogeneticCRFNSubstrate

DT = 0.02
GAIN = 2.5
JITTER = 0.03
MAX_STEPS = 600
N_DEMOS = 10

Policy = Callable[[dict, int], list]


def sensor_waves(maze, pos, t: float, blind: bool = False) -> dict:
    """8 ports: 4 wall-proximity rays + 4 goal-beacon directions (the beacon is an ORACLE)."""
    if blind:
        return {i: 0j for i in range(8)}
    rd = maze.get_ray_distances(pos, n_rays=4)
    dxg, dyg = maze.goal_pos[0] - pos[0], maze.goal_pos[1] - pos[1]
    cd = float(np.hypot(dxg, dyg))
    s = {i: (1.0 / (1.0 + rd[i])) * np.exp(1j * 2.0 * t) for i in range(4)}
    s[4] = float(max(0.0, dxg) / (cd + 0.1)) * np.exp(1j * 2.1 * t)
    s[5] = float(max(0.0, dyg) / (cd + 0.1)) * np.exp(1j * 2.1 * t)
    s[6] = float(max(0.0, -dxg) / (cd + 0.1)) * np.exp(1j * 2.1 * t)
    s[7] = float(max(0.0, -dyg) / (cd + 0.1)) * np.exp(1j * 2.1 * t)
    return s


def make_substrate(seed: int = 42) -> MorphogeneticCRFNSubstrate:
    np.random.seed(seed)  # StuartLandauSoma draws initial phases from the GLOBAL numpy RNG
    return MorphogeneticCRFNSubstrate(
        n_initial_neurons=8, n_sensory_ports=8, n_actuator_ports=4, base_omega=2.0, base_mu=0.50,
        gamma_discount=0.90, lr_rl=1.80, lr_pred=0.08, theta_grow=0.85, theta_prune=0.10,
        exploration_std=0.03, max_neurons=18, seed=seed)


def snapshot(sub) -> dict:
    return dict(somas=copy.deepcopy(sub.somas), dendrites=copy.deepcopy(sub.dendrites),
                aff=copy.deepcopy(sub.afferent_links), rec=copy.deepcopy(sub.recurrent_links),
                eff=copy.deepcopy(sub.efferent_links))


def restore(sub, snap: dict, seed: int):
    sub.somas = copy.deepcopy(snap["somas"])
    sub.dendrites = copy.deepcopy(snap["dendrites"])
    sub.afferent_links = copy.deepcopy(snap["aff"])
    sub.recurrent_links = copy.deepcopy(snap["rec"])
    sub.efferent_links = copy.deepcopy(snap["eff"])
    np.random.seed(seed)
    sub.rng = np.random.RandomState(seed)
    sub.reset()


def train_with_demonstrations(sub, maze, waypoints, path, n_demos: int = N_DEMOS,
                              seed: int = 7, shuffle_teacher: bool = False) -> dict:
    """Mirror of the benchmark's demonstration phase (teacher flux + shaped reward).

    ``shuffle_teacher=True`` is the decoupled-teacher control: the teacher's commands are drawn from a random
    permutation of the same demonstration, so the marginal action statistics are identical but the coupling
    between what the agent senses and what the teacher does is destroyed.
    """
    np.random.seed(seed)
    perm_rng = np.random.RandomState(seed)
    n = len(path)
    vels = [(path[min(i + 1, n - 1)] - path[i]) / DT for i in range(n)]
    for _ in range(n_demos):
        sub.reset()
        order = perm_rng.permutation(n) if shuffle_teacher else np.arange(n)
        prev = float(np.hypot(path[0][0] - maze.goal_pos[0], path[0][1] - maze.goal_pos[1]))
        wp_idx = 1
        for step in range(n):
            t = step * DT
            pos = path[step]
            v = vels[int(order[step])]
            carrier = 0.9 * np.exp(1j * 2.0 * t)
            tf = {}
            if v[0] > 0.1:
                tf[0] = carrier * min(1.0, v[0] / 0.5)
            elif v[0] < -0.1:
                tf[1] = carrier * min(1.0, abs(v[0]) / 0.5)
            if v[1] > 0.1:
                tf[2] = carrier * min(1.0, v[1] / 0.5)
            elif v[1] < -0.1:
                tf[3] = carrier * min(1.0, abs(v[1]) / 0.5)
            cd = float(np.hypot(maze.goal_pos[0] - pos[0], maze.goal_pos[1] - pos[1]))
            reward = (prev - cd) * 2.0 - 0.002
            prev = cd
            wp = np.array(waypoints[min(wp_idx, len(waypoints) - 1)])
            if np.hypot(pos[0] - wp[0], pos[1] - wp[1]) < 0.25:
                reward += 1.5
                wp_idx = min(wp_idx + 1, len(waypoints) - 1)
            if maze.check_goal(pos) or step == n - 1:
                reward += 3.0
            sub.step(sensor_waves(maze, pos, t), reward=reward, dt=DT, teacher_actuator_fluxes=tf)
    return snapshot(sub)


def rollout(policy: Policy, maze, physics: str = "slide", seed: int = 0, jitter: float = JITTER,
            gain: float = GAIN, max_steps: int = MAX_STEPS) -> bool:
    """One autonomous episode. Returns True iff the goal zone is reached."""
    rng = np.random.RandomState(seed)
    pos = maze.reset_agent() + rng.uniform(-jitter, jitter, size=2)
    vel = np.zeros(2)
    for step in range(max_steps):
        f = policy(sensor_waves(maze, pos, step * DT), step)
        vel[0] += ((f[0] - f[1]) * gain - 0.85 * vel[0]) * DT
        vel[1] += ((f[2] - f[3]) * gain - 0.85 * vel[1]) * DT
        prop = pos + vel * DT
        prop[0] = max(maze.agent_radius, min(maze.width - maze.agent_radius, prop[0]))
        prop[1] = max(maze.agent_radius, min(maze.height - maze.agent_radius, prop[1]))
        if physics == "slide":
            cx = maze.check_collision(np.array([prop[0], pos[1]]))
            cy = maze.check_collision(np.array([pos[0], prop[1]]))
            if not cx and not cy and not maze.check_collision(prop):
                pos = prop
            else:
                if cx:
                    vel[0] *= -0.2
                else:
                    pos[0] = prop[0]
                if cy:
                    vel[1] *= -0.2
                else:
                    pos[1] = prop[1]
        elif physics == "bounce":  # the ORIGINAL physics (before commit 684fb18)
            if maze.check_collision(prop):
                vel = -0.3 * vel
            else:
                pos = prop
        else:
            raise ValueError(physics)
        if maze.check_goal(pos):
            return True
    return False


# ----------------------------------------------------------------------------- policies
def network_policy(sub, snap, seed: int, blind: bool = False) -> Policy:
    restore(sub, snap, seed)

    def pol(s, step):
        if blind:
            s = {i: 0j for i in range(8)}
        return sub.step(s, reward=0.0, dt=DT)["actuator_forces"]
    return pol


def constant_policy(forces) -> Policy:
    return lambda s, step: list(forces)


def beacon_policy() -> Policy:
    """No network, no learning: thrust toward the goal using only the goal-beacon ports."""
    return lambda s, step: [abs(s[4]) ** 2, abs(s[6]) ** 2, abs(s[5]) ** 2, abs(s[7]) ** 2]


CONSTANT_DIRECTIONS = {
    "NE": (0.5, 0, 0.5, 0), "NW": (0, 0.5, 0.5, 0), "SE": (0.5, 0, 0, 0.5), "SW": (0, 0.5, 0, 0.5),
    "E": (0.5, 0, 0, 0), "W": (0, 0.5, 0, 0), "N": (0, 0, 0.5, 0), "S": (0, 0, 0, 0.5),
}


def maze1() -> Continuous2DMaze:
    return Continuous2DMaze(width=4.0, height=4.0)


def open_arena(goal, start=(2.0, 2.0)) -> ConfigurableMaze:
    outer = [WallSegment(0, 0, 4, 0), WallSegment(4, 0, 4, 4), WallSegment(4, 4, 0, 4), WallSegment(0, 4, 0, 0)]
    return ConfigurableMaze("open-arena", outer, start_pos=list(start), goal_pos=list(goal))


def success_outcomes(make_policy: Callable[[int], Policy], maze, n: int, physics: str = "slide", seed0: int = 100):
    out = []
    for i in range(n):
        np.random.seed(seed0 + i)
        out.append(rollout(make_policy(seed0 + i), maze, physics=physics, seed=seed0 + i))
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
_ = (CORRIDOR_WAYPOINTS, build_demonstration_path)  # re-exported for the tests
