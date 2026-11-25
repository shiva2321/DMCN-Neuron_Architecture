import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.sensorimotor_arena import (
    Continuous3DArenaEnvironment,
    ContinuousMotorAttractorField,
    CAPN3DSensorimotorAgent,
    Blind3DSensorimotorAgent,
    ReactiveGreedy3DAgent,
    RandomWalk3DAgent,
    generate_3d_ray_directions,
)


def test_3d_environment_stop_and_rebound_physics():
    """Verify stop-and-rebound physics zeroes velocity on collision without wall-sliding."""
    env = Continuous3DArenaEnvironment()
    # Place agent right in front of central hazard sphere (center=[5, 5, 5], r=1.8)
    env.pos = np.array([3.1, 5.0, 5.0])
    env.vel = np.array([1.0, 0.0, 0.0])

    # Drive directly into the sphere
    act = np.array([1.0, 0.0, 0.0])
    obs, reward, done, reached, info = env.step(act)

    assert info["collision"] is True
    # Velocity must be stopped
    assert np.allclose(env.vel, 0.0)
    # Agent must remain outside sphere radius (r=1.8 + r_agent=0.25 = 2.05)
    dist_to_center = np.linalg.norm(env.pos - np.array([5.0, 5.0, 5.0]))
    assert dist_to_center >= 1.8


def test_3d_environment_ray_casting():
    """Verify 14-directional 3D ray-casting detects boundaries and spheres."""
    env = Continuous3DArenaEnvironment()
    env.pos = np.array([5.0, 5.0, 1.0])  # Near bottom floor z=0

    obs = env._get_obs()
    dists = obs["distances"]
    assert len(dists) == 14
    # Ray pointing down along -z (index 5) should hit floor around z=1.0
    assert 0.8 <= dists[5] <= 1.2


def test_capn_3d_agent_navigates_around_hazard():
    """Verify CAPN prospective agent successfully navigates 3D arena with zero collisions."""
    env = Continuous3DArenaEnvironment(max_steps=220)
    agent = CAPN3DSensorimotorAgent(rng=np.random.RandomState(42))
    obs = env.reset(seed=42)
    agent.reset()

    done = False
    reached = False
    while not done:
        act = agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)

    assert reached is True
    assert info["collision_count"] == 0
    assert info["final_dist_goal"] < 1.0


def test_blind_3d_agent_collides_and_fails():
    """Verify blind agent cannot detect hazards, colliding continuously and failing."""
    env = Continuous3DArenaEnvironment(max_steps=220)
    agent = Blind3DSensorimotorAgent(rng=np.random.RandomState(42))
    obs = env.reset(seed=42)
    agent.reset()

    done = False
    reached = False
    while not done:
        act = agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)

    assert reached is False
    assert info["collision_count"] > 10


def test_reactive_greedy_3d_agent_fails():
    """Verify reactive greedy agent without prospective lookahead gets trapped and fails."""
    env = Continuous3DArenaEnvironment(max_steps=220)
    agent = ReactiveGreedy3DAgent(rng=np.random.RandomState(42))
    obs = env.reset(seed=42)
    agent.reset()

    done = False
    reached = False
    while not done:
        act = agent.select_action(obs)
        obs, reward, done, reached, info = env.step(act)

    assert reached is False
    assert info["final_dist_goal"] > 3.0


def test_continuous_motor_attractor_bifurcation():
    """Verify motor attractor field points toward goal in free space and bifurcates under hazard inhibition."""
    ray_dirs = np.array(generate_3d_ray_directions())
    attractor = ContinuousMotorAttractorField(ray_dirs)

    dir_goal = np.array([1.0, 0.0, 0.0])
    prev_act = np.array([1.0, 0.0, 0.0])

    # Case 1: Free space (all distances 5.0) -> Population vector aligns strongly with goal
    free_dists = np.full(14, 5.0)
    affordances_free = attractor.compute_affordances(dir_goal, free_dists, prev_act)
    pop_vec_free = affordances_free[0]
    cos_sim = float(np.dot(pop_vec_free, dir_goal))
    assert cos_sim > 0.95

    # Case 2: Obstacle directly ahead (ray index 0 points +x, distance = 0.3)
    blocked_dists = np.full(14, 5.0)
    blocked_dists[0] = 0.3  # Imminent obstacle directly ahead
    affordances_blocked = attractor.compute_affordances(dir_goal, blocked_dists, prev_act)

    # Motor attractor must generate lateral affordance peaks pointing away from collision
    assert len(affordances_blocked) >= 4
    lateral_affordances = affordances_blocked[3:]
    for aff in lateral_affordances:
        # Must possess substantial lateral (non-x) component to steer around obstacle
        lateral_mag = float(np.linalg.norm(aff[1:]))
        assert lateral_mag > 0.5

    # Full agent evaluates these neural affordances and selects a safe diagonal bypass
    agent = CAPN3DSensorimotorAgent()
    obs_blocked = {
        "distances": blocked_dists,
        "dir_to_goal": dir_goal,
        "dist_to_goal": 8.0,
        "pos": np.array([1.0, 1.0, 1.0]),
    }
    selected_act = agent.select_action(obs_blocked)
    assert np.linalg.norm(selected_act[1:]) > 0.5  # Successfully diverts laterally
