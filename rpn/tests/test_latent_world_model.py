import sys
from pathlib import Path
REPO_RPN = Path(__file__).resolve().parents[1]
if str(REPO_RPN) not in sys.path:
    sys.path.insert(0, str(REPO_RPN))

import numpy as np
import pytest
from neuron_v2.capn.latent_world_model import LatentWorldModel


def test_factorized_manifold_encoding():
    """Verify factorized cognitive manifold z = [z_map, z_entity, z_dyn, u] construction."""
    lwm = LatentWorldModel(d_map=32, d_entity=16, d_dyn=8, d_unc=8, arena_size=10.0)

    pos = np.array([2.5, 4.0])
    vel = np.array([0.5, 0.0])
    goal = np.array([8.0, 5.0])
    barrier_box = (4.0, 6.0, 3.0, 7.0)

    state = lwm.encode_latent_state(pos, vel, goal, barrier_box, sensory_precision=1.0)

    assert state["z_map"].shape == (32,)
    assert state["z_entity"].shape == (16,)
    assert state["z_dyn"].shape == (8,)
    assert state["u"].shape == (8,)
    assert state["z_full"].shape == (64,)

    # Place field activations must be positive and normalized
    assert np.all(state["z_map"] >= 0.0)
    assert np.isclose(np.linalg.norm(state["z_map"]), 1.0, atol=1e-5)


def test_mental_rollout_prunes_trap():
    """Verify counterfactual imagination prunes deceptive hazard trajectory before action."""
    lwm = LatentWorldModel(arena_size=10.0, dt=0.5)

    pos_current = np.array([2.0, 5.0])
    vel_current = np.zeros(2)
    goal = np.array([8.0, 5.0])
    xmin, xmax, ymin, ymax = 4.5, 5.5, 3.2, 6.8
    barrier_box = (xmin, xmax, ymin, ymax)

    # Candidate deflection angles
    deflection_angles = [0.0, 18.0, -18.0, 35.0, -35.0, 55.0, -55.0]

    result = lwm.imagine_and_plan_intents(
        pos_current,
        vel_current,
        goal,
        barrier_box,
        deflection_angles,
        rollout_horizon=25,
        scramble_dynamics=False,
    )

    # Straight intent heading into obstacle (0.0 deg) must hit hazard
    assert result["intent_hazard_hits"][0] is True

    # The chosen deflection angle must NOT be 0.0 (the straight collision path)
    assert result["chosen_angle_deg"] != 0.0
    # Pruning margin between worst (collision) and chosen must be > 100
    assert result["pruning_margin"] > 100.0


def test_reactive_greedy_policy_falls_into_trap():
    """Verify reactive agent without forward rollout greedily falls into the barrier trap."""
    pos_current = np.array([2.0, 5.0])
    goal = np.array([8.0, 5.0])
    barrier_box = (4.5, 5.5, 3.2, 6.8)

    # Greedy policy directly aims towards goal
    heading_to_goal = (goal - pos_current) / np.linalg.norm(goal - pos_current)

    pos = pos_current.copy()
    hit_trap = False
    for _ in range(10):
        pos = pos + heading_to_goal * 0.5
        xmin, xmax, ymin, ymax = barrier_box
        if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
            hit_trap = True
            break

    assert hit_trap is True


def test_scrambled_dynamics_null_control():
    """Verify scrambled dynamics disrupt causal predictive rollout."""
    lwm = LatentWorldModel(arena_size=10.0, dt=0.5)

    pos_current = np.array([2.0, 5.0])
    vel_current = np.zeros(2)
    goal = np.array([8.0, 5.0])
    barrier_box = (4.5, 5.5, 3.2, 6.8)

    deflection_angles = [0.0, 18.0, -18.0, 35.0, -35.0, 55.0, -55.0]

    result_intact = lwm.imagine_and_plan_intents(
        pos_current, vel_current, goal, barrier_box, deflection_angles, scramble_dynamics=False
    )
    result_scrambled = lwm.imagine_and_plan_intents(
        pos_current, vel_current, goal, barrier_box, deflection_angles, scramble_dynamics=True
    )

    # Scrambled trajectories diverge from physical reality
    traj_intact = np.array(result_intact["predicted_trajectories"][0])
    traj_scrambled = np.array(result_scrambled["predicted_trajectories"][0])
    assert not np.allclose(traj_intact, traj_scrambled)


def test_3d_latent_world_model_planning_and_hazard_avoidance():
    """Verify 3D continuous counterfactual planning avoids spherical 3D hazards."""
    lwm = LatentWorldModel(arena_size=10.0, dt=0.5)

    pos_current = np.array([2.0, 5.0, 5.0])
    vel_current = np.zeros(3)
    goal = np.array([8.0, 5.0, 5.0])
    # 3D spherical hazard blocking direct straight line
    hazard_spheres = [(np.array([5.0, 5.0, 5.0]), 1.5)]

    candidate_actions = [
        np.array([1.0, 0.0, 0.0]),     # Straight through hazard
        np.array([0.7, 0.0, 0.9]),     # Climb up over hazard
        np.array([0.7, 0.0, -0.9]),    # Dive down under hazard
        np.array([0.7, 0.9, 0.0]),     # Deflect left around hazard
        np.array([0.7, -0.9, 0.0]),    # Deflect right around hazard
    ]

    result = lwm.imagine_and_plan_intents_3d(
        pos_current, vel_current, goal, hazard_spheres, candidate_actions, rollout_horizon=25
    )

    # Straight intention must hit hazard sphere
    assert result["intent_hazard_hits"][0] is True
    # Winning intention must NOT be the straight collision path
    assert result["chosen_action_idx"] != 0
    # Pruning margin must be positive and substantial
    assert result["pruning_margin"] > 50.0

    # 3D Scrambled dynamics test
    result_scrambled = lwm.imagine_and_plan_intents_3d(
        pos_current, vel_current, goal, hazard_spheres, candidate_actions, scramble_dynamics=True
    )
    traj_intact = np.array(result["predicted_trajectories"][0])
    traj_scrambled = np.array(result_scrambled["predicted_trajectories"][0])
    assert not np.allclose(traj_intact, traj_scrambled)
