"""Latent World Model and Counterfactual Imagination Engine (CAPN Pillar 4).

Integrates:
- Factorized Latent Cognitive Manifold z = [z_map, z_entity, z_dyn, u].
- Internal Forward Transition Engine F_theta(z, a) that simulates hypothetical futures
  in latent space while sensory gating is decoupled.
- Counterfactual Pruning: Evaluates cumulative energy landscapes of competing trajectories,
  pruning deceptive traps internally before physical action is committed.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


class LatentWorldModel:
    """Continuous latent world model with counterfactual mental simulation."""

    def __init__(
        self,
        d_map: int = 32,
        d_entity: int = 16,
        d_dyn: int = 8,
        d_unc: int = 8,
        arena_size: float = 10.0,
        dt: float = 0.5,
    ) -> None:
        """Initialize latent world model.

        Args:
            d_map: Dimension of spatial place field features.
            d_entity: Dimension of entity/affordance features.
            d_dyn: Dimension of dynamics/velocity features.
            d_unc: Dimension of epistemic uncertainty features.
            arena_size: Physical arena dimension [0, arena_size]^2.
            dt: Simulation time-step.
        """
        self.d_map = d_map
        self.d_entity = d_entity
        self.d_dyn = d_dyn
        self.d_unc = d_unc
        self.arena_size = arena_size
        self.dt = dt

        # Place field centers distributed across the continuous arena
        grid_side = int(np.ceil(np.sqrt(d_map)))
        coords = np.linspace(1.0, arena_size - 1.0, grid_side)
        xx, yy = np.meshgrid(coords, coords)
        self.place_centers = np.column_stack([xx.ravel(), yy.ravel()])[:d_map]
        self.place_sigma = (arena_size / grid_side) * 0.8

        # Entity basis centers for distance/heading to entities
        self.entity_dists = np.linspace(0.0, arena_size * np.sqrt(2.0), d_entity // 2)
        self.entity_sigma = (arena_size * np.sqrt(2.0) / len(self.entity_dists)) * 0.8

    def encode_latent_state(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        barrier_box: Tuple[float, float, float, float],
        sensory_precision: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Encode physical and environmental observables into factorized manifold z.

        z = [z_map, z_entity, z_dyn, u]

        Returns:
            Dictionary containing individual factorized manifolds and full concatenated z.
        """
        # 1. z_map: Spatial place field activations
        z_map = self.encode_spatial_state(pos)

        # 2. z_entity: Goal beacon and hazard entity features
        dist_goal = np.linalg.norm(pos - goal)
        goal_features = np.exp(-((self.entity_dists - dist_goal) ** 2) / (2.0 * self.entity_sigma ** 2))
        goal_features /= (np.linalg.norm(goal_features) + 1e-12)

        xmin, xmax, ymin, ymax = barrier_box
        barrier_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        dist_barrier = np.linalg.norm(pos - barrier_center)
        barrier_features = np.exp(-((self.entity_dists - dist_barrier) ** 2) / (2.0 * self.entity_sigma ** 2))
        barrier_features /= (np.linalg.norm(barrier_features) + 1e-12)

        z_entity = np.concatenate([goal_features, barrier_features])[:self.d_entity]

        # 3. z_dyn: Directional velocity and momentum representation
        speed = float(np.linalg.norm(vel))
        heading = float(np.arctan2(vel[1], vel[0])) if speed > 1e-6 else 0.0
        dyn_angles = np.linspace(-np.pi, np.pi, self.d_dyn, endpoint=False)
        z_dyn = np.exp(-((dyn_angles - heading) ** 2) / (2.0 * (np.pi / self.d_dyn) ** 2)) * np.tanh(speed)
        norm_dyn = np.linalg.norm(z_dyn)
        if norm_dyn > 1e-12:
            z_dyn /= norm_dyn

        # 4. u: Epistemic uncertainty representation (inverse sensory precision)
        uncertainty_level = 1.0 / (sensory_precision + 1e-6)
        u = np.full(self.d_unc, uncertainty_level * 0.1, dtype=np.float64)

        z_full = np.concatenate([z_map, z_entity, z_dyn, u])

        return {
            "z_map": z_map,
            "z_entity": z_entity,
            "z_dyn": z_dyn,
            "u": u,
            "z_full": z_full,
        }

    def encode_spatial_state(self, pos: np.ndarray) -> np.ndarray:
        """Encode continuous 2D position into non-linear place field manifold.

        Args:
            pos: 2D continuous coordinate [x, y].

        Returns:
            z_map feature vector (d_map,).
        """
        dists_sq = np.sum((self.place_centers - pos) ** 2, axis=1)
        z_map = np.exp(-dists_sq / (2.0 * self.place_sigma ** 2))
        norm_z = np.linalg.norm(z_map)
        return z_map / (norm_z + 1e-12)

    def latent_transition(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: np.ndarray,
        scramble_dynamics: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next physical/latent state under action intent across any Euclidean dimension D (e.g. 2D or 3D).

        Args:
            pos: Current D-dimensional position (x, y) or (x, y, z).
            vel: Current D-dimensional velocity.
            action: D-dimensional action vector.
            scramble_dynamics: Null control; if True, applies scrambled orthogonal physics.

        Returns:
            Tuple of (pos_next, vel_next).
        """
        if scramble_dynamics:
            # Control: causal dynamics scrambled by orthogonal rotation
            if len(action) == 2:
                rotated_action = np.array([-action[1], action[0]])
            elif len(action) == 3:
                # 3D orthogonal rotation via cross product with non-parallel reference axis
                axis = np.array([0.0, 0.0, 1.0]) if abs(action[2]) < 0.85 * (np.linalg.norm(action) + 1e-8) else np.array([0.0, 1.0, 0.0])
                rotated_action = np.cross(action, axis)
                norm_act = float(np.linalg.norm(action))
                norm_rot = float(np.linalg.norm(rotated_action))
                if norm_rot > 1e-8:
                    rotated_action = rotated_action * (norm_act / norm_rot)
            else:
                rotated_action = np.roll(action, 1)

            vel_next = 0.7 * vel + 0.3 * rotated_action
            pos_next = pos + vel_next * self.dt
        else:
            # Physical continuous forward transition in D-dimensional Euclidean space
            vel_next = 0.7 * vel + 0.3 * action
            pos_next = pos + vel_next * self.dt

        pos_next = np.clip(pos_next, 0.0, self.arena_size)
        return pos_next, vel_next

    def evaluate_latent_energy(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        barrier_box: Tuple[float, float, float, float],
    ) -> Tuple[float, bool]:
        """Evaluate latent energy landscape: distance to goal + hazard penalty.

        Args:
            pos: 2D position.
            goal: 2D target goal.
            barrier_box: Bounding box of deceptive barrier (xmin, xmax, ymin, ymax).

        Returns:
            Tuple of (energy_scalar, is_hazard_hit).
        """
        xmin, xmax, ymin, ymax = barrier_box
        is_hazard = (xmin <= pos[0] <= xmax) and (ymin <= pos[1] <= ymax)

        dist_goal = float(np.linalg.norm(pos - goal))
        hazard_penalty = 200.0 if is_hazard else 0.0
        total_energy = dist_goal + hazard_penalty

        return total_energy, is_hazard

    def compute_intent_action(
        self,
        pos_current: np.ndarray,
        goal: np.ndarray,
        deflection_deg: float,
        bias_active: bool = True,
        speed: float = 1.5,
    ) -> np.ndarray:
        """Compute continuous action vector with lateral deflection angle.

        Args:
            pos_current: Current 2D position.
            goal: 2D goal location.
            deflection_deg: Lateral deflection angle in degrees.
            bias_active: Whether deflection bias is active (turned off once barrier is cleared).
            speed: Action drive magnitude.

        Returns:
            2D continuous acceleration vector.
        """
        diff = goal - pos_current
        theta_goal = np.arctan2(diff[1], diff[0])
        theta = (theta_goal + np.deg2rad(deflection_deg)) if bias_active else theta_goal
        return np.array([np.cos(theta), np.sin(theta)]) * speed

    def imagine_and_plan_intents(
        self,
        pos_current: np.ndarray,
        vel_current: np.ndarray,
        goal: np.ndarray,
        barrier_box: Tuple[float, float, float, float],
        deflection_angles: List[float],
        rollout_horizon: int = 25,
        scramble_dynamics: bool = False,
    ) -> Dict[str, Any]:
        """Execute counterfactual mental rollouts across candidate directional intents.

        Decouples sensory drive and evaluates hypothetical forward trajectories.

        Args:
            pos_current: Current 2D position.
            vel_current: Current 2D velocity.
            goal: 2D target goal.
            barrier_box: Bounding box of deceptive hazard.
            deflection_angles: List of deflection angles in degrees.
            rollout_horizon: Number of forward simulation steps.
            scramble_dynamics: Null control flag.

        Returns:
            Dictionary with chosen deflection angle, pruning margin, and simulated costs.
        """
        xmin, xmax, ymin, ymax = barrier_box
        intent_costs: List[float] = []
        intent_hazard_hits: List[bool] = []
        predicted_trajectories: List[List[np.ndarray]] = []

        for d_deg in deflection_angles:
            pos_sim = pos_current.copy()
            vel_sim = vel_current.copy()
            traj: List[np.ndarray] = [pos_sim.copy()]
            cum_cost = 0.0
            hit_hazard = False

            for _ in range(rollout_horizon):
                bias_active = (pos_sim[0] < xmax + 0.3)
                act = self.compute_intent_action(pos_sim, goal, d_deg, bias_active=bias_active)
                pos_sim, vel_sim = self.latent_transition(
                    pos_sim, vel_sim, act,
                    scramble_dynamics=scramble_dynamics,
                )
                traj.append(pos_sim.copy())
                e_step, is_haz = self.evaluate_latent_energy(pos_sim, goal, barrier_box)
                cum_cost += e_step
                if is_haz:
                    hit_hazard = True

            intent_costs.append(cum_cost)
            intent_hazard_hits.append(hit_hazard)
            predicted_trajectories.append(traj)

        best_idx = int(np.argmin(intent_costs))
        chosen_angle = deflection_angles[best_idx]
        worst_idx = int(np.argmax(intent_costs))
        pruning_margin = float(intent_costs[worst_idx] - intent_costs[best_idx])

        return {
            "chosen_angle_idx": best_idx,
            "chosen_angle_deg": chosen_angle,
            "intent_costs": intent_costs,
            "intent_hazard_hits": intent_hazard_hits,
            "pruning_margin": pruning_margin,
            "predicted_trajectories": predicted_trajectories,
        }

    def evaluate_latent_energy_3d(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        hazard_spheres: List[Tuple[np.ndarray, float]],
        safety_margin: float = 0.4,
    ) -> Tuple[float, bool]:
        """Evaluate 3D continuous latent energy: distance to goal + spherical hazard penalties."""
        dist_goal = float(np.linalg.norm(pos - goal))
        hazard_penalty = 0.0
        is_hazard = False

        for center, radius in hazard_spheres:
            dist_c = float(np.linalg.norm(pos - center))
            if dist_c < (radius + safety_margin):
                penetration = (radius + safety_margin) - dist_c
                hazard_penalty += 100.0 + 50.0 * (penetration ** 2)
                is_hazard = True

        total_energy = dist_goal + hazard_penalty
        return total_energy, is_hazard

    def imagine_and_plan_intents_3d(
        self,
        pos_current: np.ndarray,
        vel_current: np.ndarray,
        goal: np.ndarray,
        hazard_spheres: List[Tuple[np.ndarray, float]],
        candidate_actions: List[np.ndarray],
        rollout_horizon: int = 25,
        scramble_dynamics: bool = False,
    ) -> Dict[str, Any]:
        """Execute 3D counterfactual mental rollouts across candidate continuous 3D velocity/thrust intents."""
        intent_costs: List[float] = []
        intent_hazard_hits: List[bool] = []
        predicted_trajectories: List[List[np.ndarray]] = []

        for act in candidate_actions:
            pos_sim = pos_current.copy()
            vel_sim = vel_current.copy()
            traj: List[np.ndarray] = [pos_sim.copy()]
            cum_cost = 0.0
            hit_hazard = False

            for _ in range(rollout_horizon):
                pos_sim, vel_sim = self.latent_transition(
                    pos_sim, vel_sim, act,
                    scramble_dynamics=scramble_dynamics,
                )
                traj.append(pos_sim.copy())
                e_step, is_haz = self.evaluate_latent_energy_3d(pos_sim, goal, hazard_spheres)
                cum_cost += e_step
                if is_haz:
                    hit_hazard = True

            intent_costs.append(cum_cost)
            intent_hazard_hits.append(hit_hazard)
            predicted_trajectories.append(traj)

        best_idx = int(np.argmin(intent_costs))
        worst_idx = int(np.argmax(intent_costs))
        pruning_margin = float(intent_costs[worst_idx] - intent_costs[best_idx])

        return {
            "chosen_action_idx": best_idx,
            "chosen_action": candidate_actions[best_idx],
            "intent_costs": intent_costs,
            "intent_hazard_hits": intent_hazard_hits,
            "pruning_margin": pruning_margin,
            "predicted_trajectories": predicted_trajectories,
        }
