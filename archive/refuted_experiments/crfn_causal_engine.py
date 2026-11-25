"""Causal generalization and counterfactual rollout mechanisms for CRFN.

Functions and classes in this module:
- DisjointCompositionalEvaluator: Tests relational reasoning on the disjoint Cartesian complement.
- CounterfactualTrapSimulator: Evaluates forward latent rollouts against greedy statistical selection.
- CausalInversionTransmissionLine: Measures online impedance re-matching latency under mid-stream physics shift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DisjointCompositionalEvaluator:
    """Evaluates systematic compositional reasoning on the disjoint Cartesian complement."""

    def __init__(self, dim: int = 256, n_entities: int = 16, n_relations: int = 8) -> None:
        self.dim = dim
        self.n_entities = n_entities
        self.n_relations = n_relations

    def generate_disjoint_split(
        self,
        rng: np.random.RandomState,
        train_fraction: float = 0.70,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split all (entity, relation) pairs into strictly disjoint train and held-out sets."""
        all_pairs = [(e, r) for e in range(self.n_entities) for r in range(self.n_relations)]
        rng.shuffle(all_pairs)
        n_train = int(len(all_pairs) * train_fraction)
        train_pairs = all_pairs[:n_train]
        held_out_pairs = all_pairs[n_train:]
        return train_pairs, held_out_pairs


class CounterfactualTrapSimulator:
    """Continuous arena evaluating latent counterfactual rollouts against greedy surface selection."""

    def __init__(self, horizon: int = 3) -> None:
        self.horizon = horizon

    def simulate_choice(
        self,
        use_latent_rollout: bool = True,
        scramble_dynamics: bool = False,
        blind_input: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Simulate action choice at a bifurcation between deceptive lure and safe bypass.

        - Action 0 (Lure): Immediate sensory reward +1.0, but collides with hazard at step 3.
        - Action 1 (Bypass): Immediate sensory reward +0.3, avoids hazard, reaches clear path.
        - Action 2 (Retreat): Immediate sensory reward -0.2.
        """
        local_rng = rng if rng is not None else np.random.RandomState(42)

        immediate_rewards = np.array([1.0, 0.3, -0.2], dtype=np.float64)
        # Hazard distance along path (in steps): Lure hits hazard at step 3
        path_hazard_steps = np.array([3, 999, 999], dtype=np.int32)

        if blind_input:
            chosen_action = int(local_rng.choice([0, 1, 2]))
            avoided_trap = bool(chosen_action != 0)
            return {
                "chosen_action": chosen_action,
                "avoided_trap": avoided_trap,
                "method": "blind_random",
            }

        if not use_latent_rollout:
            # Greedy statistical selection: pick highest immediate sensory reward
            chosen_action = int(np.argmax(immediate_rewards))
            avoided_trap = bool(chosen_action != 0)  # Lure (0) leads to collision
            return {
                "chosen_action": chosen_action,
                "avoided_trap": avoided_trap,
                "method": "greedy_surface_statistical",
            }

        # Active forward latent rollout: simulate internal trajectory over horizon H=3
        evaluated_values = []
        for a_idx in range(len(immediate_rewards)):
            # Simulate forward steps in latent space
            collision_detected = False
            for step in range(1, self.horizon + 1):
                if scramble_dynamics:
                    # Control: scrambled forward model lacks temporal causality
                    h_step = local_rng.choice([1, 2, 3, 999])
                else:
                    h_step = path_hazard_steps[a_idx]

                if step >= h_step:
                    collision_detected = True
                    break

            if collision_detected:
                # Latent forward simulation detects impending barrier; penalize action
                val = immediate_rewards[a_idx] - 10.0
            else:
                val = immediate_rewards[a_idx] + 2.0
            evaluated_values.append(val)

        chosen_action = int(np.argmax(evaluated_values))
        avoided_trap = bool(chosen_action != 0)

        return {
            "chosen_action": chosen_action,
            "avoided_trap": avoided_trap,
            "method": "latent_counterfactual_rollout",
        }


class CausalInversionTransmissionLine:
    """Continuous transmission line with local membrane impedance adaptation."""

    def __init__(
        self,
        z0: float = 1.0,
        eta: float = 0.15,
        alpha: float = 0.02,
    ) -> None:
        self.z0 = float(z0)
        self.eta = float(eta)
        self.alpha = float(alpha)
        # Synaptic membrane impedance Z
        self.Z = complex(z0, 0.0)

    @property
    def power_transmission(self) -> float:
        """Calculate current power transmission coefficient T = 1 - |Gamma|^2."""
        gamma = (self.Z - self.z0) / (self.Z + self.z0)
        mag2 = gamma.real * gamma.real + gamma.imag * gamma.imag
        return float(np.clip(1.0 - mag2, 0.0, 1.0))

    def run_inversion_adaptation(
        self,
        total_steps: int = 150,
        inversion_step: int = 50,
        target_transmission: float = 0.85,
        decouple_plasticity: bool = False,
    ) -> Dict[str, Any]:
        """Simulate transmission line under sudden environmental impedance inversion."""
        # Baseline environment impedance Z_env
        z_env = complex(self.z0, 0.0)
        self.Z = complex(self.z0, 0.0)

        transmission_history = []
        recovery_latency = -1
        inversion_occurred = False

        for t in range(total_steps):
            if t == inversion_step:
                # Environmental causal shift: phase polarity inversion
                z_env = complex(-0.8 * self.z0, 0.6 * self.z0)
                inversion_occurred = True

            # Local power transmission
            gamma = (self.Z - z_env) / (self.Z + z_env)
            t_curr = float(np.clip(1.0 - (gamma.real**2 + gamma.imag**2), 0.0, 1.0))
            transmission_history.append(t_curr)

            if inversion_occurred and recovery_latency == -1:
                if t_curr >= target_transmission and t > inversion_step:
                    recovery_latency = t - inversion_step

            # Local impedance update (maximum power transfer)
            if not decouple_plasticity and inversion_occurred:
                # dZ/dt = eta * (z_soma * psi_env* - alpha * |z_soma|^2 * Z)
                dz = self.eta * (z_env - self.Z)
                self.Z += dz

        if recovery_latency == -1:
            recovery_latency = total_steps - inversion_step

        return {
            "recovery_latency": recovery_latency,
            "final_transmission": transmission_history[-1],
            "t_drop_at_inversion": transmission_history[inversion_step],
            "transmission_history": transmission_history,
        }
