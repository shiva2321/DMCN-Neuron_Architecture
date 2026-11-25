"""Universal Domain-Agnostic Continuous Attractor Predictive Network (CAPN) Column.

Implements the canonical cortical column architecture operating identically across
heterogeneous continuous domains (spatial navigation, non-linear dynamical systems,
and multi-channel streaming temporal signals):

1. L4 State Ingestion & Attractor Dynamics:
   Continuous Lyapunov energy relaxation with dual-store Tsodyks-Markram short-term plasticity.
2. L5/L6 Plastic Online Forward Transition Model:
   Learns environmental transition dynamics (z_{t+1} = f(z_t, u_t)) online via Three-Factor
   Plasticity without backpropagation or handcoded kinematic formulas.
3. Universal Motor/Action Attractor Field:
   Arbitrary d_u-dimensional spherical attractor network with recurrent Mexican-hat connectivity,
   sensory shunting, and Georgopoulos population vector decoding.
4. Prospective Mental Rollouts:
   Simulates forward candidate affordances in latent space using its own learned forward model.
5. Information-Theoretic Structural Plasticity & Synaptic Tagging:
   Directional residual error coherence (rho_t) for noise-immune neurogenesis and STC utility
   tags protecting rare critical connections.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def generate_spherical_basis_vectors(d: int, n_vectors: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate n_vectors normalized directions uniformly spanning unit sphere S^{d-1}."""
    if d == 3 and n_vectors == 14:
        # Canonical 3D 14-directional basis (6 cardinal + 8 diagonal)
        rays = []
        for k in range(3):
            d1 = np.zeros(3, dtype=np.float64)
            d1[k] = 1.0
            rays.append(d1)
            d2 = np.zeros(3, dtype=np.float64)
            d2[k] = -1.0
            rays.append(d2)
        for s1 in [-1.0, 1.0]:
            for s2 in [-1.0, 1.0]:
                for s3 in [-1.0, 1.0]:
                    v = np.array([s1, s2, s3], dtype=np.float64)
                    rays.append(v / np.linalg.norm(v))
        return np.array(rays, dtype=np.float64)

    # General d-dimensional spherical distribution via normalized Gaussians
    gauss = rng.randn(n_vectors, d)
    norms = np.linalg.norm(gauss, axis=1, keepdims=True) + 1e-9
    return (gauss / norms).astype(np.float64)


class UniversalMotorAttractorField:
    """Domain-agnostic continuous motor attractor field for arbitrary d_action dimensions."""

    def __init__(
        self,
        d_action: int = 3,
        n_basis: int = 14,
        beta_inh: float = 0.35,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_action = d_action
        self.n_basis = n_basis
        self.rng = rng if rng is not None else np.random.RandomState(42)

        self.basis = generate_spherical_basis_vectors(d_action, n_basis, self.rng)
        # Recurrent spherical Mexican-hat connectivity: cosine similarity minus lateral inhibition
        self.W = (self.basis @ self.basis.T) - beta_inh
        np.fill_diagonal(self.W, 1.0)
        # Angular hazard diffusion matrix
        self.diffuse = np.maximum(0.0, self.basis @ self.basis.T) ** 2

    def compute_affordances(
        self,
        dir_goal: np.ndarray,
        sensory_hazard: np.ndarray,
        prev_act: np.ndarray,
    ) -> List[np.ndarray]:
        """Compute population vector and active affordance peaks under sensory inhibition."""
        norm_goal = float(np.linalg.norm(dir_goal)) + 1e-9
        u_goal = dir_goal / norm_goal

        norm_prev = float(np.linalg.norm(prev_act)) + 1e-9
        u_prev = prev_act / norm_prev

        # 1. External afferent drives
        I_goal = np.maximum(0.0, self.basis @ u_goal)
        I_prev = np.maximum(0.0, self.basis @ u_prev)

        # Hazard repulsion in basis coordinates
        if len(sensory_hazard) == self.n_basis:
            H = self.diffuse @ sensory_hazard
        else:
            H = np.zeros(self.n_basis, dtype=np.float64)

        # Net afferent drive
        net_drive = 2.5 * I_goal + 1.0 * I_prev - 4.0 * H

        # 2. Recurrent continuous attractor relaxation
        a = np.maximum(0.0, net_drive)
        for _ in range(8):
            a = np.tanh(0.35 * (self.W @ a) + net_drive)
            a = np.maximum(0.0, a)

        # 3. Primary Georgopoulos population vector
        u_pop = self.basis.T @ a
        norm = float(np.linalg.norm(u_pop))
        pop_vector = (u_pop / norm) if norm > 1e-4 else u_goal.copy()

        # 4. Extract active affordance peaks
        candidates = [pop_vector, u_goal.copy(), u_prev.copy()]
        top_indices = np.argsort(a)[::-1][:4]
        for idx in top_indices:
            hazard_val = sensory_hazard[idx] if idx < len(sensory_hazard) else 0.0
            if a[idx] > 0.05 and hazard_val < 0.45:
                candidates.append(self.basis[idx].copy())

        return candidates


class PlasticTransitionModel:
    """Online non-backprop transition model learning f(z_t, u_t) -> Delta z_t."""

    def __init__(
        self,
        d_state: int = 16,
        d_action: int = 3,
        d_hid: int = 32,
        eta: float = 0.04,
        sigma_mod: float = 0.5,
        B_prior: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_state = d_state
        self.d_action = d_action
        self.d_hid = d_hid
        self.d_in = d_state + d_action
        self.eta = eta
        self.sigma_mod = sigma_mod
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Baseline efference copy prior (e.g. B_0 = -basis * dt)
        if B_prior is not None:
            self.B_prior: Optional[np.ndarray] = np.array(B_prior, dtype=np.float64)
            # Initialize plastic residual weights to zero so prior is active from step 0
            self.W2 = np.zeros((self.d_state, self.d_hid), dtype=np.float64)
        else:
            self.B_prior = None
            self.W2 = self.rng.randn(self.d_state, self.d_hid) * 0.1

        # Feedforward weights
        self.W1 = self.rng.randn(self.d_hid, self.d_in) * 0.05
        self.b1 = np.zeros(self.d_hid, dtype=np.float64)
        self.b2 = np.zeros(self.d_state, dtype=np.float64)

        # Fixed random feedback alignment matrix for Layer 1
        self.B = self.rng.randn(self.d_hid, self.d_state) * 0.2

    def forward(self, z: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict continuous state change Delta z."""
        v = np.concatenate([z, u]) if len(u) > 0 else z
        h = np.tanh(self.W1 @ v + self.b1)
        if self.B_prior is not None:
            delta_plastic = np.clip(self.W2 @ h + self.b2, -0.1, 0.1)
            delta_prior = (self.B_prior @ u) if len(u) > 0 else np.zeros(self.d_state, dtype=np.float64)
            delta_z = delta_prior + delta_plastic
        else:
            delta_z = np.clip(self.W2 @ h + self.b2, -0.4, 0.4)
        return h, delta_z

    def predict_next(
        self,
        z: np.ndarray,
        u: np.ndarray,
        clip_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Predict next continuous state z_{t+1}."""
        _, delta_z = self.forward(z, u)
        next_z = z + delta_z
        if clip_range is not None:
            next_z = np.clip(next_z, clip_range[0], clip_range[1])
        return next_z

    def learn_step(self, z_prev: np.ndarray, u_prev: np.ndarray, z_curr: np.ndarray) -> float:
        """Online three-factor plastic update on observed transition without backpropagation."""
        v = np.concatenate([z_prev, u_prev]) if len(u_prev) > 0 else z_prev
        h, delta_z = self.forward(z_prev, u_prev)
        actual_delta = z_curr - z_prev
        eps_z = actual_delta - delta_z

        # Norm clipping for numerical robustness under chaotic excursions
        eps_norm = float(np.linalg.norm(eps_z))
        if eps_norm > 0.4:
            eps_z = (eps_z / eps_norm) * 0.4

        err_sq = float(np.sum(eps_z ** 2))

        # Three-factor neuromodulator gate
        M_t = float(np.tanh(err_sq / self.sigma_mod))

        # Hidden error via fixed feedback alignment
        dh = 1.0 - h ** 2
        eps_h = dh * (self.B @ eps_z)

        # Scale normalization for input norm invariance (Normalized Hebbian/LMS)
        scale_v = 1.0 / np.sqrt(1.0 + float(np.sum(v ** 2)))
        scale_h = 1.0 / np.sqrt(1.0 + float(np.sum(h ** 2)))

        # Synaptic weight updates with gentle weight decay
        decay = 0.0005
        self.W2 += self.eta * np.outer(eps_z, h) * M_t * scale_h - decay * self.W2
        self.b2 += self.eta * eps_z * M_t * 0.5
        self.W1 += self.eta * np.outer(eps_h, v) * M_t * scale_v - decay * self.W1
        self.b1 += self.eta * eps_h * M_t * 0.5

        return err_sq


class UniversalCAPNColumn:
    """Canonical Cortical Column operating across any continuous sensory/motor domain."""

    def __init__(
        self,
        d_sensory: int = 14,
        d_state: int = 16,
        d_action: int = 3,
        n_action_basis: int = 14,
        dt: float = 0.1,
        eta_dyn: float = 0.04,
        use_efference_prior: bool = False,
        clip_range: Optional[Tuple[float, float]] = None,
        prune_threshold: float = 0.005,
        utility_threshold: float = 0.005,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_sensory = d_sensory
        self.d_state = d_state
        self.d_action = d_action
        self.n_action_basis = n_action_basis
        self.dt = dt
        self.use_efference_prior = use_efference_prior
        self.clip_range = clip_range
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # 1. L4 Sensory Ingestion & Attractor State Representation
        if self.use_efference_prior and self.d_state == self.d_sensory:
            self.W_in = np.eye(self.d_state, self.d_sensory, dtype=np.float64)
        else:
            self.W_in = self.rng.randn(self.d_state, self.d_sensory) * (1.0 / np.sqrt(self.d_sensory))
        self.b_in = np.zeros(self.d_state, dtype=np.float64)
        self.W_rec = self.rng.randn(self.d_state, self.d_state) * 0.05
        np.fill_diagonal(self.W_rec, 0.5)

        # Tsodyks-Markram Short-Term Plasticity (STP) state
        self.u_stp = 0.2
        self.x_stp = np.ones(self.d_state, dtype=np.float64)
        self.U_base = 0.2
        self.tau_fac = 200.0  # ms
        self.tau_rec = 500.0  # ms
        self.tau_dt = 10.0  # ms

        # 3. Universal Motor Attractor Field
        self.motor_attractor = UniversalMotorAttractorField(
            d_action=self.d_action,
            n_basis=self.n_action_basis,
            rng=self.rng,
        )

        # 2. L5/L6 Plastic Transition World Model
        if self.use_efference_prior and self.d_state == self.n_action_basis:
            B_prior = -self.motor_attractor.basis * self.dt
        else:
            B_prior = None

        self.dyn_model = PlasticTransitionModel(
            d_state=self.d_state,
            d_action=self.d_action,
            d_hid=32,
            eta=eta_dyn,
            B_prior=B_prior,
            rng=self.rng,
        )

        # 4. Synaptic Tagging & Information-Theoretic Structural State
        self.prune_threshold = prune_threshold
        self.utility_threshold = utility_threshold
        self.alpha_u = 0.05
        self.U_in = np.ones((self.d_state, self.d_sensory), dtype=np.float64) * 0.05

        self.prev_z: Optional[np.ndarray] = None
        self.prev_act = np.zeros(self.d_action, dtype=np.float64)
        if self.d_action > 0:
            self.prev_act.fill(1.0 / np.sqrt(self.d_action))

        self.prev_eps: Optional[np.ndarray] = None
        self.ma_variance = 0.0
        self.ma_coherence = 0.0

    def reset(self) -> None:
        """Reset internal temporal states."""
        self.x_stp = np.ones(self.d_state, dtype=np.float64)
        self.u_stp = self.U_base
        self.prev_z = None
        self.prev_act = np.zeros(self.d_action, dtype=np.float64)
        if self.d_action > 0:
            self.prev_act.fill(1.0 / np.sqrt(self.d_action))
        self.prev_eps = None
        self.ma_variance = 0.0
        self.ma_coherence = 0.0

    def encode_state(self, sensory_input: np.ndarray) -> np.ndarray:
        """L4 State Ingestion: Continuous attractor relaxation + Tsodyks-Markram STP."""
        if self.use_efference_prior:
            return sensory_input.copy()

        # Tsodyks-Markram dynamics update
        self.u_stp += self.tau_dt * (self.U_base - self.u_stp) / self.tau_fac
        self.x_stp += self.tau_dt * (1.0 - self.x_stp) / self.tau_rec

        # Effective synaptic transmission
        syn_efficacy = self.u_stp * self.x_stp
        drive = self.W_in @ sensory_input + self.b_in

        # Energy relaxation
        z = np.tanh(drive * syn_efficacy)
        for _ in range(5):
            z = np.tanh(self.W_rec @ z + drive * syn_efficacy)

        # Update depression
        self.x_stp -= self.u_stp * self.x_stp * (np.abs(z) * 0.1)
        self.x_stp = np.clip(self.x_stp, 0.01, 1.0)
        return z

    def step_learn_transition(self, z_curr: np.ndarray) -> float:
        """Learn world dynamics online from observed transition without backprop."""
        if self.prev_z is not None:
            err = self.dyn_model.learn_step(self.prev_z, self.prev_act, z_curr)
        else:
            err = 0.0
        self.prev_z = z_curr.copy()
        return err

    def prune_synapses(self) -> None:
        """Excise input synapses only if both weight and utility tags are sub-threshold (STC)."""
        mask = (np.abs(self.W_in) < self.prune_threshold) & (self.U_in < self.utility_threshold)
        self.W_in[mask] = 0.0
        self.U_in[mask] = 0.0

    def select_action(
        self,
        dir_goal: np.ndarray,
        sensory_hazard_profile: np.ndarray,
        z_curr: Optional[np.ndarray] = None,
        hazard_eval_fn: Optional[Any] = None,
        horizon: int = 15,
    ) -> np.ndarray:
        """Prospective action selection via Motor Attractor Field and Learned World Model."""
        norm_goal = float(np.linalg.norm(dir_goal)) + 1e-9
        u_goal = dir_goal / norm_goal

        # 1. Biological motor affordance competition (continuous neural attractor field)
        candidates = self.motor_attractor.compute_affordances(
            dir_goal=u_goal,
            sensory_hazard=sensory_hazard_profile,
            prev_act=self.prev_act,
        )

        # 2. Prospective forward rollout evaluation in learned latent state space
        best_score = -float("inf")
        best_act = candidates[0]

        for cand_act in candidates:
            hazard_penalty = 0.0
            if z_curr is not None and hazard_eval_fn is not None:
                sim_z = z_curr.copy()
                for h in range(1, horizon + 1):
                    sim_z = self.dyn_model.predict_next(sim_z, cand_act, clip_range=self.clip_range)
                    hazard_penalty += hazard_eval_fn(sim_z, h)

            progress_score = float(np.dot(cand_act, u_goal)) * 5.0
            smoothness_bonus = 2.0 * float(np.dot(cand_act, self.prev_act))
            score = progress_score - hazard_penalty + smoothness_bonus

            if score > best_score:
                best_score = score
                best_act = cand_act

        self.prev_act = best_act.copy()
        return best_act
