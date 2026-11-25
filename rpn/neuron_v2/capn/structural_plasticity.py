"""Information-Theoretic Structural Plasticity Engine (CAPN Pillar 6).

Implements genuine activity-dependent neurogenesis and metabolic pruning:
1. Directional Residual Error Coherence:
   Distinguishes learnable capacity deficits (systematic distribution shifts)
   from irreducible high-variance stationary noise by measuring residual error correlation:
   rho_t = <eps(t), eps(t-1)> / (||eps(t)|| * ||eps(t-1)||).
2. Capacity-Overflow Neurogenesis:
   Spawns a new hidden unit iff residual variance > theta_growth AND error coherence > theta_rho
   sustained over a temporal window. Zero timers, zero clocks.
3. Metabolic Synaptic Pruning:
   Connections with near-zero functional impact decay and are excised.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StructuralPlasticityNetwork:
    """Network with surprise-driven neurogenesis and metabolic synaptic pruning."""

    def __init__(
        self,
        d_in: int = 8,
        d_initial_hid: int = 4,
        d_out: int = 4,
        max_hid: int = 32,
        eta: float = 0.03,
        theta_growth: float = 0.30,
        theta_rho: float = 0.38,
        sustained_steps: int = 4,
        refractory_steps: int = 25,
        prune_threshold: float = 0.005,
        utility_threshold: float = 0.005,
        alpha_u: float = 0.05,
        alpha_ma: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_in = d_in
        self.d_hid = d_initial_hid
        self.d_out = d_out
        self.max_hid = max_hid
        self.eta = eta
        self.theta_growth = theta_growth
        self.theta_rho = theta_rho
        self.sustained_steps = sustained_steps
        self.refractory_steps = refractory_steps
        self.prune_threshold = prune_threshold
        self.utility_threshold = utility_threshold
        self.alpha_u = alpha_u
        self.alpha_ma = alpha_ma
        self.rng = rng if rng is not None else np.random.RandomState(42)

        # Dynamic weight matrices
        self.W1 = self.rng.randn(self.d_hid, self.d_in) * 0.1
        self.b1 = np.zeros(self.d_hid, dtype=np.float64)
        self.W2 = self.rng.randn(self.d_out, self.d_hid) * 0.1
        self.b2 = np.zeros(self.d_out, dtype=np.float64)
        self.B = self.rng.randn(self.d_hid, self.d_out) * 0.2

        # Functional utility traces (synaptic tags) protecting rare critical synapses
        self.U1 = np.ones((self.d_hid, self.d_in), dtype=np.float64) * 0.05
        self.U2 = np.ones((self.d_out, self.d_hid), dtype=np.float64) * 0.05

        # Information-theoretic monitoring state
        self.prev_eps: Optional[np.ndarray] = None
        self.ma_variance = 0.0
        self.ma_coherence = 0.0
        self.sustained_counter = 0
        self.last_spawn_step = 0
        self.spawn_events: List[int] = []

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward activation."""
        h = np.tanh(self.W1 @ x + self.b1)
        y_hat = self.W2 @ h + self.b2
        return h, y_hat

    def spawn_unit(self, x: np.ndarray, eps_y: np.ndarray, step: int) -> None:
        """Spawn a new hidden neuron aligned with current residual error and input."""
        if self.d_hid >= self.max_hid:
            return

        norm_x = float(np.linalg.norm(x)) + 1e-8
        norm_eps = float(np.linalg.norm(eps_y)) + 1e-8

        # Input weights initialized along current input pattern
        w1_new = (x / norm_x) * 0.1
        b1_new = 0.0

        # Output weights initialized along residual prediction error
        w2_new = (eps_y / norm_eps) * 0.1

        # Random feedback projection
        b_new = self.rng.randn(1, self.d_out) * 0.2

        self.W1 = np.vstack([self.W1, w1_new])
        self.b1 = np.append(self.b1, b1_new)
        self.W2 = np.hstack([self.W2, w2_new[:, np.newaxis]])
        self.B = np.vstack([self.B, b_new])

        # New synapses receive initial active structural utility tags
        u1_new = np.ones(self.d_in, dtype=np.float64) * 0.05
        u2_new = np.ones(self.d_out, dtype=np.float64) * 0.05
        self.U1 = np.vstack([self.U1, u1_new])
        self.U2 = np.hstack([self.U2, u2_new[:, np.newaxis]])

        self.d_hid += 1
        self.spawn_events.append(step)
        self.sustained_counter = 0

    def prune_synapses(self) -> None:
        """Excise synaptic connections with BOTH sub-threshold conductance AND depleted utility tags.
        
        Protects rarely activating but critical synapses that were tagged by high-surprise neuromodulatory events.
        """
        prune_mask1 = (np.abs(self.W1) < self.prune_threshold) & (self.U1 < self.utility_threshold)
        prune_mask2 = (np.abs(self.W2) < self.prune_threshold) & (self.U2 < self.utility_threshold)
        self.W1[prune_mask1] = 0.0
        self.U1[prune_mask1] = 0.0
        self.W2[prune_mask2] = 0.0
        self.U2[prune_mask2] = 0.0

    def step(self, x: np.ndarray, y: np.ndarray, step: int) -> Tuple[np.ndarray, float, bool]:
        """Process single online sample, update weights, and check structural plasticity."""
        h, y_hat = self.forward(x)
        eps_y = y - y_hat
        err_sq = float(np.sum(eps_y ** 2))

        # 1. Compute directional coherence rho_t
        if self.prev_eps is not None:
            norm_curr = float(np.linalg.norm(eps_y))
            norm_prev = float(np.linalg.norm(self.prev_eps))
            if norm_curr > 1e-6 and norm_prev > 1e-6:
                rho = float(np.dot(eps_y, self.prev_eps) / (norm_curr * norm_prev))
            else:
                rho = 0.0
        else:
            rho = 0.0
        self.prev_eps = eps_y.copy()

        # Update exponential moving averages
        self.ma_variance = (1.0 - self.alpha_ma) * self.ma_variance + self.alpha_ma * err_sq
        self.ma_coherence = (1.0 - self.alpha_ma) * self.ma_coherence + self.alpha_ma * rho

        # 2. Check neurogenesis condition: large error AND directional coherence
        spawned = False
        in_refractory = (step - self.last_spawn_step) < self.refractory_steps
        if not in_refractory and (self.ma_variance > self.theta_growth) and (self.ma_coherence > self.theta_rho):
            self.sustained_counter += 1
            if self.sustained_counter >= self.sustained_steps:
                self.spawn_unit(x, eps_y, step)
                self.last_spawn_step = step
                spawned = True
        else:
            self.sustained_counter = max(0, self.sustained_counter - 1)

        # 3. Three-factor synaptic plasticity update
        # Re-compute h and error if new unit spawned
        if spawned:
            h, y_hat = self.forward(x)
            eps_y = y - y_hat

        dh = 1.0 - h ** 2
        eps_h = dh * (self.B @ eps_y)
        M_t = float(np.tanh(err_sq / 0.5))

        # Update functional utility traces (synaptic tags)
        impact1 = np.abs(np.outer(eps_h, x)) * M_t
        impact2 = np.abs(np.outer(eps_y, h)) * M_t
        self.U1 = (1.0 - self.alpha_u) * self.U1 + self.alpha_u * impact1
        self.U2 = (1.0 - self.alpha_u) * self.U2 + self.alpha_u * impact2

        # Excised synapses do not receive updates unless created by synaptogenesis
        self.prune_synapses()
        mask1 = (self.W1 != 0.0).astype(np.float64)
        mask2 = (self.W2 != 0.0).astype(np.float64)

        self.W2 += mask2 * (self.eta * np.outer(eps_y, h) * M_t)
        self.b2 += self.eta * eps_y * M_t
        self.W1 += mask1 * (self.eta * np.outer(eps_h, x) * M_t)
        self.b1 += self.eta * eps_h * M_t

        # 4. Metabolic synaptic pruning after plasticity
        self.prune_synapses()

        return y_hat, err_sq, spawned

    def process_stream(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Process an entire stream sample by sample."""
        n_samples = len(X)
        errors: List[float] = []
        for step in range(n_samples):
            _, err_sq, _ = self.step(X[step], Y[step], step)
            errors.append(err_sq)

        return {
            "n_spawn": len(self.spawn_events),
            "final_hid": self.d_hid,
            "spawn_events": list(self.spawn_events),
            "mean_error": float(np.mean(errors)),
        }


class TimerBasedBaselineNetwork:
    """Legacy comparative baseline: periodic timer-based neurogenesis (spawns every K steps)."""

    def __init__(
        self,
        d_in: int = 8,
        d_initial_hid: int = 4,
        d_out: int = 4,
        spawn_period: int = 50,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.d_in = d_in
        self.d_hid = d_initial_hid
        self.d_out = d_out
        self.spawn_period = spawn_period
        self.rng = rng if rng is not None else np.random.RandomState(42)
        self.spawn_events: List[int] = []

    def process_stream(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        n_samples = len(X)
        for step in range(n_samples):
            if step > 0 and step % self.spawn_period == 0:
                self.d_hid += 1
                self.spawn_events.append(step)

        return {
            "n_spawn": len(self.spawn_events),
            "final_hid": self.d_hid,
            "spawn_events": list(self.spawn_events),
        }
