"""Canonical 3-Layer Cortical Predictive Microcircuit.

Implements the classical predictive coding laminar architecture (Rao & Ballard 1999;
Bastos et al. 2012; Friston 2005; Larkum 2013):
- Layer 4 (L4): Granular sensory relay receiving feedforward continuous sensory input y(t).
- Layer 2/3 (L2/3): Supragranular prediction error comparator computing:
      epsilon(t) = y(t) - y_hat(t)
  When top-down predictions match sensory input, prediction error is cancelled (predictive silencing).
- Layer 5 (L5): Infragranular generative state-space predictor. Maintains continuous temporal
  history via Continuous Legendre Memory and projects top-down prediction y_hat(t) back to L2/3.
  Weights adapt via local predictive coding plasticity:
      dw/dt = eta * epsilon(t) * m(t)
"""
from __future__ import annotations

import numpy as np
from neuron_v2.biophysics.continuous_delay import ContinuousLegendreDelayEngine


class PredictiveCorticalMicrocircuit:
    """Canonical 3-Layer Cortical Predictive Microcircuit."""

    def __init__(
        self,
        order: int = 8,
        theta: float = 40.0,
        dt: float = 0.5,
        lr: float = 0.05,
    ):
        self.order = int(order)
        self.theta = float(theta)
        self.dt = float(dt)
        self.lr = float(lr)

        # Layer 5 continuous temporal state-space engine
        self.l5_engine = ContinuousLegendreDelayEngine(order=self.order, theta=self.theta, dt=self.dt)
        # Layer 5 generative prediction weights
        self.w_pred = np.zeros(self.order, dtype=np.float64)

    def step(
        self,
        y_sensory: float,
        top_down_override: float | None = None,
        learn: bool = True,
    ) -> dict[str, float | np.ndarray]:
        """Execute one continuous microcircuit step.
        
        y_sensory: feedforward sensory input from Layer 4.
        top_down_override: optional custom prediction (e.g. for decoupled null control).
        learn: if True, updates predictive coding weights locally.
        """
        # 1. Advance Layer 5 continuous state space with sensory stream
        m_t = self.l5_engine.step(y_sensory)

        # 2. Layer 5 top-down prediction to Layer 2/3 apical dendrite
        if top_down_override is not None:
            y_hat = float(top_down_override)
        else:
            y_hat = float(np.dot(self.w_pred, m_t))

        # 3. Layer 2/3 prediction error comparator
        eps = float(y_sensory - y_hat)

        # 4. Local predictive coding plasticity: dw = lr * eps * m_t / (||m||^2 + eps0)
        if learn:
            norm_m = float(np.dot(m_t, m_t)) + 1e-4
            self.w_pred += (self.lr / norm_m) * eps * m_t

        return {
            "sensory": float(y_sensory),
            "prediction": y_hat,
            "error": eps,
            "l5_state": m_t.copy(),
            "w_pred": self.w_pred.copy(),
        }

    def reset(self):
        """Reset internal temporal states and generative weights."""
        self.l5_engine.reset()
        self.w_pred.fill(0.0)
