"""Resonant Compartmental Predictive Neuron (RCPN).

Unifies:
1. Multi-branch basal dendrites with local NMDA plateaus and human dCaAP sub-linear XOR gating.
2. Competitive lateral branch inhibition for autonomous symmetry breaking (no pre-wired routing).
3. Apical nexus coincidence detection (Larkum apical amplification).
4. Continuous-time Legendre Memory Unit (LMU) state-space delay integration without external clocks.
5. Tripartite astrocytic homeostatic gain scaling.
6. Predictive soma with surprise-gated event emission and rectified Born confidence filtering.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from neuron_v2.substrate.phasor import normalise, bind, unbind, bundle, sim, sim_batch


def legendre_matrices(order: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytical continuous-time state-space matrices for the Legendre Memory Unit (Voelker & Eliasmith, 2018).

    Maps continuous input history x(t - tau) for tau in [0, theta] onto orthogonal
    shifted Legendre polynomials of degree < order.
    """
    A = np.zeros((order, order), dtype=np.float64)
    B = np.zeros((order, 1), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            if i < j:
                A[i, j] = -(2 * i + 1) / theta
            else:
                A[i, j] = ((2 * i + 1) / theta) * ((-1) ** (i - j + 1))
        B[i, 0] = ((2 * i + 1) / theta) * ((-1) ** i)
    return A, B


class ContinuousTimeDelay:
    """Continuous-time state-space delay and history projection without external clocks."""

    def __init__(self, order: int = 6, theta: float = 100.0, dt: float = 1.0, n_substeps: int = 10):
        self.order = order
        self.theta = theta
        self.dt = dt
        self.n_substeps = n_substeps
        self.A, self.B = legendre_matrices(order, theta)
        self.h = np.zeros((order, 1), dtype=np.float64)
        # Learnable delay readout weights for arbitrary target delays
        self.W_readout = np.zeros((1, order), dtype=np.float64)

    def step(self, x_scalar: float) -> np.ndarray:
        """Advance the continuous-time state-space via Euler sub-stepping."""
        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            dh = (self.A @ self.h) + (self.B * x_scalar)
            self.h += sub_dt * dh
        return self.h.flatten()

    def reconstruct_delay(self, tau: float) -> float:
        """Analytically reconstruct the delayed input x(t - tau) from the Legendre state."""
        tau = np.clip(tau, 0.0, self.theta)
        # Map delay tau in [0, theta] to Legendre domain z in [-1, 1]
        z = 1.0 - (2.0 * tau / self.theta)
        # Evaluate shifted Legendre polynomials at z
        P = np.zeros(self.order, dtype=np.float64)
        if self.order > 0:
            P[0] = 1.0
        if self.order > 1:
            P[1] = z
        for i in range(1, self.order - 1):
            P[i + 1] = ((2 * i + 1) * z * P[i] - i * P[i - 1]) / (i + 1)

        # Expansion: x(t - tau) = sum_i (-1)^i * P_i(z) * h_i
        recon = 0.0
        for i in range(self.order):
            recon += ((-1) ** i) * P[i] * self.h[i, 0]
        return float(recon)

    def learn_readout(self, target_value: float, lr: float = 0.05) -> float:
        """Autonomously learn linear readout weights for an active delay task without a clock."""
        pred = float((self.W_readout @ self.h)[0, 0])
        err = target_value - pred
        self.W_readout += lr * err * self.h.T
        return pred

    def reset(self):
        self.h.fill(0.0)


class BasalDendriticBranch:
    """Active compartmental dendritic branch.

    Implements:
    - Multi-channel complex phasor receptive fields.
    - Local NMDA/dCaAP non-linear voltage plateaus.
    - Shunting lateral inhibition for autonomous routing/symmetry breaking.
    - Gated Delta-Rule synaptic plasticity.
    """

    def __init__(self, D: int, branch_id: int, n_channels: int, seed: int = 42):
        self.D = D
        self.id = branch_id
        self.n_channels = n_channels
        rng = np.random.RandomState(seed + branch_id * 101)
        # Random unit complex synaptic weights connecting all channels to this branch
        phases = rng.uniform(-np.pi, np.pi, (n_channels, D)).astype(np.float32)
        self.W = np.exp(1j * phases).astype(np.complex64)

        self.voltage = 0.0
        self.plateau_timer = 0
        self.plateau_duration = 5  # ticks
        self.plateau_current = 3.0
        self.threshold = 1.25  # Requires co-activation of 2 channels (each ~1.0) to cross
        self.adaptation = 0.0  # Homeostatic adaptation to ensure symmetry breaking
        self.lateral_inhibition = 0.0

    def forward(
        self,
        channel_inputs: list[np.ndarray],
        g_astro: float,
        competing_inhibition: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Integrate incoming channel hypervectors and compute plateau current injection."""
        branch_current = 0.0
        projected_hvs = []
        for ch_idx, hv in enumerate(channel_inputs):
            # Check if channel is active (non-zero norm)
            if np.linalg.norm(hv) > 1e-4:
                # Cosine alignment with synaptic receptive field
                current = float(np.real(np.vdot(self.W[ch_idx], hv))) / self.D
                branch_current += max(0.0, current)
                projected_hvs.append(self.W[ch_idx] * hv)

        # Net current with shunting lateral inhibition
        net_current = max(0.0, branch_current * g_astro - competing_inhibition)
        self.voltage = 0.5 * self.voltage + 0.5 * net_current

        # Decay adaptation slowly
        self.adaptation *= 0.90
        eff_threshold = self.threshold + self.adaptation

        # NMDA / dCaAP Voltage Plateau Initiation (requires co-activation >= threshold)
        if self.voltage >= eff_threshold and self.plateau_timer == 0:
            self.plateau_timer = self.plateau_duration
            self.adaptation += 0.25  # Local fatigue prevents single-branch monopoly

        if self.plateau_timer > 0:
            inj_current = self.plateau_current
            self.plateau_timer -= 1
        else:
            inj_current = 0.15 * self.voltage

        composite_pred = normalise(np.sum(projected_hvs, axis=0)) if projected_hvs else np.zeros(self.D, dtype=np.complex64)
        return composite_pred, inj_current

    def learn_delta(
        self,
        channel_inputs: list[np.ndarray],
        target_hv: np.ndarray,
        lr: float,
        g_astro: float,
        reinforce_signal: float = 1.0,
    ):
        """Oja-Hebbian synaptic alignment: rotates receptive fields toward co-active inputs."""
        for ch_idx, hv in enumerate(channel_inputs):
            if np.linalg.norm(hv) > 1e-4:
                # Receptive field rotates toward input under positive reinforcement
                dW = lr * g_astro * reinforce_signal * hv
                self.W[ch_idx] = normalise(self.W[ch_idx] + dW)


class ApicalNexus:
    """Apical tuft and nexus for top-down contextual coincidence detection (Larkum, 2013)."""

    def __init__(self, D: int):
        self.D = D
        self.coincidence_threshold = 0.4
        self.amplification_factor = 2.5

    def evaluate_coincidence(
        self,
        top_down_context: Optional[np.ndarray],
        basal_prediction: np.ndarray,
    ) -> float:
        """Detect coincidence between top-down prediction and bottom-up basal drive."""
        if top_down_context is None:
            return 1.0  # Baseline gain
        s = sim(normalise(top_down_context), normalise(basal_prediction))
        if s >= self.coincidence_threshold:
            # Apical calcium plateau triggers burst amplification
            return self.amplification_factor
        return 1.0


class AstrocyteEnvelope:
    """Tripartite glial homeostatic gain regulator."""

    def __init__(self, target_ca: float = 1.0, alpha: float = 0.05, beta: float = 0.02):
        self.calcium = 0.0
        self.target_ca = target_ca
        self.alpha = alpha
        self.beta = beta

    def update(self, total_synaptic_activity: float, somatic_spiked: bool) -> float:
        """Update intracellular calcium and compute adenosine-mediated synaptic scaling."""
        drive = total_synaptic_activity + (2.0 if somatic_spiked else 0.0)
        self.calcium += self.alpha * drive - self.beta * self.calcium
        self.calcium = max(0.0, self.calcium)

        # Multiplicative scaling factor: g_astro decreases as calcium exceeds target
        gain = 1.0 - 0.4 * np.tanh(0.8 * (self.calcium - self.target_ca))
        return float(np.clip(gain, 0.15, 1.25))

    def reset(self):
        self.calcium = 0.0


class PredictiveSoma:
    """Predictive soma with leaky integrate-and-fire dynamics and Born confidence filtering."""

    def __init__(self, v_threshold: float = 1.0, tau_m: float = 15.0, v_reset: float = 0.0):
        self.v = 0.0
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.v_reset = v_reset
        self.refractory_timer = 0
        self.refractory_period = 2

    def step(self, total_injected_current: float, surprise: float) -> tuple[bool, float]:
        """Integrate injected current and emit spike only if threshold and surprise gates pass."""
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.v = self.v_reset
            return False, self.v

        # Leaky integration
        dv = (-self.v + total_injected_current) / (self.tau_m / 10.0)
        self.v += dv

        spiked = False
        # Somatic threshold check
        if self.v >= self.v_threshold:
            # Active inference check: fires if unexpected or highly confident
            if surprise >= 0.10:
                spiked = True
                self.v = self.v_reset
                self.refractory_timer = self.refractory_period
            else:
                # Sub-threshold clamp: expected inputs are quietly assimilated
                self.v = self.v_reset
        return spiked, self.v

    def reset(self):
        self.v = self.v_reset
        self.refractory_timer = 0


class RCPNeuron:
    """The Complete Resonant Compartmental Predictive Neuron.

    Full integration of basal dendritic branches, apical nexus, astrocyte,
    continuous-time delay state space, and predictive soma.
    """

    def __init__(
        self,
        D: int = 256,
        n_branches: int = 2,
        n_channels: int = 4,
        theta_delay: float = 100.0,
        lr: float = 0.15,
        seed: int = 42,
    ):
        self.D = D
        self.lr = lr
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.branches = [
            BasalDendriticBranch(D, b_id, n_channels, seed=seed)
            for b_id in range(n_branches)
        ]
        self.nexus = ApicalNexus(D)
        self.astrocyte = AstrocyteEnvelope()
        self.soma = PredictiveSoma()
        self.delay_engine = ContinuousTimeDelay(order=6, theta=theta_delay)
        self.last_g_astro = 1.0

    def step(
        self,
        channel_inputs: list[np.ndarray],
        top_down_context: Optional[np.ndarray] = None,
        analog_stream: float = 0.0,
    ) -> dict:
        """Execute one complete forward predictive and somatic integration cycle."""
        # 1. Update continuous-time state-space memory
        self.delay_engine.step(analog_stream)

        # 2. Update astrocytic homeostatic gain
        activity = sum(float(np.linalg.norm(inp)) for inp in channel_inputs) / np.sqrt(self.D)
        g_astro = self.astrocyte.update(activity, somatic_spiked=False)
        self.last_g_astro = g_astro

        # 3. Compute competitive lateral branch inhibition
        # Branch with highest local current inhibits other branches (breaking symmetry)
        pre_voltages = [b.voltage for b in self.branches]
        max_v = max(pre_voltages) if pre_voltages else 0.0

        branch_preds = []
        total_injected_current = 0.0

        for b_idx, branch in enumerate(self.branches):
            # Compute inhibition from competing branches
            competing = 0.4 * (max_v - branch.voltage) if branch.voltage < max_v else 0.0
            pred_hv, inj = branch.forward(channel_inputs, g_astro, competing_inhibition=competing)
            branch_preds.append(pred_hv)
            total_injected_current += inj

        # 4. Apical Coincidence Amplification (Larkum Nexus)
        basal_composite = normalise(np.sum(branch_preds, axis=0))
        amp_factor = self.nexus.evaluate_coincidence(top_down_context, basal_composite)
        total_injected_current *= amp_factor

        # 5. Predictive Surprise Computation
        if top_down_context is not None:
            surprise = 1.0 - sim(normalise(top_down_context), basal_composite)
        else:
            surprise = 0.5  # Neutral default

        # 6. Somatic Integration and Firing
        spiked, soma_v = self.soma.step(total_injected_current, surprise=surprise)
        if spiked:
            # Inform astrocyte of somatic firing
            self.astrocyte.update(0.0, somatic_spiked=True)

        return {
            "spiked": spiked,
            "soma_voltage": soma_v,
            "total_injected_current": total_injected_current,
            "g_astro": g_astro,
            "surprise": surprise,
            "branch_voltages": [b.voltage for b in self.branches],
            "branch_preds": branch_preds,
        }

    def learn_autonomous_xor(
        self,
        channel_inputs: list[np.ndarray],
        target_spike: bool,
        spiked: bool,
    ):
        """Autonomous symmetry-breaking learning across branches.

        Uses competitive winner-take-all lateral shunting: only the most active branch
        learns to associate the active input pattern with excitation.
        """
        voltages = [b.voltage for b in self.branches]
        # Winner selection incorporates homeostatic adaptation to break symmetry
        net_activations = [b.voltage - b.adaptation for b in self.branches]
        winner_idx = int(np.argmax(net_activations))

        if target_spike:
            # Positive reinforcement: strengthen winning branch
            boost = 0.4 if spiked else 1.2
            self.branches[winner_idx].learn_delta(
                channel_inputs,
                None,
                lr=self.lr,
                g_astro=self.last_g_astro,
                reinforce_signal=boost,
            )
            self.branches[winner_idx].adaptation += 0.25
        else:
            # Target is SILENT: if it spiked falsely, depress active branches
            if spiked:
                for b_idx in range(self.n_branches):
                    if voltages[b_idx] >= 0.4:
                        self.branches[b_idx].learn_delta(
                            channel_inputs,
                            None,
                            lr=self.lr,
                            g_astro=self.last_g_astro,
                            reinforce_signal=-1.0,
                        )

    def reset(self):
        """Reset internal potential and state variables across all compartments."""
        for branch in self.branches:
            branch.voltage = 0.0
            branch.plateau_timer = 0
        self.astrocyte.reset()
        self.soma.reset()
        self.delay_engine.reset()
