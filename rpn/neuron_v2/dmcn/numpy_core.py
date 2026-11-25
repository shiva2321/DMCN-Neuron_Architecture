"""NumPy reference implementation of the Dendritic Multiplexed Cascade Neuron (DMCN).

Implements:
1. Pillar 1: Strictly causal online divisive & intrinsic excitability homeostasis.
2. Pillar 2: Causal diagonal multi-timescale spine ODE bank (exact ZOH for arbitrary dt).
3. Pillar 3: Surprise-gated dormant branch recruitment, causal smooth non-monotonic
   dendritic conductance, and branch-isolated Gated-Delta + zero-buffer Benna-Fusi cascade.
4. Pillar 4: Single-phase multiplexed apical-basal local plasticity with activity-gated
   Kolen-Pollack feedback alignment and 3-factor eligibility traces for delayed-reward RL.
5. Exact per-step MAC/FLOP and memory byte accounting via count_layer_ops().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .config import AblationConfig, DMCNConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -30.0, 20.0))))


@dataclass
class OpsReport:
    """Exact analytical operation and memory footprint breakdown for one step."""

    forward_macs: int
    forward_flops: int
    learn_macs: int
    learn_flops: int
    total_flops_per_step: int
    parameter_count: int
    state_count: int
    memory_bytes_fp64: int
    memory_bytes_fp32: int


def count_layer_ops(cfg: DMCNConfig) -> OpsReport:
    """Compute exact analytical MACs, FLOPs, parameter count, and memory footprint (sparse top-k)."""
    d_in = cfg.d_in
    d_out = cfg.d_out
    b = cfg.n_branches
    k = 0 if cfg.ablation.ablate_spine_traces else cfg.n_spine_scales
    d_psi = d_in * (1 + k)
    k_act = min(b, max(1, cfg.top_k_branches))

    # Forward MACs (sparse active-branch evaluation using cached W_eff):
    # - Homeostasis: 2 * d_in
    # - Spine ZOH update: 2 * k * d_in
    # - Prototype routing (only when B > 1): b * d_psi
    # - Sparse active-branch projection: k_act * d_out * d_psi
    # - Somatic aggregation: k_act * d_out
    route_macs = (b * d_psi) if b > 1 else 0
    fwd_macs = (2 * d_in) + (2 * k * d_in) + route_macs + (k_act * d_out * d_psi) + (k_act * d_out)
    fwd_extra = (4 * d_in) + (3 * k * d_in) + (4 * b if b > 1 else 0) + (6 * k_act * d_out) + (4 * d_out)
    fwd_flops = 2 * fwd_macs + fwd_extra

    # Local learning MACs (strictly on k_act active branches):
    # - Fast delta outer product & decay: 2 * k_act * d_out * d_psi
    # - Cascade consolidation (when enabled): k_act * d_out * d_psi
    # - Prototype update (when B > 1): k_act * d_psi
    # - Feedback projection + Kolen-Pollack update: 2 * d_in * d_out
    consol_macs = (k_act * d_out * d_psi) if (not cfg.ablation.ablate_cascade and cfg.eta_consol > 0.0) else 0
    proto_macs = (k_act * d_psi) if b > 1 else 0
    prec_macs = (2 * d_psi * d_psi) if (cfg.use_precision_plasticity and not cfg.ablation.ablate_precision) else 0
    lrn_macs = (2 * k_act * d_out * d_psi) + consol_macs + proto_macs + prec_macs + (2 * d_in * d_out)
    lrn_flops = 2 * lrn_macs + (3 * k_act * d_out * d_psi)

    # Parameters: W_slow, W_fast, b_dend, prototypes, theta_soma, B_fb, dend params
    param_count = (2 * b * d_out * d_psi) + (b * d_out) + (b * d_psi if b > 1 else 0) + d_out + (d_in * d_out)
    if cfg.use_nonmonotonic and not cfg.ablation.ablate_nonmonotonic:
        param_count += 3 * b * d_out

    # Dynamic state registers (spine_state, ema_var, ema_act, elig, optional P_prec)
    prec_states = (d_psi * d_psi) if (cfg.use_precision_plasticity and not cfg.ablation.ablate_precision) else 0
    state_count = (k * d_in) + d_in + d_out + (b * d_out * d_psi) + prec_states

    total_floats = param_count + state_count
    return OpsReport(
        forward_macs=int(fwd_macs),
        forward_flops=int(fwd_flops),
        learn_macs=int(lrn_macs),
        learn_flops=int(lrn_flops),
        total_flops_per_step=int(fwd_flops + lrn_flops),
        parameter_count=int(param_count),
        state_count=int(state_count),
        memory_bytes_fp64=int(total_floats * 8),
        memory_bytes_fp32=int(total_floats * 4),
    )


class DMCNLayer:
    """Single-cell or multi-unit Dendritic Multiplexed Cascade Neuron layer (NumPy)."""

    def __init__(self, cfg: DMCNConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.d_in = int(cfg.d_in)
        self.d_out = int(cfg.d_out)
        self.n_branches = int(cfg.n_branches)
        self.k_scales = 0 if cfg.ablation.ablate_spine_traces else int(cfg.n_spine_scales)
        self.d_psi = self.d_in * (1 + self.k_scales)

        # Pillar 1: Strictly causal online RMS divisive & intrinsic homeostasis states
        self.ema_mean = np.zeros(self.d_in, dtype=np.float64)
        self.ema_var = np.ones(self.d_in, dtype=np.float64)
        self.homeo_initialized = False
        self.ema_out_var = np.ones(self.d_out, dtype=np.float64)
        self.out_scale_initialized = False
        self.ema_act = np.zeros(self.d_out, dtype=np.float64)
        self.theta_soma = np.zeros(self.d_out, dtype=np.float64)

        # Pillar 2: Diagonal multi-timescale spine ODE decay rates
        if self.k_scales > 0:
            if self.k_scales == 1:
                taus = np.array([cfg.tau_min], dtype=np.float64)
            else:
                taus = np.geomspace(cfg.tau_min, cfg.tau_max, self.k_scales)
            self.lambdas = 1.0 / taus
        else:
            self.lambdas = np.zeros(0, dtype=np.float64)
        self.spine_state = np.zeros((self.k_scales, self.d_in), dtype=np.float64)

        # Pillar 3: Active dendritic compartments & dual-cascade memory
        scale = cfg.weight_init_scale / np.sqrt(max(1, self.d_psi))
        self.W_slow = self.rng.randn(self.n_branches, self.d_out, self.d_psi) * scale
        self.W_fast = np.zeros((self.n_branches, self.d_out, self.d_psi), dtype=np.float64)
        self.W_eff = self.W_slow.copy()
        self.b_dend = np.zeros((self.n_branches, self.d_out), dtype=np.float64)

        # Smooth non-monotonic conductance parameters per (branch, neuron)
        init_rho_c = float(np.clip(cfg.init_rho, 1e-4, 1.0 - 1e-4))
        self.rho_logit = np.full(
            (self.n_branches, self.d_out),
            np.log(init_rho_c / (1.0 - init_rho_c)),
            dtype=np.float64,
        )
        self.theta_on = np.full((self.n_branches, self.d_out), cfg.init_theta_on, dtype=np.float64)
        self.delta_theta_raw = np.full(
            (self.n_branches, self.d_out),
            np.log(np.expm1(max(1e-3, cfg.init_delta_theta))),
            dtype=np.float64,
        )

        # Surprise-gated dormant branch recruitment state
        self.prototypes = np.zeros((self.n_branches, self.d_psi), dtype=np.float64)
        self.recruited = np.zeros(self.n_branches, dtype=bool)
        self.recruited[0] = True
        self.proto0_initialized = False

        # Local error coherence & recruitment monitors
        self.prev_delta_out: Optional[np.ndarray] = None
        self.ema_err_sq = 0.0
        self.ema_err_coh = 0.0
        self.sustained_recruit = 0
        self.steps_since_recruit = cfg.recruit_cooldown
        self.recruit_events: List[int] = []
        self.step_count = 0

        # Synaptic Tagging & Capture (STC) cascade coherence traces
        self.ema_dW_fast = np.zeros((self.n_branches, self.d_out, self.d_psi), dtype=np.float64)
        self.tag_coherence = np.zeros(self.n_branches, dtype=np.float64)

        # Pillar 4: Apical feedback alignment matrix B_fb (d_in x d_out) & eligibility traces
        fb_scale = cfg.weight_init_scale / np.sqrt(max(1, self.d_out))
        self.B_fb = self.rng.randn(self.d_in, self.d_out) * fb_scale
        self.elig_W = np.zeros((self.n_branches, self.d_out, self.d_psi), dtype=np.float64)

        # Optional local synaptic precision matrix (Sherman-Morrison orthogonal metaplasticity)
        self.P_prec: Optional[np.ndarray] = (
            (1.0 / max(1e-6, cfg.precision_ridge)) * np.eye(self.d_psi, dtype=np.float64)
            if (cfg.use_precision_plasticity and not cfg.ablation.ablate_precision)
            else None
        )

        # Cached forward intermediates for strictly local online plasticity
        self._last_x_norm = np.zeros(self.d_in, dtype=np.float64)
        self._last_psi = np.zeros(self.d_psi, dtype=np.float64)
        self._last_ctx_hat = np.zeros(self.d_psi, dtype=np.float64)
        self._last_p = np.zeros(self.n_branches, dtype=np.float64)
        self._last_p[0] = 1.0
        self._last_u = np.zeros((self.n_branches, self.d_out), dtype=np.float64)
        self._last_dphi = np.ones((self.n_branches, self.d_out), dtype=np.float64)
        self._last_dy = np.ones(self.d_out, dtype=np.float64)
        self._last_y = np.zeros(self.d_out, dtype=np.float64)
        self._last_max_sim = 1.0

    def reset_temporal_state(self) -> None:
        """Reset fast temporal traces (spine ODE bank and RL eligibility) between episodes."""
        self.spine_state.fill(0.0)
        self.elig_W.fill(0.0)
        self.prev_delta_out = None

    def reset_fast_memory(self) -> None:
        """Clear fast episodic weight trace while preserving consolidated W_slow."""
        self.W_fast.fill(0.0)
        self.W_eff = self.W_slow.copy()
        self.ema_dW_fast.fill(0.0)
        self.tag_coherence.fill(0.0)

    def _compute_routing(self, ctx_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute sparse competitive routing weights p over recruited branches."""
        if self.n_branches == 1:
            return np.array([1.0], dtype=np.float64), 1.0

        rec_idx = np.flatnonzero(self.recruited)
        sims = self.prototypes[rec_idx] @ ctx_hat
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

        logits = self.cfg.beta_route * (sims - max_sim)
        exp_l = np.exp(np.clip(logits, -30.0, 0.0))
        probs_rec = exp_l / (float(np.sum(exp_l)) + 1e-12)

        p = np.zeros(self.n_branches, dtype=np.float64)
        p[rec_idx] = probs_rec

        # Optional top-k sparsification over recruited branches
        k_top = self.cfg.top_k_branches
        if 0 < k_top < len(rec_idx):
            keep = np.argsort(p)[::-1][:k_top]
            p_sparse = np.zeros_like(p)
            p_sparse[keep] = p[keep]
            s_p = float(np.sum(p_sparse))
            if s_p > 1e-12:
                p = p_sparse / s_p

        return p, max_sim

    def _dendritic_activation_branch(self, b: int, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate causal dendritic transfer phi(u_b) and derivative dphi/du_b for one branch."""
        if self.cfg.ablation.ablate_nonmonotonic or not self.cfg.use_nonmonotonic:
            if self.cfg.soma_activation == "linear" and not self.cfg.use_nonmonotonic:
                return ub.copy(), np.ones_like(ub)
            t_u = np.tanh(ub)
            return t_u, 1.0 - t_u * t_u

        rho = _sigmoid(self.rho_logit[b])
        theta_off = self.theta_on[b] + _softplus(self.delta_theta_raw[b])
        beta = self.cfg.beta_dend

        s_on = _sigmoid(beta * (ub - self.theta_on[b]))
        s_off = _sigmoid(beta * (ub - theta_off))
        bump = s_on - s_off
        dbump = beta * (s_on * (1.0 - s_on) - s_off * (1.0 - s_off))

        t_u = np.tanh(ub)
        dt_u = 1.0 - t_u * t_u

        phi = (1.0 - rho) * t_u + rho * bump
        dphi = (1.0 - rho) * dt_u + rho * dbump
        return phi, dphi

    def _somatic_activation(self, i_soma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        act = self.cfg.soma_activation
        if act == "linear":
            return i_soma.copy(), np.ones_like(i_soma)
        if act == "sigmoid":
            y = _sigmoid(i_soma)
            return y, y * (1.0 - y)
        y = np.tanh(i_soma)
        return y, 1.0 - y * y

    def step(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Execute one strictly causal online forward step."""
        x_arr = np.asarray(x, dtype=np.float64).reshape(self.d_in)
        self.step_count += 1
        self.steps_since_recruit += 1

        # Pillar 1: Strictly causal RMS divisive homeostasis (normalizes with t-dt power first!)
        if not self.cfg.ablation.ablate_homeostasis:
            if not self.homeo_initialized:
                x_norm = np.clip(x_arr, -8.0, 8.0)
                if float(np.dot(x_arr, x_arr)) > 1e-12:
                    self.ema_mean = x_arr.copy()
                    self.ema_var = np.maximum(np.zeros_like(x_arr) if self.cfg.center_homeostasis else x_arr * x_arr, 1.0)
                    self.homeo_initialized = True
            else:
                if self.cfg.center_homeostasis:
                    x_norm = np.clip(
                        (x_arr - self.ema_mean) / np.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    ah = self.cfg.alpha_homeo
                    diff = x_arr - self.ema_mean
                    self.ema_mean = (1.0 - ah) * self.ema_mean + ah * x_arr
                    self.ema_var = (1.0 - ah) * self.ema_var + ah * (diff * diff)
                else:
                    x_norm = np.clip(
                        x_arr / np.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    ah = self.cfg.alpha_homeo
                    self.ema_var = (1.0 - ah) * self.ema_var + ah * (x_arr * x_arr)
        else:
            x_norm = x_arr

        # Pillar 2: Causal diagonal multi-timescale spine ODE bank (exact ZOH)
        if self.k_scales > 0:
            decay = np.exp(-self.lambdas[:, None] * float(dt))
            self.spine_state = decay * self.spine_state + (1.0 - decay) * x_norm[None, :]
            psi = np.concatenate([x_norm, self.spine_state.reshape(-1)])
        else:
            psi = x_norm

        # Context normalization for branch routing
        psi_norm_l2 = float(np.linalg.norm(psi))
        if psi_norm_l2 > 1e-9:
            ctx_hat = psi / psi_norm_l2
            if not self.proto0_initialized:
                self.prototypes[0] = ctx_hat.copy()
                self.proto0_initialized = True
        else:
            ctx_hat = np.zeros_like(psi)

        p, max_sim = self._compute_routing(ctx_hat)
        psi_proj = ctx_hat if self.cfg.use_unit_key_norm else psi

        # Sparse O(k_top) active-branch projection via cached W_eff (zero full-B allocation!)
        u = self._last_u
        dphi = self._last_dphi

        if self.n_branches == 1:
            u0 = self.W_eff[0] @ psi_proj - self.b_dend[0]
            u[0] = u0
            phi0, dphi0 = self._dendritic_activation_branch(0, u0)
            dphi[0] = dphi0
            i_soma = phi0
        elif self.cfg.top_k_branches == 1:
            b = int(np.argmax(p))
            ub = self.W_eff[b] @ psi_proj - self.b_dend[b]
            u[b] = ub
            phib, dphib = self._dendritic_activation_branch(b, ub)
            dphi[b] = dphib
            i_soma = phib if p[b] == 1.0 else (p[b] * phib)
        else:
            active_idx = np.flatnonzero(p > 1e-6)
            i_soma = np.zeros(self.d_out, dtype=np.float64)
            for b_i in active_idx:
                b = int(b_i)
                ub = self.W_eff[b] @ psi_proj - self.b_dend[b]
                u[b] = ub
                phib, dphib = self._dendritic_activation_branch(b, ub)
                dphi[b] = dphib
                i_soma += p[b] * phib

        if not self.cfg.ablation.ablate_homeostasis:
            i_soma = i_soma - self.theta_soma
        y_int, dy = self._somatic_activation(i_soma)

        # Causal output-scale homeostasis (projects scale-invariant y_int to physical target scale)
        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis and self.out_scale_initialized:
            out_scale = np.sqrt(self.ema_out_var + self.cfg.eps_homeo)
            y = y_int * out_scale
        else:
            y = y_int

        # Optional k-Winner-Take-All lateral inhibition (orthogonalizes overlapping features)
        if 0 < self.cfg.k_wta < self.d_out:
            top_idx = np.argpartition(np.abs(y), -self.cfg.k_wta)[-self.cfg.k_wta :]
            mask = np.zeros(self.d_out, dtype=np.float64)
            mask[top_idx] = 1.0
            y = y * mask
            dy = dy * mask

        # Intrinsic excitability homeostasis
        if not self.cfg.ablation.ablate_homeostasis and self.cfg.eta_theta > 0.0:
            aa = self.cfg.alpha_act
            self.ema_act = (1.0 - aa) * self.ema_act + aa * y_int
            self.theta_soma += self.cfg.eta_theta * (self.ema_act - self.cfg.target_activity)

        # Cache local cellular state for single-phase local plasticity
        self._last_x_norm = x_norm
        self._last_psi = psi_proj
        self._last_ctx_hat = ctx_hat
        self._last_p = p
        self._last_u = u
        self._last_dphi = dphi
        self._last_dy = dy
        self._last_y = y
        self._last_max_sim = max_sim
        return y.copy()

    def _maybe_recruit_branch(self, delta_out: np.ndarray) -> bool:
        """Surprise-gated dormant branch recruitment (zero timers, noise-immune)."""
        if self.n_branches <= 1 or bool(np.all(self.recruited)):
            return False

        err_sq = float(np.mean(delta_out * delta_out))
        if self.prev_delta_out is not None:
            n_curr = float(np.linalg.norm(delta_out))
            n_prev = float(np.linalg.norm(self.prev_delta_out))
            if n_curr > 1e-8 and n_prev > 1e-8:
                coh = float(np.dot(delta_out, self.prev_delta_out) / (n_curr * n_prev))
            else:
                coh = 0.0
        else:
            coh = 0.0
        self.prev_delta_out = delta_out.copy()

        ae = self.cfg.alpha_err_ema
        self.ema_err_sq = (1.0 - ae) * self.ema_err_sq + ae * err_sq
        self.ema_err_coh = (1.0 - ae) * self.ema_err_coh + ae * coh

        # Never recruit on zero input or during refractory cooldown
        if float(np.linalg.norm(self._last_ctx_hat)) < 1e-6:
            self.sustained_recruit = 0
            return False
        if self.steps_since_recruit < self.cfg.recruit_cooldown:
            return False

        novel_trigger = (self._last_max_sim < self.cfg.theta_novel) and (err_sq > self.cfg.theta_err_novel)
        coherent_trigger = (self.ema_err_sq > self.cfg.theta_err) and (self.ema_err_coh > self.cfg.theta_coh)

        if novel_trigger or coherent_trigger:
            self.sustained_recruit += 1
            if self.sustained_recruit >= self.cfg.recruit_sustained:
                free_idx = int(np.flatnonzero(~self.recruited)[0])
                self.recruited[free_idx] = True
                self.prototypes[free_idx] = self._last_ctx_hat.copy()
                self.sustained_recruit = 0
                self.steps_since_recruit = 0
                self.recruit_events.append(self.step_count)
                # Re-route immediately to the newly recruited compartment
                self._last_p, self._last_max_sim = self._compute_routing(self._last_ctx_hat)
                return True
        else:
            self.sustained_recruit = max(0, self.sustained_recruit - 1)
        return False

    def learn_local(self, delta_out: np.ndarray, mod: float = 1.0) -> np.ndarray:
        """Single-phase sparse O(k_top) local dendritic plasticity + cascade consolidation."""
        delta_arr = np.asarray(delta_out, dtype=np.float64).reshape(self.d_out)

        # Causal output-scale homeostasis: normalize teaching residual & update target RMS scale
        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis:
            obs_target = self._last_y + delta_arr
            if not self.out_scale_initialized:
                self.ema_out_var = np.maximum(obs_target * obs_target, 1e-4)
                self.out_scale_initialized = True
                delta_eff = delta_arr / np.sqrt(self.ema_out_var + self.cfg.eps_homeo)
            else:
                out_scale = np.sqrt(self.ema_out_var + self.cfg.eps_homeo)
                delta_eff = np.clip(delta_arr / out_scale, -8.0, 8.0)
                ao = self.cfg.alpha_out_scale
                self.ema_out_var = (1.0 - ao) * self.ema_out_var + ao * (obs_target * obs_target)
        else:
            delta_eff = delta_arr

        self._maybe_recruit_branch(delta_eff)

        p = self._last_p
        psi = self._last_psi
        ctx_hat = self._last_ctx_hat
        delta_soma = delta_eff * self._last_dy
        if self.n_branches == 1:
            active_idx = (0,)
        elif self.cfg.top_k_branches == 1:
            active_idx = (int(np.argmax(p)),)
        else:
            active_idx = tuple(int(i) for i in np.flatnonzero(p > 1e-6))

        if self.P_prec is not None:
            p_psi = self.P_prec @ psi
            denom_p = 1.0 + float(np.dot(psi, p_psi))
            psi_norm = p_psi / denom_p
            self.P_prec -= np.outer(psi_norm, p_psi)
        else:
            psi_sq = float(np.dot(psi, psi))
            denom = (psi_sq + 1e-6) if self.cfg.use_unit_key_norm else (1.0 + psi_sq)
            psi_norm = psi / denom
        ctx_norm = float(np.sqrt(np.dot(ctx_hat, ctx_hat))) if self.n_branches > 1 else 0.0

        for b in active_idx:
            pb = float(p[b])

            # Competitive prototype update on active branch + Boids separation from other branches
            if self.n_branches > 1 and ctx_norm > 1e-8:
                updated = self.prototypes[b] + self.cfg.eta_proto * pb * (ctx_hat - self.prototypes[b])
                if self.cfg.boids_repulsion > 0.0:
                    for other_b in np.flatnonzero(self.recruited):
                        ob = int(other_b)
                        if ob != b:
                            sim_bo = float(np.dot(self.prototypes[b], self.prototypes[ob]))
                            if sim_bo > 0.2:
                                updated -= self.cfg.boids_repulsion * sim_bo * self.prototypes[ob]
                n_u = float(np.sqrt(np.dot(updated, updated)))
                if n_u > 1e-8:
                    self.prototypes[b] = updated / n_u

            delta_b = (pb * delta_soma) * self._last_dphi[b]

            if self.cfg.ablation.ablate_fast_delta:
                self.W_slow[b] += (mod * self.cfg.eta_slow) * np.outer(delta_b, psi_norm)
                self.W_eff[b] = self.W_slow[b]
            else:
                dW_b = np.outer((mod * self.cfg.eta_fast) * delta_b, psi_norm)
                wf_b = self.W_fast[b]
                if self.cfg.gamma_forget > 0.0:
                    wf_b *= (1.0 - pb * self.cfg.gamma_forget)
                wf_b += dW_b

                if not self.cfg.ablation.ablate_cascade and self.cfg.eta_consol > 0.0:
                    at = self.cfg.alpha_tag
                    ema_b = self.ema_dW_fast[b]
                    dw_flat = dW_b.ravel()
                    ema_flat = ema_b.ravel()
                    n_dw2 = float(np.dot(dw_flat, dw_flat))
                    n_ema2 = float(np.dot(ema_flat, ema_flat))
                    if n_dw2 > 1e-18 and n_ema2 > 1e-18:
                        cos_tag = max(0.0, float(np.dot(dw_flat, ema_flat) / np.sqrt(n_dw2 * n_ema2)))
                    else:
                        cos_tag = 0.0
                    ema_b *= (1.0 - at)
                    ema_b += at * dW_b
                    tc = (1.0 - at) * self.tag_coherence[b] + at * cos_tag
                    self.tag_coherence[b] = tc
                    c_rate = pb * self.cfg.eta_consol * tc
                    self.W_slow[b] *= (1.0 - c_rate * self.cfg.lambda_eq)
                    self.W_slow[b] += c_rate * wf_b

                np.multiply(wf_b, self.cfg.kappa_fast, out=self.W_eff[b])
                self.W_eff[b] += self.W_slow[b]

            if self.P_prec is None:
                self.b_dend[b] -= (mod * self.cfg.eta_fast * 0.25) * delta_b

            if self.cfg.use_nonmonotonic and not self.cfg.ablation.ablate_nonmonotonic:
                ub = self._last_u[b]
                rho = _sigmoid(self.rho_logit[b])
                theta_off = self.theta_on[b] + _softplus(self.delta_theta_raw[b])
                beta = self.cfg.beta_dend
                s_on = _sigmoid(beta * (ub - self.theta_on[b]))
                s_off = _sigmoid(beta * (ub - theta_off))
                bump = s_on - s_off
                t_u = np.tanh(ub)
                err_b = pb * delta_soma
                d_rho = err_b * (bump - t_u) * rho * (1.0 - rho)
                d_ton = err_b * rho * (-beta * (s_on * (1.0 - s_on) - s_off * (1.0 - s_off)))
                self.rho_logit[b] = np.clip(self.rho_logit[b] + mod * self.cfg.eta_dend_params * d_rho, -4.0, 4.0)
                self.theta_on[b] = np.clip(self.theta_on[b] + mod * self.cfg.eta_dend_params * d_ton, -2.0, 3.0)

        # Pillar 4: Compute apical feedback for previous layer & activity-gated Kolen-Pollack update
        apical_prev = self.B_fb @ delta_soma
        if self.cfg.ablation.shuffle_apical:
            apical_prev = self.rng.permutation(apical_prev)

        if not self.cfg.ablation.freeze_random_feedback:
            x_n = self._last_x_norm
            x_sq = float(np.dot(x_n, x_n))
            denom_fb = 1.0 + x_sq
            decay_fb = min(0.5, self.cfg.lambda_fb * x_sq / denom_fb)
            self.B_fb += (mod * self.cfg.eta_fb / denom_fb) * np.outer(x_n, delta_soma) - decay_fb * self.B_fb

        return apical_prev

    def accumulate_eligibility(self, action_grad: np.ndarray) -> None:
        """Accumulate local 3-factor synaptic eligibility trace for delayed-reward RL."""
        g_arr = np.asarray(action_grad, dtype=np.float64).reshape(self.d_out)
        p = self._last_p
        psi = self._last_psi
        denom = 1.0 + float(np.dot(psi, psi))
        delta_dend = p[:, None] * (g_arr * self._last_dy)[None, :] * self._last_dphi
        step_elig = np.einsum("bo,i->boi", delta_dend, psi / denom)

        if self.cfg.ablation.ablate_eligibility:
            self.elig_W = step_elig
        else:
            self.elig_W = self.cfg.gamma_elig * self.elig_W + step_elig

    def apply_rpe(self, rpe: float) -> None:
        """Convert accumulated eligibility trace into local fast-weight plasticity via scalar RPE."""
        if self.cfg.ablation.ablate_fast_delta:
            self.W_slow += self.cfg.eta_rl * float(rpe) * self.elig_W
            self.W_eff = self.W_slow.copy()
        else:
            if self.cfg.gamma_forget > 0.0:
                self.W_fast *= (1.0 - self.cfg.gamma_forget)
            self.W_fast += self.cfg.eta_rl * float(rpe) * self.elig_W
            # Gentle consolidation of reinforced policies into slow scaffold
            if not self.cfg.ablation.ablate_cascade and self.cfg.eta_consol > 0.0:
                self.W_slow += self.cfg.eta_consol * (self.W_fast - self.cfg.lambda_eq * self.W_slow)
            self.W_eff = self.W_slow + self.cfg.kappa_fast * self.W_fast

    def ops_report(self) -> OpsReport:
        return count_layer_ops(self.cfg)


class DMCNNetwork:
    """Multi-layer gradient-free network of DMCNLayer modules with apical-basal plasticity."""

    def __init__(self, layer_configs: Sequence[DMCNConfig]) -> None:
        self.layers: List[DMCNLayer] = [DMCNLayer(cfg) for cfg in layer_configs]

    def reset_temporal_state(self) -> None:
        for layer in self.layers:
            layer.reset_temporal_state()

    def reset_fast_memory(self) -> None:
        for layer in self.layers:
            layer.reset_fast_memory()

    def step(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = layer.step(h, dt=dt)
        return h

    def learn_local(self, target: np.ndarray, mod: float = 1.0) -> Tuple[np.ndarray, float]:
        """Single-phase top-down apical-basal local update across all layers (zero backprop)."""
        y_hat = self.layers[-1]._last_y
        err = np.asarray(target, dtype=np.float64).reshape(y_hat.shape) - y_hat
        mse = float(np.mean(err * err))

        delta = err
        for layer in reversed(self.layers):
            delta = layer.learn_local(delta, mod=mod)
        return y_hat.copy(), mse

    def total_ops_report(self) -> Dict[str, int]:
        reports = [layer.ops_report() for layer in self.layers]
        return {
            "forward_macs": sum(r.forward_macs for r in reports),
            "forward_flops": sum(r.forward_flops for r in reports),
            "learn_macs": sum(r.learn_macs for r in reports),
            "learn_flops": sum(r.learn_flops for r in reports),
            "total_flops_per_step": sum(r.total_flops_per_step for r in reports),
            "parameter_count": sum(r.parameter_count for r in reports),
            "state_count": sum(r.state_count for r in reports),
            "memory_bytes_fp64": sum(r.memory_bytes_fp64 for r in reports),
            "memory_bytes_fp32": sum(r.memory_bytes_fp32 for r in reports),
        }
