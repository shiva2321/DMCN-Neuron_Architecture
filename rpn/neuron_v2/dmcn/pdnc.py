"""Prospective Dendritic Natural-Credit (PDNC) learning algorithm for DMCN.

Unifies supervised feature discovery, replay-free continual learning, and delayed-reward
multi-layer RL under one local, strictly causal, gradient-free (.backward()-free) and
weight-transport-free dendritic plasticity framework:

  1. Prospective Apical Activity Settlement (inspired by Song et al., Nature Neurosci. 2024
     & MuPC prospective relaxation): top-down apical feedback nudges hidden compartments
     toward reducing downstream error *before* readout consolidation, updating each upper
     layer on settled post-plasticity representations so layers cooperate instead of
     overshooting.
  2. Symmetric Reciprocal Kolen-Pollack Alignment (zero weight transport): forward basal
     synapses W and reciprocal apical synapses B_fb are initialized with independent RNG
     streams and updated strictly from the same two local endpoint signals (pre, post)
     with matched homeostatic decay, driving B_fb -> W^T without ever reading W.
  3. Dual-Regime Local Synaptic Precision Conditioning:
     - 'diagonal' (BONG / spine-RMS second-moment conditioning): prevents dense features
       from starving sparse informative inputs (critical for raw-pixel RL and high-noise
       feature discovery).
     - 'woodbury' with attenuation alpha_null: when alpha_null == 1.0, performs exact
       order-invariant RLS/Woodbury readout updates; when 0 < alpha_null < 1.0 on hidden
       layers, acts as a continuous, SVD-free, boundary-free online null-space projector
       that lets hidden dendrites acquire new task features while locking previously
       visited input subspaces against drift.
  4. Multi-Layer Multiplexed 3-Factor TD(lambda) Eligibility: propagates Actor + Critic
     local sensitivity through reciprocal apical synapses into hidden dendritic eligibility
     traces modulated by scalar TD error (RPE) in single-pass causal time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -25.0, 0.0))
    return e / (float(np.sum(e)) + 1e-12)


@dataclass
class PDNCConfig:
    """Configuration for a single PDNCLayer (NumPy and PyTorch/CUDA backends)."""

    d_in: int = 8
    d_out: int = 4
    activation: str = "tanh"  # "tanh", "relu", "silu", "linear"
    k_wta: int = 0

    # Optional causal transient spine bank [x_t, scale * (x_t - spine_ema)]
    use_transient_spine: bool = False
    tau_spine: float = 2.0
    transient_scale: float = 1.5

    # Local synaptic precision geometry:
    # - "diagonal": BONG/AdamW-style 1st-moment momentum + 2nd-moment diagonal Fisher + decoupled wd
    # - "subspace": diagonal natural step projected through online Woodbury null-space P_prec
    # - "woodbury": exact or damped Sherman-Morrison/Woodbury RLS update
    # - "none": plain normalized local update
    precision_mode: str = "diagonal"
    precision_ridge: float = 1.0
    alpha_null: float = 1.0
    scale_null_by_batch: bool = False
    beta_mom: float = 0.90
    beta_rms: float = 0.99
    eps_prec: float = 1e-6
    weight_decay: float = 0.01

    # Plasticity & symmetric reciprocal Kolen-Pollack rates
    eta: float = 0.05
    eta_fb: Optional[float] = None
    lambda_kp: float = 0.01
    prospective_beta: float = 0.5
    gamma_elig: float = 0.85

    # Independent initialization scales (strictly disjoint RNG streams -> zero weight transport)
    weight_init_scale: float = 1.0
    bias_init_scale: float = 0.0
    fb_init_scale: float = 0.25

    # Optional 2D local receptive-field wiring mask on W: shape (d_out, d_ext)
    receptive_field_mask: Optional[np.ndarray] = None

    # Explicit ablation switches for causal falsification
    freeze_weights: bool = False
    freeze_random_feedback: bool = False
    shuffle_apical: bool = False
    ablate_eligibility: bool = False
    seed: int = 42


def make_patch_receptive_field_mask(
    height: int,
    width: int,
    patch_size: int = 4,
    stride: int = 2,
    compartments_per_patch: int = 3,
    include_transient: bool = True,
    include_bias: bool = True,
) -> np.ndarray:
    """Construct sparse 2D local receptive-field connectivity mask (d_out, d_ext) without weight sharing."""
    patches = []
    for r in range(0, height - patch_size + 1, stride):
        for c in range(0, width - patch_size + 1, stride):
            win = np.zeros((height, width), dtype=np.float64)
            win[r : r + patch_size, c : c + patch_size] = 1.0
            patches.append(win.reshape(height * width))
    base = np.repeat(np.stack(patches, axis=0), compartments_per_patch, axis=0)
    parts = [base, base] if include_transient else [base]
    if include_bias:
        parts.append(np.ones((base.shape[0], 1), dtype=np.float64))
    return np.concatenate(parts, axis=1)


class PDNCLayer:
    """NumPy implementation of a Prospective Dendritic Natural-Credit (PDNC) layer."""

    def __init__(self, cfg: PDNCConfig) -> None:
        self.cfg = cfg
        self.d_in = int(cfg.d_in)
        self.d_out = int(cfg.d_out)
        self.d_psi = (2 * self.d_in) if cfg.use_transient_spine else self.d_in
        self.d_ext = self.d_psi + 1  # includes constant dendritic bias spine 1.0

        # Disjoint RNG streams guarantee zero weight transport at initialization
        self.rng_w = np.random.RandomState(cfg.seed)
        self.rng_b = np.random.RandomState(cfg.seed + 7919)
        self.rng_shuf = np.random.RandomState(cfg.seed + 15485)

        w_scale = cfg.weight_init_scale / np.sqrt(max(1, self.d_ext))
        self.W = self.rng_w.randn(self.d_out, self.d_ext) * w_scale
        if cfg.bias_init_scale > 0.0:
            self.W[:, -1] = self.rng_w.randn(self.d_out) * cfg.bias_init_scale
        elif cfg.weight_init_scale > 0.0:
            self.W[:, -1] = 0.0

        self.rf_mask: Optional[np.ndarray] = None
        if cfg.receptive_field_mask is not None:
            m_arr = np.asarray(cfg.receptive_field_mask, dtype=np.float64)
            assert m_arr.shape == (self.d_out, self.d_ext), f"mask shape {m_arr.shape} != {(self.d_out, self.d_ext)}"
            self.rf_mask = m_arr
            self.W *= self.rf_mask

        fb_scale = cfg.fb_init_scale / np.sqrt(max(1, self.d_out))
        self.B_fb = self.rng_b.randn(self.d_in, self.d_out) * fb_scale

        # Causal transient spine state
        self.decay_spine = float(np.exp(-1.0 / max(1e-6, cfg.tau_spine)))
        self.spine_ema = np.zeros(self.d_in, dtype=np.float64)
        self.spine_init = False

        # Precision & momentum conditioning state
        if cfg.precision_mode == "woodbury":
            self.P_prec: Optional[np.ndarray] = (1.0 / max(1e-6, cfg.precision_ridge)) * np.eye(
                self.d_ext, dtype=np.float64
            )
            self.m_w: Optional[np.ndarray] = None
            self.v_rms: Optional[np.ndarray] = None
            self.m_b: Optional[np.ndarray] = None
            self.v_fb_rms: Optional[np.ndarray] = None
        elif cfg.precision_mode in ("diagonal", "subspace"):
            self.P_prec = (
                np.eye(self.d_ext, dtype=np.float64) if cfg.precision_mode == "subspace" else None
            )
            self.m_w = np.zeros((self.d_out, self.d_ext), dtype=np.float64)
            self.v_rms = np.zeros((self.d_out, self.d_ext), dtype=np.float64)
            self.m_b = np.zeros((self.d_in, self.d_out), dtype=np.float64)
            self.v_fb_rms = np.zeros((self.d_in, self.d_out), dtype=np.float64)
        else:
            self.P_prec = None
            self.m_w = None
            self.v_rms = None
            self.m_b = None
            self.v_fb_rms = None

        self.rms_steps = 0

        # 3-factor eligibility traces (forward basal + reciprocal apical) & trace accumulators
        self.elig_W = np.zeros((self.d_out, self.d_ext), dtype=np.float64)
        self.elig_B = np.zeros((self.d_in, self.d_out), dtype=np.float64)
        self.acc_grad_W = np.zeros((self.d_out, self.d_ext), dtype=np.float64)
        self.acc_grad_B = np.zeros((self.d_in, self.d_out), dtype=np.float64)
        self.acc_steps = 0

        # Cached forward states for local plasticity & prospective recomputation
        self._last_psi_ext: np.ndarray = np.zeros((1, self.d_ext), dtype=np.float64)
        self._last_dphi: np.ndarray = np.ones((1, self.d_out), dtype=np.float64)
        self._last_h: np.ndarray = np.zeros((1, self.d_out), dtype=np.float64)
        self._was_1d: bool = True

    def reset_temporal_state(self) -> None:
        """Clear fast episode spine and 3-factor eligibility traces while keeping learned synapses."""
        self.spine_ema.fill(0.0)
        self.spine_init = False
        self.elig_W.fill(0.0)
        self.elig_B.fill(0.0)
        self.acc_grad_W.fill(0.0)
        self.acc_grad_B.fill(0.0)
        self.acc_steps = 0

    def _activate(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        act = self.cfg.activation
        if act == "linear":
            h = u.copy()
            dphi = np.ones_like(u)
        elif act == "relu":
            h = np.maximum(0.0, u)
            dphi = (u > 0.0).astype(np.float64)
        elif act == "silu":
            u_c = np.clip(u, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-u_c))
            h = u * sig
            dphi = sig + u * sig * (1.0 - sig)
        else:
            h = np.tanh(u)
            dphi = 1.0 - h * h

        if 0 < self.cfg.k_wta < self.d_out:
            mask = np.zeros_like(h)
            top_idx = np.argpartition(np.abs(h), -self.cfg.k_wta, axis=1)[:, -self.cfg.k_wta :]
            rows = np.arange(h.shape[0])[:, None]
            mask[rows, top_idx] = 1.0
            h = h * mask
            dphi = dphi * mask
        return h, dphi

    def _compute_from_psi_ext(self, psi_ext: np.ndarray) -> np.ndarray:
        u = psi_ext @ self.W.T
        h, dphi = self._activate(u)
        self._last_psi_ext = psi_ext
        self._last_dphi = dphi
        self._last_h = h
        return h[0].copy() if self._was_1d else h.copy()

    def recompute_post_plasticity(self) -> np.ndarray:
        """Re-evaluate somatic output using already-cached basal spine input after a local weight update."""
        return self._compute_from_psi_ext(self._last_psi_ext)

    def step(self, x: np.ndarray, dt: float = 1.0, stateless_eval: bool = False) -> np.ndarray:
        """Strictly causal forward pass for a single vector (d_in,) or batch (B, d_in)."""
        arr = np.asarray(x, dtype=np.float64)
        self._was_1d = arr.ndim == 1
        x_mat = arr.reshape(1, self.d_in) if self._was_1d else arr.reshape(-1, self.d_in)
        bsz = x_mat.shape[0]

        if self.cfg.use_transient_spine:
            if bsz == 1:
                x0 = x_mat[0]
                if not self.spine_init:
                    if not stateless_eval:
                        self.spine_ema = x0.copy()
                        self.spine_init = True
                    tr = np.zeros_like(x_mat)
                else:
                    tr = ((x0 - self.spine_ema) * self.cfg.transient_scale).reshape(1, self.d_in)
                    if not stateless_eval:
                        decay = float(np.exp(-float(dt) / max(1e-6, self.cfg.tau_spine)))
                        self.spine_ema = decay * self.spine_ema + (1.0 - decay) * x0
                psi = np.concatenate([x_mat, tr], axis=1)
            else:
                psi = np.concatenate([x_mat, np.zeros_like(x_mat)], axis=1)
        else:
            psi = x_mat

        ones = np.ones((bsz, 1), dtype=np.float64)
        psi_ext = np.concatenate([psi, ones], axis=1)
        return self._compute_from_psi_ext(psi_ext)

    def project_apical(self, delta_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local somatic error delta_soma and project apical signal to layer l-1."""
        d_arr = np.asarray(delta_out, dtype=np.float64)
        d_mat = d_arr.reshape(1, self.d_out) if d_arr.ndim == 1 else d_arr.reshape(-1, self.d_out)
        delta_soma = d_mat * self._last_dphi
        apical = delta_soma @ self.B_fb.T
        if self.cfg.shuffle_apical:
            perm = self.rng_shuf.permutation(self.d_in)
            apical = apical[:, perm]
        return delta_soma, (apical[0] if self._was_1d else apical)

    def _update_subspace_projector(self, psi_ext: np.ndarray) -> None:
        """Advance online SVD-free Woodbury null-space projector P_prec toward orthogonal complement."""
        if self.P_prec is None:
            return
        bsz = float(psi_ext.shape[0])
        scale_null = np.sqrt(self.cfg.alpha_null / (bsz if self.cfg.scale_null_by_batch else 1.0))
        psi_null = scale_null * psi_ext
        k_mat = self.P_prec @ psi_null.T
        s_mat = np.eye(psi_ext.shape[0], dtype=np.float64) + psi_null @ k_mat
        g_null = np.linalg.solve(s_mat, k_mat.T).T
        p_new = self.P_prec - g_null @ k_mat.T
        self.P_prec = 0.5 * (p_new + p_new.T)

    def update_synapses(self, delta_soma: np.ndarray, mod: float = 1.0) -> None:
        """Apply local precision-conditioned basal plasticity + symmetric Kolen-Pollack apical update."""
        if self.cfg.freeze_weights:
            return
        ds = delta_soma.reshape(-1, self.d_out)
        psi_ext = self._last_psi_ext
        bsz = float(ds.shape[0])
        eta_w = float(mod) * self.cfg.eta
        eta_b = float(mod) * (self.cfg.eta if self.cfg.eta_fb is None else self.cfg.eta_fb)

        if self.cfg.precision_mode == "woodbury" and self.P_prec is not None:
            scale_null = np.sqrt(self.cfg.alpha_null / (bsz if self.cfg.scale_null_by_batch else 1.0))
            psi_null = scale_null * psi_ext
            k_mat = self.P_prec @ psi_null.T
            s_mat = np.eye(ds.shape[0], dtype=np.float64) + psi_null @ k_mat
            g_null = np.linalg.solve(s_mat, k_mat.T).T
            p_new = self.P_prec - g_null @ k_mat.T
            self.P_prec = 0.5 * (p_new + p_new.T)

            if self.cfg.alpha_null == 1.0 and not self.cfg.scale_null_by_batch:
                step_w = eta_w * (ds.T @ g_null.T)
                step_b = eta_b * (g_null[: self.d_in, :] @ ds)
            else:
                psi_proj = psi_ext @ self.P_prec
                step_w = (eta_w / bsz) * (ds.T @ psi_proj)
                step_b = (eta_b / bsz) * (psi_proj[:, : self.d_in].T @ ds)

            self.W += step_w
            if not self.cfg.freeze_random_feedback:
                self.B_fb += step_b
            return

        grad_w = (ds.T @ psi_ext) / bsz
        grad_b = (psi_ext[:, : self.d_in].T @ ds) / bsz

        if self.m_w is not None and self.v_rms is not None and self.m_b is not None and self.v_fb_rms is not None:
            self.rms_steps += 1
            b1 = self.cfg.beta_mom
            b2 = self.cfg.beta_rms
            bc1 = 1.0 - (b1 ** self.rms_steps)
            bc2 = 1.0 - (b2 ** self.rms_steps)

            self.m_w = b1 * self.m_w + (1.0 - b1) * grad_w
            self.v_rms = b2 * self.v_rms + (1.0 - b2) * (grad_w * grad_w)
            step_w = eta_w * ((self.m_w / bc1) / (np.sqrt(self.v_rms / bc2) + self.cfg.eps_prec))

            self.m_b = b1 * self.m_b + (1.0 - b1) * grad_b
            self.v_fb_rms = b2 * self.v_fb_rms + (1.0 - b2) * (grad_b * grad_b)
            step_b = eta_b * ((self.m_b / bc1) / (np.sqrt(self.v_fb_rms / bc2) + self.cfg.eps_prec))
        else:
            psi_sq = float(np.mean(np.sum(psi_ext * psi_ext, axis=1)))
            step_w = (eta_w / (1.0 + psi_sq)) * grad_w
            step_b = (eta_b / (1.0 + psi_sq)) * grad_b

        if self.cfg.precision_mode == "subspace" and self.P_prec is not None:
            # Project weight update through current P_null BEFORE absorbing current subspace into P_prec
            step_w = step_w @ self.P_prec
            step_b = self.P_prec[: self.d_in, : self.d_in] @ step_b
            self._update_subspace_projector(psi_ext)

        wd_w = min(0.2, eta_w * (self.cfg.weight_decay + self.cfg.lambda_kp))
        if wd_w > 0.0:
            self.W[:, : self.d_psi] *= 1.0 - wd_w
        self.W += step_w
        if self.rf_mask is not None:
            self.W *= self.rf_mask

        if not self.cfg.freeze_random_feedback:
            wd_b = min(0.2, eta_b * self.cfg.lambda_kp)
            if wd_b > 0.0:
                self.B_fb *= 1.0 - wd_b
            self.B_fb += step_b

    def accumulate_eligibility(self, post_sensitivity: np.ndarray) -> np.ndarray:
        """Accumulate local 3-factor eligibility (basal + reciprocal apical) and return apical signal to l-1."""
        s_arr = np.asarray(post_sensitivity, dtype=np.float64).reshape(1, self.d_out)
        delta_soma = s_arr * self._last_dphi  # (1, d_out)
        psi_ext = self._last_psi_ext  # (1, d_ext)

        apical = (delta_soma @ self.B_fb.T)[0]
        if self.cfg.shuffle_apical:
            apical = apical[self.rng_shuf.permutation(self.d_in)]

        step_ew = delta_soma.T @ psi_ext
        step_eb = psi_ext[:, : self.d_in].T @ delta_soma

        if self.cfg.ablate_eligibility:
            self.elig_W = step_ew
            self.elig_B = step_eb
        else:
            g = self.cfg.gamma_elig
            self.elig_W = g * self.elig_W + step_ew
            self.elig_B = g * self.elig_B + step_eb

        return apical

    def _apply_local_gradients(self, grad_w: np.ndarray, grad_b: np.ndarray, eta_w: float, eta_b: float) -> None:
        """Apply precision-conditioned natural update + symmetric Kolen-Pollack update from local gradients."""
        if self.m_w is not None and self.v_rms is not None and self.m_b is not None and self.v_fb_rms is not None:
            self.rms_steps += 1
            b1 = self.cfg.beta_mom
            b2 = self.cfg.beta_rms
            bc1 = 1.0 - (b1 ** self.rms_steps)
            bc2 = 1.0 - (b2 ** self.rms_steps)

            self.m_w = b1 * self.m_w + (1.0 - b1) * grad_w
            self.v_rms = b2 * self.v_rms + (1.0 - b2) * (grad_w * grad_w)
            step_w = eta_w * ((self.m_w / bc1) / (np.sqrt(self.v_rms / bc2) + self.cfg.eps_prec))

            self.m_b = b1 * self.m_b + (1.0 - b1) * grad_b
            self.v_fb_rms = b2 * self.v_fb_rms + (1.0 - b2) * (grad_b * grad_b)
            step_b = eta_b * ((self.m_b / bc1) / (np.sqrt(self.v_fb_rms / bc2) + self.cfg.eps_prec))
        else:
            norm_scale = 1.0 / (1.0 + float(np.mean(grad_w * grad_w)) * self.d_ext)
            step_w = (eta_w * norm_scale) * grad_w
            step_b = (eta_b * norm_scale) * grad_b

        if self.cfg.precision_mode == "subspace" and self.P_prec is not None:
            step_w = step_w @ self.P_prec
            step_b = self.P_prec[: self.d_in, : self.d_in] @ step_b

        wd_w = min(0.2, eta_w * (self.cfg.weight_decay + self.cfg.lambda_kp))
        if wd_w > 0.0:
            self.W[:, : self.d_psi] *= 1.0 - wd_w
        self.W += step_w
        if self.rf_mask is not None:
            self.W *= self.rf_mask

        if not self.cfg.freeze_random_feedback:
            wd_b = min(0.2, eta_b * self.cfg.lambda_kp)
            if wd_b > 0.0:
                self.B_fb *= 1.0 - wd_b
            self.B_fb += step_b

    def apply_rpe(self, rpe: float, consolidate: bool = True) -> None:
        """Modulate 3-factor eligibility traces by scalar RPE; optionally consolidate accumulated episode gradients."""
        if self.cfg.freeze_weights:
            return
        r = float(rpe)
        self.acc_grad_W += r * self.elig_W
        self.acc_grad_B += r * self.elig_B
        self.acc_steps += 1

        if not consolidate or self.acc_steps == 0:
            return

        inv_s = 1.0 / float(self.acc_steps)
        grad_w = self.acc_grad_W * inv_s
        grad_b = self.acc_grad_B * inv_s
        self.acc_grad_W.fill(0.0)
        self.acc_grad_B.fill(0.0)
        self.acc_steps = 0

        eta_w = self.cfg.eta
        eta_b = self.cfg.eta if self.cfg.eta_fb is None else self.cfg.eta_fb
        self._apply_local_gradients(grad_w, grad_b, eta_w, eta_b)

    def feedback_alignment_cosine(self) -> float:
        """Measure cosine alignment between forward basal weights W[:, :d_in]^T and reciprocal apical B_fb."""
        w_in_t = self.W[:, : self.d_in].T
        nw = float(np.linalg.norm(w_in_t))
        nb = float(np.linalg.norm(self.B_fb))
        if nw < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.sum(w_in_t * self.B_fb) / (nw * nb))


class PDNCNetwork:
    """Multi-layer gradient-free Prospective Dendritic Natural-Credit network (NumPy)."""

    def __init__(
        self,
        layer_configs: Sequence[PDNCConfig],
        ablate_prospective_settle: bool = False,
    ) -> None:
        self.layers: List[PDNCLayer] = [PDNCLayer(cfg) for cfg in layer_configs]
        self.ablate_prospective_settle = bool(ablate_prospective_settle)

    def reset_temporal_state(self) -> None:
        for layer in self.layers:
            layer.reset_temporal_state()

    def step(self, x: np.ndarray, dt: float = 1.0, stateless_eval: bool = False) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = layer.step(h, dt=dt, stateless_eval=stateless_eval)
        return h

    def learn_prospective(self, target: np.ndarray, mod: float = 1.0) -> Tuple[np.ndarray, float]:
        """Two-sweep local prospective settlement + null-space/precision update (zero backprop)."""
        top = self.layers[-1]
        y_pre = top._last_h[0] if top._was_1d else top._last_h
        tgt = np.asarray(target, dtype=np.float64).reshape(y_pre.shape)
        err_pre = tgt - y_pre
        mse_pre = float(np.mean(err_pre * err_pre))

        n_layers = len(self.layers)
        if n_layers == 1:
            ds0, _ = top.project_apical(err_pre)
            top.update_synapses(ds0, mod=mod)
            return y_pre.copy(), mse_pre

        # Phase 1: Top-down apical prospective target inference (no weights mutated yet)
        delta_somas: List[Optional[np.ndarray]] = [None] * n_layers
        apical = err_pre
        for l_idx in range(n_layers - 1, -1, -1):
            ds_l, apical_prev = self.layers[l_idx].project_apical(apical)
            if l_idx < n_layers - 1:
                ds_l = self.layers[l_idx].cfg.prospective_beta * ds_l
            delta_somas[l_idx] = ds_l
            apical = apical_prev

        # Phase 2: Bottom-up prospective settlement & precision-conditioned weight plasticity
        for l_idx in range(n_layers - 1):
            layer = self.layers[l_idx]
            ds_l = delta_somas[l_idx]
            assert ds_l is not None
            layer.update_synapses(ds_l, mod=mod)
            if not self.ablate_prospective_settle and not layer.cfg.freeze_weights:
                h_post = layer.recompute_post_plasticity()
                for k_idx in range(l_idx + 1, n_layers):
                    h_post = self.layers[k_idx].step(h_post, stateless_eval=True)

        # Final readout layer updates on the settled post-plasticity hidden representation!
        y_settled = top._last_h[0] if top._was_1d else top._last_h
        err_settled = tgt - y_settled
        ds_top, _ = top.project_apical(err_settled)
        top.update_synapses(ds_top, mod=mod)
        return y_pre.copy(), mse_pre


class PDNCActorCritic:
    """Multi-layer PDNC TD(lambda) Actor-Critic (Layer-1 feature discovery + Actor & Critic heads).

    Zero .backward(), zero episode buffer, zero weight transport:
      - Layer 1 (`enc`) receives multiplexed apical sensitivity from BOTH `actor` and `critic`
        via symmetric Kolen-Pollack reciprocal synapses and accumulates 3-factor dendritic
        eligibility modulated by scalar TD error.
    """

    def __init__(
        self,
        enc_cfg: PDNCConfig,
        n_actions: int = 3,
        eta_actor: float = 0.015,
        eta_critic: float = 0.025,
        gamma: float = 0.96,
        lam: float = 0.88,
        critic_apical_weight: float = 0.5,
        beta_ent: float = 0.015,
        eps_explore: float = 0.04,
        precision_mode_heads: str = "diagonal",
        seed: int = 42,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.critic_apical_weight = float(critic_apical_weight)
        self.beta_ent = float(beta_ent)
        self.eps_explore = float(eps_explore)
        self.n_actions = int(n_actions)

        enc_cfg.gamma_elig = self.gamma * self.lam
        self.enc = PDNCLayer(enc_cfg)

        d_h = enc_cfg.d_out
        self.actor = PDNCLayer(
            PDNCConfig(
                d_in=d_h,
                d_out=self.n_actions,
                activation="linear",
                precision_mode=precision_mode_heads,
                beta_rms=enc_cfg.beta_rms,
                eps_prec=enc_cfg.eps_prec,
                eta=eta_actor,
                lambda_kp=enc_cfg.lambda_kp,
                gamma_elig=self.gamma * self.lam,
                weight_init_scale=0.0,
                fb_init_scale=enc_cfg.fb_init_scale,
                freeze_random_feedback=enc_cfg.freeze_random_feedback,
                shuffle_apical=enc_cfg.shuffle_apical,
                ablate_eligibility=enc_cfg.ablate_eligibility,
                seed=seed + 303,
            )
        )
        self.critic = PDNCLayer(
            PDNCConfig(
                d_in=d_h,
                d_out=1,
                activation="linear",
                precision_mode=precision_mode_heads,
                beta_rms=enc_cfg.beta_rms,
                eps_prec=enc_cfg.eps_prec,
                eta=eta_critic,
                lambda_kp=enc_cfg.lambda_kp,
                gamma_elig=self.gamma * self.lam,
                weight_init_scale=0.0,
                fb_init_scale=enc_cfg.fb_init_scale,
                freeze_random_feedback=enc_cfg.freeze_random_feedback,
                shuffle_apical=enc_cfg.shuffle_apical,
                ablate_eligibility=enc_cfg.ablate_eligibility,
                seed=seed + 606,
            )
        )

    def reset_episode(self) -> None:
        self.enc.reset_temporal_state()
        self.actor.reset_temporal_state()
        self.critic.reset_temporal_state()

    def step_policy(self, obs: np.ndarray, greedy: bool = False) -> Tuple[int, float]:
        """Causal forward step + immediate multi-layer 3-factor eligibility trace accumulation."""
        h = self.enc.step(obs, stateless_eval=greedy)
        logits = self.actor.step(h, stateless_eval=greedy)
        val = float(self.critic.step(h, stateless_eval=greedy)[0])
        probs = _softmax_np(logits)

        if greedy:
            act = int(np.argmax(logits))
            return act, val

        probs_exp = (1.0 - self.eps_explore) * probs + self.eps_explore / float(self.n_actions)
        act = int(self.rng.choice(self.n_actions, p=probs_exp))

        # Local Actor policy score + local entropy regularizer
        log_p = np.log(np.clip(probs, 1e-8, 1.0))
        ent = -float(np.dot(probs, log_p))
        ent_grad = -probs * (log_p + ent)
        score_pi = -probs.copy()
        score_pi[act] += 1.0
        score_pi = score_pi + self.beta_ent * ent_grad

        # Propagate apical sensitivity into Layer-1 dendritic eligibility before stepping env
        apical_pi = self.actor.accumulate_eligibility(score_pi)
        apical_v = self.critic.accumulate_eligibility(np.array([1.0], dtype=np.float64))
        apical_total = apical_pi + self.critic_apical_weight * apical_v
        self.enc.accumulate_eligibility(apical_total)
        return act, val

    def value_only(self, obs: np.ndarray) -> float:
        """Evaluate V(s_{t+1}) causally without mutating spine or eligibility state."""
        h_next = self.enc.step(obs, stateless_eval=True)
        return float(self.critic.step(h_next, stateless_eval=True)[0])

    def apply_td_update(
        self,
        v_curr: float,
        reward: float,
        v_next: float,
        done: bool,
        consolidate: Optional[bool] = None,
    ) -> float:
        """Compute local TD error and modulate Layer-1, Actor, and Critic eligibility traces.

        By default (`consolidate=None`), accumulates TD-modulated eligibility locally per frame
        (`O(1)` in trajectory length) and fires the natural-gradient consolidation step at
        `done == True`, eliminating step-to-terminal variance mismatch while remaining buffer-free.
        """
        td_err = float(reward + (0.0 if done else self.gamma * v_next) - v_curr)
        td_clip = float(np.clip(td_err, -2.0, 2.0))
        do_cons = bool(done) if consolidate is None else bool(consolidate)
        self.actor.apply_rpe(td_clip, consolidate=do_cons)
        self.critic.apply_rpe(td_clip, consolidate=do_cons)
        self.enc.apply_rpe(td_clip, consolidate=do_cons)
        return td_err


class PDNCTorchLayer(nn.Module):
    """GPU/CUDA-ready Prospective Dendritic Natural-Credit layer (strictly zero autograd)."""

    def __init__(
        self,
        cfg: PDNCConfig,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_in = int(cfg.d_in)
        self.d_out = int(cfg.d_out)
        self.d_psi = (2 * self.d_in) if cfg.use_transient_spine else self.d_in
        self.d_ext = self.d_psi + 1

        ref = PDNCLayer(cfg)
        self.register_buffer("W", torch.tensor(ref.W, dtype=dtype))
        self.register_buffer("B_fb", torch.tensor(ref.B_fb, dtype=dtype))
        self.register_buffer("spine_ema", torch.zeros(self.d_in, dtype=dtype))
        self.spine_init = False

        if cfg.precision_mode == "woodbury" and ref.P_prec is not None:
            self.register_buffer("P_prec", torch.tensor(ref.P_prec, dtype=dtype))
            self.m_w = None
            self.v_rms = None
            self.m_b = None
            self.v_fb_rms = None
        elif cfg.precision_mode in ("diagonal", "subspace"):
            if cfg.precision_mode == "subspace" and ref.P_prec is not None:
                self.register_buffer("P_prec", torch.tensor(ref.P_prec, dtype=dtype))
            else:
                self.P_prec = None
            self.register_buffer("m_w", torch.zeros((self.d_out, self.d_ext), dtype=dtype))
            self.register_buffer("v_rms", torch.zeros((self.d_out, self.d_ext), dtype=dtype))
            self.register_buffer("m_b", torch.zeros((self.d_in, self.d_out), dtype=dtype))
            self.register_buffer("v_fb_rms", torch.zeros((self.d_in, self.d_out), dtype=dtype))
        else:
            self.P_prec = None
            self.m_w = None
            self.v_rms = None
            self.m_b = None
            self.v_fb_rms = None

        self.rms_steps = 0
        self._shuf_gen = torch.Generator(device="cpu")
        self._shuf_gen.manual_seed(cfg.seed + 15485)

        self._last_psi_ext: Optional[torch.Tensor] = None
        self._last_dphi: Optional[torch.Tensor] = None
        self._last_h: Optional[torch.Tensor] = None
        self._was_1d: bool = True

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def reset_temporal_state(self) -> None:
        self.spine_ema.zero_()
        self.spine_init = False

    @torch.no_grad()
    def _activate(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        act = self.cfg.activation
        if act == "linear":
            h = u.clone()
            dphi = torch.ones_like(u)
        elif act == "relu":
            h = torch.relu(u)
            dphi = (u > 0.0).to(dtype=u.dtype)
        elif act == "silu":
            sig = torch.sigmoid(u)
            h = u * sig
            dphi = sig + u * sig * (1.0 - sig)
        else:
            h = torch.tanh(u)
            dphi = 1.0 - h * h

        if 0 < self.cfg.k_wta < self.d_out:
            _, top_idx = torch.topk(torch.abs(h), k=self.cfg.k_wta, dim=1)
            mask = torch.zeros_like(h)
            mask.scatter_(1, top_idx, 1.0)
            h = h * mask
            dphi = dphi * mask
        return h, dphi

    @torch.no_grad()
    def _compute_from_psi_ext(self, psi_ext: torch.Tensor) -> torch.Tensor:
        u = psi_ext @ self.W.T
        h, dphi = self._activate(u)
        self._last_psi_ext = psi_ext
        self._last_dphi = dphi
        self._last_h = h
        return h[0].clone() if self._was_1d else h.clone()

    @torch.no_grad()
    def recompute_post_plasticity(self) -> torch.Tensor:
        assert self._last_psi_ext is not None
        return self._compute_from_psi_ext(self._last_psi_ext)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, dt: float = 1.0, stateless_eval: bool = False) -> torch.Tensor:
        x_dev = x.to(device=self.W.device, dtype=self.W.dtype)
        self._was_1d = x_dev.ndim == 1
        x_mat = x_dev.reshape(1, self.d_in) if self._was_1d else x_dev.reshape(-1, self.d_in)
        bsz = x_mat.shape[0]

        if self.cfg.use_transient_spine:
            if bsz == 1:
                x0 = x_mat[0]
                if not self.spine_init:
                    if not stateless_eval:
                        self.spine_ema.copy_(x0)
                        self.spine_init = True
                    tr = torch.zeros_like(x_mat)
                else:
                    tr = ((x0 - self.spine_ema) * self.cfg.transient_scale).reshape(1, self.d_in)
                    if not stateless_eval:
                        decay = float(np.exp(-float(dt) / max(1e-6, self.cfg.tau_spine)))
                        self.spine_ema.mul_(decay).add_((1.0 - decay) * x0)
                psi = torch.cat([x_mat, tr], dim=1)
            else:
                psi = torch.cat([x_mat, torch.zeros_like(x_mat)], dim=1)
        else:
            psi = x_mat

        ones = torch.ones((bsz, 1), device=x_mat.device, dtype=x_mat.dtype)
        psi_ext = torch.cat([psi, ones], dim=1)
        return self._compute_from_psi_ext(psi_ext)

    @torch.no_grad()
    def project_apical(self, delta_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._last_dphi is not None
        d_dev = delta_out.to(device=self.W.device, dtype=self.W.dtype)
        d_mat = d_dev.reshape(1, self.d_out) if d_dev.ndim == 1 else d_dev.reshape(-1, self.d_out)
        delta_soma = d_mat * self._last_dphi
        apical = delta_soma @ self.B_fb.T
        if self.cfg.shuffle_apical:
            perm = torch.randperm(self.d_in, generator=self._shuf_gen).to(apical.device)
            apical = apical[:, perm]
        return delta_soma, (apical[0] if self._was_1d else apical)

    @torch.no_grad()
    def _update_subspace_projector(self, psi_ext: torch.Tensor) -> None:
        if self.P_prec is None:
            return
        bsz = float(psi_ext.shape[0])
        scale_null = float(np.sqrt(self.cfg.alpha_null / (bsz if self.cfg.scale_null_by_batch else 1.0)))
        psi_null = scale_null * psi_ext
        k_mat = self.P_prec @ psi_null.T
        s_mat = torch.eye(psi_ext.shape[0], device=psi_ext.device, dtype=psi_ext.dtype) + psi_null @ k_mat
        g_null = torch.linalg.solve(s_mat, k_mat.T).T
        p_new = self.P_prec - g_null @ k_mat.T
        self.P_prec.copy_(0.5 * (p_new + p_new.T))

    @torch.no_grad()
    def update_synapses(self, delta_soma: torch.Tensor, mod: float = 1.0) -> None:
        if self.cfg.freeze_weights:
            return
        assert self._last_psi_ext is not None
        ds = delta_soma.reshape(-1, self.d_out)
        psi_ext = self._last_psi_ext
        bsz = float(ds.shape[0])
        eta_w = float(mod) * self.cfg.eta
        eta_b = float(mod) * (self.cfg.eta if self.cfg.eta_fb is None else self.cfg.eta_fb)

        if self.cfg.precision_mode == "woodbury" and self.P_prec is not None:
            scale_null = float(np.sqrt(self.cfg.alpha_null / (bsz if self.cfg.scale_null_by_batch else 1.0)))
            psi_null = scale_null * psi_ext
            k_mat = self.P_prec @ psi_null.T
            s_mat = torch.eye(ds.shape[0], device=ds.device, dtype=ds.dtype) + psi_null @ k_mat
            g_null = torch.linalg.solve(s_mat, k_mat.T).T
            p_new = self.P_prec - g_null @ k_mat.T
            self.P_prec.copy_(0.5 * (p_new + p_new.T))

            if self.cfg.alpha_null == 1.0 and not self.cfg.scale_null_by_batch:
                step_w = eta_w * (ds.T @ g_null.T)
                step_b = eta_b * (g_null[: self.d_in, :] @ ds)
            else:
                psi_proj = psi_ext @ self.P_prec
                step_w = (eta_w / bsz) * (ds.T @ psi_proj)
                step_b = (eta_b / bsz) * (psi_proj[:, : self.d_in].T @ ds)

            self.W.add_(step_w)
            if not self.cfg.freeze_random_feedback:
                self.B_fb.add_(step_b)
            return

        grad_w = (ds.T @ psi_ext) / bsz
        grad_b = (psi_ext[:, : self.d_in].T @ ds) / bsz

        if self.m_w is not None and self.v_rms is not None and self.m_b is not None and self.v_fb_rms is not None:
            self.rms_steps += 1
            b1 = self.cfg.beta_mom
            b2 = self.cfg.beta_rms
            bc1 = 1.0 - (b1 ** self.rms_steps)
            bc2 = 1.0 - (b2 ** self.rms_steps)

            self.m_w.mul_(b1).add_((1.0 - b1) * grad_w)
            self.v_rms.mul_(b2).add_((1.0 - b2) * (grad_w * grad_w))
            step_w = eta_w * ((self.m_w / bc1) / (torch.sqrt(self.v_rms / bc2) + self.cfg.eps_prec))

            self.m_b.mul_(b1).add_((1.0 - b1) * grad_b)
            self.v_fb_rms.mul_(b2).add_((1.0 - b2) * (grad_b * grad_b))
            step_b = eta_b * ((self.m_b / bc1) / (torch.sqrt(self.v_fb_rms / bc2) + self.cfg.eps_prec))
        else:
            psi_sq = float((psi_ext * psi_ext).sum(dim=1).mean().item())
            step_w = (eta_w / (1.0 + psi_sq)) * grad_w
            step_b = (eta_b / (1.0 + psi_sq)) * grad_b

        if self.cfg.precision_mode == "subspace" and self.P_prec is not None:
            step_w = step_w @ self.P_prec
            step_b = self.P_prec[: self.d_in, : self.d_in] @ step_b
            self._update_subspace_projector(psi_ext)

        wd_w = min(0.2, eta_w * (self.cfg.weight_decay + self.cfg.lambda_kp))
        if wd_w > 0.0:
            self.W[:, : self.d_psi].mul_(1.0 - wd_w)
        self.W.add_(step_w)

        if not self.cfg.freeze_random_feedback:
            wd_b = min(0.2, eta_b * self.cfg.lambda_kp)
            if wd_b > 0.0:
                self.B_fb.mul_(1.0 - wd_b)
            self.B_fb.add_(step_b)

    @torch.no_grad()
    def feedback_alignment_cosine(self) -> float:
        w_in_t = self.W[:, : self.d_in].T
        nw = float(torch.linalg.norm(w_in_t).item())
        nb = float(torch.linalg.norm(self.B_fb).item())
        if nw < 1e-12 or nb < 1e-12:
            return 0.0
        return float(torch.sum(w_in_t * self.B_fb).item() / (nw * nb))


class PDNCTorchNetwork(nn.Module):
    """GPU/CUDA-ready multi-layer Prospective Dendritic Natural-Credit network (zero autograd)."""

    def __init__(
        self,
        layer_configs: Sequence[PDNCConfig],
        ablate_prospective_settle: bool = False,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([PDNCTorchLayer(cfg, device=device, dtype=dtype) for cfg in layer_configs])
        self.ablate_prospective_settle = bool(ablate_prospective_settle)

    @torch.no_grad()
    def reset_temporal_state(self) -> None:
        for layer in self.layers:
            layer.reset_temporal_state()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, dt: float = 1.0, stateless_eval: bool = False) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, dt=dt, stateless_eval=stateless_eval)
        return h

    @torch.no_grad()
    def learn_prospective(self, target: torch.Tensor, mod: float = 1.0) -> Tuple[torch.Tensor, float]:
        top = self.layers[-1]
        assert top._last_h is not None
        y_pre = top._last_h[0] if top._was_1d else top._last_h
        tgt = target.to(device=y_pre.device, dtype=y_pre.dtype).reshape(y_pre.shape)
        err_pre = tgt - y_pre
        mse_pre = float(torch.mean(err_pre * err_pre).item())

        n_layers = len(self.layers)
        if n_layers == 1:
            ds0, _ = top.project_apical(err_pre)
            top.update_synapses(ds0, mod=mod)
            return y_pre.clone(), mse_pre

        delta_somas: List[Optional[torch.Tensor]] = [None] * n_layers
        apical = err_pre
        for l_idx in range(n_layers - 1, -1, -1):
            ds_l, apical_prev = self.layers[l_idx].project_apical(apical)
            if l_idx < n_layers - 1:
                ds_l = self.layers[l_idx].cfg.prospective_beta * ds_l
            delta_somas[l_idx] = ds_l
            apical = apical_prev

        for l_idx in range(n_layers - 1):
            layer = self.layers[l_idx]
            ds_l = delta_somas[l_idx]
            assert ds_l is not None
            layer.update_synapses(ds_l, mod=mod)
            if not self.ablate_prospective_settle and not layer.cfg.freeze_weights:
                h_post = layer.recompute_post_plasticity()
                for k_idx in range(l_idx + 1, n_layers):
                    h_post = self.layers[k_idx](h_post, stateless_eval=True)

        assert top._last_h is not None
        y_settled = top._last_h[0] if top._was_1d else top._last_h
        err_settled = tgt - y_settled
        ds_top, _ = top.project_apical(err_settled)
        top.update_synapses(ds_top, mod=mod)
        return y_pre.clone(), mse_pre
