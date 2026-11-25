"""DMCN v3 Adaptive Extensions: Self-Gating Pathways, Factorized Bilinear Dendrites, and Unsupervised Oja Vision.

All three mechanisms are strictly local, causal, and gradient-free (zero backpropagation):
  1. SelfGatingDMCNLayer: 4 parallel causal pathways (raw, RMS homeostatic, spine ODE, transient)
     combined via local Exponential-Weights (Hedge) self-gating based on each pathway's own causal
     prediction error — zero human pillar toggling.
  2. FactorizedBilinearDMCNLayer: Unsupervised competitive sparse Oja/L1 dendritic factorization
     paired with additive + multiplicative (bilinear coincidence) somatic binding for zero-shot
     compositional generalization without hardcoded factor indices.
  3. UnsupOjaDendriticLayer: Motion-driven k-WTA competitive Oja feature discovery on raw pixel
     streams with homeostatic fatigue so Layer-1 neurons self-organize into localized object detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import AblationConfig, DMCNConfig
from .numpy_core import DMCNLayer


@dataclass
class SelfGatingConfig:
    d_in: int = 4
    d_out: int = 1
    d_hid: int = 32
    beta_gate: float = 4.0
    alpha_loss: float = 0.08
    alpha_homeo: float = 0.05
    tau_spine: float = 2.5
    eta_readout: float = 0.12
    use_precision: bool = False
    precision_ridge: float = 2.0
    ablate_self_gating: bool = False
    seed: int = 42


class SelfGatingDMCNLayer:
    """Darwinian Self-Gating DMCN: 4 local candidate pathways competing via causal Hedge weights.

    Pathways:
      0: raw x_t (preserves physical amplitude, e.g. Lorenz chaos / Pong position)
      1: causal RMS homeostatic x_norm (absorbs 10x-50x sudden sensor gain shocks)
      2: causal spine ODE memory [x_norm, spine_1, spine_2] (preserves delayed cues)
      3: causal high-pass transient [x_t, 3*(x_t - spine_fast)] (extracts velocity/change)
    """

    PATHWAY_NAMES = ("raw", "homeo", "spine", "transient")

    def __init__(self, cfg: SelfGatingConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.d_in = int(cfg.d_in)
        self.d_out = int(cfg.d_out)
        self.d_hid = int(cfg.d_hid)

        # Shared causal sensory preprocessors (strictly causal, zero future peeking)
        self.ema_var = np.ones(self.d_in, dtype=np.float64)
        self.homeo_init = False
        self.peak_e = 0.0
        self.peak_init = False

        self.taus = np.array([2.0, 8.0], dtype=np.float64)
        self.spine_homeo = np.zeros((2, self.d_in), dtype=np.float64)
        self.spine_raw = np.zeros(self.d_in, dtype=np.float64)
        self.raw_spine_init = False

        # 4 independent pathway experts (random dendritic tanh expansion + local readout)
        dims = [self.d_in, self.d_in, 3 * self.d_in, 2 * self.d_in]
        self.encoders: List[DMCNLayer] = []
        self.readouts: List[DMCNLayer] = []
        for m, d_p in enumerate(dims):
            enc = DMCNLayer(DMCNConfig(
                d_in=d_p,
                d_out=self.d_hid,
                n_branches=1,
                n_spine_scales=0,
                soma_activation="tanh",
                weight_init_scale=1.8,
                seed=cfg.seed + 100 * (m + 1),
                ablation=AblationConfig(ablate_homeostasis=True),
            ))
            enc.b_dend[0] = self.rng.normal(0.0, 0.30, size=self.d_hid)
            ro = DMCNLayer(DMCNConfig(
                d_in=self.d_hid + 1,
                d_out=self.d_out,
                n_branches=1,
                n_spine_scales=0,
                soma_activation="linear",
                eta_fast=1.0 if cfg.use_precision else cfg.eta_readout,
                gamma_forget=0.0,
                eta_consol=0.0,
                weight_init_scale=0.0,
                use_precision_plasticity=cfg.use_precision,
                precision_ridge=cfg.precision_ridge,
                seed=cfg.seed + 500 + m,
                ablation=AblationConfig(ablate_homeostasis=True),
            ))
            self.encoders.append(enc)
            self.readouts.append(ro)

        # Causal Hedge pathway loss trackers & gate weights
        self.ema_loss = np.zeros(4, dtype=np.float64)
        self.loss_init = False
        self.gates = np.full(4, 0.25, dtype=np.float64)
        self._last_preds = np.zeros((4, self.d_out), dtype=np.float64)
        self._last_y = np.zeros(self.d_out, dtype=np.float64)
        self._last_feats: List[np.ndarray] = [np.zeros(self.d_hid + 1, dtype=np.float64) for _ in range(4)]

    def reset_temporal_state(self) -> None:
        self.spine_homeo.fill(0.0)
        self.spine_raw.fill(0.0)
        self.raw_spine_init = False
        self.peak_e = 0.0
        self.peak_init = False

    def _build_pathways(self, x: np.ndarray, dt: float = 1.0, episodic_peak: bool = False) -> List[np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64).reshape(self.d_in)

        # Pathway 0: raw unscaled input (soft-clipped only at extreme 50.0 to prevent overflow)
        p0 = np.clip(x_arr, -50.0, 50.0)

        # Pathway 1: strictly causal RMS divisive homeostasis
        if episodic_peak:
            e_now = float(np.mean(x_arr * x_arr))
            if not self.peak_init:
                self.peak_e = e_now
                self.peak_init = True
            else:
                self.peak_e = max(0.98 * self.peak_e, e_now)
            p1 = np.clip(x_arr / np.sqrt(self.peak_e + 1e-6), -8.0, 8.0)
        else:
            if not self.homeo_init:
                p1 = np.clip(x_arr, -8.0, 8.0)
                if float(np.dot(x_arr, x_arr)) > 1e-12:
                    self.ema_var = np.maximum(x_arr * x_arr, 1e-2)
                    self.homeo_init = True
            else:
                p1 = np.clip(x_arr / np.sqrt(self.ema_var + 1e-4), -8.0, 8.0)
                ah = self.cfg.alpha_homeo
                self.ema_var = (1.0 - ah) * self.ema_var + ah * (x_arr * x_arr)

        # Pathway 2: causal multi-timescale spine ODE bank over homeostatic signal
        dec = np.exp(-float(dt) / self.taus)[:, None]
        self.spine_homeo = dec * self.spine_homeo + (1.0 - dec) * p1[None, :]
        p2 = np.concatenate([p1, self.spine_homeo.reshape(-1)])

        # Pathway 3: causal high-pass transient derivative [x_raw, 3*(x_raw - spine_raw)]
        if not self.raw_spine_init:
            self.spine_raw = p0.copy()
            self.raw_spine_init = True
            tr = np.zeros_like(p0)
        else:
            dec_r = float(np.exp(-float(dt) / self.cfg.tau_spine))
            tr = np.clip((p0 - self.spine_raw) * 3.0, -20.0, 20.0)
            self.spine_raw = dec_r * self.spine_raw + (1.0 - dec_r) * p0
        p3 = np.concatenate([p0, tr])

        return [p0, p1, p2, p3]

    def _compute_gates(self) -> np.ndarray:
        if self.cfg.ablate_self_gating or not self.loss_init:
            return np.full(4, 0.25, dtype=np.float64)
        scale = float(np.mean(self.ema_loss)) + 1e-8
        norm_l = (self.ema_loss - float(np.min(self.ema_loss))) / scale
        logits = -self.cfg.beta_gate * norm_l
        e = np.exp(np.clip(logits, -30.0, 0.0))
        return e / (float(np.sum(e)) + 1e-12)

    def step(self, x: np.ndarray, dt: float = 1.0, episodic_peak: bool = False) -> np.ndarray:
        pathways = self._build_pathways(x, dt=dt, episodic_peak=episodic_peak)
        self.gates = self._compute_gates()
        for m in range(4):
            h = np.append(self.encoders[m].step(pathways[m]), 1.0)
            self._last_feats[m] = h
            self._last_preds[m] = self.readouts[m].step(h)
        y = np.sum(self.gates[:, None] * self._last_preds, axis=0)
        self._last_y = y
        return y.copy()

    def learn_local(self, target: np.ndarray) -> Tuple[np.ndarray, float]:
        tgt = np.asarray(target, dtype=np.float64).reshape(self.d_out)
        al = self.cfg.alpha_loss
        for m in range(4):
            err_m = tgt - self._last_preds[m]
            mse_m = float(np.mean(err_m * err_m))
            if not self.loss_init:
                self.ema_loss[m] = mse_m
            else:
                self.ema_loss[m] = (1.0 - al) * self.ema_loss[m] + al * mse_m
            self.readouts[m].learn_local(err_m)
        self.loss_init = True
        overall_mse = float(np.mean((tgt - self._last_y) ** 2))
        return self._last_y.copy(), overall_mse

    def gate_dict(self) -> Dict[str, float]:
        return {name: float(self.gates[i]) for i, name in enumerate(self.PATHWAY_NAMES)}


class FactorizedBilinearDMCNLayer:
    """Unsupervised Sparse Competitive Dendritic Factorization + Bilinear Coincidence Binding.

    Discovers sparse factor-specialized dendritic branches via local competitive Oja + L1 shrinkage
    (zero hardcoded factor indices, zero backpropagation), then feeds both:
      - Additive factorized branch activations (zero-shot independent attribute composition)
      - Pairwise bilinear coincidence products u_i * u_j across distinct branches (zero-shot Rule x Operand)
    into a local Sherman-Morrison precision readout.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_branches: int = 16,
        top_k_oja: int = 2,
        eta_oja: float = 0.04,
        l1_shrink: float = 0.015,
        ridge: float = 1.0,
        ablate_factorization: bool = False,
        ablate_bilinear: bool = False,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.n_branches = int(n_branches)
        self.top_k_oja = int(min(n_branches, max(1, top_k_oja)))
        self.eta_oja = float(eta_oja)
        self.l1_shrink = float(l1_shrink)
        self.ablate_factorization = bool(ablate_factorization)
        self.ablate_bilinear = bool(ablate_bilinear)

        # Dendritic branch weights W_fac (n_branches x d_in), unit-normalized rows
        w = self.rng.randn(self.n_branches, self.d_in)
        w /= (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        self.W_fac = w
        self.b_fac = np.zeros(self.n_branches, dtype=np.float64)
        self.win_counts = np.zeros(self.n_branches, dtype=np.float64)

        # Pairwise indices for bilinear coincidence binding (b1 < b2)
        self._pair_i, self._pair_j = np.triu_indices(self.n_branches, k=1)
        self.n_pairs = len(self._pair_i)
        d_feat = (self.n_branches + 1) if ablate_bilinear else (self.n_branches + self.n_pairs + 1)

        self.readout = DMCNLayer(DMCNConfig(
            d_in=d_feat,
            d_out=self.d_out,
            n_branches=1,
            n_spine_scales=0,
            soma_activation="linear",
            eta_fast=1.0,
            gamma_forget=0.0,
            eta_consol=0.0,
            weight_init_scale=0.0,
            use_precision_plasticity=True,
            precision_ridge=ridge,
            seed=seed + 101,
            ablation=AblationConfig(ablate_homeostasis=True),
        ))

    def observe_unsupervised(self, x: np.ndarray) -> None:
        """Strictly local competitive sparse Oja update to discover independent input factors."""
        if self.ablate_factorization:
            return
        x_arr = np.asarray(x, dtype=np.float64).reshape(self.d_in)
        xn = float(np.linalg.norm(x_arr))
        if xn < 1e-8:
            return
        x_u = x_arr / xn

        # Homeostatic fatigue encourages all branches to cover distinct sparse factors
        acts = self.W_fac @ x_u
        scores = np.abs(acts) - 0.05 * (self.win_counts / (float(np.mean(self.win_counts)) + 1.0))
        winners = np.argsort(scores)[::-1][: self.top_k_oja]

        for b in winners:
            y_b = float(acts[b])
            self.W_fac[b] += self.eta_oja * y_b * (x_u - y_b * self.W_fac[b])
            # Soft L1 shrinkage prunes cross-factor leakage weights to zero!
            w_b = self.W_fac[b]
            self.W_fac[b] = np.sign(w_b) * np.maximum(0.0, np.abs(w_b) - self.l1_shrink)
            nb = float(np.linalg.norm(self.W_fac[b]))
            if nb > 1e-8:
                self.W_fac[b] /= nb
            self.win_counts[b] += 1.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).reshape(self.d_in)
        u_lin = self.W_fac @ x_arr + self.b_fac
        u_nl = np.tanh(1.5 * u_lin)
        if self.ablate_bilinear:
            return np.append(u_nl, 1.0)
        bilin = u_lin[self._pair_i] * u_lin[self._pair_j]
        return np.concatenate([u_nl, bilin, np.array([1.0], dtype=np.float64)])

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.readout.step(self.encode(x))

    def learn_local(self, target: np.ndarray) -> np.ndarray:
        tgt = np.asarray(target, dtype=np.float64).reshape(self.d_out)
        err = tgt - self.readout._last_y
        self.readout.learn_local(err)
        return self.readout._last_y.copy()


class UnsupOjaDendriticLayer:
    """Motion-driven k-WTA competitive Oja Layer-1 visual feature discovery (zero backprop)."""

    def __init__(
        self,
        d_in: int = 256,
        d_hid: int = 128,
        k_wta: int = 8,
        eta_oja: float = 0.08,
        seed: int = 42,
        ablate_unsup: bool = False,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.d_in = int(d_in)
        self.d_hid = int(d_hid)
        self.k_wta = int(min(d_hid, max(1, k_wta)))
        self.eta_oja = float(eta_oja)
        self.ablate_unsup = bool(ablate_unsup)

        w = self.rng.randn(self.d_hid, self.d_in)
        w = np.maximum(0.0, w)  # non-negative visual templates
        w /= (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        self.W = w
        self.win_ema = np.full(self.d_hid, self.k_wta / float(self.d_hid), dtype=np.float64)
        self.spine_ema = np.zeros(self.d_in, dtype=np.float64)
        self.spine_init = False
        self.decay = float(np.exp(-1.0 / 2.0))
        self.unsup_steps = 0

    def reset_temporal_state(self) -> None:
        self.spine_ema.fill(0.0)
        self.spine_init = False

    def step(self, frame: np.ndarray, learn_unsup: bool = True) -> np.ndarray:
        f = np.asarray(frame, dtype=np.float64).reshape(self.d_in)
        if not self.spine_init:
            self.spine_ema = f.copy()
            self.spine_init = True
            tr = np.zeros_like(f)
        else:
            tr = f - self.spine_ema
            self.spine_ema = self.decay * self.spine_ema + (1.0 - self.decay) * f

        # Motion-highlighted unsupervised drive: raw active pixels + 2x moving pixel changes
        x_drive = np.maximum(0.0, f + 2.0 * np.abs(tr))
        drive_n = float(np.linalg.norm(x_drive))

        raw_a = self.W @ f
        tr_a = self.W @ tr

        if learn_unsup and not self.ablate_unsup and drive_n > 1e-6 and self.unsup_steps < 2500:
            self.unsup_steps += 1
            eta_now = self.eta_oja / (1.0 + 0.005 * self.unsup_steps)
            x_u = x_drive / drive_n
            a_d = self.W @ x_u
            # Homeostatic competition spreads detectors across all ball & paddle positions
            scores = a_d - 0.35 * (self.win_ema - self.k_wta / float(self.d_hid))
            winners = np.argpartition(scores, -self.k_wta)[-self.k_wta :]
            win_vec = np.zeros(self.d_hid, dtype=np.float64)
            win_vec[winners] = 1.0
            self.win_ema = 0.99 * self.win_ema + 0.01 * win_vec
            for idx in winners:
                yi = float(a_d[idx])
                self.W[idx] = np.maximum(0.0, self.W[idx] + eta_now * yi * (x_u - yi * self.W[idx]))
                ni = float(np.linalg.norm(self.W[idx]))
                if ni > 1e-8:
                    self.W[idx] /= ni

        # Sparse k-WTA lateral inhibition: only top-k winning object detectors fire (removes DC offset!)
        raw_centered = raw_a - float(np.mean(raw_a))
        k_out = min(self.d_hid, max(8, 2 * self.k_wta))
        top_out = np.argpartition(raw_centered, -k_out)[-k_out:]
        h_pos = np.zeros(self.d_hid, dtype=np.float64)
        h_pos[top_out] = np.tanh(4.0 * np.maximum(0.0, raw_centered[top_out]))
        h_tr = np.tanh(4.0 * tr_a)
        return np.concatenate([h_pos, h_tr])
