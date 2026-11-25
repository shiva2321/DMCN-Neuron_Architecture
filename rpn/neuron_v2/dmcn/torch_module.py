"""PyTorch CUDA-ready nn.Module implementation of DMCN (gradient-free local plasticity)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DMCNConfig
from .numpy_core import DMCNLayer, OpsReport, count_layer_ops


class DMCNTorchLayer(nn.Module):
    """GPU/CUDA-ready Dendritic Multiplexed Cascade Neuron layer (zero autograd)."""

    def __init__(self, cfg: DMCNConfig, device: Optional[str | torch.device] = None, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_in = int(cfg.d_in)
        self.d_out = int(cfg.d_out)
        self.n_branches = int(cfg.n_branches)
        self.k_scales = 0 if cfg.ablation.ablate_spine_traces else int(cfg.n_spine_scales)
        self.d_psi = self.d_in * (1 + self.k_scales)

        rng = np.random.RandomState(cfg.seed)
        scale = cfg.weight_init_scale / np.sqrt(max(1, self.d_psi))
        fb_scale = cfg.weight_init_scale / np.sqrt(max(1, self.d_out))

        # Pillar 1: Causal RMS homeostasis buffers
        self.register_buffer("ema_mean", torch.zeros(self.d_in, dtype=dtype))
        self.register_buffer("ema_var", torch.ones(self.d_in, dtype=dtype))
        self.register_buffer("ema_out_var", torch.ones(self.d_out, dtype=dtype))
        self.register_buffer("ema_act", torch.zeros(self.d_out, dtype=dtype))
        self.register_buffer("theta_soma", torch.zeros(self.d_out, dtype=dtype))
        self.homeo_initialized = False
        self.out_scale_initialized = False

        # Pillar 2: Spine ODE decay rates & states
        if self.k_scales > 0:
            taus = np.array([cfg.tau_min]) if self.k_scales == 1 else np.geomspace(cfg.tau_min, cfg.tau_max, self.k_scales)
            lambdas_np = (1.0 / taus).astype(np.float64)
        else:
            lambdas_np = np.zeros(0, dtype=np.float64)
        self.register_buffer("lambdas", torch.tensor(lambdas_np, dtype=dtype))
        self.register_buffer("spine_state", torch.zeros(self.k_scales, self.d_in, dtype=dtype))

        # Pillar 3: Active dendritic weights & cascade states
        if cfg.use_sparse_ternary:
            p_nz = float(np.clip(cfg.ternary_density, 1e-4, 1.0))
            u_mask = rng.rand(self.n_branches, self.d_out, self.d_psi)
            tern = np.where(u_mask < 0.5 * p_nz, -1.0, np.where(u_mask > 1.0 - 0.5 * p_nz, 1.0, 0.0))
            w_scale_t = cfg.weight_init_scale / np.sqrt(max(1e-6, p_nz * self.d_psi))
            w_slow_np = tern * w_scale_t
        else:
            w_slow_np = rng.randn(self.n_branches, self.d_out, self.d_psi) * scale
        self.register_buffer("W_slow", torch.tensor(w_slow_np, dtype=dtype))
        self.register_buffer("W_fast", torch.zeros(self.n_branches, self.d_out, self.d_psi, dtype=dtype))
        self.register_buffer("W_eff", torch.tensor(w_slow_np, dtype=dtype))
        self.register_buffer("b_dend", torch.zeros(self.n_branches, self.d_out, dtype=dtype))

        init_rho_c = float(np.clip(cfg.init_rho, 1e-4, 1.0 - 1e-4))
        self.register_buffer(
            "rho_logit",
            torch.full((self.n_branches, self.d_out), float(np.log(init_rho_c / (1.0 - init_rho_c))), dtype=dtype),
        )
        self.register_buffer("theta_on", torch.full((self.n_branches, self.d_out), float(cfg.init_theta_on), dtype=dtype))
        self.register_buffer(
            "delta_theta_raw",
            torch.full((self.n_branches, self.d_out), float(np.log(np.expm1(max(1e-3, cfg.init_delta_theta)))), dtype=dtype),
        )

        self.register_buffer("prototypes", torch.zeros(self.n_branches, self.d_psi, dtype=dtype))
        recruited = torch.zeros(self.n_branches, dtype=torch.bool)
        recruited[0] = True
        self.register_buffer("recruited", recruited)
        self.proto0_initialized = False

        self.register_buffer("ema_dW_fast", torch.zeros(self.n_branches, self.d_out, self.d_psi, dtype=dtype))
        self.register_buffer("tag_coherence", torch.zeros(self.n_branches, dtype=dtype))

        # Pillar 4: Apical feedback alignment matrix & eligibility trace
        b_fb_np = rng.randn(self.d_in, self.d_out) * fb_scale
        self.register_buffer("B_fb", torch.tensor(b_fb_np, dtype=dtype))
        self.register_buffer("elig_W", torch.zeros(self.n_branches, self.d_out, self.d_psi, dtype=dtype))

        # Optional local synaptic precision matrix (Woodbury exact per-step vs Gram-Cholesky checkpoint)
        if cfg.use_precision_plasticity and not cfg.ablation.ablate_precision:
            if cfg.precision_update_mode == "gram_cholesky":
                self.P_prec = None
                self.register_buffer("G_gram", float(max(1e-6, cfg.precision_ridge)) * torch.eye(self.d_psi, dtype=dtype))
                self.register_buffer("H_cross", torch.zeros(self.d_psi, self.d_out, dtype=dtype))
            else:
                p_init = (1.0 / max(1e-6, cfg.precision_ridge)) * torch.eye(self.d_psi, dtype=dtype)
                self.register_buffer("P_prec", p_init)
                self.G_gram = None
                self.H_cross = None
        else:
            self.P_prec = None
            self.G_gram = None
            self.H_cross = None

        # Recruitment tracking scalars
        self.prev_delta_out: Optional[torch.Tensor] = None
        self.ema_err_sq = 0.0
        self.ema_err_coh = 0.0
        self.sustained_recruit = 0
        self.steps_since_recruit = cfg.recruit_cooldown
        self.recruit_events: List[int] = []
        self.step_count = 0

        # Cached local cellular intermediates
        self._last_x_norm: Optional[torch.Tensor] = None
        self._last_psi: Optional[torch.Tensor] = None
        self._last_ctx_hat: Optional[torch.Tensor] = None
        self._last_p: Optional[torch.Tensor] = None
        self._last_u: Optional[torch.Tensor] = None
        self._last_dphi: Optional[torch.Tensor] = None
        self._last_dy: Optional[torch.Tensor] = None
        self._last_y: Optional[torch.Tensor] = None
        self._last_max_sim: float = 1.0

        self._torch_rng = torch.Generator(device="cpu")
        self._torch_rng.manual_seed(cfg.seed)

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def load_from_numpy(self, np_layer: DMCNLayer) -> None:
        """Copy exact parameter and state tensors from a NumPy DMCNLayer for parity checks."""
        dev = self.W_slow.device
        dt = self.W_slow.dtype
        self.W_slow.copy_(torch.tensor(np_layer.W_slow, device=dev, dtype=dt))
        self.W_fast.copy_(torch.tensor(np_layer.W_fast, device=dev, dtype=dt))
        self.W_eff.copy_(torch.tensor(np_layer.W_eff, device=dev, dtype=dt))
        self.b_dend.copy_(torch.tensor(np_layer.b_dend, device=dev, dtype=dt))
        self.B_fb.copy_(torch.tensor(np_layer.B_fb, device=dev, dtype=dt))
        self.prototypes.copy_(torch.tensor(np_layer.prototypes, device=dev, dtype=dt))
        self.recruited.copy_(torch.tensor(np_layer.recruited, device=dev, dtype=torch.bool))
        self.ema_mean.copy_(torch.tensor(np_layer.ema_mean, device=dev, dtype=dt))
        self.ema_var.copy_(torch.tensor(np_layer.ema_var, device=dev, dtype=dt))
        self.ema_out_var.copy_(torch.tensor(np_layer.ema_out_var, device=dev, dtype=dt))
        self.theta_soma.copy_(torch.tensor(np_layer.theta_soma, device=dev, dtype=dt))
        self.spine_state.copy_(torch.tensor(np_layer.spine_state, device=dev, dtype=dt))
        self.homeo_initialized = np_layer.homeo_initialized
        self.out_scale_initialized = np_layer.out_scale_initialized
        self.proto0_initialized = np_layer.proto0_initialized
        if self.P_prec is not None and np_layer.P_prec is not None:
            self.P_prec.copy_(torch.tensor(np_layer.P_prec, device=dev, dtype=dt))

    @torch.no_grad()
    def reset_temporal_state(self) -> None:
        self.spine_state.zero_()
        self.elig_W.zero_()
        self.prev_delta_out = None

    @torch.no_grad()
    def _compute_routing(self, ctx_hat: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.n_branches == 1:
            return torch.ones(1, device=ctx_hat.device, dtype=ctx_hat.dtype), 1.0

        sims = torch.mv(self.prototypes, ctx_hat)
        masked_sims = torch.where(self.recruited, sims, torch.full_like(sims, -1e9))
        max_sim = float(torch.max(masked_sims).item())
        logits = self.cfg.beta_route * (masked_sims - max_sim)
        p = torch.softmax(logits, dim=0)
        p = torch.where(self.recruited, p, torch.zeros_like(p))

        k_top = self.cfg.top_k_branches
        n_rec = int(self.recruited.sum().item())
        if 0 < k_top < n_rec:
            _, top_idx = torch.topk(p, k=k_top)
            p_sparse = torch.zeros_like(p)
            p_sparse[top_idx] = p[top_idx]
            s_p = float(p_sparse.sum().item())
            if s_p > 1e-12:
                p = p_sparse / s_p
        return p, max_sim

    @torch.no_grad()
    def _compute_routing_batch(self, ctx_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sz = ctx_hat.shape[0]
        if self.n_branches == 1:
            return (
                torch.ones((batch_sz, 1), device=ctx_hat.device, dtype=ctx_hat.dtype),
                torch.ones(batch_sz, device=ctx_hat.device, dtype=ctx_hat.dtype),
            )

        sims = ctx_hat @ self.prototypes.T  # (B, n_branches)
        rec_mask = self.recruited.unsqueeze(0)
        masked_sims = torch.where(rec_mask, sims, torch.full_like(sims, -1e9))
        max_sim = torch.max(masked_sims, dim=1, keepdim=True).values
        logits = self.cfg.beta_route * (masked_sims - max_sim)
        p = torch.softmax(logits, dim=1)
        p = torch.where(rec_mask, p, torch.zeros_like(p))

        k_top = self.cfg.top_k_branches
        n_rec = int(self.recruited.sum().item())
        if 0 < k_top < n_rec:
            _, top_idx = torch.topk(p, k=k_top, dim=1)
            p_sparse = torch.zeros_like(p)
            p_sparse.scatter_(1, top_idx, torch.gather(p, 1, top_idx))
            s_p = p_sparse.sum(dim=1, keepdim=True)
            p = torch.where(s_p > 1e-12, p_sparse / s_p.clamp_min(1e-12), p)
        return p, max_sim.squeeze(1)

    @torch.no_grad()
    def _dendritic_activation_branch(self, b: int, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.ablation.ablate_nonmonotonic or not self.cfg.use_nonmonotonic:
            if self.cfg.soma_activation == "linear" and not self.cfg.use_nonmonotonic:
                return ub.clone(), torch.ones_like(ub)
            t_u = torch.tanh(ub)
            return t_u, 1.0 - t_u * t_u

        rho = torch.sigmoid(self.rho_logit[b])
        theta_off = self.theta_on[b] + F.softplus(self.delta_theta_raw[b])
        beta = self.cfg.beta_dend

        s_on = torch.sigmoid(beta * (ub - self.theta_on[b]))
        s_off = torch.sigmoid(beta * (ub - theta_off))
        bump = s_on - s_off
        dbump = beta * (s_on * (1.0 - s_on) - s_off * (1.0 - s_off))
        t_u = torch.tanh(ub)
        phi = (1.0 - rho) * t_u + rho * bump
        dphi = (1.0 - rho) * (1.0 - t_u * t_u) + rho * dbump
        return phi, dphi

    @torch.no_grad()
    def _somatic_activation(self, i_soma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        act = self.cfg.soma_activation
        if act == "linear":
            return i_soma.clone(), torch.ones_like(i_soma)
        if act == "sigmoid":
            y = torch.sigmoid(i_soma)
            return y, y * (1.0 - y)
        y = torch.tanh(i_soma)
        return y, 1.0 - y * y

    @torch.no_grad()
    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Native vectorized GPU/CPU forward pass for mini-batch (B, d_in) with k_scales == 0."""
        x_mat = x.to(device=self.W_slow.device, dtype=self.W_slow.dtype)
        batch_sz = int(x_mat.shape[0])
        self.step_count += batch_sz
        self.steps_since_recruit += batch_sz

        # Pillar 1: Causal homeostasis (batch-vectorized)
        if not self.cfg.ablation.ablate_homeostasis:
            if not self.homeo_initialized:
                x_norm = torch.clamp(x_mat, -8.0, 8.0)
                if float((x_mat * x_mat).sum().item()) > 1e-12:
                    self.ema_mean.copy_(x_mat.mean(dim=0))
                    if self.cfg.center_homeostasis:
                        self.ema_var.fill_(1.0)
                    else:
                        self.ema_var.copy_(torch.clamp((x_mat * x_mat).mean(dim=0), min=1.0))
                    self.homeo_initialized = True
            else:
                ah_eff = float(min(1.0, 1.0 - (1.0 - self.cfg.alpha_homeo) ** batch_sz))
                if self.cfg.center_homeostasis:
                    x_norm = torch.clamp(
                        (x_mat - self.ema_mean) / torch.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    diff = x_mat - self.ema_mean
                    self.ema_mean.mul_(1.0 - ah_eff).add_(ah_eff * x_mat.mean(dim=0))
                    self.ema_var.mul_(1.0 - ah_eff).add_(ah_eff * (diff * diff).mean(dim=0))
                else:
                    x_norm = torch.clamp(
                        x_mat / torch.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    self.ema_var.mul_(1.0 - ah_eff).add_(ah_eff * (x_mat * x_mat).mean(dim=0))
        else:
            x_norm = x_mat

        psi = x_norm
        psi_l2 = torch.linalg.norm(psi, dim=1, keepdim=True)
        ctx_hat = torch.where(psi_l2 > 1e-9, psi / psi_l2.clamp_min(1e-9), torch.zeros_like(psi))
        if not self.proto0_initialized:
            nonzero_rows = torch.nonzero(psi_l2.squeeze(1) > 1e-9, as_tuple=False).reshape(-1)
            if nonzero_rows.numel() > 0:
                self.prototypes[0].copy_(ctx_hat[int(nonzero_rows[0].item())])
                self.proto0_initialized = True

        p, max_sim_vec = self._compute_routing_batch(ctx_hat)
        psi_proj = ctx_hat if self.cfg.use_unit_key_norm else psi

        if self.n_branches == 1:
            u0 = psi_proj @ self.W_eff[0].T - self.b_dend[0]
            phi0, dphi0 = self._dendritic_activation_branch(0, u0)
            i_soma = phi0
            u = u0.unsqueeze(1)
            dphi = dphi0.unsqueeze(1)
        else:
            u = torch.zeros((batch_sz, self.n_branches, self.d_out), device=x_mat.device, dtype=x_mat.dtype)
            dphi = torch.ones((batch_sz, self.n_branches, self.d_out), device=x_mat.device, dtype=x_mat.dtype)
            i_soma = torch.zeros((batch_sz, self.d_out), device=x_mat.device, dtype=x_mat.dtype)
            rec_idx = torch.nonzero(self.recruited, as_tuple=False).reshape(-1)
            for idx_t in rec_idx:
                b = int(idx_t.item())
                ub = psi_proj @ self.W_eff[b].T - self.b_dend[b]
                phib, dphib = self._dendritic_activation_branch(b, ub)
                u[:, b, :] = ub
                dphi[:, b, :] = dphib
                i_soma = i_soma + p[:, b : b + 1] * phib

        if not self.cfg.ablation.ablate_homeostasis:
            i_soma = i_soma - self.theta_soma
        y_int, dy = self._somatic_activation(i_soma)

        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis and self.out_scale_initialized:
            y = y_int * torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
        else:
            y = y_int

        if 0 < self.cfg.k_wta < self.d_out:
            _, top_idx = torch.topk(torch.abs(y), k=self.cfg.k_wta, dim=1)
            mask = torch.zeros_like(y)
            mask.scatter_(1, top_idx, 1.0)
            y = y * mask
            dy = dy * mask

        if not self.cfg.ablation.ablate_homeostasis and self.cfg.eta_theta > 0.0:
            aa_eff = float(min(1.0, 1.0 - (1.0 - self.cfg.alpha_act) ** batch_sz))
            self.ema_act.mul_(1.0 - aa_eff).add_(aa_eff * y_int.mean(dim=0))
            self.theta_soma.add_(self.cfg.eta_theta * (self.ema_act - self.cfg.target_activity))

        self._last_x_norm = x_norm
        self._last_psi = psi_proj
        self._last_ctx_hat = ctx_hat
        self._last_p = p
        self._last_u = u
        self._last_dphi = dphi
        self._last_dy = dy
        self._last_y = y
        self._last_max_sim = float(max_sim_vec.mean().item())
        return y.clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Strictly causal online forward step for a single sample (d_in,) or batch (N, d_in)."""
        if x.ndim == 2:
            if self.k_scales == 0:
                return self._forward_batch(x)
            outs = [self.forward(x[i], dt=dt) for i in range(x.shape[0])]
            return torch.stack(outs, dim=0)

        x_vec = x.to(device=self.W_slow.device, dtype=self.W_slow.dtype).reshape(self.d_in)
        self.step_count += 1
        self.steps_since_recruit += 1

        # Pillar 1: Strictly causal RMS divisive homeostasis
        if not self.cfg.ablation.ablate_homeostasis:
            if not self.homeo_initialized:
                x_norm = torch.clamp(x_vec, -8.0, 8.0)
                if float(torch.dot(x_vec, x_vec).item()) > 1e-12:
                    self.ema_mean.copy_(x_vec)
                    init_v = torch.ones_like(x_vec) if self.cfg.center_homeostasis else torch.clamp(x_vec * x_vec, min=1.0)
                    self.ema_var.copy_(init_v)
                    self.homeo_initialized = True
            else:
                if self.cfg.center_homeostasis:
                    x_norm = torch.clamp(
                        (x_vec - self.ema_mean) / torch.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    ah = self.cfg.alpha_homeo
                    diff = x_vec - self.ema_mean
                    self.ema_mean.mul_(1.0 - ah).add_(ah * x_vec)
                    self.ema_var.mul_(1.0 - ah).add_(ah * (diff * diff))
                else:
                    x_norm = torch.clamp(
                        x_vec / torch.sqrt(self.ema_var + self.cfg.eps_homeo),
                        -8.0,
                        8.0,
                    )
                    ah = self.cfg.alpha_homeo
                    self.ema_var.mul_(1.0 - ah).add_(ah * (x_vec * x_vec))
        else:
            x_norm = x_vec

        # Pillar 2: Diagonal spine ODE bank
        if self.k_scales > 0:
            decay = torch.exp(-self.lambdas.unsqueeze(1) * float(dt))
            self.spine_state.mul_(decay).add_((1.0 - decay) * x_norm.unsqueeze(0))
            psi = torch.cat([x_norm, self.spine_state.reshape(-1)], dim=0)
        else:
            psi = x_norm

        psi_l2 = float(torch.linalg.norm(psi).item())
        if psi_l2 > 1e-9:
            ctx_hat = psi / psi_l2
            if not self.proto0_initialized:
                self.prototypes[0].copy_(ctx_hat)
                self.proto0_initialized = True
        else:
            ctx_hat = torch.zeros_like(psi)

        p, max_sim = self._compute_routing(ctx_hat)
        psi_proj = ctx_hat if self.cfg.use_unit_key_norm else psi

        u = torch.zeros(self.n_branches, self.d_out, device=x_vec.device, dtype=x_vec.dtype)
        dphi = torch.ones(self.n_branches, self.d_out, device=x_vec.device, dtype=x_vec.dtype)
        i_soma = torch.zeros(self.d_out, device=x_vec.device, dtype=x_vec.dtype)

        if self.n_branches == 1:
            u0 = torch.mv(self.W_eff[0], psi_proj) - self.b_dend[0]
            u[0] = u0
            phi0, dphi0 = self._dendritic_activation_branch(0, u0)
            dphi[0] = dphi0
            i_soma = phi0
        else:
            active_idx = torch.nonzero(p > 1e-6, as_tuple=False).reshape(-1)
            for idx_t in active_idx:
                b = int(idx_t.item())
                ub = torch.mv(self.W_eff[b], psi_proj) - self.b_dend[b]
                u[b] = ub
                phib, dphib = self._dendritic_activation_branch(b, ub)
                dphi[b] = dphib
                i_soma = i_soma + p[b] * phib

        if not self.cfg.ablation.ablate_homeostasis:
            i_soma = i_soma - self.theta_soma
        y_int, dy = self._somatic_activation(i_soma)

        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis and self.out_scale_initialized:
            out_scale = torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
            y = y_int * out_scale
        else:
            y = y_int

        if 0 < self.cfg.k_wta < self.d_out:
            _, top_idx = torch.topk(torch.abs(y), k=self.cfg.k_wta)
            mask = torch.zeros_like(y)
            mask[top_idx] = 1.0
            y = y * mask
            dy = dy * mask

        if not self.cfg.ablation.ablate_homeostasis and self.cfg.eta_theta > 0.0:
            aa = self.cfg.alpha_act
            self.ema_act.mul_(1.0 - aa).add_(aa * y_int)
            self.theta_soma.add_(self.cfg.eta_theta * (self.ema_act - self.cfg.target_activity))

        self._last_x_norm = x_norm
        self._last_psi = psi_proj
        self._last_ctx_hat = ctx_hat
        self._last_p = p
        self._last_u = u
        self._last_dphi = dphi
        self._last_dy = dy
        self._last_y = y
        self._last_max_sim = max_sim
        return y.clone()

    @torch.no_grad()
    def _maybe_recruit_branch(self, delta_out: torch.Tensor) -> bool:
        if self.n_branches <= 1 or bool(self.recruited.all().item()):
            return False

        err_sq = float(torch.mean(delta_out * delta_out).item())
        if self.prev_delta_out is not None:
            n_curr = float(torch.linalg.norm(delta_out).item())
            n_prev = float(torch.linalg.norm(self.prev_delta_out).item())
            coh = float(torch.dot(delta_out, self.prev_delta_out).item() / (n_curr * n_prev)) if (n_curr > 1e-8 and n_prev > 1e-8) else 0.0
        else:
            coh = 0.0
        self.prev_delta_out = delta_out.clone()

        ae = self.cfg.alpha_err_ema
        self.ema_err_sq = (1.0 - ae) * self.ema_err_sq + ae * err_sq
        self.ema_err_coh = (1.0 - ae) * self.ema_err_coh + ae * coh

        if self._last_ctx_hat is None or float(torch.linalg.norm(self._last_ctx_hat).item()) < 1e-6:
            self.sustained_recruit = 0
            return False
        if self.steps_since_recruit < self.cfg.recruit_cooldown:
            return False

        novel_trigger = (self._last_max_sim < self.cfg.theta_novel) and (err_sq > self.cfg.theta_err_novel)
        coherent_trigger = (self.ema_err_sq > self.cfg.theta_err) and (self.ema_err_coh > self.cfg.theta_coh)
        if novel_trigger or coherent_trigger:
            self.sustained_recruit += 1
            if self.sustained_recruit >= self.cfg.recruit_sustained:
                free_idx = int(torch.nonzero(~self.recruited, as_tuple=False)[0].item())
                self.recruited[free_idx] = True
                self.prototypes[free_idx].copy_(self._last_ctx_hat)
                self.sustained_recruit = 0
                self.steps_since_recruit = 0
                self.recruit_events.append(self.step_count)
                self._last_p, self._last_max_sim = self._compute_routing(self._last_ctx_hat)
                return True
        else:
            self.sustained_recruit = max(0, self.sustained_recruit - 1)
        return False

    @torch.no_grad()
    def _learn_local_batch(self, delta_out: torch.Tensor, mod: float = 1.0) -> torch.Tensor:
        """Native vectorized GPU/CPU local plasticity for mini-batch (B, d_out).

        When P_prec is enabled, uses the exact Woodbury matrix identity:
            K = P @ X^T, S = I + X @ K, G = solve(S, K^T)^T,
            W += (E^T @ G^T), P -= G @ K^T,
        which is algebraically identical to sequential single-sample Sherman-Morrison.
        """
        delta_mat = delta_out.to(device=self.W_slow.device, dtype=self.W_slow.dtype)
        batch_sz = int(delta_mat.shape[0])
        inv_b = 1.0 / float(max(1, batch_sz))

        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis:
            obs_target = self._last_y + delta_mat
            if not self.out_scale_initialized:
                self.ema_out_var.copy_(torch.clamp((obs_target * obs_target).mean(dim=0), min=1e-4))
                self.out_scale_initialized = True
                delta_eff = delta_mat / torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
            else:
                out_scale = torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
                delta_eff = torch.clamp(delta_mat / out_scale, -8.0, 8.0)
                ao_eff = float(min(1.0, 1.0 - (1.0 - self.cfg.alpha_out_scale) ** batch_sz))
                self.ema_out_var.mul_(1.0 - ao_eff).add_(ao_eff * (obs_target * obs_target).mean(dim=0))
        else:
            delta_eff = delta_mat

        p = self._last_p
        psi = self._last_psi
        ctx_hat = self._last_ctx_hat
        delta_soma = delta_eff * self._last_dy

        if self.G_gram is not None and self.H_cross is not None:
            y_tgt = self._last_y + delta_eff
            self.G_gram.addmm_(psi.T, psi)
            self.H_cross.addmm_(psi.T, y_tgt)
            apical_prev = delta_soma @ self.B_fb.T
            return apical_prev

        if self.P_prec is not None:
            k_mat = self.P_prec @ psi.T  # (d_psi, B)
            s_mat = torch.eye(batch_sz, device=psi.device, dtype=psi.dtype) + psi @ k_mat  # (B, B)
            g_mat = torch.linalg.solve(s_mat, k_mat.T).T  # (d_psi, B)
            self.P_prec.sub_(g_mat @ k_mat.T)
            self.P_prec.copy_(0.5 * (self.P_prec + self.P_prec.T))
            psi_norm = None
        else:
            g_mat = None
            psi_sq = (psi * psi).sum(dim=1, keepdim=True)
            denom = (psi_sq + 1e-6) if self.cfg.use_unit_key_norm else (1.0 + psi_sq)
            psi_norm = psi / denom

        rec_idx = torch.nonzero(self.recruited, as_tuple=False).reshape(-1)
        for idx_t in rec_idx:
            b = int(idx_t.item())
            pb_col = p[:, b : b + 1]
            pb_mean = float(pb_col.mean().item())
            if pb_mean <= 1e-6:
                continue

            if self.n_branches > 1:
                w_ctx = (pb_col * ctx_hat).mean(dim=0)
                upd = self.prototypes[b] + self.cfg.eta_proto * (w_ctx - pb_mean * self.prototypes[b])
                if self.cfg.boids_repulsion > 0.0:
                    for ob_t in rec_idx:
                        ob = int(ob_t.item())
                        if ob != b:
                            sim_bo = float(torch.dot(self.prototypes[b], self.prototypes[ob]).item())
                            if sim_bo > 0.2:
                                upd = upd - self.cfg.boids_repulsion * sim_bo * self.prototypes[ob]
                nu = float(torch.linalg.norm(upd).item())
                if nu > 1e-8:
                    self.prototypes[b].copy_(upd / nu)

            delta_b = (pb_col * delta_soma) * self._last_dphi[:, b, :]

            if self.P_prec is not None:
                dW_b = (mod * self.cfg.eta_fast) * (delta_b.T @ g_mat.T)
            else:
                dW_b = (mod * self.cfg.eta_fast * inv_b) * (delta_b.T @ psi_norm)

            if self.cfg.ablation.ablate_fast_delta:
                scale_slow = self.cfg.eta_slow / max(1e-12, self.cfg.eta_fast)
                self.W_slow[b].add_(scale_slow * dW_b)
                self.W_eff[b].copy_(self.W_slow[b])
            else:
                if self.cfg.gamma_forget > 0.0:
                    self.W_fast[b].mul_(1.0 - pb_mean * self.cfg.gamma_forget)
                self.W_fast[b].add_(dW_b)

                if not self.cfg.ablation.ablate_cascade and self.cfg.eta_consol > 0.0:
                    at = self.cfg.alpha_tag
                    ema_b = self.ema_dW_fast[b]
                    n_dw = float(torch.linalg.norm(dW_b).item())
                    n_ema = float(torch.linalg.norm(ema_b).item())
                    cos_tag = max(0.0, float(torch.sum(dW_b * ema_b).item() / (n_dw * n_ema))) if (n_dw > 1e-9 and n_ema > 1e-9) else 0.0
                    self.ema_dW_fast[b].mul_(1.0 - at).add_(at * dW_b)
                    self.tag_coherence[b] = (1.0 - at) * self.tag_coherence[b] + at * cos_tag
                    self.W_slow[b].add_(
                        pb_mean
                        * self.cfg.eta_consol
                        * self.tag_coherence[b]
                        * (self.W_fast[b] - self.cfg.lambda_eq * self.W_slow[b])
                    )

                self.W_eff[b].copy_(self.W_slow[b] + self.cfg.kappa_fast * self.W_fast[b])

            if self.P_prec is None:
                self.b_dend[b].sub_((mod * self.cfg.eta_fast * 0.25 * inv_b) * delta_b.sum(dim=0))

            if self.cfg.use_nonmonotonic and not self.cfg.ablation.ablate_nonmonotonic:
                ub = self._last_u[:, b, :]
                rho = torch.sigmoid(self.rho_logit[b])
                theta_off = self.theta_on[b] + F.softplus(self.delta_theta_raw[b])
                beta = self.cfg.beta_dend
                s_on = torch.sigmoid(beta * (ub - self.theta_on[b]))
                s_off = torch.sigmoid(beta * (ub - theta_off))
                bump = s_on - s_off
                t_u = torch.tanh(ub)
                err_b = pb_col * delta_soma
                d_rho = (err_b * (bump - t_u)).mean(dim=0) * rho * (1.0 - rho)
                d_ton = (err_b * (-beta * (s_on * (1.0 - s_on) - s_off * (1.0 - s_off)))).mean(dim=0) * rho
                self.rho_logit[b].add_(mod * self.cfg.eta_dend_params * d_rho).clamp_(-4.0, 4.0)
                self.theta_on[b].add_(mod * self.cfg.eta_dend_params * d_ton).clamp_(-2.0, 3.0)

        apical_prev = delta_soma @ self.B_fb.T
        if self.cfg.ablation.shuffle_apical:
            perm = torch.randperm(self.d_in, generator=self._torch_rng).to(apical_prev.device)
            apical_prev = apical_prev[:, perm]

        if not self.cfg.ablation.freeze_random_feedback:
            x_n = self._last_x_norm
            x_sq = (x_n * x_n).sum(dim=1, keepdim=True)
            denom_fb = 1.0 + x_sq
            decay_fb = min(0.5, float((self.cfg.lambda_fb * x_sq / denom_fb).mean().item()))
            self.B_fb.add_((mod * self.cfg.eta_fb * inv_b) * ((x_n / denom_fb).T @ delta_soma) - decay_fb * self.B_fb)

        return apical_prev

    @torch.no_grad()
    def learn_local(self, delta_out: torch.Tensor, mod: float = 1.0) -> torch.Tensor:
        """Single-phase sparse O(k_top) local dendritic plasticity + cascade consolidation."""
        if delta_out.ndim == 2:
            return self._learn_local_batch(delta_out, mod=mod)
        delta_vec = delta_out.to(device=self.W_slow.device, dtype=self.W_slow.dtype).reshape(self.d_out)

        if self.cfg.adapt_output_scale and not self.cfg.ablation.ablate_homeostasis:
            obs_target = self._last_y + delta_vec
            if not self.out_scale_initialized:
                self.ema_out_var.copy_(torch.clamp(obs_target * obs_target, min=1e-4))
                self.out_scale_initialized = True
                delta_eff = delta_vec / torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
            else:
                out_scale = torch.sqrt(self.ema_out_var + self.cfg.eps_homeo)
                delta_eff = torch.clamp(delta_vec / out_scale, -8.0, 8.0)
                ao = self.cfg.alpha_out_scale
                self.ema_out_var.mul_(1.0 - ao).add_(ao * (obs_target * obs_target))
        else:
            delta_eff = delta_vec

        self._maybe_recruit_branch(delta_eff)

        p = self._last_p
        psi = self._last_psi
        ctx_hat = self._last_ctx_hat
        delta_soma = delta_eff * self._last_dy

        if self.P_prec is not None:
            p_psi = torch.mv(self.P_prec, psi)
            denom_p = 1.0 + float(torch.dot(psi, p_psi).item())
            psi_norm = p_psi / denom_p
            self.P_prec.sub_(torch.outer(psi_norm, p_psi))
        else:
            psi_sq = float(torch.dot(psi, psi).item())
            denom = (psi_sq + 1e-6) if self.cfg.use_unit_key_norm else (1.0 + psi_sq)
            psi_norm = psi / denom
        ctx_norm = float(torch.linalg.norm(ctx_hat).item()) if self.n_branches > 1 else 0.0

        active_idx = torch.nonzero(p > 1e-6, as_tuple=False).reshape(-1)
        for idx_t in active_idx:
            b = int(idx_t.item())
            pb = float(p[b].item())

            if self.n_branches > 1 and ctx_norm > 1e-8:
                upd = self.prototypes[b] + self.cfg.eta_proto * pb * (ctx_hat - self.prototypes[b])
                if self.cfg.boids_repulsion > 0.0:
                    rec_all = torch.nonzero(self.recruited, as_tuple=False).reshape(-1)
                    for ob_t in rec_all:
                        ob = int(ob_t.item())
                        if ob != b:
                            sim_bo = float(torch.dot(self.prototypes[b], self.prototypes[ob]).item())
                            if sim_bo > 0.2:
                                upd = upd - self.cfg.boids_repulsion * sim_bo * self.prototypes[ob]
                nu = float(torch.linalg.norm(upd).item())
                if nu > 1e-8:
                    self.prototypes[b].copy_(upd / nu)

            delta_b = (pb * delta_soma) * self._last_dphi[b]

            if self.cfg.ablation.ablate_fast_delta:
                self.W_slow[b].add_((mod * self.cfg.eta_slow) * torch.outer(delta_b, psi_norm))
                self.W_eff[b].copy_(self.W_slow[b])
            else:
                dW_b = (mod * self.cfg.eta_fast) * torch.outer(delta_b, psi_norm)
                if self.cfg.gamma_forget > 0.0:
                    self.W_fast[b].mul_(1.0 - pb * self.cfg.gamma_forget)
                self.W_fast[b].add_(dW_b)

                if not self.cfg.ablation.ablate_cascade and self.cfg.eta_consol > 0.0:
                    at = self.cfg.alpha_tag
                    ema_b = self.ema_dW_fast[b]
                    n_dw = float(torch.linalg.norm(dW_b).item())
                    n_ema = float(torch.linalg.norm(ema_b).item())
                    cos_tag = max(0.0, float(torch.sum(dW_b * ema_b).item() / (n_dw * n_ema))) if (n_dw > 1e-9 and n_ema > 1e-9) else 0.0
                    self.ema_dW_fast[b].mul_(1.0 - at).add_(at * dW_b)
                    self.tag_coherence[b] = (1.0 - at) * self.tag_coherence[b] + at * cos_tag
                    self.W_slow[b].add_(
                        pb
                        * self.cfg.eta_consol
                        * self.tag_coherence[b]
                        * (self.W_fast[b] - self.cfg.lambda_eq * self.W_slow[b])
                    )

                self.W_eff[b].copy_(self.W_slow[b] + self.cfg.kappa_fast * self.W_fast[b])

            if self.P_prec is None:
                self.b_dend[b].sub_((mod * self.cfg.eta_fast * 0.25) * delta_b)

            if self.cfg.use_nonmonotonic and not self.cfg.ablation.ablate_nonmonotonic:
                ub = self._last_u[b]
                rho = torch.sigmoid(self.rho_logit[b])
                theta_off = self.theta_on[b] + F.softplus(self.delta_theta_raw[b])
                beta = self.cfg.beta_dend
                s_on = torch.sigmoid(beta * (ub - self.theta_on[b]))
                s_off = torch.sigmoid(beta * (ub - theta_off))
                bump = s_on - s_off
                t_u = torch.tanh(ub)
                err_b = pb * delta_soma
                d_rho = err_b * (bump - t_u) * rho * (1.0 - rho)
                d_ton = err_b * rho * (-beta * (s_on * (1.0 - s_on) - s_off * (1.0 - s_off)))
                self.rho_logit[b].add_(mod * self.cfg.eta_dend_params * d_rho).clamp_(-4.0, 4.0)
                self.theta_on[b].add_(mod * self.cfg.eta_dend_params * d_ton).clamp_(-2.0, 3.0)

        apical_prev = torch.mv(self.B_fb, delta_soma)
        if self.cfg.ablation.shuffle_apical:
            perm = torch.randperm(self.d_in, generator=self._torch_rng).to(apical_prev.device)
            apical_prev = apical_prev[perm]

        if not self.cfg.ablation.freeze_random_feedback:
            x_n = self._last_x_norm
            x_sq = float(torch.dot(x_n, x_n).item())
            denom_fb = 1.0 + x_sq
            decay_fb = min(0.5, self.cfg.lambda_fb * x_sq / denom_fb)
            self.B_fb.add_((mod * self.cfg.eta_fb / denom_fb) * torch.outer(x_n, delta_soma) - decay_fb * self.B_fb)

        return apical_prev

    @torch.no_grad()
    def consolidate_gram(self) -> None:
        """Factorize accumulated Gram matrix X^T X + lambda*I via Cholesky and update W_slow/W_eff."""
        if self.G_gram is None or self.H_cross is None:
            return
        L = torch.linalg.cholesky(self.G_gram)
        w_sol = torch.cholesky_solve(self.H_cross, L).T
        self.W_slow[0].copy_(w_sol)
        self.W_eff[0].copy_(w_sol)

    def ops_report(self) -> OpsReport:
        return count_layer_ops(self.cfg)


class DMCNTorchNetwork(nn.Module):
    """Multi-layer GPU/CUDA-ready DMCN network with local apical-basal learning."""

    def __init__(
        self,
        layer_configs: Sequence[DMCNConfig],
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DMCNTorchLayer(cfg, device=device, dtype=dtype) for cfg in layer_configs])

    @torch.no_grad()
    def reset_temporal_state(self) -> None:
        for layer in self.layers:
            layer.reset_temporal_state()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, dt=dt)
        return h

    @torch.no_grad()
    def learn_local(self, target: torch.Tensor, mod: float = 1.0) -> Tuple[torch.Tensor, float]:
        y_hat = self.layers[-1]._last_y
        tgt = target.to(device=y_hat.device, dtype=y_hat.dtype).reshape(y_hat.shape)
        err = tgt - y_hat
        mse = float(torch.mean(err * err).item())
        delta = err
        for layer in reversed(self.layers):
            delta = layer.learn_local(delta, mod=mod)
        return y_hat.clone(), mse

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
