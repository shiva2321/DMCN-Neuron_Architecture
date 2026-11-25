"""Configuration and ablation controls for the Dendritic Multiplexed Cascade Neuron (DMCN)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """Explicit per-pillar ablation switches for falsifying every DMCN mechanism."""

    ablate_homeostasis: bool = False
    ablate_spine_traces: bool = False
    ablate_nonmonotonic: bool = False
    ablate_fast_delta: bool = False
    ablate_cascade: bool = False
    freeze_random_feedback: bool = False
    shuffle_apical: bool = False
    ablate_eligibility: bool = False
    ablate_precision: bool = False


@dataclass
class DMCNConfig:
    """Hyperparameters for a single DMCNLayer (NumPy and PyTorch backends)."""

    d_in: int = 8
    d_out: int = 4
    n_branches: int = 4
    top_k_branches: int = 1
    n_spine_scales: int = 4
    tau_min: float = 1.0
    tau_max: float = 16.0

    # Pillar 1: Causal Homeostasis
    alpha_homeo: float = 0.05
    eps_homeo: float = 1e-4
    eta_theta: float = 0.005
    alpha_act: float = 0.05
    target_activity: float = 0.0
    adapt_output_scale: bool = False
    alpha_out_scale: float = 0.08
    center_homeostasis: bool = False

    # Pillar 3: Routing, Active Dendrites & Dual-Cascade Memory
    beta_route: float = 8.0
    eta_proto: float = 0.05
    boids_repulsion: float = 0.0
    k_wta: int = 0
    theta_novel: float = 0.25
    theta_err_novel: float = 0.08
    theta_err: float = 0.20
    theta_coh: float = 0.35
    recruit_sustained: int = 4
    recruit_cooldown: int = 15
    alpha_err_ema: float = 0.15

    use_nonmonotonic: bool = False
    beta_dend: float = 4.0
    init_rho: float = 0.5
    init_theta_on: float = 0.4
    init_delta_theta: float = 1.2
    eta_dend_params: float = 0.05

    soma_activation: str = "tanh"  # "linear", "tanh", "sigmoid"
    kappa_fast: float = 1.0
    eta_fast: float = 0.15
    eta_slow: float = 0.02
    gamma_forget: float = 0.02
    eta_consol: float = 0.05
    lambda_eq: float = 1.0
    alpha_tag: float = 0.15
    weight_init_scale: float = 0.15
    use_sparse_ternary: bool = False
    ternary_density: float = 0.05
    use_unit_key_norm: bool = False
    use_precision_plasticity: bool = False
    precision_ridge: float = 1.0
    precision_update_mode: str = "woodbury"  # "woodbury" (exact per-step) | "gram_cholesky" (fast checkpoint)

    # Pillar 4: Apical-Basal & 3-Factor RL
    eta_fb: float = 0.05
    lambda_fb: float = 0.01
    gamma_elig: float = 0.90
    eta_rl: float = 0.10

    seed: int = 42
    ablation: AblationConfig = field(default_factory=AblationConfig)
