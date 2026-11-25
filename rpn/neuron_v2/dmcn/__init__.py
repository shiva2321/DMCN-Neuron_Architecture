"""Dendritic Multiplexed Cascade Neuron (DMCN) unified package (NumPy + PyTorch)."""
from .adaptive_v3 import (
    FactorizedBilinearDMCNLayer,
    SelfGatingConfig,
    SelfGatingDMCNLayer,
    UnsupOjaDendriticLayer,
)
from .config import AblationConfig, DMCNConfig
from .numpy_core import DMCNLayer, DMCNNetwork, OpsReport, count_layer_ops
from .pdnc import (
    PDNCActorCritic,
    PDNCConfig,
    PDNCLayer,
    PDNCNetwork,
    PDNCTorchLayer,
    PDNCTorchNetwork,
    make_patch_receptive_field_mask,
)
from .swarm_boids import SwarmBoidNetwork, SwarmConfig
from .torch_module import DMCNTorchLayer, DMCNTorchNetwork

__all__ = [
    "AblationConfig",
    "DMCNConfig",
    "DMCNLayer",
    "DMCNNetwork",
    "DMCNTorchLayer",
    "DMCNTorchNetwork",
    "FactorizedBilinearDMCNLayer",
    "OpsReport",
    "PDNCActorCritic",
    "PDNCConfig",
    "PDNCLayer",
    "PDNCNetwork",
    "PDNCTorchLayer",
    "PDNCTorchNetwork",
    "SelfGatingConfig",
    "SelfGatingDMCNLayer",
    "SwarmBoidNetwork",
    "SwarmConfig",
    "UnsupOjaDendriticLayer",
    "count_layer_ops",
    "make_patch_receptive_field_mask",
]
