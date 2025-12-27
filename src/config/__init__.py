"""
Central exports for configuration schemas and loaders.

This module simply re-exports the two main config entry points:

- PrepConfig / load_prep_config   : data preparation (LoDoPaB + LIDC)
- CrossDomainTrainConfig / load_train_config : cross-domain training
"""

from __future__ import annotations

from .prep import PrepConfig, load_config as load_prep_config
from .train import CrossDomainTrainConfig, load_train_config

__all__ = [
    "PrepConfig",
    "load_prep_config",
    "CrossDomainTrainConfig",
    "load_train_config"
]