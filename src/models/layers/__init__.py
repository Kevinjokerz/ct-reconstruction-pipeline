"""
Building blocks ("layers") for CT reconstruction models.

This subpackage exposes:
- Regularization layers (DropPath)
- ConvNeXt building blocks
- UNeXt encoder/decoder stages
"""

from __future__ import annotations

from .drop_path import DropPath, DropPathConfig
from .convnext_block import ConvNeXtBlock, ConvNeXtBlockConfig
from .stages import (
    EncoderStage,
    EncoderStageConfig,
    DecoderStage,
    DecoderStageConfig,
)


__all__ = [
    # DropPath
    "DropPath",
    "DropPathConfig",

    # ConvNeXt
    "ConvNeXtBlock",
    "ConvNeXtBlockConfig",

    # UNeXt stages
    "EncoderStage",
    "EncoderStageConfig",
    "DecoderStage",
    "DecoderStageConfig",
]