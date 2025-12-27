from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from src.models.layers.convnext_block import ConvNeXtBlock, ConvNeXtBlockConfig

Tensor = torch.Tensor

@dataclass(frozen=True)
class EncoderStageConfig:
    """
    Config for one encoder stage in ConvNeXt-UNet.

    Attributes
    ----------
    in_dim : int
        Input channels.
    out_dim : int
        Output channels.
    depth : int
        Number of ConvNeXt blocks.
    drop_path_rates : Sequence[float]
        Per-block DropPath rates, length == depth.
    norm_eps : float
        Epsilon for LayerNorm inside ConvNeXtBlock.
    is_first : bool
        If True, do not downsample spatially (stride=1).
    """
    in_dim: int
    out_dim: int
    depth: int
    drop_path_rates: Sequence[float]
    norm_eps: float = 1e-6
    is_first: bool = False

@dataclass(frozen=True)
class DecoderStageConfig:
    """
    Config for one decoder stage in ConvNeXt-UNet.

    Attributes
    ----------
    in_dim : int
        Channels of low-resolution input feature.
    skip_dim : int
        Channels of skip connection.
    out_dim : int
        Output channels after fusion.
    depth : int
        Number of ConvNeXt blocks.
    norm_eps : float
        Epsilon for LayerNorm in ConvNeXt blocks.
    """
    in_dim: int
    skip_dim: int
    out_dim: int
    depth: int
    norm_eps: float = 1e-6


class EncoderStage(nn.Module):
    """
    One encoder stage: optional downsampling + ConvNeXt blocks.

    If cfg.is_first is True:
        - no spatial downsample (stride=1), only channel projection.
    Else:
        - stride=2 to downsample H,W by 2.
    """

    def __init__(self, cfg: EncoderStageConfig) -> None:
        super().__init__()

        assert len(cfg.drop_path_rates) == cfg.depth, (
            "[STAGE:encoder] len(drop_path_rates) must equal depth."
        )
        self.cfg = cfg

        if cfg.is_first:
            self.downsample = nn.Conv2d(
                in_channels=cfg.in_dim,
                out_channels=cfg.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.downsample = nn.Conv2d(
                in_channels=cfg.in_dim,
                out_channels=cfg.out_dim,
                kernel_size=2,
                stride=2,
                padding=0,
            )

        blocks: List[nn.Module] = []
        for i in range(cfg.depth):
            block_cfg = ConvNeXtBlockConfig(
                dim=cfg.out_dim,
                drop_path=float(cfg.drop_path_rates[i]),
                norm_eps=float(cfg.norm_eps),
            )
            block = ConvNeXtBlock(block_cfg)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)


    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, in_dim, H, W] -> out : [B, out_dim, H', W'].
        """

        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DecoderStage(nn.Module):
    """
    One decoder stage: upsample + concat skip + ConvNeXt blocks.

    Inputs
    ------
    x    : [B, C_in,  H,  W]   (low-res)
    skip : [B, C_skip, 2H, 2W] (encoder skip)

    Output
    ------
    out  : [B, C_out, 2H, 2W]
    """

    def __init__(self, cfg: DecoderStageConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.upsample = nn.ConvTranspose2d(
            in_channels=cfg.in_dim,
            out_channels=cfg.out_dim,
            kernel_size=2,
            stride=2,
        )

        self.fuse = nn.Conv2d(
            in_channels=cfg.out_dim + cfg.skip_dim,
            out_channels=cfg.out_dim,
            kernel_size=1,
        )

        blocks: List[nn.Module] = []
        for _ in range(cfg.depth):
            block_cfg = ConvNeXtBlockConfig(
                dim=cfg.out_dim,
                drop_path=0.0,
                norm_eps=cfg.norm_eps,
            )
            block = ConvNeXtBlock(block_cfg)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        x    : [B, C_in,  H,  W]
        skip : [B, C_skip, 2H, 2W]
        """
        assert x.shape[1] == self.cfg.in_dim, "[STAGE:decoder] channel dim mismatch before upsample"
        assert skip.shape[1] == self.cfg.skip_dim, "[STAGE:decoder] skip channel dim mismatch"

        x = self.upsample(x)

        assert x.shape[-2:] == skip.shape[-2:], (
            f"[STAGE:decoder] upsampled {x.shape[-2:]} != skip {skip.shape[-2:]}"
        )

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x
