from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn

from src.models.layers import (
    EncoderStage,
    EncoderStageConfig,
    DecoderStage,
    DecoderStageConfig,
    ConvNeXtBlock,
    ConvNeXtBlockConfig
)

Tensor = torch.Tensor

# =========================
# Config
# =========================
@dataclass(frozen=True)
class UnetConvNeXtConfig:
    """
    ConvNeXt-UNet backbone config for low-dose CT reconstruction.

    Assumptions
    -----------
    - Input:  [B, 1, H, W], float32, normalized to [0, 1].
    - Output: [B, 1, H, W], same spatial size.

    Attributes
    ----------
    in_channels, out_channels :
        Usually 1 for CT grayscale.
    depths :
        Number of ConvNeXt blocks per encoder stage.
    dims :
        Number of channels per stage.
    """
    in_channels: int = 1
    out_channels: int = 1

    depths: Tuple[int, ...] = (2, 2, 2, 2)
    dims: Tuple[int, ...] = (64, 128, 256, 512)

    drop_path_rate: float = 0.1
    norm_eps: float = 1e-6

    bottleneck_depth: int = 2
    bottleneck_drop_path_rate: float | None = None

    use_checkpoint: bool = False
    final_activation: str | None = "sigmoid"        # "sigmoid" / "tanh" / None


class UNetConvNeXt(nn.Module):
    """
    ConvNeXt-UNet backbone.

    x : [B, in_channels, H, W] -> y : [B, out_channels, H, W]
    """

    def __init__(self, cfg: UnetConvNeXtConfig):
        super().__init__()
        self.cfg = cfg

        assert len(cfg.depths) == len(cfg.dims) == 4, "[INIT:UNeXt] depths and dims must have length 4 (4 stages)"

        # ---- Stem ----
        self.stem = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ---- Encoder ----
        total_blocks = sum(cfg.depths)
        dpr = torch.linspace(0, cfg.drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        encoders: List[nn.Module] = []
        in_dim = cfg.dims[0]

        for stage_idx, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
            stage_dpr = dpr[dp_idx: dp_idx + depth]
            dp_idx += depth

            enc_cfg = EncoderStageConfig(
                in_dim=in_dim,
                out_dim=dim,
                depth=depth,
                drop_path_rates=stage_dpr,
                norm_eps=cfg.norm_eps,
                is_first=(stage_idx == 0),
            )
            enc = EncoderStage(enc_cfg)
            encoders.append(enc)
            in_dim = dim

        self.encoders = nn.ModuleList(encoders)

        # ---- Bottleneck ----
        bn_depth = self.cfg.bottleneck_depth
        assert bn_depth >= 1, f"[INIT:UNeXt] bottleneck depth must be >= 1."

        if cfg.bottleneck_drop_path_rate is not None:
            bn_dpr = torch.linspace(
                0.0,
                cfg.bottleneck_drop_path_rate,
                bn_depth,
            ).tolist()
        else:
            bn_dpr = [0.0] * bn_depth

        bn_blocks: List[nn.Module] = []
        for i in range(bn_depth):
            block_cfg = ConvNeXtBlockConfig(
                dim=in_dim,
                drop_path=float(bn_dpr[i]),
                norm_eps=cfg.norm_eps,
            )
            block = ConvNeXtBlock(block_cfg)
            bn_blocks.append(block)
        self.bottleneck = nn.Sequential(*bn_blocks)

        # ---- Decoder ----
        rev_dims = list(cfg.dims[::-1])
        rev_depths = list(cfg.depths[::-1])

        decoders: List[nn.Module] = []
        for i in range(len(rev_dims) - 1):
            in_dim_dec = rev_dims[i]
            skip_dim = rev_dims[i + 1]
            out_dim_dec = rev_dims[i + 1]

            dec_cfg = DecoderStageConfig(
                in_dim=in_dim_dec,
                skip_dim=skip_dim,
                out_dim=out_dim_dec,
                depth=rev_depths[i],
                norm_eps=cfg.norm_eps,
            )
            dec = DecoderStage(dec_cfg)
            decoders.append(dec)

        self.decoders = nn.ModuleList(decoders)

        # ---- Head ----
        self.head = nn.Conv2d(
            in_channels=cfg.dims[0],
            out_channels=cfg.out_channels,
            kernel_size=1,
        )

        if cfg.final_activation == "sigmoid":
            self._act = nn.Sigmoid()
        elif cfg.final_activation == "tanh":
            self._act = nn.Tanh()
        else:
            self._act = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, in_channels, H, W] -> out : [B, out_channels, H, W]
        """
        assert x.ndim == 4, "[FORWARD:UNeXt] x must be [B, in_channels, H, W]."
        assert x.shape[1] == self.cfg.in_channels, "[FORWARD:UNeXt] in_channels mismatch"

        # ---- Stem ----
        h = self.stem(x)

        # ---- Encoder ----
        skips: List[Tensor] = []
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # ---- Bottleneck ----
        h = self.bottleneck(h)

        # ---- Decoder (use reversed skips except deepest) ----
        for dec_idx, dec in enumerate(self.decoders):
            skip = skips[-(dec_idx + 2)]
            h = dec(h, skip)

        # ---- Head + activation ----
        out = self.head(h)
        out = self._act(out)
        return out
