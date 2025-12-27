from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.models.layers.drop_path import DropPath, DropPathConfig

Tensor = torch.Tensor

@dataclass(frozen=True)
class ConvNeXtBlockConfig:
    """
    Config for a single ConvNeXt block.

    Attributes
    ----------
    dim : int
        Number of channels C.
    drop_path : float
        DropPath probability for the residual branch.
    norm_eps : float
        Epsilon for LayerNorm.
    layer_scale_init : float
        Initial value for LayerScale gamma.
    """
    dim: int
    drop_path: float = 0.0
    norm_eps: float = 1e-6
    layer_scale_init: float = 1e-6

class ConvNeXtBlock(nn.Module):
    """
    Simplified ConvNeXt block.

    Pipeline
    --------
    x -> depthwise conv (7x7) -> LayerNorm (channels-last)
      -> Linear(dim -> 4*dim) -> GELU -> Linear(4*dim -> dim)
      -> LayerScale -> DropPath -> + residual
    """

    def __init__(self, cfg: ConvNeXtBlockConfig) -> None:
        super().__init__()

        self.cfg = cfg
        dim = cfg.dim

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=True,
        )

        self.norm = nn.LayerNorm(normalized_shape=dim, eps=cfg.norm_eps)

        hidden_dim = 4 * dim
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()

        self.gamma = nn.Parameter(torch.ones(dim) * cfg.layer_scale_init, requires_grad=True)

        if cfg.drop_path > 0.0:
            dp_cfg = DropPathConfig(drop_prob=cfg.drop_path, inplace=False)
            self.drop_path: nn.Module = DropPath(dp_cfg)
        else:
            self.drop_path = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, C=dim, H, W] -> same shape.
        """
        assert x.dim() == 4, "[BLOCK:convNext] x must be 4D."
        assert x.shape[1] == self.cfg.dim, "[BLOCK:convNeXt] channel dim mismatch"

        residual = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x)
        out = residual + x

        assert out.shape == residual.shape, "[BLOCK:convNeXt] output shape mismatch"
        return out

