from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class DropPathConfig:
    """
    Config for DropPath (stochastic depth).

    Attributes
    ----------
    drop_prob : float
        Probability of dropping the residual branch.
    inplace : bool
        If True, scale input in-place.
    """
    drop_prob: float = 0.0
    inplace: bool = False


class DropPath(nn.Module):
    """
    Stochastic depth / DropPath.

    Expected usage
    --------------
    out = x + drop_path(f(x))
    """
    def __init__(self, cfg: Optional[DropPathConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DropPathConfig()
        self.drop_prob: float = max(0.0, min(cfg.drop_prob, 1.0))
        self.inplace: bool = cfg.inplace
        assert 0.0 <= self.drop_prob <= 1.0, f"[INIT:drop_path] drop probability must be between 0 and 1."

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply DropPath to residual branch.

        Parameters
        ----------
        x : Tensor
            [B, ...] input tensor.

        Returns
        -------
        Tensor
            Same shape as x.
        """

        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        if keep_prob <= 0.0:
            return torch.zeros_like(x)

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()

        if self.inplace:
            x.mul_(binary_mask).div_(keep_prob)
            return x
        else:
            return x * binary_mask / keep_prob