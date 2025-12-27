from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.unext import UNetConvNeXt, UnetConvNeXtConfig

Tensor = torch.Tensor

@dataclass(frozen=True)
class MoEReconConfig:
    """
    Mixture-of-Experts config for CT reconstruction.

    Assumptions
    -----------
    - domain_id:
        0 -> LoDoPaB
        1 -> LIDC
      (extensible to more domains).
    """

    num_experts: int = 2
    num_domains: int = 2

    in_channels: int = 1
    out_channels: int = 1

    gating_hidden_dim: int = 32
    gating_temp: float = 1.0

    expert_cfg: UnetConvNeXtConfig = UnetConvNeXtConfig()


class DomainGating(nn.Module):
    """
    Simple domain-based gating network.

    Input
    -----
    domain_ids : LongTensor [B] with values in {0, ..., num_domains-1}.

    Output
    ------
    weights : FloatTensor [B, E], softmax over experts.
    """

    def __init__(self, cfg: MoEReconConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Domain Embedding ----
        self.domain_emb = nn.Embedding(
            num_embeddings=cfg.num_domains,
            embedding_dim=cfg.gating_hidden_dim,
        )

        # ---- Small MLP -> logits over experts ----
        self.mlp = nn.Sequential(
            nn.Linear(cfg.gating_hidden_dim, cfg.gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.gating_hidden_dim, cfg.num_experts),
        )

    def forward(self, domain_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        domain_ids :
            [B], LongTensor.

        Returns
        -------
        weights :
            [B, num_experts]
        """

        assert domain_ids.dtype == torch.long, f"[FORWARD:Domain_gating] domain ids must be long tensor. (got {domain_ids.dtype})"
        assert domain_ids.ndim == 1, f"[FORWARD:Domain_gating] domain ids must be 1D [B]."

        domain_ids = domain_ids.to(next(self.parameters()).device)

        # ---- Embed domain IDs ----
        h = self.domain_emb(domain_ids)

        # ---- MLP -> logits ----
        logits = self.mlp(h)

        # ---- Temperature + softmax ----
        temp = max(self.cfg.gating_temp, 1e-6)
        logits = logits / temp
        weights = torch.softmax(logits, dim=-1)

        return weights


class MoEReconModel(nn.Module):
    """
    Mixture-of-Experts model for CT reconstruction.

    - `experts`: list of UNetConvNeXt (same architecture, different weights).
    - `gating`: DomainGating that returns per-sample expert weights.
    """

    def __init__(self, cfg: MoEReconConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Build experts ----
        experts = []
        for _ in range(cfg.num_experts):
            expert = UNetConvNeXt(cfg.expert_cfg)
            experts.append(expert)

        self.experts = nn.ModuleList(experts)

        # ---- Instantiate gating ----
        self.gating = DomainGating(cfg)


    def forward(
            self,
            x: Tensor,
            domain_ids: Tensor,
            *,
            return_all: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        x :
            [B, C, H, W]
        domain_ids :
            [B], LongTensor.

        Returns
        -------
        recon :
            [B, C, H, W]
        aux :
            - "gating_weights": [B, E]
            - "expert_outputs": [E, B, C, H, W] (if return_all=True)
        """

        assert x.ndim == 4, f"[FORWARD:MoEReconModel] x must be 4D tensor [B, C, H, W]."
        assert x.shape[1] == self.cfg.in_channels, f"[FORWARD:MoEReconModel] expected C={self.cfg.in_channels} (got {x.shape[1]})"
        assert domain_ids.shape[0] == x.shape[0], f"[FORWARD:MoEReconModel] domain_ids batch != x batch"
        if self.cfg.expert_cfg.use_checkpoint:
            assert x.requires_grad, f"[FORWARD:MoEReconModel] x.requires_grad=False but use_checkpoint=True"

        # ---- Run each expert on the same input ----
        outputs: List[Tensor] = []
        for expert in self.experts:
            if self.training and self.cfg.expert_cfg.use_checkpoint:
                y = grad_checkpoint(expert, x)
            else:
                y = expert(x)                                           # [B, C, H, W]
            outputs.append(y)

        expert_stack = torch.stack(outputs, dim=0)                  # [E, B, C, H, W]

        # ---- Compute gating weights ----
        weights = self.gating(domain_ids)                           # [B, E]
        assert weights.shape[1] == len(self.experts), "[FORWARD:MoEReconModel] gating E != num_experts."

        # ---- Weight mixture ----
        expert_b = expert_stack.permute(1, 0, 2, 3, 4)              # [B, E, C, H, W]
        w = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
        recon = (w * expert_b).sum(dim=1)                           # [B, C, H, W]

        aux: Dict[str, Tensor] = {"gating_weights": weights}
        if return_all:
            aux["expert_outputs"] = expert_stack

        return recon, aux
