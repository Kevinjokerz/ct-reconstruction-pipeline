"""
recon_loss.py — Hybrid loss functions for CT reconstruction / denoising.

This module defines a configurable reconstruction loss combining:

- L1 (MAE)
- L2 (MSE)
- (1 - SSIM) or (1 - MS-SSIM)
- Total variation (TV) regularizer

The goal is to better preserve anatomical structures in low-dose CT
reconstruction than pure MSE, while remaining simple enough for
ablation studies and transfer to other datasets (e.g. LIDC).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from src.eval.metrics_recon import ssim2d, ms_ssim2d

@dataclass(frozen=True)
class ReconstructionLossConfig:
    """
    Configuration for the hybrid reconstruction loss.

    Parameters
    ----------
    l1_weight :
        Weight for the mean absolute error (L1) term.
    mse_weight :
        Weight for the mean squared error (MSE) term.
    ssim_weight :
        Weight for the structural dissimilarity term,
        implemented as ``ssim_weight * (1 - SSIM)`` or
        ``ssim_weight * (1 - MS-SSIM)``.
    tv_weight :
        Weight for the total variation (TV) penalty on the prediction.
        Set to zero to disable TV.
    data_range :
        Intensity range of the input/target images (e.g. 1.0 for [0, 1]).
    use_ms_ssim :
        If True, use multi-scale SSIM; otherwise single-scale SSIM.
    """
    l1_weight: float = 0.5
    mse_weight: float = 0.0
    ssim_weight: float = 0.5
    tv_weight: float = 0.0
    data_range: float = 1.0
    use_ms_ssim: bool = False

class ReconstructionLoss:
    """
    Hybrid loss for low-dose CT reconstruction.

    Example
    -------
    >>> cfg = ReconstructionLossConfig(l1_weight=0.5, ssim_weight=0.5)
    >>> criterion = ReconstructionLoss(cfg)
    >>> loss = criterion(pred, target)
    """

    def __init__(self, cfg: ReconstructionLossConfig):
        self.cfg = cfg
    
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"l1={self.cfg.l1_weight}, "
            f"mse={self.cfg.mse_weight}, "
            f"ssim={self.cfg.ssim_weight}, "
            f"tv={self.cfg.tv_weight}, "
            f"ms_ssim={self.cfg.use_ms_ssim}, "
            f"range={self.cfg.data_range})"
        )
    # ============================================================
    # Internal helpers
    # ============================================================

    def _validate_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """
        Validate prediction and target tensors for reconstruction loss.

        Parameters
        ----------
        pred, target :
            FloatTensor [B, C, H, W]. C is typically 1 for CT.

        Raises
        ------
        AssertionError
            If shapes, dtype, or basic assumptions are violated.
        """
        assert pred.shape == target.shape, f"[LOSS] pred shape {pred.shape} != target shape {target.shape}"
        assert pred.ndim == 4, f"[LOSS] expected 4D tensors, got {pred.ndim}D"
        assert pred.dtype == target.dtype, "[LOSS] pred/target dtype mismatch"
        assert pred.dtype.is_floating_point, (
            "[LOSS] expected floating-point tensors."
        )

    # ---------------------------
    # Individual loss components
    # ---------------------------

    def _l1 (
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean absolute error (L1) term.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        diff = pred - target
        loss = diff.abs().mean()
        return loss

    def _mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean squared error (L2 / MSE) term.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """

        diff = pred - target
        loss = (diff ** 2).mean()
        return loss

    def _ssim_term(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Structural dissimilarity term (1 - SSIM or 1 - MS-SSIM).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor, averaged over the batch.
        """
        cfg = self.cfg

        pred_clamped = torch.clamp(pred, 0.0, cfg.data_range)
        target_clamped = torch.clamp(target, 0.0, cfg.data_range)

        if cfg.use_ms_ssim:
            ms_ssim_2d = ms_ssim2d(pred_clamped, target_clamped, data_range=cfg.data_range)
            loss = 1.0 - ms_ssim_2d
        else:
            ssim_score = ssim2d(pred_clamped, target_clamped, data_range=cfg.data_range)
            loss = 1.0 - ssim_score
        return loss

    def _tv(
        self,
        pred: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Isotropic total variation (TV) regularizer on the prediction.

        TV encourages piecewise-smooth reconstructions while still
        allowing sharp edges.

        Parameters
        ----------
        pred :
            FloatTensor [B, C, H, W].
        eps :
            Small constant for numerical stability.

        Returns
        -------
        torch.Tensor
            Scalar TV regularizer.
        """
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        dx_c = dx[:, :, :-1, :]
        dy_c = dy[:, :, :, :-1]

        tv_map = torch.sqrt(dx_c**2 + dy_c**2 + eps)
        tv = tv_map.mean()

        return tv

    # ============================================================
    # Public API
    # ============================================================

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the hybrid reconstruction loss.

        Parameters
        ----------
        pred, target :
            FloatTensor [B, C, H, W], roughly in [0, data_range].

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        self._validate_inputs(pred, target)

        cfg = self.cfg
        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

        if cfg.l1_weight > 0.0:
            l1 = self._l1(pred, target)
            loss = loss + cfg.l1_weight * l1

        if cfg.mse_weight > 0.0:
            mse = self._mse(pred, target)
            loss = loss + cfg.mse_weight * mse

        if cfg.ssim_weight > 0.0:
            ssim_loss = self._ssim_term(pred, target)
            loss = loss + cfg.ssim_weight * ssim_loss

        if cfg.tv_weight > 0.0:
            tv = self._tv(pred)
            loss = loss + cfg.tv_weight * tv

        assert loss.ndim == 0, f"[Loss] Expected scalar loss, got {loss.shape}"
        return loss
