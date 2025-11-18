"""
metrics_recon.py — PSNR / SSIM utilities for CT reconstruction.

These helpers are used in:
- training loops (to log PSNR / SSIM over validation sets),
- evaluation scripts (offline analysis, plots),
- ReconstructionLoss (for the (1 - SSIM) term).

Assumptions
-----------
- Inputs are 4D tensors of shape [B, C, H, W].
- CT slices are normalized to [0, data_range]; typically [0, 1].
"""

from __future__ import annotations
from typing import Literal, List
import torch
import torch.nn.functional as F

Reduction = Literal["none", "mean"]

def _validate_batch_tensors(pred: torch.Tensor, target: torch.Tensor) -> None:
    """
    Lightweight validation of prediction/target tensors.

    Parameters
    ----------
    pred, target :
        FloatTensor of shape [B, C, H, W].

    Notes
    -----
    This function is intentionally strict for early failure
    during metric computation; relax checks if needed.
    """
    assert pred.shape == target.shape, (
        f"[METRICS] pred shape {pred.shape} != target shape {target.shape}"
    )

    assert pred.ndim == 4, f"[METRICS] expected 4D tensors, got {pred.ndim}D"
    assert pred.dtype == target.dtype, "[METRICS] pred/target dtype mismatch"
    assert pred.dtype.is_floating_point, (
        "[METRICS] expected floating-point tensors."
    )

# ============================================================
# PSNR
# ============================================================

def psnr_batch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, reduction: Reduction = "mean") -> torch.Tensor:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) over a batch of images.

    PSNR is defined as:
        PSNR = 10 * log10( data_range**2 / MSE )

    where MAX is `data_range` and MSE is the mean squared error
    between prediction and target.

    Parameters
    ----------
    pred, target :
        FloatTensor with shape [B, C, H, W].
    data_range :
        Intensity range of inputs (MAX in the PSNR formula).
        Use 1.0 if images are normalized to [0, 1].
    reduction :
        - ``"mean"`` : return scalar PSNR averaged over the batch.
        - ``"none"`` : return PSNR per-sample with shape [B].

    Returns
    -------
    torch.Tensor
        PSNR value(s) in dB.
    """
    _validate_batch_tensors(pred, target)

    eps = 1e-8
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)

    if reduction == "mean":
        return psnr.mean()
    return psnr

# ============================================================
# PSNR
# ============================================================

def _gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel normalized to sum = 1.

    Parameters
    ----------
    kernel_size :
        Odd integer window size (e.g. 11).
    sigma :
        Standard deviation of the Gaussian (e.g. 1.5).
    device, dtype :
        Tensor device and dtype for the resulting kernel.

    Returns
    -------
    torch.Tensor
        1D tensor of shape [kernel_size].
    """
    assert kernel_size % 2 == 1, "[GAUSS1D] kernel_size should be odd (e.g., 11)"
    assert kernel_size > 1, "[GAUSS1D] kernel_size must be > 1"
    
    half = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    x = coords - half

    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()

    return g

def _gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create a 2D separable Gaussian kernel.

    Returns
    -------
    torch.Tensor
        Kernel of shape [1, 1, kernel_size, kernel_size].
    """
    k1d = _gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    k2d = k1d[:, None] * k1d[None, :]

    k2d = k2d / k2d.sum()
    kernel = k2d.view(1, 1, kernel_size, kernel_size)

    total = float(kernel.sum())
    assert abs(total - 1.0) < 1e-6, f"[GAUSS2D] kernel sum != 1 (got {total})"
    return kernel

def _gaussian_conv2d(
    x: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    """
    Apply Gaussian blur with `kernel` to tensor x using depthwise conv.

    Parameters
    ----------
    x :
        FloatTensor [B, C, H, W].
    kernel :
        Gaussian kernel [1, 1, k, k].

    Returns
    -------
    torch.Tensor
        Blurred tensor with same shape as `x`.
    """
    B, C, H, W = x.shape
    k = kernel.shape[-1]

    weight = kernel.expand(C, 1, k, k)

    out = F.conv2d(
        x,
        weight=weight,
        bias=None,
        stride=1,
        padding=k // 2,
        groups=C,
    )
    return out

# ============================================================
# SSIM core
# ============================================================


def _ssim_map(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute SSIM map per pixel for 2D images.

    Parameters
    ----------
    pred, target :
        FloatTensor [B, C, H, W] in [0, data_range].

    Returns
    -------
    torch.Tensor
        SSIM map with shape [B, C, H, W], values in [0, 1].
    """
    _validate_batch_tensors(pred, target)
    
    # 1) Build Gaussian kernel:
    kernel = _gaussian_kernel_2d(win_size, win_sigma, pred.device, pred.dtype)

    # 2) Compute local means:
    mu_x = _gaussian_conv2d(pred, kernel)
    mu_y = _gaussian_conv2d(target, kernel)

    # 3) Compute local variances / covariance:
    sigma_x2 = _gaussian_conv2d(pred * pred, kernel) - mu_x**2
    sigma_y2 = _gaussian_conv2d(target * target, kernel) - mu_y**2
    sigma_xy = _gaussian_conv2d(pred * target, kernel) - mu_x * mu_y

    # 4) Stabilize variances:
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    # 5) Compute SSIM map using standard constants:
    k1, k2 = 0.01, 0.03
    L = data_range
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    den = torch.clamp(den, min=1e-6)

    ssim_map = num / den

    ssim_map = torch.clamp(ssim_map, 0.0, 1.0)

    return ssim_map

def ssim2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute single-scale SSIM over a batch of 2D images.

    Parameters
    ----------
    pred, target :
        FloatTensor [B, C, H, W] in [0, data_range].
    data_range :
        Intensity range (max - min).
    reduction :
        - "mean" : return batch-averaged SSIM scalar.
        - "none" : return per-sample SSIM with shape [B].
    win_size, win_sigma :
        Gaussian window size and standard deviation.

    Returns
    -------
    torch.Tensor
        SSIM score(s) in [0, 1].
    """
    ssim_map = _ssim_map(pred, target, data_range, win_size, win_sigma)
    per_sample = ssim_map.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return per_sample.mean()
    return per_sample

# ============================================================
# MS-SSIM
# ============================================================
def _downsample(x: torch.Tensor) -> torch.Tensor:
    """
    Downsample image by a factor of 2 using average pooling.

    Parameters
    ----------
    x :
        FloatTensor [B, C, H, W].

    Returns
    -------
    torch.Tensor
        Downsampled tensor [B, C, H//2, W//2].
    """

    out = F.avg_pool2d(x, kernel_size=2, stride=2)
    return out

def ms_ssim2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
    win_size: int = 11,
    win_sigma: float = 1.5,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute multi-scale SSIM (MS-SSIM) over a batch of 2D images.

    Implementation sketch
    ---------------------
    - Follow Wang et al., 2003 MS-SSIM:
      * At each scale, compute SSIM components.
      * Downsample images between scales.
      * Aggregate across scales using predefined weights.

    Parameters
    ----------
    pred, target :
        FloatTensor [B, C, H, W] in [0, data_range].
    data_range :
        Intensity range (max - min).
    reduction :
        Same semantics as :func:`ssim2d`.
    weights :
        Optional 1D tensor of scale weights. If None, use default
        5-scale weights as in the original MS-SSIM paper.

    Returns
    -------
    torch.Tensor
        MS-SSIM score(s) in [0, 1].
    """
    _validate_batch_tensors(pred, target)
    if weights is None:
        # Default 5-scale weights from Wang et al. (2003)
        default_w = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(
            default_w,
            device=pred.device,
            dtype=pred.dtype
        )
    else:
        weights =weights.to(device=pred.device, dtype=pred.dtype)
    
    num_scales = int(weights.numel())
    assert num_scales >= 1, "[MS-SSIM] weights must contain at least 1 scale."

    current_pred = pred
    current_target = target

    per_scale_ssim: List[torch.Tensor] = []

    for s in range(num_scales):
        ssim_s = ssim2d(
            current_pred,
            current_target,
            data_range=data_range,
            reduction="none",
            win_size=win_size,
            win_sigma=win_sigma
        )
        per_scale_ssim.append(ssim_s)

        if s < num_scales - 1:
            current_pred = _downsample(current_pred)
            current_target = _downsample(current_target)
    
    ssim_stack = torch.stack(per_scale_ssim, dim=0)
    ssim_stack = torch.clamp(ssim_stack, min=1e-6, max=1.0)

    weights_exp = weights.view(num_scales, 1)
    log_ms = (weights_exp * torch.log(ssim_stack)).sum(dim=0)
    ms_ssim_per_sample = torch.exp(log_ms)

    if reduction == "mean":
        return ms_ssim_per_sample.mean()
    if reduction == "none":
        return ms_ssim_per_sample
    
    raise ValueError(f"[MS-SSIM] Unknown reduction={reduction!r}")

