from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# Data
from src.datasets.lodopab_fbp_recon_dataset import (
    LoDoPaBFBPConfig,
    LoDoPaBFBPDataset,
)

from src.datasets.lidc_recon_dataset import (
    LIDCReconConfig,
    LIDCReconDataset,
)

# Models
from src.models.unext import UnetConvNeXtConfig, UNetConvNeXt
from src.models.moe import MoEReconModel, MoEReconConfig

# Loss / metrics
from src.models.losses.recon_loss import ReconstructionLoss, ReconstructionLossConfig
from src.eval.metrics_recon import psnr_batch, ssim2d

# Utils
from src.config.train import CrossDomainTrainConfig

Tensor = torch.Tensor

# ============================================================
# Helpers
# ============================================================
def cosine_warmup_lambda(
    epoch: int,
    *,
    max_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """
    Piecewise LR schedule with linear warmup and cosine decay.

    Parameters
    ----------
    epoch :
        Zero-based epoch index in [0, max_epochs - 1].
    max_epochs :
        Total number of training epochs.
    warmup_epochs :
        Number of warmup epochs at the beginning of training.
    min_lr_ratio :
        Ratio between final LR and base LR, e.g. 1e-2 means decay to 1% LR.

    Returns
    -------
    float
        Multiplicative LR factor to be applied on top of base LR.
    """
    if not 0 <= epoch < max_epochs:
        raise ValueError(f"Epoch={epoch} out of range 0..{max_epochs-1}.")

    if epoch < warmup_epochs and warmup_epochs > 0:
        factor = (epoch + 1) / warmup_epochs
    else:
        denom = max(1, max_epochs - max(warmup_epochs, 1))
        t = (epoch - warmup_epochs) / float(denom)
        t = min(max(t, 0.0), 1.0)
        factor = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * t))

    return factor

# ============================================================
# Dataloaders
# ============================================================

def build_dataloaders(
        cfg: CrossDomainTrainConfig,
        lodopab_manifest_rel: str,
        lidc_manifest_rel: str,
        *,
        shuffle: bool,
        drop_last: bool,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for LoDoPaB-FBP and LIDC synthetic-noise datasets.

    This function is split-agnostic: callers pass in the manifest paths
    they want to use (e.g. train/val/test). The same factory is reused
    for all splits.

    Parameters
    ----------
    cfg :
        Resolved cross-domain training configuration.
    lodopab_manifest_rel :
        Relative path to the LoDoPaB manifest CSV for this split.
        Example: ``'data/prepared/lodopab_fbp/train/fbp_manifest.csv'``.
    lidc_manifest_rel :
        Relative path to the LIDC manifest CSV for this split.
        Example: ``'data/prepared/lidc/splits/lidc_train_manifest.csv'``.
    shuffle :
        Whether to shuffle the dataset in the DataLoader.
        Typically ``True`` for training and ``False`` for eval.
    drop_last :
        Whether to drop the last incomplete batch.
        Typically ``True`` for training (for stable batch norm) and
        ``False`` for eval.

    Returns
    -------
    dict
        {
            "lodopab": DataLoader over LoDoPaBFBPDataset,
            "lidc": DataLoader over LIDCReconDataset,
        }
    """
    # LoDoPaB FBP dataset: FBP noisy -> GT
    lodopab_ds_cfg = LoDoPaBFBPConfig(
        manifest_rel=lodopab_manifest_rel,
        patch_size=cfg.lodopab_patch_size,
        seed=cfg.seed,
    )

    lodopab_ds = LoDoPaBFBPDataset(lodopab_ds_cfg)

    # LIDC synthetic-noise dataset: synthetic noisy -> GT
    lidc_ds_cfg = LIDCReconConfig(
        manifest_rel=lidc_manifest_rel,
        patch_size=cfg.lidc_patch_size,
        noise_mode=cfg.lidc_noise_mode,
        noise_std=cfg.lidc_noise_std,
        poisson_scale=cfg.lidc_poisson_scale,
        seed=cfg.seed,
    )

    lidc_ds = LIDCReconDataset(lidc_ds_cfg)

    pin_memory = cfg.device.startswith("cuda")

    lodopab_loader = DataLoader(
        lodopab_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    lidc_loader = DataLoader(
        lidc_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    loaders: Dict[str, DataLoader] = {
        "lodopab": lodopab_loader,
        "lidc": lidc_loader,
    }

    return loaders

# ============================================================
# Model / loss / optimizer builders / scheduler
# ============================================================
def build_model(cfg: CrossDomainTrainConfig) -> nn.Module:
    """
    Build reconstruction model (plain UNetConvNeXt or MoEReconModel).

    Parameters
    ----------
    cfg :
        Cross-domain training configuration.

    Returns
    -------
    nn.Module
        Initialized model moved to cfg.device.
    """
    device = torch.device(cfg.device)

    backbone_cfg = UnetConvNeXtConfig(
        in_channels=1,
        out_channels=1,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate,
        final_activation=cfg.final_activation,
    )

    if cfg.use_moe:
        moe_cfg = MoEReconConfig(
            num_experts=cfg.num_experts,
            num_domains=cfg.num_domains,
            in_channels=1,
            out_channels=1,
            expert_cfg=backbone_cfg,
            gating_hidden_dim=cfg.gating_hidden_dim,
            gating_temp=cfg.gating_temp,
        )
        model: nn.Module = MoEReconModel(moe_cfg)
    else:
        model = UNetConvNeXt(backbone_cfg)

    model = model.to(device)
    return model

def build_loss(cfg: CrossDomainTrainConfig) -> ReconstructionLoss:
    """
    Build reconstruction loss module.

    Parameters
    ----------
    cfg :
        Cross-domain training configuration.

    Returns
    -------
    ReconstructionLoss
        Composite reconstruction loss (L1/MSE/SSIM/TV).
    """
    loss_cfg = ReconstructionLossConfig(
        l1_weight=cfg.l1_weight,
        mse_weight=cfg.mse_weight,
        ssim_weight=cfg.ssim_weight,
        tv_weight=cfg.tv_weight,
        data_range=cfg.data_range,
    )
    criterion = ReconstructionLoss(loss_cfg)
    return criterion

def build_optimizer(
    cfg: CrossDomainTrainConfig,
    model: nn.Module,
) -> optim.Optimizer:
    """
    Build optimizer for reconstruction model.

    Parameters
    ----------
    cfg :
        Cross-domain training configuration.
    model :
        Reconstruction model whose parameters will be optimized.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer instance (Adam).
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    return optimizer

def build_scheduler(
    cfg: CrossDomainTrainConfig,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Build learning-rate scheduler for cross-domain reconstruction.

    Strategy
    --------
    - Default: cosine decay with linear warmup on total epoch budget.
    - Optional: ReduceLROnPlateau on val_loss_avg for ablations.
    - Can be disabled by setting scheduler_type='none'.
    """
    if cfg.scheduler_type == "none":
        return None

    if cfg.scheduler_type == "cosine_warmup":
        min_lr_ratio = cfg.min_lr / cfg.lr

        lr_lambda = lambda epoch: cosine_warmup_lambda(
            epoch=epoch,
            max_epochs=cfg.max_epochs,
            warmup_epochs=cfg.warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        )

        scheduler: _LRScheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
        )
        return scheduler

    if cfg.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            patience=cfg.plateau_patience,
            factor=cfg.plateau_factor,
        )
        return scheduler

    raise ValueError(f"Scheduler type {cfg.scheduler_type} not supported.")


# ============================================================
# Regularizer
# ============================================================
def moe_entropy_regularizer(aux: Dict[str, Tensor]) -> Tensor:
    """
    Entropy-based regularizer on MoE gating.

    Assumes aux["gating_weights"] has shape [B, num_experts].
    We encourage *high entropy* (diverse expert usage) by
    minimizing -H(p), where p are the gating probabilities.

    Returns
    -------
    Tensor
        Scalar tensor on the same device as gating weights.
    """
    if "gating_weights" not in aux:
        # Fallback: zero regularizer on a sane device
        if not aux:
            return torch.tensor(0.0)
        return torch.zeros((), device=next(iter(aux.values())).device)

    weights = aux["gating_weights"]              # [B, E]
    probs = weights.clamp_min(1e-8)             # avoid log(0)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [B]

    reg = -entropy.mean()                       # smaller => higher entropy
    return reg

# ============================================================
# Training / evaluation loops
# ============================================================
def train_one_epoch(
    cfg: CrossDomainTrainConfig,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: ReconstructionLoss,
    optimizer: optim.Optimizer,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Run one cross-domain training epoch over LoDoPaB and LIDC.

    The epoch length is defined as ``max(len(lodopab_loader), len(lidc_loader))``.
    At each iteration, a mini-batch is drawn from each domain. If one loader is
    exhausted earlier, its iterator is re-created so that its batches are cycled
    (with reshuffling if ``shuffle=True`` in the DataLoader).

    For each paired mini-batch:
    1. Forward pass on LoDoPaB (domain id = 0) and LIDC (domain id = 1),
       optionally with MoE domain-aware gating.
    2. Compute reconstruction loss for each domain via ``criterion``.
    3. Combine domain losses and an optional MoE entropy regularizer into a
       single scalar loss, backpropagate, and perform one optimizer step.
    4. Accumulate average PSNR and SSIM across both domains for monitoring.

    Parameters
    ----------
    cfg : CrossDomainTrainConfig
        Resolved cross-domain training configuration. Must contain device
        string (e.g. ``"cuda:0"``), MoE regularization weight, and
        reconstruction hyperparameters consistent with ``criterion``.
    model : nn.Module
        Reconstruction model. Can be either ``UNetConvNeXt`` or
        ``MoEReconModel``. Expected input / output tensors have shape
        ``[B, 1, H, W]`` and dtype ``float32``.
    loaders : dict of {str: DataLoader}
        Mapping with exactly two entries:
        - ``"lodopab"``: DataLoader yielding batches of LoDoPaB patches.
        - ``"lidc"``   : DataLoader yielding batches of LIDC patches.
        Each batch must be a dict with keys ``"input"`` and ``"target"``,
        both of shape ``[B, 1, H, W]``.
    criterion : ReconstructionLoss
        Composite reconstruction loss used to compute per-domain loss
        (e.g., weighted combination of L1, MSE, SSIM, TV) on
        ``(prediction, target)`` pairs.
    optimizer : optim.Optimizer
        Optimizer instance (e.g. ``torch.optim.Adam``) responsible for
        updating ``model.parameters()`` at each iteration.
    epoch : int
        Current epoch index (0-based). Only used for logging.
    logger : logging.Logger
        Logger used to report aggregated epoch statistics.

    Returns
    -------
    Dict[str, float]
        Dictionary of scalar epoch-level statistics:
        - ``"loss"`` : float
            Mean training loss over all optimization steps.
        - ``"psnr"`` : float
            Mean PSNR (in dB) averaged over both domains and all steps.
        - ``"ssim"`` : float
            Mean SSIM averaged over both domains and all steps.

    Raises
    ------
    RuntimeError
        If either domain loader is empty, or if no training steps were
        executed (e.g. due to inconsistent loader configuration).

    Notes
    -----
    - This function assumes that all inputs and targets are pre-normalized
      to the same intensity range (e.g. [0, 1]) consistent with
      ``cfg.data_range`` used in PSNR/SSIM.
    - When ``model`` is a ``MoEReconModel``, domain ids are set to 0 for
      LoDoPaB and 1 for LIDC, and an entropy-based regularizer on the
      gating distribution is added with weight ``cfg.moe_reg_weight``.
    """
    device = torch.device(cfg.device)
    model.train()

    lodopab_loader = loaders["lodopab"]
    lidc_loader = loaders["lidc"]

    len_lodo = len(lodopab_loader)
    len_lidc = len(lidc_loader)

    if len_lodo == 0 or len_lidc == 0:
        raise RuntimeError("[TRAIN] One of the domain loaders is empty.")

    num_batches = max(len_lodo, len_lidc)

    iter_lodo = iter(lodopab_loader)
    iter_lidc = iter(lidc_loader)

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_steps = 0

    for step in range(num_batches):
        try:
            batch_lodo = next(iter_lodo)
        except StopIteration:
            iter_lodo = iter(lodopab_loader)
            batch_lodo = next(iter_lodo)

        try:
            batch_lidc = next(iter_lidc)
        except StopIteration:
            iter_lidc = iter(lidc_loader)
            batch_lidc = next(iter_lidc)

        optimizer.zero_grad(set_to_none=True)

        # ---------------------------
        # Domain 0: LoDoPaB (FBP)
        # ---------------------------
        in_lodo = batch_lodo["input"].to(device, non_blocking=True)
        tgt_lodo = batch_lodo["target"].to(device, non_blocking=True)

        if isinstance(model, MoEReconModel):
            dom_lodo = torch.zeros(
                in_lodo.shape[0],
                dtype=torch.long,
                device=device,
            )
            pred_lodo, aux_lodo = model(in_lodo, domain_ids=dom_lodo)
        else:
            pred_lodo = model(in_lodo)
            aux_lodo: Dict[str, Any] = {}

        loss_lodo = criterion(pred_lodo, tgt_lodo)

        # ---------------------------
        # Domain 1: LIDC (synthetic)
        # ---------------------------
        in_lidc = batch_lidc["input"].to(device, non_blocking=True)
        tgt_lidc = batch_lidc["target"].to(device, non_blocking=True)

        if isinstance(model, MoEReconModel):
            dom_lidc = torch.ones(
                in_lidc.shape[0],
                dtype=torch.long,
                device=device,
            )
            pred_lidc, aux_lidc = model(in_lidc, domain_ids=dom_lidc)
        else:
            pred_lidc = model(in_lidc)
            aux_lidc = {}

        loss_lidc = criterion(pred_lidc, tgt_lidc)

        # ---------------------------
        # MoE regularizer
        # ---------------------------
        reg_term = torch.tensor(0.0, device=device)
        if isinstance(model, MoEReconModel) and cfg.moe_reg_weight > 0.0:
            reg_lodo = moe_entropy_regularizer(aux_lodo)
            reg_lidc = moe_entropy_regularizer(aux_lidc)
            reg_term = 0.5 * (reg_lodo + reg_lidc)

        loss = 0.5 * (loss_lodo + loss_lidc) + cfg.moe_reg_weight * reg_term
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            psnr_lodo = psnr_batch(pred_lodo, tgt_lodo, data_range=cfg.data_range)
            psnr_lidc = psnr_batch(pred_lidc, tgt_lidc, data_range=cfg.data_range)
            ssim_lodo = ssim2d(pred_lodo, tgt_lodo, data_range=cfg.data_range)
            ssim_lidc = ssim2d(pred_lidc, tgt_lidc, data_range=cfg.data_range)

            batch_psnr = 0.5 * (psnr_lodo + psnr_lidc)
            batch_ssim = 0.5 * (ssim_lodo + ssim_lidc)

        running_loss += float(loss.item())
        running_psnr += float(batch_psnr)
        running_ssim += float(batch_ssim)
        num_steps += 1

    if num_steps == 0:
        raise RuntimeError("[TRAIN] No steps executed in train_one_epoch.")

    epoch_loss = running_loss / num_steps
    epoch_psnr = running_psnr / num_steps
    epoch_ssim = running_ssim / num_steps

    logger.info("[TRAIN] epoch=%s | loss=%.4e | psnr=%.2f dB | ssim=%.4f", epoch, epoch_loss, epoch_psnr, epoch_ssim)
    stats = {
        "loss": epoch_loss,
        "psnr": epoch_psnr,
        "ssim": epoch_ssim,
    }
    return stats

def evaluate_on_loader(
    cfg: CrossDomainTrainConfig,
    model: nn.Module,
    loader: DataLoader,
    domain_id: int,
    criterion: Optional[ReconstructionLoss] = None,
    split: str = "val",
    collect_samples: bool = False,
    max_samples: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate model on a single domain loader.

    If split == "test" and collect_samples=True, also return a few
    (noisy, pred, target) triplets for visualization.

    Returns
    -------
    dict
        {
            "loss": float,
            "psnr": float,
            "ssim": float,
            # Optional:
            # "samples": List[Dict[str, np.ndarray]]
        }
    """
    device = torch.device(cfg.device)
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_steps = 0

    collected: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            if isinstance(model, MoEReconModel):
                dom = torch.full(
                    (inp.shape[0],),
                    fill_value=domain_id,
                    dtype=torch.long,
                    device=device,
                )
                pred, _ = model(inp, domain_ids=dom)
            else:
                pred = model(inp)


            if criterion is not None:
                loss_val = float(criterion(pred, tgt).item())
            else:
                loss_val = 0.0

            psnr_val = psnr_batch(pred, tgt, data_range=cfg.data_range)
            ssim_val = ssim2d(pred, tgt, data_range=cfg.data_range)

            total_loss += float(loss_val)
            total_psnr += float(psnr_val)
            total_ssim += float(ssim_val)
            num_steps += 1

            if collect_samples and split == "test" and len(collected) < max_samples:
                remaining = max_samples - len(collected)
                b = inp.shape[0]
                num_to_take = min(b, remaining)

                for i in range(num_to_take):
                    noisy_np = inp[i].detach().cpu().numpy()
                    pred_np = pred[i].detach().cpu().numpy()
                    tgt_np = tgt[i].detach().cpu().numpy()

                    sample = {
                        "noisy": noisy_np,
                        "pred": pred_np,
                        "target": tgt_np,
                    }
                    collected.append(sample)

    if num_steps == 0:
        raise RuntimeError("[EVAL] No steps executed in evaluate_on_loader.")

    metrics: Dict[str, Any] = {
        "loss": total_loss / num_steps,
        "psnr": total_psnr / num_steps,
        "ssim": total_ssim / num_steps,
    }

    if collect_samples and split == "test":
        metrics["samples"] = collected

    return metrics