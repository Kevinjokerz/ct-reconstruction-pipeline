from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Any, Dict

import torch
import pandas as pd
from tqdm.auto import tqdm

from src.config import CrossDomainTrainConfig, load_train_config
from src.training.loops import (
    build_dataloaders,
    build_model,
    build_loss,
    build_optimizer,
    train_one_epoch,
    evaluate_on_loader,
    build_scheduler,
)
from src.utils.logger import setup_logger
from src.utils.paths import get_project_root, safe_join_rel, to_relpath
from src.utils.repro import set_seed
from src.utils.io import save_df_csv

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for cross-domain CT reconstruction training.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments. These will later be merged into
        :class:`CrossDomainTrainConfig` via ``load_train_config``.
    """
    parser = argparse.ArgumentParser(
        description="Cross-domain low-dose CT reconstruction (LoDoPaB + LIDC)."
    )

    # ---- Config location ----
    parser.add_argument("--config", type=str, default=None, help=(
        "Optional YAML config path (project-root-relative or absolute path). "
        "If omitted, dataclass default are used."
    ))

    # ---- Run naming ----
    parser.add_argument("--run_name", type=str, default=None, help=(
        "Optional run name. If not provided, will be derived from "
        "backbone / MoE flag / model_size."
    ))

    # ---- Typical overrides (all optional, merged into config) ----
    parser.add_argument("--seed", type=int, default=None, help="Override global random seed.")
    parser.add_argument("--device", type=str, default=None, help="Override device string from config.")
    parser.add_argument("--use_moe", type=str, default=None, help="Override MoE usage flag (e.g., 'true' / 'false').")

    args = parser.parse_args()
    return args

def resolve_run_name(cfg: CrossDomainTrainConfig, cli_run_name: str | None) -> str:
    """
    Derive a human-readable run name for logging/checkpointing.

    Parameters
    ----------
    cfg :
        Resolved training configuration.
    cli_run_name :
        Run name passed via CLI, if any.

    Returns
    -------
    str
        Run name such as ``'unext_base'`` or ``'moe_large'``.
    """
    if cli_run_name is not None:
        return cli_run_name

    backbone_tag = "moe" if cfg.use_moe else "unext"
    size_tag = cfg.model_size or "base"

    run_name = f"{backbone_tag}_{size_tag}"
    return run_name

def main() -> None:
    """
    Entry point for cross-domain training script.

    High-level pipeline
    -------------------
    1. Parse CLI and resolve :class:`CrossDomainTrainConfig`.
    2. Set up logging, seeds, and project-root-relative paths.
    3. Build DataLoaders, model, loss, and optimizer using ``loops.py`` helpers.
    4. Train for ``cfg.max_epochs``, evaluating after each epoch.
    5. Select the best model using validation metrics and save a checkpoint.
    6. Persist epoch-wise metrics into a CSV under ``reports/``.
    """
    args = parse_args()

    # ----------------------------------------------------
    # Build configuration (defaults + YAML + CLI overrides)
    # ----------------------------------------------------
    cfg: CrossDomainTrainConfig = load_train_config(cli_args=args, yaml_path=args.config)

    project_root: Path = get_project_root()
    run_name: str = resolve_run_name(cfg=cfg, cli_run_name=args.run_name)

    # ----------------------------------------------------
    # Logger and reproducibility
    # ----------------------------------------------------
    log_dir: Path = safe_join_rel(cfg.log_dir_rel, root=project_root)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name=f"train_cross_domain_{run_name}", log_dir=to_relpath(log_dir))
    logger.info("[INIT] Cross-domain training started.")
    logger.info("[CFG] %s", cfg)

    seed_used: int = set_seed(seed=cfg.seed)
    logger.info("[SEED] Seed used: %d", seed_used)

    # ----------------------------------------------------
    # Data: build training loaders
    # ----------------------------------------------------
    train_loaders: Dict[str, Any] = build_dataloaders(
        cfg=cfg,
        lodopab_manifest_rel=cfg.lodopab_manifest_rel,
        lidc_manifest_rel=cfg.lidc_manifest_rel,
        shuffle=True,
        drop_last=True,
    )

    val_loaders: Dict[str, Any] = build_dataloaders(
        cfg=cfg,
        lodopab_manifest_rel=cfg.lodopab_val_manifest_rel,
        lidc_manifest_rel=cfg.lidc_val_manifest_rel,
        shuffle=False,
        drop_last=False,
    )

    batch_lodo = next(iter(train_loaders["lodopab"]))
    batch_lidc = next(iter(train_loaders["lidc"]))
    logger.info("[SHAPE] LoDoPaB input=%s target=%s", batch_lodo["input"].shape, batch_lodo["target"].shape)
    logger.info("[SHAPE] LIDC input=%s target=%s", batch_lidc["input"].shape, batch_lidc["target"].shape)

    # ----------------------------------------------------
    # Model / loss / optimizer
    # ----------------------------------------------------
    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # ----------------------------------------------------
    # Checkpoints & reports dirs
    # ----------------------------------------------------
    ckpt_dir: Path = safe_join_rel(cfg.ckpt_dir_rel, run_name, root=project_root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    reports_dir: Path = safe_join_rel(cfg.reports_dir_rel, run_name, root=project_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_val_metric: float = float("-inf")
    best_epoch: int = -1

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    epoch_iter = tqdm(
        range(cfg.max_epochs),
        desc=f"Training {run_name}",
        dynamic_ncols=True,
    )

    for epoch in epoch_iter:
        # 1) Train one epoch across both domains
        train_stats: Dict[str, float] = train_one_epoch(
            cfg=cfg,
            model=model,
            loaders=train_loaders,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            logger=logger,
        )

        # 2) Validation per domain
        val_lodo: Dict[str, float] = evaluate_on_loader(
            cfg=cfg,
            model=model,
            loader=val_loaders["lodopab"],
            domain_id=0,
            criterion=criterion,
            split="val",
            collect_samples=False,
        )

        val_lidc: Dict[str, float] = evaluate_on_loader(
            cfg=cfg,
            model=model,
            loader=val_loaders["lidc"],
            domain_id=1,
            criterion=criterion,
            split="val",
            collect_samples=False,
        )

        # 3) Aggregate metrics (simple average across two domains)
        val_loss_avg = 0.5 * (val_lodo["loss"] + val_lidc["loss"])
        val_psnr_avg = 0.5 * (val_lodo["psnr"] + val_lidc["psnr"])
        val_ssim_avg = 0.5 * (val_lodo["ssim"] + val_lidc["ssim"])

        epoch_iter.set_postfix(
            {
                "train_psnr": f"{train_stats['psnr']:.2f}",
                "val_psnr": f"{val_psnr_avg:.2f}",
            }
        )

        logger.info(
            "[EPOCH %03d/%03d] "
            "train_loss=%.4e | train_psnr=%.2f | train_ssim=%.4f | "
            "val_loss=%.4e | val_psnr=%.2f | val_ssim=%.4f",
            epoch + 1,
            cfg.max_epochs,
            train_stats["loss"],
            train_stats["psnr"],
            train_stats["ssim"],
            val_loss_avg,
            val_psnr_avg,
            val_ssim_avg,
        )

        # 4) Model selection (currently on val_psnr_avg)
        current_val_metric: float = float(val_psnr_avg)
        is_best: bool = current_val_metric > best_val_metric

        if is_best:
            best_val_metric = current_val_metric
            best_epoch = epoch

            ckpt_path: Path = ckpt_dir / f"best_epoch_{epoch:03d}.pth"

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(cfg),
                "val_metric": current_val_metric,
            }
            torch.save(state, ckpt_path)

        # 5) Append metrics to history
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_stats["loss"]),
                "train_psnr": float(train_stats["psnr"]),
                "train_ssim": float(train_stats["ssim"]),
                "val_loss_lodopab": float(val_lodo["loss"]),
                "val_psnr_lodopab": float(val_lodo["psnr"]),
                "val_ssim_lodopab": float(val_lodo["ssim"]),
                "val_loss_lidc": float(val_lidc["loss"]),
                "val_psnr_lidc": float(val_lidc["psnr"]),
                "val_ssim_lidc": float(val_lidc["ssim"]),
                "val_loss_avg": float(val_loss_avg),
                "val_psnr_avg": float(val_psnr_avg),
                "val_ssim_avg": float(val_ssim_avg),
                "is_best": bool(is_best),
            }
        )

        if scheduler is not None:
            if cfg.scheduler_type == "plateau":
                scheduler.step(val_loss_avg)
            else:
                scheduler.step()

    # ----------------------------------------------------
    # Save training history CSV
    # ----------------------------------------------------
    history_df = pd.DataFrame(history)
    history_csv_path: Path = reports_dir / "train_history.csv"

    save_df_csv(
        df=history_df,
        path=history_csv_path,
        sanitize_paths=False,
        index=False
    )

    logger.info(
        "[DONE] Training finished. "
        "Best val metric (PSNR avg) = %.3f at epoch=%d. "
        "History saved to %s",
        best_val_metric,
        best_epoch,
        history_csv_path.as_posix(),
    )

if __name__ == "__main__":
    main()




