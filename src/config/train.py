from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Literal, Optional, Tuple

import argparse
import yaml

from src.utils.paths import get_project_root

@dataclass(frozen=True)
class CrossDomainTrainConfig:
    """
    Configuration for cross-domain CT reconstruction training (LoDoPaB + LIDC).

    Resolution order (for each field)
    ---------------------------------
    1) Dataclass default
    2) YAML file (if provided via --config)
    3) CLI arguments (if not None)

    Notes
    -----
    - All paths are project-root-relative (see get_project_root()).
    - Domains:
        0 -> LoDoPaB
        1 -> LIDC
      but num_domains is kept as a field for future extension.
    """
    # General
    seed: int = 42
    device: str = "cuda"

    # Data (train)
    lodopab_manifest_rel: str = "data/prepared/lodopab_fbp/train/fbp_manifest.csv"
    lidc_manifest_rel: str = "data/prepared/lidc/splits/lidc_train_manifest.csv"

    # Data (val)
    lodopab_val_manifest_rel: str = "data/prepared/lodopab_fbp/val/fbp_manifest.csv"
    lidc_val_manifest_rel: str = "data/prepared/lidc/splits/lidc_val_manifest.csv"

    # Data (test)
    lodopab_test_manifest_rel: str = "data/prepared/lodopab_fbp/test/fbp_manifest.csv"
    lidc_test_manifest_rel: str = "data/prepared/lidc/splits/lidc_test_manifest.csv"



    batch_size: int = 4
    num_workers: int = 4

    # LoDoPaB FBP patches
    lodopab_patch_size: Optional[int] = None        # e.g., 256 for patch training

    # LIDC synthetic noise patches + noise model
    lidc_patch_size: Optional[int] = None
    lidc_noise_mode: str = "gaussian"               # "identity" / "gaussian" / "poisson"
    lidc_noise_std: float = 0.02                    # for gaussian
    lidc_poisson_scale: float = 1e4                 # for poisson

    # Scheduler configuration
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    plateau_patience: int = 5
    plateau_factor: float = 0.5


    # Optimization
    max_epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Logging / ckpt
    log_dir_rel: str = "logs"
    ckpt_dir_rel: str = "models/checkpoints"
    reports_dir_rel: str = "reports"

    # Model choice
    use_moe: bool = False
    num_experts: int = 2
    num_domains: int = 2

    gating_hidden_dim: int = 128
    gating_temp: float = 1.0

    # MoE regularizer
    moe_reg_weight: float = 0.0

    # Optional size preset: "debug" / "base" / "large"
    model_size: Optional[Literal["debug", "base", "large"]] = None

    # Backbone hyperparams
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    dims: Tuple[int, ...] = (64, 128, 256, 512)
    drop_path_rate: float = 0.1
    final_activation: Optional[str] = "sigmoid"

    # Loss hyperparams
    l1_weight: float = 0.0
    mse_weight: float = 1.0
    ssim_weight: float = 0.0
    tv_weight: float = 0.0
    data_range: float = 1.0

# ============================================================
# Internal helpers
# ============================================================

def _namespace_to_dict(ns: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert argparse.Namespace -> dict, skipping None values.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Key/value mapping for non-None CLI args.
    """

    raw_dict = vars(ns)

    cli_dict: Dict[str, Any] = {
        key: value
        for key, value in raw_dict.items()
        if value is not None
    }

    return cli_dict

def _load_yaml_dict(yaml_path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML config file if provided.

    Parameters
    ----------
    yaml_path : str or None
        Relative or absolute path to YAML config.

    Returns
    -------
    dict
        Parsed key/value pairs from YAML, or {} if no file.
    """
    yaml_dict: Dict[str, Any] = {}

    if yaml_path is None:
        return yaml_dict

    root = get_project_root()
    cfg_path = (root / yaml_path).resolve()

    with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return yaml_dict

    if not isinstance(loaded, dict):
        raise ValueError(f"[CONFIG] Expected YAML top-level to be a mapping, got {type(loaded)}.")

    yaml_dict = loaded
    return yaml_dict

def _coerce_config_types(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce raw config dictionary to types expected by CrossDomainTrainConfig.

    Examples
    --------
    - YAML may provide depths/dims as lists -> convert to tuples.
    - Booleans given as strings -> parse if necessary.

    Parameters
    ----------
    raw : dict
        Merged configuration before type coercion.

    Returns
    -------
    dict
        Configuration with corrected types.
    """

    cleaned = dict(raw)

    # ---- depths: list -> tuple[int, ...] ----
    if "depths" in cleaned:
        depths_val = cleaned["depths"]
        if isinstance(depths_val, list):
            try:
                cleaned["depths"] = tuple(map(int, depths_val))
            except (TypeError, ValueError) as e:
                raise ValueError(f"[CONFIG] depths must be list of ints, got {depths_val!r}.") from e
        elif isinstance(depths_val, tuple):
            pass
        else:
            raise ValueError(f"[CONFIG] depths must be list/tuple of ints, got {type(depths_val)}.")

    # ---- dims: list -> tuple[int, ...] ----
    if "dims" in cleaned:
        dims_val = cleaned["dims"]
        if isinstance(dims_val, list):
            try:
                cleaned["dims"] = tuple(map(int, dims_val))
            except (TypeError, ValueError) as e:
                raise ValueError(f"[CONFIG] dims must be list of ints, got {dims_val!r}.") from e
        elif isinstance(dims_val, tuple):
            pass
        else:
            raise ValueError(f"[CONFIG] dims must be list/tuple of ints, got {type(dims_val)}.")

    # ---- Int fields ----
    int_keys = [
        "num_experts",
        "num_domains",
        "batch_size",
        "num_workers",
        "max_epochs",
        "seed",
        "lodopab_patch_size",
        "lidc_patch_size",
        "gating_hidden_dim"
    ]
    for key in int_keys:
        if key in cleaned and cleaned[key] is not None:
            val = cleaned[key]
            if isinstance(val, int):
                continue
            try:
                cleaned[key] = int(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"[CONFIG] {key} must be int, got {val!r} (type {type(val)}).") from e

    # ---- Float fields ----
    float_keys = [
        "lr",
        "weight_decay",
        "drop_path_rate",
        "l1_weight",
        "mse_weight",
        "ssim_weight",
        "tv_weight",
        "data_range",
        "lidc_noise_std",
        "lidc_poisson_scale",
        "gating_temp",
        "moe_reg_weight"
    ]
    for key in float_keys:
        if key in cleaned and cleaned[key] is not None:
            val = cleaned[key]
            if isinstance(val, float):
                continue
            try:
                cleaned[key] = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"[CONFIG] {key} must be float, got {val!r} (type {type(val)}).") from e

    # ---- Bool fields ----
    bool_keys = [
        "use_moe",
    ]
    for key in bool_keys:
        if key in cleaned and cleaned[key] is not None:
            val = cleaned[key]
            if isinstance(val, bool):
                continue
            if isinstance(val, str):
                lower = val.lower()
                if lower in ("true", "yes", "y", "1"):
                    cleaned[key] = True
                    continue
                if lower in ("false", "no", "n", "0"):
                    cleaned[key] = False
                    continue
            raise ValueError(f"[CONFIG] {key} must be bool or bool-like string, got {val!r} (type {type(val)}).")

    return cleaned


# ============================================================
# Public loader
# ============================================================

def load_train_config(
        cli_args: argparse.Namespace,
        *,
        yaml_path: Optional[str] = None,
) -> CrossDomainTrainConfig:
    """
    Build CrossDomainTrainConfig from defaults, YAML, and CLI overrides.

    Parameters
    ----------
    cli_args :
        Parsed argparse.Namespace from train_cross_domain CLI.
    yaml_path :
        Optional path to YAML config file (from --config).

    Returns
    -------
    CrossDomainTrainConfig
        Fully-resolved training configuration.
    """

    # ---- Start from dataclass default ----
    base_dict: Dict[str, Any] = asdict(CrossDomainTrainConfig())

    # ---- Merge YAML (if any) ----
    yaml_dict = _load_yaml_dict(yaml_path)
    base_dict.update(yaml_dict)

    # ---- Merge CLI ----
    cli_all = _namespace_to_dict(cli_args)
    valid_keys = {f.name for f in fields(CrossDomainTrainConfig)}
    cli_dict = {k: v for k, v in cli_all.items() if k in valid_keys}
    base_dict.update(cli_dict)

    # ---- Coerce type ----
    base_dict = _coerce_config_types(base_dict)

    # ---- Construct CrossDomainTrainConfig ----
    cfg = CrossDomainTrainConfig(**base_dict)
    assert len(cfg.depths) == len(cfg.dims) == 4, \
        f"[CONFIG] depths and dims must be 4D (got depth={len(cfg.depths)}D, dims={len(cfg.dims)}D)."

    return cfg

