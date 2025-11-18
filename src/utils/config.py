from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import os
from dotenv import load_dotenv

# =============================================================================
# Dataclass definition
# =============================================================================

@dataclass(frozen=True)
class PrepConfig:
    """
    Configuration for the CT prepare_data pipeline (LoDoPaB + LIDC).

    Resolution order
    ----------------
    1) CLI args (highest priority)
    2) Environment (.env)
    3) Hard-coded defaults (lowest priority)

    Notes
    -----
    - Avoid leaking absolute paths in logs/reports; prefer project-relative
      paths (see `paths.py` + `sanitize.py`).
    - This config is **CT-specific** and does not include any text/ICD fields.
    """
    seed: int                               # Random seed for preproducibility
    source: Literal["lodopab", "dicom"]     # Data source & geometry

    # Intensity handling (HU or normalized)
    clip_lo: float
    clip_hi: float
    norm: Literal["none", "minmax", "zscore"]
    max_items: int                           # Sampling / debug

    lodopab_root: str                        # e.g. "data/raw/lodopab"
    dicom_root: str                          # e.g. "data/raw/lidc"
    out_root: str                            # e.g. "data/prepared"
    log_dir: str                             # e.g. "logs"
    reports_dir: str                         # e.g. "reports"



# =============================================================================
# Helpers
# =============================================================================

def _as_bool(value: str) -> bool:
    """Robust boolean parser for environment values"""
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# =============================================================================
# Main loader
# =============================================================================

def load_config(cli_args: Any) -> PrepConfig:
    """
    Load configuration from CLI + .env with proper precedence.

    Parameters
    ----------
    cli_args    :   argparse.Namespace
        Parsed CLI arguments from prepare_data.py

    Returns
    -------
    PrepConfig
        Immutable configuration object for the pipeline.
    """
    load_dotenv()

    env = {
        "SEED": int(os.getenv("SEED", "42")),
        "SOURCE": os.getenv("SOURCE", "lodopab"),
        "CLIP_LO": float(os.getenv("CLIP_LO", "-1000.0")),
        "CLIP_HI": float(os.getenv("CLIP_HI", "1000.0")),
        "NORM": os.getenv("NORM", "minmax"),
        "MAX_ITEMS": int(os.getenv("MAX_ITEMS", "0")),
        "LODOPAB_ROOT": os.getenv("LODOPAB_ROOT", "data/raw/lodopab"),
        "DICOM_ROOT": os.getenv("DICOM_ROOT", "data/raw/lidc"),
        "OUT_ROOT": os.getenv("OUT_ROOT", "data/prepared"),
        "LOG_DIR": os.getenv("LOG_DIR", "logs"),
        "REPORTS_DIR": os.getenv("REPORTS_DIR", "reports")
    }

    seed = getattr(cli_args, "seed", None)
    seed = env["SEED"] if seed is None else int(seed)

    # source
    source = getattr(cli_args, "source", None) or env["SOURCE"]

    # clip bound
    clip_lo = getattr(cli_args, "clip_lo", None)
    clip_lo = env["CLIP_LO"] if clip_lo is None else float(clip_lo)

    clip_hi = getattr(cli_args, "clip_hi", None)
    clip_hi = env["CLIP_HI"] if clip_hi is None else float(clip_hi)

    # normalization
    norm = getattr(cli_args, "norm", None) or env["NORM"]

    # sampling
    max_items = getattr(cli_args, "max_items", None)
    max_items = env["MAX_ITEMS"] if max_items is None else int(max_items)

    # directories
    lodopab_root = getattr(cli_args, "lodopab_root", None) or env["LODOPAB_ROOT"]
    dicom_root = getattr(cli_args, "dicom_root", None) or env["DICOM_ROOT"]
    out_root = getattr(cli_args, "out_root", None) or env["OUT_ROOT"]

    log_dir = env["LOG_DIR"]
    reports_dir = env["REPORTS_DIR"]

    valid_sources = {"lodopab", "dicom", "lidc_meta"}
    if source not in valid_sources:
        raise ValueError(f"SOURCE must be one of {valid_sources}, got {source!r}")
    
    valid_norms = {"none", "minmax", "zscore"}
    if norm not in valid_norms:
        raise ValueError(f"NORM myst be one of {valid_norms}, got {norm!r}")
    
    if clip_hi <= clip_lo:
        raise ValueError(f"CLIP_HI must be > CLI_LO (got {clip_hi}..{clip_lo})")
    
    if max_items < 0:
        raise ValueError(f"MAX_ITEMS must be >= 0, got {max_items}")

    return PrepConfig(
        seed=seed,
        source=source,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        norm=norm,
        max_items=max_items,
        lodopab_root=lodopab_root,
        dicom_root=dicom_root,
        out_root=out_root,
        log_dir=log_dir,
        reports_dir=reports_dir,
    )