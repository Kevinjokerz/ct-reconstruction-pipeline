"""
utils — Unified Public API for Core Utilities (MedIm CT)

Purpose
-------
Expose only *stable* utilities used across the CT reconstruction pipeline:

- repro      : seeds & determinism
- paths      : project-root-relative helpers (no absolute leak)
- sanitize   : JSON/path sanitization + assertions
- io         : CSV I/O helpers with path-safety (manifests, metrics)
- logger     : opt-in setup (no side effects on import)
- config     : CT-specific PrepConfig loaded from CLI + .env

Notes
-----
- Keep this surface minimal and stable to avoid breaking imports elsewhere.
- Do NOT initialize logging or config automatically on import.
"""


# --- Reproducibility ---------------------------------------------------------
from .repro import DEFAULT_SEED, set_seed

# --- Paths -------------------------------------------------------------------
from .paths import get_project_root, looks_absolute, to_relpath, safe_join_rel

# --- Sanitization ------------------------------------------------------------
from .sanitize import (
    sanitize_json_like,
    sanitize_path_columns,
    assert_no_abs_in_df,
    assert_no_abs_in_json_series,
    to_list_str,
)

# --- I/O ---------------------------------------------------------------------
from .io import (
    ensure_parent_dir,
    read_csv,
    save_df_csv,
)

# --- Logging -----------------------------------------------------------------
from .logger import setup_logger

# --- Config ------------------------------------------------------------------
from .config import PrepConfig, load_config

# --- Public API (explicit) ---------------------------------------------------
__all__ = [
    # repro
    "DEFAULT_SEED",
    "set_seed",

    # paths
    "get_project_root",
    "looks_absolute",
    "to_relpath",
    "safe_join_rel",

    # sanitize
    "sanitize_json_like",
    "sanitize_path_columns",
    "assert_no_abs_in_df",
    "assert_no_abs_in_json_series",
    "to_list_str",

    # io
    "ensure_parent_dir",
    "read_csv",
    "save_df_csv",

    # logger
    "setup_logger",

    # config
    "PrepConfig",
    "load_config",
]
