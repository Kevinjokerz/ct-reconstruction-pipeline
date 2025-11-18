"""
I/O Utilities — MedIm CT Project

Purpose
-------
Portable, PHI-safe helpers to read/write CSV tables for CT reconstruction:
- slice-level manifests (LoDoPaB, LIDC),
- metrics tables (PSNR, SSIM, NRMSE, etc.).

Guard rails
-----------
- Forbid absolute path leakage in saved CSVs.
- Normalize any path-like columns to project-relative POSIX paths.

Convention
----------
- Reproducibility: deterministic; minimal side effects.
- Portability: use project-relative paths in logs; assert no absolute paths
  in manifest/metrics tables.
- Single responsibility: each function does exactly one thing.
- Logging: INFO-level "[READ]" / "[SAVE]" with rel_path, rows, and cols.

Typical inputs
--------------
- CSV manifests with at least a 'path' or '*_path' column.
- CSV metrics with slice- or volume-level statistics.

Typical outputs
---------------
- CSV files under 'data/prepared/' or 'reports/' with only relative paths.

Examples
--------
>>> from pathlib import Path
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "slice_idx": [0, 1],
...     "path": [
...         "data/prepared/lodopab/train/000001.npy",
...         "data/prepared/lodopab/train/000002.npy",
...     ],
... })
>>> save_df_csv(df, Path("reports/manifest_debug.csv"))
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence
import logging
import pandas as pd
from .paths import get_project_root, looks_absolute, to_relpath, safe_join_rel
from .sanitize import sanitize_path_columns, assert_no_abs_in_df

logger = logging.getLogger(__name__)

def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """
    Ensure parent directory for a file path exists.

    Parameters
    ----------
    path : str | Path
        Target file path (may be relative to project root).

    Returns
    -------
    Path
        Normalized Path object pointing to the same file.
    """
    if path is None:
        raise ValueError("Path cannot be None.")
    
    p = Path(path)
    parent = p.parent
    p.parent.mkdir(parents=True, exist_ok=True)
    assert parent.is_dir()
    return p


def read_csv(
    path: Union[str, Path], 
    *,
    nrows: Optional[int] = None, 
    usecols: Optional[Sequence[str]] = None, 
    encoding: str = "utf-8", 
    dtype: Optional[Union[str, dict]] = None,
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame with basic logging.

    Parameters
    ----------
    path : str | Path
        CSV path (absolute or relative). Absolute paths are allowed here,
        but they will be logged relative to project root.
    nrows : int, optional
        Number of rows to read (for sampling/debugging).
    usecols : sequence of str, optional
        Subset of columns to load.
    dtype : str, optional
        Force dtype for all columns (e.g., 'str' for text-heavy tables).
    encoding : str, default='utf-8'
        File encoding.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    
    df = pd.read_csv(
        p, nrows=nrows, usecols=usecols or None, 
        encoding=encoding, encoding_errors="replace", 
        low_memory=False, dtype=dtype
    )
    rel = to_relpath(p) if p.is_absolute() else p.as_posix()
    logger.info("[READ] %s rows=%s col=%s", rel, f"{len(df):,}", df.shape[1])
    assert isinstance(df, pd.DataFrame)
    return df

def save_df_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    sanitize_paths: bool = True,
    index: bool = False,
) -> Path:
    """
    Save a DataFrame as CSV with path-sanitization and logging.

    This is intended for:
    - manifest tables (LoDoPaB, LIDC),
    - metrics tables (PSNR/SSIM per slice/volume).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str | Path
        Output path (relative to project root or absolute).
    sanitize_paths : bool, default=True
        If True, apply `sanitize_path_columns` before saving.
    index : bool, default=False
        Whether to write the DataFrame index.

    Returns
    -------
    Path
        Absolute path to the saved CSV.
    """

    p = ensure_parent_dir(path)
    df_out = df.copy()

    if sanitize_paths:
        df_out = sanitize_path_columns(df_out)
        assert_no_abs_in_df(df_out)
    
    df_out.to_csv(p, index=index)
    rel = to_relpath(p) if p.is_absolute() else p.as_posix()
    logger.info("[SAVE] %s rows=%s cols=%s", rel, f"{len(df_out):,}", df_out.shape[1])

    assert p.is_file()
    return p