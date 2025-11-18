from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.utils import (
    get_project_root,
    read_csv,
    save_df_csv,
    setup_logger,
    set_seed,
    to_relpath,
)

@dataclass
class LIDCMergeConfig:
    """
    Merge per-series LIDC manifest.csv files into a single slice-level manifest.
    """

    prepared_root_rel: str = "data/prepared/lidc"
    out_manifest_rel: str = "data/prepared/lidc/all_slices_manifest.csv"
    seed: int = 42

def _iter_manifest_paths(root: Path) -> List[Path]:
    """
    Recursively collect all per-series ``manifest.csv`` files under root.

    Expected layout
    ---------------
    root/
      LIDC-IDRI-0001/
        <SeriesUID1>/
          manifest.csv
          slice_0000.npy
          ...
        <SeriesUID2>/
          manifest.csv
          ...
      LIDC-IDRI-0002/
        ...
    """
    manifest_paths: List[Path] = []
    for p in root.rglob("manifest.csv"):
        manifest_paths.append(p)
    
    return manifest_paths

def merge_manifests(cfg: LIDCMergeConfig) -> None:
    logger = setup_logger("lidc_merge_manifests")
    set_seed(cfg.seed)

    root = get_project_root()
    prepared_root = (root / cfg.prepared_root_rel).resolve()
    out_manifest_path = root / cfg.out_manifest_rel
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_paths = _iter_manifest_paths(prepared_root)
    logger.info("[LIDC:merge] Found %d LIDC manifest.csv files under %s", len(manifest_paths), to_relpath(prepared_root))

    all_rows: List[pd.DataFrame] = []

    for mpath in manifest_paths:
        series_dir = mpath.parent
        series_uid = series_dir.name
        case_id = series_dir.parent.name

        df = read_csv(mpath)
        assert "index" in df.columns and "path" in df.columns

        df["case_id"] = case_id
        df["series_uid"] = series_uid

        all_rows.append(df)
    
    if not all_rows:
        logger.warning("[LIDC:merge] No manifest.csv files found; nothing to merge.")
        return
    
    merged = pd.concat(all_rows, ignore_index=True)

    logger.info("[LIDC:merge] Merged manifest shape: %s", merged.shape)
    logger.info("[LIDC:merge] Merged head:\n%s", merged.head().to_string())

    save_df_csv(merged, out_manifest_path)
    logger.info("[LIDC:merge] Saved merged LIDC manifest to %s", to_relpath(out_manifest_path))

def main() -> None:
    cfg = LIDCMergeConfig()
    merge_manifests(cfg)


if __name__ == "__main__":
    main()
