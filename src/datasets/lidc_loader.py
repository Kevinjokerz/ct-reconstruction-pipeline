"""
lidc_loader.py — PyTorch Dataset for prepared LIDC/DICOM CT slices.

Purpose
-------
Load normalized CT slices (e.g., LIDC-IDRI volumes) saved by
`src.data.prepare_data.prepare_dicom` from a manifest CSV + .npy files.

Typical manifest schema (per `prepare_dicom`):
    index,path,min,max

- `path` is a **project-root-relative** POSIX path
  (e.g. "data/prepared/ct_debug/slice_0000.npy").
- Values in .npy are float32, typically in [0, 1] after minmax normalization
  or z-scored (depending on config).

This Dataset exposes slices as:
    - image : torch.float32 tensor of shape [1, H, W]
    - meta  : dict with index, stats, spacing, and absolute path (optional)

Usage (example)
---------------
>>> from src.datasets.lidc_loader import LIDCDicomConfig, LIDCDicomSliceDataset
>>> cfg = LIDCDicomConfig(
...     manifest_rel="data/prepared/ct_debug/manifest.csv"
... )
>>> ds = LIDCDicomSliceDataset(cfg)
>>> sample = ds[0]
>>> img = sample["image"]        # torch.float32 [1, H, W]
>>> meta = sample["meta"]        # dict

Notes
-----
- No data augmentation is done here; pass callable `transform` to apply
  random crops/flips outside this core Dataset.
- This loader is agnostic to the original scanner/vendor; all geometry /
  quantitative info comes from the prepared meta.json (if present).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import (
    read_csv,
    safe_join_rel,
    get_project_root,
)

@dataclass(frozen=True)
class LIDCDicomConfig:
    """
    Configuration for LIDC/DICOM slice Dataset.

    Parameters
    ----------
    manifest_rel : str
        Relative path (from project root) to the manifest CSV produced by
        `prepare_dicom`. Example:
            "data/prepared/ct_debug/manifest.csv"
    meta_rel     : Optional[str]
        Relative path to the JSON metadata file. If None, we will
        look for a file named "meta.json" in the same directory as
        `manifest_rel`.
    transform    : Optional[Callable]
        Optional transform applied to the image tensor **after** loading.
        Signature: transform(image: torch.Tensor) -> torch.Tensor
    """

    manifest_rel: str = "data/prepared/ct/manifest.csv"
    meta_rel: Optional[str] = None
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

class LIDCDicomSliceDataset(Dataset):
    """
    PyTorch Dataset for prepared LIDC/DICOM CT slices.

    Each __getitem__ returns a dict:
        {
            "image": torch.float32 [1, H, W],
            "meta": {
                "index": int,
                "min": float,
                "max": float,
                "path_rel": str,
                "path_abs": str,
                "spacing_mm": {"px": float, "py": float, "dz": float} | None,
                "quant_meta": {...} | None,
            },
        }
    """
    def __init__(self, cfg: LIDCDicomConfig) -> None:
        super().__init__()
        self.cfg = cfg

        project_root = get_project_root()
        self._project_root: Path = project_root

        self.manifest_path: Path = safe_join_rel(cfg.manifest_rel, root=project_root)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"[DATASET:LIDC] CT manifest not found: {self.manifest_path}")
        
        df = read_csv(self.manifest_path)
        required_cols = {"index", "path", "min", "max"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"[DATASET:LIDC] Manifest {self.manifest_path} missing columns: {missing}"
            )
        
        self._paths_rel = df["path"].astype(str).tolist()
        self._mins = df["min"].astype(float).to_numpy()
        self._maxs = df["max"].astype(float).to_numpy()

        self._means = df["mean"].astype(float).to_numpy() if "mean" in df.columns else None
        self._stds = df["std"].astype(float).to_numpy() if "std" in df.columns else None

        self._spacing_mm: Optional[Dict[str, Any]] = None
        self._quant_meta: Optional[Dict[str, Any]] = None

        meta_path = None
        if cfg.meta_rel is not None:
            meta_path = safe_join_rel(cfg.meta_rel, root=project_root)
        else:
            meta_path = self.manifest_path.parent / "meta.json"
        
        if meta_path is not None and Path(meta_path).is_file():
            with open(meta_path, "r") as f:
                meta_json = json.load(f)
            
            self._spacing_mm = meta_json.get("spacing_mm", None)
            self._quant_meta = meta_json.get("quant_meta", None)
        
        self._transform = cfg.transform

    def __len__(self) -> int:
        return len(self._paths_rel)
    
    def _load_slice(self, idx: int) -> np.ndarray:
        """
        Load a single .npy slice as float32 ndarray [H, W].

        Notes
        -----
        - Enforces dtype float32.
        - Asserts no NaNs/Infs.
        """
        rel = self._paths_rel[idx]
        path_abs = safe_join_rel(rel, root=self._project_root)

        arr = np.load(path_abs)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[DATASET:LIDC] Non-finite values in slice at {path_abs}"
            )
        
        return arr
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        arr = self._load_slice(idx)

        img = torch.from_numpy(arr).unsqueeze(0)

        if self._transform is not None:
            img = self._transform(img)

        rel = self._paths_rel[idx]
        path_abs = safe_join_rel(rel, root=self._project_root)

        meta: Dict[str, Any] = {
            "index": int(idx),
            "min": float(self._mins[idx]),
            "max": float(self._maxs[idx]),
            "path_rel": str(rel),
            "path_abs": str(path_abs),
        }

        if self._means is not None and self._stds is not None:
            meta["mean"] = float(self._means[idx])
            meta["std"] = float(self._stds[idx])

        if self._spacing_mm is not None:
            meta["spacing_mm"] = self._spacing_mm
        
        if self._quant_meta is not None:
            meta["quant_meta"] = self._quant_meta
        
        return {
            "image": img,
            "meta": meta,
        }
    