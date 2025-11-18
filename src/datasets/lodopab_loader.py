"""
lodopab_loader.py — PyTorch Dataset for prepared LoDoPaB slices.

Purpose
-------
Load normalized LoDoPaB ground-truth images saved by `src.data.prepare_data`
from a manifest CSV + .npy files.

Typical manifest schema (per `prepare_lodopab`):
    index,path,min,max[,mean,std]

- `path` is a **project-root-relative** POSIX path
  (e.g. "data/prepared/lodopab_debug/train/train_000000.npy").
- Values in .npy are float32, typically in [0, 1] after minmax normalization.

This Dataset exposes slices as:
    - image : torch.float32 tensor of shape [1, H, W]
    - meta  : dict with index, stats, and absolute path (optional)

Usage (example)
---------------
>>> from src.datasets.lodopab_loader import LoDoPaBSliceDataset, LoDoPaBConfig
>>> cfg = LoDoPaBConfig(
...     manifest_rel="data/prepared/lodopab_debug/train/train_manifest.csv"
... )
>>> ds = LoDoPaBSliceDataset(cfg)
>>> sample = ds[0]
>>> img = sample["image"]        # torch.float32 [1, H, W]
>>> meta = sample["meta"]        # dict

Notes
-----
- No data augmentation is done here; pass callable `transform` to apply
  random crops/flips outside this core Dataset.
- No "low-dose" input is modeled yet — this Dataset **only** yields GT images.
  Later we can extend to (input, target) pairs (e.g. FBP vs GT).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import (
    read_csv,
    safe_join_rel,
    get_project_root
)


@dataclass(frozen=True)
class LoDoPaBConfig:
    """
    Configuration for LoDoPaB slice Dataset.

    Parameters
    ----------
    manifest_rel : str
        Relative path (from project root) to the manifest CSV produced by
        `prepare_lodopab`. Example:
            "data/prepared/lodopab_debug/train/train_manifest.csv"
    transform    : Optional[Callable]
        Optional transform applied to the image tensor **after** loading.
        Signature: transform(image: torch.Tensor) -> torch.Tensor
    """

    manifest_rel: str = "data/prepared/lodopab/train/train_manifest.csv"
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

class LoDoPaBSliceDataset(Dataset):
    """
    PyTorch Dataset for prepared LoDoPaB slices.

    Each __getitem__ returns a dict:
        {
            "image": torch.float32 [1, H, W],
            "meta": {
                "index": int,
                "min": float,
                "max": float,
                "rel": str,
                "path_abs": str,
            },
        }
    """
    def __init__(self, cfg: LoDoPaBConfig) -> None:
        super().__init__()
        self.cfg = cfg

        project_root = get_project_root()
        self.manifest_path: Path = safe_join_rel(cfg.manifest_rel, root=project_root)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"[DATASET:LoDoPaB] LoDoPaB manifest not found: {self.manifest_path}")
        
        df = read_csv(self.manifest_path)
        required_cols = {"index", "path", "min", "max"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"[DATASET:LoDoPaB] Manifest {self.manifest_path} missing columns: {missing}")
        
        self._paths_rel = df["path"].astype(str).tolist()
        self._mins = df["min"].astype(float).to_numpy()
        self._maxs = df["max"].astype(float).to_numpy()
        self._means = df["mean"].astype(float).to_numpy() if "mean" in df.columns else None
        self._std = df["std"].astype(float).to_numpy() if "std" in df.columns else None

        self._project_root = project_root
        self._transforms = cfg.transform

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
            raise ValueError(f"[DATASET:LoDoPaB] Non-finite values in slice at {path_abs}")
        
        return arr
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        arr = self._load_slice(idx)

        img = torch.from_numpy(arr).unsqueeze(0)

        if self._transforms is not None:
            img = self._transforms(img)

        rel = self._paths_rel[idx]
        path_abs = safe_join_rel(rel, root=self._project_root)

        meta: Dict[str, Any] = {
            "index": idx,
            "min": float(self._mins[idx]),
            "max": float(self._maxs[idx]),
            "path_rel": str(rel),
            "path_abs": str(path_abs),
        }

        if self._means is not None and self._std is not None:
            meta["mean"] = float(self._means[idx])
            meta["std"] = float(self._std[idx])


        sample: Dict[str, Any] = {
            "image": img,
            "meta": meta,
        }

        return sample
