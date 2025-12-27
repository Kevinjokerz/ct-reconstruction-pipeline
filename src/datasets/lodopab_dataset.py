"""
lodopab_dataset.py — LoDoPaB 2D slice dataset for reconstruction training.

This module provides a Dataset wrapper around preprocessed LoDoPaB slices
stored as .npy files + manifest.csv (see prepare_data.py).

Design goals
------------
- Phase 1: full-slice supervision (no patches).
- Phase 2: optional patch-based sampling via `patch_size`.
- Interface compatible with future LIDC patch dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.utils import (
    get_project_root,
    read_csv,
    setup_logger,
    to_relpath
)

@dataclass
class LoDoPaBSliceConfig:
    """
    Configuration for the LoDoPaB 2D slice dataset.

    All paths are interpreted relative to the project root.
    """
        
    manifest_rel: str = (
        "data/prepared/lodopab/train/train_manifest.csv"
    )

    split: str = "train"
    patch_size: Optional[int] = None    # None -> full slice

    noise_mode: str = "identity"        # "identity", "gaussian", "poisson"
    noise_std: float = 0.01             # used for Gaussian noise

    # Poisson noise is controlled via a pseudo photon-count scale.
    # Larger values -> higher SNR (weaker noise).
    poisson_scale: float = 1e4

    seed: int = 42

class LoDoPaBSliceDataset(Dataset):
    """
    LoDoPaB 2D slice dataset.

    Each item is a dict with:

    - ``"input"``  : noisy or clean input image
    - ``"target"`` : clean ground-truth image
    - ``"meta"``   : metadata dict (index, path)

    Notes
    -----
    - Assumes slices are pre-normalized to [0, 1] by prepare_lodopab().
    - For Phase 1, noise_mode = "identity" so input == target.
      Later, we can replace input by FBP(low-dose) or add synthetic noise.
    """

    def __init__(self, cfg: LoDoPaBSliceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = get_project_root()
        self.logger = setup_logger(f"lodopab_dataset_{cfg.split}")

        manifest_path = self.root / cfg.manifest_rel
        self.df = self._load_manifest(manifest_path)

        self.rng = np.random.default_rng(cfg.seed)

        self.logger.info(
            "[LoDoPaB:dataset] split=%s rows=%d manifest=%s",
            cfg.split,
            len(self.df),
            to_relpath(manifest_path),
        )
    
    def _load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        """
        Load and validate the LoDoPaB manifest CSV.
        """
        df = read_csv(manifest_path)
        required_cols = {"index", "path", "min", "max"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"[LoDoPaB:manifest] Missing required columns: {sorted(missing)}")
        
        return df
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_slice_np(self, idx: int) -> np.ndarray:
        """
        Load a LoDoPaB slice as a float32 array with shape (H, W).

        Parameters
        ----------
        idx :
            Row index into the manifest DataFrame.

        Returns
        -------
        np.ndarray
            2D image array, dtype float32, normalized to [0, 1].
        """

        row = self.df.iloc[idx]
        rel_path = str(row["path"])
        abs_path = self.root / rel_path

        if not abs_path.is_file():
            raise FileNotFoundError(f"[LoDoPaB:load_slice] slice not found at {rel_path}")
        
        x = np.load(abs_path)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        
        assert 0.0 <= x.min() + 1e-6 and x.max() <= 1.0 + 1e-6, (
            "[LoDoPaB:load_slice] expected normalized slice in [0, 1]"
        )
        return x
    
    def _maybe_random_crop(self, x: np.ndarray) -> np.ndarray:
        """
        Optionally apply a random square crop of size `patch_size`.

        Phase 1: patch_size is None -> return full slice.
        """
        if self.cfg.patch_size is None:
            return x
        
        patch = int(self.cfg.patch_size)
        h, w = x.shape
        if patch > min(h, w):
            raise ValueError(
                f"patch_size={patch} larger than image size {x.shape}"
            )
        
        r0 = self.rng.integers(0, h - patch + 1)
        c0 = self.rng.integers(0, w - patch + 1)
        x = x[r0 : r0 + patch, c0 : c0 + patch]
        return x
    
    def _apply_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Apply synthetic noise to a normalized slice.
        """

        if self.cfg.noise_mode == "identity":
            return x
        
        if self.cfg.noise_mode == "gaussian":
            noise = self.rng.normal(0.0, self.cfg.noise_std, size=x.shape)
            noisy = x + noise.astype(np.float32)
            noisy = np.clip(noisy, 0.0, 1.0)
            return noisy
        
        if self.cfg.noise_mode == "poisson":
            scale = float(self.cfg.poisson_scale)
            if scale <= 0.0:
                raise ValueError(
                    f"[LoDoPaB:noise] poisson_scale must be > 0, got {scale}"
                )
            
            lam = x.astype(np.float64) * scale
            lam = np.clip(lam, 1e-6, None)

            counts = self.rng.poisson(lam)
            noisy = counts.astype(np.float32) / scale
            noisy = np.clip(noisy, 0.0, 1.0)

            return noisy
        
        raise ValueError(f"Unknown noise_mode={self.cfg.noise_mode}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single training sample.

        Returns
        -------
        dict
            Dictionary with keys ``"input"``, ``"target"``, and ``"meta"``.
        """
        x = self._load_slice_np(idx)
        x = self._maybe_random_crop(x)

        target = x
        noisy = self._apply_noise(x)

        inp_t = torch.from_numpy(noisy)[None, ...]
        tgt_t = torch.from_numpy(target)[None, ...]

        sample = {
            "input": inp_t,
            "target": tgt_t,
            "meta": {
                "manifest_index": int(self.df.iloc[idx]["index"]),
                "path": str(self.df.iloc[idx]["path"]),
                "split": self.cfg.split,
            },
        }

        return sample

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    cfg = LoDoPaBSliceConfig(
        manifest_rel="data/prepared/lodopab/train/train_manifest.csv",
        split="train",
        noise_mode="identity",   # thử "gaussian" / "poisson" sau
    )

    ds = LoDoPaBSliceDataset(cfg)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    batch = next(iter(dl))
    print(batch["input"].shape, batch["input"].dtype)
    print(batch["target"].shape, batch["target"].dtype)
    print(batch["meta"])
