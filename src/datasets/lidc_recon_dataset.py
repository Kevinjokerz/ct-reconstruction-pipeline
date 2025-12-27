from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.lidc_loader import LIDCDicomSliceDataset, LIDCDicomConfig

@dataclass(frozen=True)
class LIDCReconConfig:
    """
    Configuration for LIDCReconDataset.

    Parameters
    ----------
    manifest_rel :
        Relative path (from project root) to the LIDC manifest CSV.
        Example: "data/prepared/lidc/splits/lidc_train_manifest.csv".
    patch_size :
        Optional square patch size in pixels. If None, use full slice.
    noise_mode :
        Which synthetic noise to apply to build the input image.
        Supported: "identity", "gaussian", "poisson".
    noise_std :
        Standard deviation for Gaussian noise when noise_mode="gaussian".
    poisson_scale :
        Photon-count scale for Poisson noise when noise_mode="poisson".
    seed :
        Random seed for this dataset (controls crop + noise RNG).
    """
    manifest_rel: str = "data/prepared/lidc/splits/lidc_train_manifest.csv"

    patch_size: Optional[int] = None
    noise_mode: str = "identity"
    noise_std: float = 0.01
    poisson_scale: float = 1e4
    seed: int = 42

class LIDCReconDataset(Dataset):
    """
    LIDC reconstruction dataset: synthetic low-dose -> clean target.

    Base loader: LIDCDicomSliceDataset (GT slice already normalized).

    Each sample
    -----------
    Returns a dict with:
        "input"  : torch.float32 tensor [1, Hc, Wc]
            Synthetic noisy LIDC image.
        "target" : torch.float32 tensor [1, Hc, Wc]
            Clean LIDC slice (GT).
        "meta"   : dict
            Metadata from base loader plus domain + noise info.
    """

    def __init__(self, cfg: LIDCReconConfig) -> None:
        super().__init__()
        self.cfg = cfg

        base_cfg = LIDCDicomConfig(
            manifest_rel=cfg.manifest_rel,
            meta_rel=None,
            transform=None,
        )
        self.base_ds = LIDCDicomSliceDataset(base_cfg)

        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.base_ds)

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _maybe_random_crop(self, x: np.ndarray) -> np.ndarray:
        """
        Random square crop if self.cfg.patch_size is not None.

        Parameters
        ----------
        x :
            2D GT slice, float32, shape (H, W).

        Returns
        -------
        np.ndarray
            2D patch (Hc, Wc) or original slice if no cropping.
        """
        ps = self.cfg.patch_size
        if ps is None:
            return x

        H, W = x.shape
        assert ps <= H and ps <= W, "[DATASET:random_crop] patch_size larger than image"

        i0 = self.rng.integers(0, H - ps + 1)
        j0 = self.rng.integers(0, W - ps + 1)

        return x[i0:i0 + ps, j0:j0 + ps]

    def _apply_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Apply synthetic noise according to cfg.noise_mode.

        Notes
        -----
        - Assumes x is already normalized (e.g., minmax ~ [0, 1])
          if you used minmax in prepare_data.
        - If using z-score, adjust clipping / scaling logic accordingly.

        Parameters
        ----------
        x :
            2D patch GT after cropping, float32.

        Returns
        -------
        np.ndarray
            Noisy input patch, same shape & dtype as x.
        """
        mode = self.cfg.noise_mode.lower()

        if mode == "identity":
            return x.copy()

        if mode == "gaussian":
            noise = self.rng.normal(loc=0.0, scale=self.cfg.noise_std, size=x.shape).astype(np.float32)
            y = x + noise
            y = np.clip(y, 0.0, 1.0)
            return y

        if mode == "poisson":
            scale = float(self.cfg.poisson_scale)
            x_clip = np.clip(x, 0.0, 1.0)
            photons = self.rng.poisson(x_clip * scale).astype(np.float32)
            y = photons / scale
            return y

        raise ValueError(f"[DATASET:apply_noise] Unknown noise mode for LIDCReconDataset: {self.cfg.noise_mode}")

    # ---------------------------------------------------------
    # __getitem__
    # ---------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Build one training sample for LIDC.

        Steps
        -----
        - Load clean slice from base loader.
        - Convert to numpy 2D.
        - Random-crop if configured.
        - Target = cropped clean patch.
        - Input  = noisy version of target.
        - Convert to torch with channel dim = 1.

        Returns
        -------
        dict
            {"input": Tensor, "target": Tensor, "meta": dict}
        """
        base_sample = self.base_ds[idx]
        img_t: Tensor = base_sample["image"]
        meta_in: Dict[str, Any] = base_sample["meta"]

        x = img_t.squeeze(0).cpu().numpy()
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        x_crop = self._maybe_random_crop(x)
        target_np = x_crop
        input_np = self._apply_noise(x_crop)

        target_t = torch.from_numpy(target_np).unsqueeze(0).float()
        input_t = torch.from_numpy(input_np).unsqueeze(0).float()

        if self.cfg.patch_size is None:
            patch_meta: int | str = "full"
        else:
            patch_meta = int(self.cfg.patch_size)

        meta_out: Dict[str, Any] = {
            **meta_in,
            "domain": "lidc",
            "noise_mode": self.cfg.noise_mode,
            "patch_size": patch_meta,
        }

        return {"input": input_t, "target": target_t, "meta": meta_out}

