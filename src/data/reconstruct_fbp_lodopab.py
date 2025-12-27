from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from skimage.transform import iradon
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.paths import get_project_root, to_relpath

from src.data.prepare_data import clip_array, normalize_minmax as prep_minmax

@dataclass(frozen=True)
class FBPReconstructConfig:
    """
    Configuration for LoDoPaB FBP reconstruction.

    Parameters
    ----------
    split :
        One of "train", "val", "test".
    raw_root_rel :
        Relative path (from project root) to the raw LoDoPaB HDF5 directory.
        Expected to contain observation_* and ground_truth_* HDF5 files.
    out_root_rel :
        Relative path for output directory, where reconstructed .npy
        and manifest CSV will be saved.
    clip_lo, clip_hi :
        Clip bounds before min-max normalization.
    """

    split: str = "train"
    raw_root_rel: str = "data/raw/lodopab"
    out_root_rel: str = "data/prepared/lodopab_fbp"
    clip_lo: float = 0.0
    clip_hi: float = 1.0


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for FBP reconstruction script.
    """
    parser = argparse.ArgumentParser(
        description="Precompute LoDoPaB FBP reconstruction from sinograms."
    )

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--raw_root", type=str, default="data/raw/lodopab", help="Relative path to LoDoPaB HDF5 directory.")
    parser.add_argument("--out_root", type=str, default="data/prepared/lodopab_fbp", help="Relative path for FBP outputs.")
    parser.add_argument("--clip_lo", type=float, default=0.0)
    parser.add_argument("--clip_hi", type=float, default=1.0)

    return parser.parse_args()

def simple_fbp(sino: np.ndarray) -> np.ndarray:
    """
    Simple FBP reconstruction using skimage.iradon.

    Parameters
    ----------
    sino :
        2D sinogram with shape (num_angles, num_detectors).

    Returns
    -------
    np.ndarray
        Reconstructed image, float32 with shape (H, W).
    """
    sino_T = sino.T
    num_angles = sino_T.shape[1]
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    recon = iradon(
        sino_T,
        theta=theta,
        filter_name="ramp",                 # Consider "hann" / "shepp-logan"
        circle=True,
    )

    recon = recon.astype(np.float32, copy=False)
    return recon

def normalize_minmax(x: np.ndarray, clip: Tuple[float, float]) -> Tuple[np.ndarray, float, float]:
    """
    Clip and min-max normalize image into [0, 1].

    Uses the same helpers as src.data.prepare_data to keep
    LoDoPaB FBP preprocessing consistent with other CT data.
    """
    clip_lo, clip_hi = clip

    x_clip = clip_array(x, clip_lo, clip_hi)

    mn = float(x_clip.min())
    mx = float(x_clip.max())

    x_norm = prep_minmax(x_clip, clip_lo, clip_hi)

    return x_norm, mn, mx

def find_hdf5_files(raw_root: Path, split: str) -> Tuple[List[Path], List[Path]]:
    """
    Locate observation_* and ground_truth_* HDF5 files for the given split.
    """
    obs_paths = sorted(raw_root.glob(f"observation_{split}_*.hdf5"))
    gt_paths  = sorted(raw_root.glob(f"ground_truth_{split}_*.hdf5"))

    assert len(obs_paths) == len(gt_paths) > 0, f"Mismatched LoDoPaB HDF5 files"
    return obs_paths, gt_paths

def main() -> None:
    """
    Entry point for FBP reconstruction.

    Responsibilities
    ----------------
    - Parse CLI args -> FBPReconstructConfig.
    - Iterate over LoDoPaB sinograms and GT images.
    - Compute FBP reconstructions.
    - Normalize & save .npy files plus manifest CSV.
    """
    args = parse_args()
    cfg = FBPReconstructConfig(
        split=args.split,
        raw_root_rel=args.raw_root,
        out_root_rel=args.out_root,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
    )
    project_root = get_project_root()
    raw_root = (project_root / cfg.raw_root_rel).resolve()
    out_root = (project_root / cfg.out_root_rel / cfg.split).resolve()

    out_root.mkdir(exist_ok=True, parents=True)

    logger = setup_logger(name=f"fbp_lodopab_{cfg.split}")
    logger.info("[FBP:lodopab_%s] FBP config %s", cfg.split, cfg)

    obs_paths, gt_paths = find_hdf5_files(raw_root, cfg.split)

    manifest_rows = []
    global_index = 0

    for obs_path, gt_path in zip(obs_paths, gt_paths):
        with h5py.File(obs_path, "r") as f_obs, h5py.File(gt_path, "r") as f_gt:
            obs_data = f_obs["data"]
            gt_data = f_gt["data"]
            num_samples = obs_data.shape[0]

            for local_idx in tqdm(range(num_samples), desc=f"{obs_path.name}", dynamic_ncols=True):
                sino = obs_data[local_idx].astype(np.float32, copy=False)
                gt = gt_data[local_idx].astype(np.float32, copy=False)

                fbp = simple_fbp(sino)

                fbp_norm, fbp_min, fbp_max = normalize_minmax(
                    fbp, (cfg.clip_lo, cfg.clip_hi)
                )
                gt_norm, gt_min, gt_max = normalize_minmax(
                    gt, (cfg.clip_lo, cfg.clip_hi)
                )

                # ---- Save .npy files ----
                fbp_name = f"fbp_{global_index:07d}.npy"
                gt_name = f"gt_{global_index:07d}.npy"

                fbp_path_abs = out_root / fbp_name
                gt_path_abs = out_root / gt_name

                np.save(fbp_path_abs, fbp_norm)
                np.save(gt_path_abs, gt_norm)

                fbp_path_rel = to_relpath(fbp_path_abs)
                gt_path_rel = to_relpath(gt_path_abs)

                manifest_rows.append(
                    {
                        "index": global_index,
                        "path_fbp": fbp_path_rel,
                        "path_gt": gt_path_rel,
                        "fbp_min": fbp_min,
                        "fbp_max": fbp_max,
                        "gt_min": gt_min,
                        "gt_max": gt_max,
                        "split": cfg.split,
                    }
                )
                global_index += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "fbp_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("[FBP:lodopab_%s] FBP manifest %s", cfg.split, manifest_path)

if __name__ == "__main__":
    main()

