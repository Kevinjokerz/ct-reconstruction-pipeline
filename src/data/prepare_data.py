"""
prepare_data.py — CT data preparation (LoDoPaB + DICOM/LIDC)

Pipeline
--------
- LoDoPaB: ground truth images -> clip -> normalize -> .npy per slice + manifest + meta
- DICOM  : 3D HU volume -> clip -> normalize -> .npy per slice + manifest + meta

Notes
-----
- All saved paths must be project-root-relative (no absolute leakage).
- Config comes from CLI + .env via PrepConfig (see utils.config).
- Intended entrypoint: `python -m src.prepare_data ...`
"""

from __future__ import annotations
import os
import csv
import glob
import json
import argparse
from typing import Iterator, Tuple, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

from src.utils import (
    set_seed,
    setup_logger,
    get_project_root,
    to_relpath,
    ensure_parent_dir
)

from src.config.prep import load_config

import numpy as np
import pandas as pd

# =========================
# Utility: I/O + validation
# =========================

def _normalize_file_location(raw_location: str) -> str:
    """
    Normalize TCIA ``File Location`` strings into POSIX-style relative paths.

    Parameters
    ----------
    raw_location :
        Raw value from the TCIA LIDC-IDRI ``File Location`` column.
        Typical pattern: ``".\\\\LIDC-IDRI\\\\LIDC-IDRI-0001\\\\01-01-2000-NA-..."``.

    Returns
    -------
    str
        Normalized path using ``/`` as separator and without a leading dot,
        e.g. ``"LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-..."``.

    Notes
    -----
    The returned path is **relative** to the LIDC raw root directory
    (see ``--raw_root`` in the CLI).
    """
    raw = str(raw_location).strip()
    cleaned = raw.lstrip(".\\/")
    cleaned = cleaned.replace("\\", "/")

    assert "\\" not in cleaned, (
        f"[FILE LOCATION:norm] Backslash still present: {cleaned!r}"
    )

    return cleaned

def _norm_rel(p: Path, root: Path) -> str:
    """
    Return POSIX-style relative path to root (portable for CSV/JSON)
    """
    project_root = get_project_root()
    rel = to_relpath(p, root=project_root)
    assert ":" not in rel and not rel.startswith("/"), f"Absolute leaked: {rel}"
    return rel


def _validate_clip_bounds(bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Ensure lo < hi and return (lo, hi)"""
    lo, hi = bounds
    if not(np.isfinite([lo, hi]).all()):
        raise ValueError(f"--clip must be finite, got {bounds}")
    if not (lo < hi):
        raise ValueError(f"--clip requires lo < hi, got {bounds}")
    else:
        return float(lo), float(hi)

# =========================
# Intensity transforms
# =========================

def clip_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Clip values to [lo, hi]. Assumes x is float array.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.clip(x, lo, hi)

def normalize_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Map [lo, hi] -> [0, 1]
    Assumes x already clipped
    """
    eps = hi - lo
    if eps == 0.0:
        eps += 1e-8
    return ((x - lo) / eps).astype(np.float32, copy=False)

def normalize_zscore(x: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    Zero-mean, unit-variance normalization.
    If mean/std None -> compute from x.
    """
    if mean is None:
        mean = float(x.mean())
    if std is None:
        std = float(x.std() + 1e-8)
    return (x - mean) / std, mean, std

def _apply_real_world_value(ds, arr_float32: np.ndarray) -> np.ndarray:
    """
    Apply quantitative mapping for CT pixel values
    1) If RealWorldValueMappingSequence exists, use slope/intercept from it.
    2) Otherwise, fall back to RescaleSlope/RescaleIntercept
    Returns float32 (commonly HU for CT)
    """
    rwvm = getattr(ds, "RealWorldValueMappingSequence", None)
    if rwvm and len(rwvm) > 0:
        item = rwvm[0]
        slope = float(getattr(item, "RealWorldValueSlope", getattr(ds, "RescaleSlope", 1.0)))
        inter = float(getattr(item, "RealWorldValueIntercept", getattr(ds, "RescaleIntercept", 0.0)))
        return (arr_float32 * slope + inter).astype(np.float32, copy=False)
    
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return (arr_float32 * slope + inter).astype(np.float32, copy=False)

# =========================
# LoDoPaB branch (no HU)
# =========================

def load_lodopab_split(split: str = "train", max_items: Optional[int] = None) -> Iterator[np.ndarray]:
    """
    Yield ground truth 2D arrays from LoDoPaB.
    """
    from dival.datasets import LoDoPaBDataset
    ds = LoDoPaBDataset(impl='skimage')

    if split == 'train':
        num_sample = ds.get_len("train")
        idx_iter = range(num_sample)
    elif split == 'val':
        num_sample = ds.get_len("validation")
        idx_iter = range(num_sample)
    else:
        num_sample = ds.get_len("test")
        idx_iter = range(num_sample)
    
    count = 0
    for i in idx_iter:
        measurement, gt = ds.get_sample(i)          #(measurement, ground_truth)
        x = np.asarray(gt, dtype=np.float32)        #ground_truth image
        assert x.ndim == 2, f"Expected 2D slice, got {x.shape}"
        yield x
        count += 1
        if max_items is not None and count >= max_items:
            break


def prepare_lodopab(
        split: str,
        clip_bounds: Tuple[float, float] = (0.0, 1.0),
        norm: str = "minmax",
        out_dir: str = os.path.join("data", "prepared", "lodopab"),
        max_items: Optional[int] = None,
) -> None:
    """
    Pipeline:
        load -> clip -> normalize -> save per-slice .npy + manifest.csv
    """
    if max_items is not None and max_items <= 0:
        print(f"[WARN] max_items <= 0 nothing to do")
        return

    out_dir = os.path.join(out_dir, split)
    out_dir_p = Path(out_dir).resolve()
    ensure_parent_dir(out_dir_p / "dummy.txt")
    lo, hi = _validate_clip_bounds(clip_bounds)

    out_dir_p = Path(out_dir).resolve()
    try:
        data_root = out_dir_p.parents[2]        # .../data
    except IndexError:
        data_root = out_dir_p

    manifest_path = out_dir_p / f"{split}_manifest.csv"

    n_saved = 0
    gen = load_lodopab_split(split=split, max_items=max_items)

    #Collect summary for meta
    first_shape = None
    m_list, s_list = [], []


    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        header =  ["index", "path", "min", "max"] + (["mean", "std"] if norm == "zscore" else [])      #add more columns later if needed
        w.writerow(header)

        for i, arr in enumerate(gen):
            if first_shape is None:
                first_shape = tuple(arr.shape)

            if not np.issubdtype(arr.dtype, np.floating) or arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)

            x = clip_array(arr, lo, hi)

            if norm == "minmax":
                x = normalize_minmax(x, lo, hi).astype(np.float32)
            elif norm == "zscore":
                x, m, s = normalize_zscore(x)
                x = x.astype(np.float32)
                m_list.append(float(m))
                s_list.append(float(s))
            else:
                raise ValueError(f"Unknown norm: {norm}")

            out_path = out_dir_p / f"{split}_{i:06d}.npy"
            np.save(out_path, x)

            rel_path = _norm_rel(out_path, data_root)

            row = [i, rel_path, float(x.min()), float(x.max())]
            if norm == "zscore":
                row += [float(m), float(s)]

            w.writerow(row)
            n_saved += 1
            if(i + 1) % 500 == 0:
                logging.getLogger(__name__).info(f"[LoDoPaB] processed {i + 1} slices...")

    norm_meta = {"kind": norm}
    if norm == "minmax":
        norm_meta.update({"range_in": [float(lo), float(hi)], "range_out": [0.0, 1.0]})
    else:
        if len(m_list) > 0:
            norm_meta.update({
                "mean_min": float(np.min(m_list)),
                "mean_max": float(np.max(m_list)),
                "std_min": float(np.min(s_list)),
                "std_max": float(np.max(s_list)),
                "scope": "per_slice"
            })
    
    meta = {
        "source": "LoDoPaB",
        "split": split,
        "num_slices": int(n_saved),
        "slice_shape": list(first_shape) if first_shape is not None else None,
        "dtype": "float32",
        "clip_bounds": [float(lo), float(hi)],
        "normalization": norm_meta,
        "paths": {
            "prepared_dir": _norm_rel(out_dir_p, data_root),
            "manifest_csv": _norm_rel(manifest_path, data_root),
        }
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[LoDoPaB] saved {n_saved} slice(s) -> {out_dir}")

# =========================
# Clinical DICOM branch (HU)
# =========================

@dataclass
class LIDCMetaConfig:
    """
    Configuration for bulk preprocessing of LIDC-IDRI via TCIA metadata.

    All paths are interpreted **relative** to the project root.
    """
    metadata_rel: str = "data/raw/lidc-idri/manifest-1600709154662/metadata.csv"
    raw_root_rel: str = "data/raw/lidc-idri/manifest-1600709154662/"
    out_root_rel: str = "data/prepared/lidc"
    collection: str = "LIDC-IDRI"
    modality: str = "CT"

    clip_bounds: Tuple[float, float] = (-1000.0, 1000.0)
    norm: str = "minmax"

    # Limit on number of series for debugging
    max_series: Optional[int] = None

def prepare_lidc_from_metadata(config: LIDCMetaConfig, logger: logging.Logger) -> None:
    """
    Run bulk slice-level preprocessing for the LIDC-IDRI dataset.

    This function reads a TCIA-style ``metadata.csv`` file and, for each
    matching CT series, calls :func:`prepare_dicom` to:

    - convert DICOM to HU,
    - apply clipping and normalization,
    - save each axial slice as ``.npy``,
    - write a per-series ``manifest.csv`` and ``meta.json``.

    Parameters
    ----------
    config :
        LIDC-IDRI bulk preprocessing configuration.

    Notes
    -----
    - All paths inside ``metadata.csv`` *must* remain dataset-relative
      (no absolute drive letters).
    - The function is idempotent: if a target output directory already
      contains a ``manifest.csv``, that series is skipped.
    """
    root = get_project_root()

    metadata_path = root / config.metadata_rel
    raw_root = (root / config.raw_root_rel).resolve()
    out_root = (root / config.out_root_rel).resolve()

    # 1) Load metadata CSV
    df = pd.read_csv(metadata_path)
    logger.info("[LIDC] metadata columns: %s", df.columns.tolist())

    # 2) Filter by collection and modality if those columns exist.
    if "Collection" in df.columns and config.collection is not None:
        df = df[df["Collection"] == config.collection]
    if "Modality" in df.columns and config.modality is not None:
        df = df[df["Modality"] == config.modality]
    
    logger.info(f"[LIDC] {len(df)} CT series after filtering.")

    processed = 0

    for _, row in df.iterrows():
        subject_id = str(row["Subject ID"])
        series_uid = str(row["Series UID"])
        raw_loc = str(row["File Location"])

        rel_series_posix = _normalize_file_location(raw_loc)
        series_dir = raw_root / rel_series_posix

        if not series_dir.is_dir():
            logger.warning(f"[LIDC] Missing series dir: {series_dir}")
            continue

        out_dir = out_root / subject_id / series_uid
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "manifest.csv"
        if manifest_path.exists():
            logger.warning(f"[LIDC] skip existing series {subject_id} | {series_uid}")
            continue

        logger.info(f"[LIDC] prepare series {subject_id} | {series_uid}")
        
        lo, hi = _validate_clip_bounds(config.clip_bounds)
        prepare_dicom(
            dicom_dir=str(series_dir),
            clip_bounds=(lo, hi),
            norm=config.norm,
            out_dir=str(out_dir)
        )
        
        processed += 1
        if config.max_series is not None and processed >= config.max_series:
            logger.info(f"[LIDC] reached max_series={config.max_series}, stopping.")
            break

    logger.info(f"[LIDC] done. processed {processed} CT series.")



def _slice_sort_key(ds) -> Tuple[float, int]:
    """
    Compute a geometric sort key for a DICOM slice.

    - Row / Column direction cosines -> normal vector
    - Project ImagePositionPatient onto the normal to get slice position
    - Use InstanceNumber as tie-breaker
    """
    #Orientation (direction cosines)
    orientation = np.array(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
    row_dir, column_dir = orientation[:3], orientation[3:]
    normal_vec = np.cross(row_dir, column_dir)

    #Position of slice (3D coordinates of first pixel)
    ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=float)
    slice_pos = float(np.dot(ipp, normal_vec))

    instance_num = int(getattr(ds, "InstanceNumber", 0))
    return (slice_pos, instance_num)

def _sorted_dicom_paths(dicom_dir: str) -> List[str]:
    """
    Collect all .dcm files in a folder and return them sorted by slice position.

    Notes
    -----
    - Groups files by SeriesInstanceUID to avoid mixing different series.
    - Uses header-only read (fast, no pixel data).
    - Sorts by geometry (slice position via IOP/IPP) and InstanceNumber.
    """
    import pydicom, collections

    #Find DICOM files
    paths = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    
    #Group by SeriesInstanceUID
    by_uid = collections.defaultdict(list)
    for p in paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        uid = getattr(ds, "SeriesInstanceUID", None)
        by_uid[uid].append(p)

    #choose the largest group
    uid, paths = max(by_uid.items(), key=lambda kv: len(kv[1]))


    keyed = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        sort_key = _slice_sort_key(ds)      #(slice_position, instance_number)
        keyed.append((sort_key, path))

    #Sort by position (and instance number as tie-breaker)
    keyed.sort(key=lambda t: t[0])
    sorted_paths = [path for _, path in keyed]

    return sorted_paths

def load_dicom_stack(dicom_dir: str) -> np.ndarray:
    """
    Load a 3D CT series and return HU volume of shape (Z, H, W), float32.
    - Sort by geometry (IOP/IPP)
    - Apply RealWorldValueMappingSequence if available; otherwise RescaleSlope/RescaleIntercept.
    - Does NOT invert MONOCHROME1 (since this is display-related, not quantiative)
    """
    import pydicom

    logger = logging.getLogger(__name__)
    paths = _sorted_dicom_paths(dicom_dir)
    if not paths:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")
    
    slices_hu = []
    slopes, intercepts = set(), set()
    rwvm_flags = []

    for p in paths:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32, copy=False)

        slopes.add(float(getattr(ds, "RescaleSlope", 1.0)))
        intercepts.add(float(getattr(ds, "RescaleIntercept", 0.0)))
        rwvm = getattr(ds, "RealWorldValueMappingSequence", None)
        rwvm_flags.append(bool(rwvm and len(rwvm) > 0))

        #Do NOT invert MONOCHROME1 for quantitative CT
        #PhotometricInterpretation is only a display convention, not a quantitative mapping
        hu = _apply_real_world_value(ds, arr)       #prefer RWVM, fallback to slope/intercept
        slices_hu.append(hu)
    
    vol = np.stack(slices_hu, axis=0).astype(np.float32, copy=False)
    assert np.isfinite(vol).all(), "Non-finite HU detected"

    vmin, vmed, vmax = float(vol.min()), float(np.median(vol)), float(vol.max())
    if vmin <= -4000 or vmax >= 10000:
        logger.warning(
            f"[DICOM] HU range suspicious in {dicom_dir} -> "
            f"min={vmin:.1f}, med={vmed:.1f}, max={vmax:.1f}"
        )

        q_lo, q_hi = np.percentile(vol, [0.5, 99.5])
        vol = np.clip(vol, q_lo, q_hi, out=vol)

    vmin2, vmed2, vmax2 = float(vol.min()), float(np.median(vol)), float(vol.max())
    logger.info(f"[DICOM] vol shape = {vol.shape} HU[min, median, max]={vmin2:.1f},{vmed2:.1f},{vmax2:.1f}")

    return vol

def _compute_spacing(dicom_dir: str) -> Tuple[float, float, float]:
    """
    Return (row_spacing_mm, col_spacing_mm, dz_mm).
    - row_spacing_mm: mm per pixel along rows (DICOM PixelSpacing[0]) ~ Y
    - col_spacing_mm: mm per pixel along cols (DICOM PixelSpacing[1]) ~ X
    - dz_mm: median slice spacing along the slice normal (always >= 0)
    """
    import pydicom
    
    files = _sorted_dicom_paths(dicom_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    
    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)

    #Orientation -> normal vector
    iop = np.array(getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal_vec = np.cross(row_dir, col_dir)
    if not np.isfinite(normal_vec).all() or np.linalg.norm(normal_vec) < 1e-8:
        normal_vec = np.array([0.0, 0.0, 1.0], dtype=float)

    #Project IPP onto normal to get slice coordinate
    positions = []
    for p in files:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=float)
        positions.append(float(np.dot(ipp, normal_vec)))

    positions = np.sort(np.array(positions, dtype=float))
    if positions.size > 1:
        diffs = np.diff(positions)
        #robust: absolute in case of descending order or tiny negatives
        dz = float(np.median(np.abs(diffs)))
    else:
        #Prefer SpacingBetweenSlices if available; else SliceThickness; else 1.0
        dz = float(getattr(ds0, "SpacingBetweenSlices", 
                           getattr(ds0, "SliceThickness", 1.0)))
    
    #PixelSpacing: [row_spacing, col_spacing] (Y, X)
    px_vals = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    #Ensure floats even if strings
    row_spacing = float(px_vals[0])
    col_spacing = float(px_vals[1])
    
    #Ensure non-negative dz
    dz = abs(dz)

    return row_spacing, col_spacing, dz

def _read_quant_meta(dicom_dir: str) -> dict:
    """
    Read basic quantiative metadata from the first slice (best-effort).
    """
    import pydicom
    paths = _sorted_dicom_paths(dicom_dir)
    if not paths:
        return {}
    ds0 = pydicom.dcmread(paths[0], stop_before_pixels=True)

    photometric = getattr(ds0, "PhotometricInterpretation", None)
    rwvm = getattr(ds0, "RealWorldValueMappingSequence", None)
    rwvm_used = bool(rwvm and len(rwvm) > 0)

    slope = float(getattr(ds0, "RescaleSlope", 1.0))
    inter = float(getattr(ds0, "RescaleIntercept", 0.0))

    return {
        "photometric": photometric,
        "rwvm_present": rwvm_used,
        "rescale": {"slope": slope, "intercept": inter}
    }


def prepare_dicom(
        dicom_dir: str,
        clip_bounds: Tuple[float, float] = (-1000.0, 1000.0),
        norm: str = "minmax",
        out_dir: str = os.path.join("data", "prepared", "ct")
) -> None:
    """
    Pipeline:
        load_dicom_stack(HU) -> clip -> normalize -> save_per_slice .npy + manifest.csv
    """
    lo, hi = _validate_clip_bounds(clip_bounds)

    #Resolve paths and infer data_root (.../data) with safe fallback
    out_dir_p = Path(out_dir).resolve()
    ensure_parent_dir(out_dir_p / "dummy.txt")
    try:
        data_root = out_dir_p.parents[2]        # .../data
    except IndexError:
        data_root = out_dir_p

    #load HU volume
    vol = load_dicom_stack(dicom_dir).astype(np.float32, copy=False)       #(Z, H, W) float32

    #Per-volume clipping & normalization (simple baseline)
    vol = clip_array(vol, lo, hi)
    norm_meta = {"kind": norm}

    if norm == "minmax":
        vol = normalize_minmax(vol, lo, hi)
        norm_meta.update({"range_in": [float(lo), float(hi)], "range_out": [0.0, 1.0]})
    elif norm == "zscore":
        vol, m, s = normalize_zscore(vol)
        vol = vol.astype(np.float32)
        norm_meta.update({"mean": float(m), "std": float(s)})
    else:
        raise ValueError(f"Unknown norm: {norm}")
    
    assert np.isfinite(vol).all(), "Non-finite after normalization"
    if norm == "minmax":
        assert vol.min() >= -1e-6 and vol.max() <= 1 + 1e-6, "MinMax out of [0, 1]"
    
    #Save slices + manifest
    manifest_path = out_dir_p / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "path", "min", "max"])
        for i in range(vol.shape[0]):
            sl = vol[i]
            sl_path_abs = out_dir_p / f"slice_{i:04d}.npy"
            np.save(sl_path_abs, sl)

            # Save relative POSIX path
            rel_path = _norm_rel(sl_path_abs, data_root)
            w.writerow([i, rel_path, float(sl.min()), float(sl.max())])
        
            if (i + 1) % 200 == 0:
                logging.getLogger(__name__).info(f"[DICOM] saved {i + 1}/{vol.shape[0]} slices...")
    
    meta = {
        "source": "DICOM",
        "num_slices": int(vol.shape[0]),
        "volume_shape": list(vol.shape),
        "dtype": "float32",
        "clip_bounds": [float(lo), float(hi)],
        "normalization": norm_meta,
        "paths": {
            "prepared_dir": _norm_rel(out_dir_p, data_root),
            "manifest_csv": _norm_rel(manifest_path, data_root),
        },
        "spacing_mm": None,
        "quant_meta": None,
    }

    try:
        px, py, dz = _compute_spacing(dicom_dir)
        meta["spacing_mm"] = {"px": px, "py": py, "dz": dz}
    except Exception as e:
        print(f"[WARN] spacing meta skipped: {e}")
    
    try:
        quant = _read_quant_meta(dicom_dir)
        meta["quant_meta"] = quant
    except Exception as e:
        print(f"[WARN] quantitative meta skipped: {e}")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print((f"[DICOM] saved {vol.shape[0]} slice(s) -> {out_dir}"))

# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CT preprocessing."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lodopab", "dicom", "lidc_meta"], required=True, help=(
        "Type of data to prepare: "
        "'lodopab' (LoDoPaB ground-truth slices), "
        "'dicom' (single CT series), or "
        "'lidc_meta' (bulk LIDC-IDRI using TCIA metadata.csv)."
    ))

    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--dicom_dir", type=str, default=None, help="Directory containing a single CT series (source=dicom).")
    ap.add_argument("--metadata", type=str, default=None, help=(
        "Relative path (from PROJECT_ROOT) to TCIA metadata.csv "
        "for LIDC-IDRI (required for source=lidc_meta)."
    ))
    ap.add_argument("--raw_root", type=str, default="data/raw/lidc-idri/manifest-1600709154662", help=(
        "Root directory that contains the 'LIDC-IDRI/' folder "
        "downloaded from TCIA (source=lidc_meta)."
    ))
    ap.add_argument("--clip", nargs=2, type=float, default=None,
                    help="clip low high. LoDoPaB default=(0,1).DICOM default(-1000,1000).")
    ap.add_argument("--norm", choices=["minmax", "zscore"], default="minmax")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=None, help="limit items (debug)")
    return ap.parse_args()

def main() -> None:
    """Entry point for the CT preprocessing CLI."""
    args = _parse_args()
    cfg = load_config(args)

    logger = setup_logger(name="prep_ct", log_dir=cfg.log_dir)
    set_seed(cfg.seed)
    logger.info(f"[CFG] {cfg}")

    if cfg.source == "lodopab":
        clip_bounds = tuple(args.clip) if args.clip else (0.0, 1.0)

        max_items = cfg.max_items if (cfg.max_items is not None and cfg.max_items > 0) else None

        prepare_lodopab(
            split=args.split,
            clip_bounds=clip_bounds,
            norm=cfg.norm,
            out_dir=cfg.out_root,
            max_items=max_items,
        )
        return
    
    if args.source == "dicom":
        if args.dicom_dir is None:
            raise ValueError(f"--dicom_dir is required for source='dicom'.")
        
        clip_bounds = tuple(args.clip) if args.clip is not None else (-1000, 1000)
        prepare_dicom(
            dicom_dir=args.dicom_dir,
            clip_bounds=clip_bounds,
            norm=cfg.norm,
            out_dir=cfg.out_root
        )

        return

    if args.source == "lidc_meta":
        if args.metadata is None:
            raise ValueError(f"--metadata is required for source='lidc_meta'.")
        
        clip_bounds = tuple(args.clip) if args.clip is not None else (-1000, 1000)
        max_series = cfg.max_items if (cfg.max_items is not None and cfg.max_items > 0) else None
        lidc_cfg = LIDCMetaConfig(
            metadata_rel=args.metadata,
            raw_root_rel=args.raw_root,
            out_root_rel=cfg.out_root,
            clip_bounds=clip_bounds,
            norm=cfg.norm,
            max_series=max_series,
        )
        prepare_lidc_from_metadata(lidc_cfg, logger)
        return
    
if __name__ == "__main__":
    main()