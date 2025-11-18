from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import argparse

import numpy as np
import pandas as pd

from src.utils import (
    get_project_root,
    read_csv,
    save_df_csv,
    setup_logger,
    set_seed,
    to_relpath
)

SplitOn = Literal["case", "slice"]

@dataclass
class LIDCSplitConfig:
    """
    Split merged LIDC manifest into train/val/test subsets.
    """
    merged_manifest_rel: str = "data/prepared/lidc_debug/all_slices_manifest.csv"
    out_dir_rel: str = "data/prepared/lidc_debug/splits"
    split_on: SplitOn = "case"
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

def assign_split_by_case(df: pd.DataFrame, cfg: LIDCSplitConfig) -> pd.DataFrame:
    """
    Add a 'split' column based on case-level random split.
    """
    if "case_id" not in df.columns:
        raise KeyError("[SPLIT:assign] Expected 'case_id' column in merged manifest.")
    
    cases = df["case_id"].unique()
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(cases)

    n_cases = len(cases)
    n_val = int(cfg.val_frac * n_cases)
    n_test = int(cfg.test_frac * n_cases)
    n_val = max(n_val, 1) if cfg.val_frac > 0.0 else 0
    n_test = max(n_test, 1) if cfg.test_frac > 0.0 else 0

    val_cases = set(cases[:n_val])
    test_cases = set(cases[n_val:n_val + n_test])
    train_cases = set(cases[n_val + n_test:])

    case_to_split: dict[str, str] = {}

    for cid in train_cases:
        case_to_split[cid] = "train"
    for cid in val_cases:
        case_to_split[cid] = "val"
    for cid in test_cases:
        case_to_split[cid] = "test"

    df_out = df.copy()
    df_out["split"] = df_out["case_id"].map(case_to_split)

    return df_out

def split_and_save(cfg: LIDCSplitConfig) -> None:
    logger = setup_logger("lidc_split_manifest")
    set_seed(cfg.seed)

    root = get_project_root()
    merged_path = root / cfg.merged_manifest_rel
    out_dir = root / cfg.out_dir_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(merged_path)
    logger.info("[LIDC:split] Loaded merged manifest with shape %s", df.shape)

    if cfg.split_on == "case":
        df = assign_split_by_case(df, cfg)
    else:
        raise NotImplementedError("slice-level split not implemented yet.")
    
    logger.info("[LIDC:split] Split counts:\n%s", df["split"].value_counts())

    for split in ("train", "val", "test"):
        df_split = df[df["split"] == split].copy()
        if df_split.empty:
            logger.warning("[LIDC:split] No slices for split=%s", split)
            continue

        out_path = out_dir / f"lidc_{split}_manifest.csv"
        save_df_csv(df_split, out_path)
        logger.info("[LIDC:split] Saved %d rows to %s", len(df_split), to_relpath(out_path))


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for LIDC train/val/test splitting.
    """
    parser = argparse.ArgumentParser(
        description="Split merged LIDC manifest into train/val/test subsets."
    )

    parser.add_argument(
        "--merged-manifest",
        type=str,
        default="data/prepared/lidc_debug/all_slices_manifest.csv",
        help=(
            "Path (relative to project root) to the merged LIDC manifest CSV. "
            "Default: data/prepared/lidc_debug/all_slices_manifest.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/prepared/lidc_debug/splits",
        help=(
            "Output directory (relative to project root) where split manifests "
            "will be written. Default: data/prepared/lidc_debug/splits"
        ),
    )
    parser.add_argument(
        "--split-on",
        choices=("case", "slice"),
        default="case",
        help=(
            "Split strategy: 'case' (preferred; no leakage across splits) or "
            "'slice' (not implemented yet). Default: case."
        ),
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of cases assigned to the validation split. Default: 0.15",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of cases assigned to the test split. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for split reproducibility. Default: 42",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for LIDC manifest splitting.
    """
    args = _parse_args()

    cfg = LIDCSplitConfig(
        merged_manifest_rel=args.merged_manifest,
        out_dir_rel=args.out_dir,
        split_on=args.split_on,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    split_and_save(cfg)


if __name__ == "__main__":
    main()