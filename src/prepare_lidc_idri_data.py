#!/usr/bin/env python3
"""
Process LIDC-IDRI dataset from downloaded DICOM files.
Handles SeriesInstanceUID-based directory structure.
Saves prepared data to data/prepared/lidc/

This script should be placed in: ct-reconstruction-pipeline/src/
It will process data from: ct-reconstruction-pipeline/data/lidc-idri-data/
And save to: ct-reconstruction-pipeline/data/prepared/lidc/

Usage:
    cd /path/to/ct-reconstruction-pipeline/src
    python3 prepare_lidc_idri_data.py
"""

import os
import sys
from pathlib import Path
import subprocess


def get_patient_id_from_dicom(dicom_dir):
    """Extract patient ID from first DICOM file."""
    try:
        dcm_files = list(dicom_dir.glob("*.dcm"))
        if dcm_files:
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            return getattr(ds, 'PatientID', None)
    except Exception as e:
        print(f"    Warning: Could not read patient ID: {e}")
    return None


def main():
    # Setup paths - get project root from src directory
    src_dir = Path(__file__).parent.resolve()
    base_dir = src_dir.parent  # Go up one level to ct-reconstruction-pipeline
    data_dir = base_dir / "data"
    lidc_data_dir = data_dir / "lidc-idri-data"
    prepare_script = data_dir / "prepare_data.py"
    output_base = data_dir / "prepared" / "lidc"

    print(f"Script location: {src_dir}")
    print(f"Project root: {base_dir}")
    print()

    # Check if required files exist
    if not prepare_script.exists():
        print(f"ERROR: prepare_data.py not found at {prepare_script}")
        print(f"Please ensure prepare_data.py exists in the data directory")
        sys.exit(1)

    if not lidc_data_dir.exists():
        print(f"ERROR: lidc-idri-data directory not found at {lidc_data_dir}")
        print("Please ensure LIDC-IDRI data is downloaded first")
        print(f"Expected location: {lidc_data_dir}")
        sys.exit(1)

    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIDC-IDRI Data Preparation")
    print("=" * 70)
    print(f"Data source: {lidc_data_dir}")
    print(f"Output directory: {output_base}")
    print(f"Using script: {prepare_script}")
    print()

    # Find all series directories (SeriesInstanceUID format)
    series_dirs = sorted([d for d in lidc_data_dir.iterdir() if d.is_dir()])

    if not series_dirs:
        print(f"ERROR: No series directories found in {lidc_data_dir}")
        sys.exit(1)

    print(f"Found {len(series_dirs)} series directories")
    print()

    # Ask for confirmation
    response = input(f"Process all {len(series_dirs)} series? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    print()

    # Process each series
    total = 0
    success = 0
    failed = 0
    skipped = 0
    failed_series = []

    for series_dir in series_dirs:
        series_uid = series_dir.name
        total += 1

        print("-" * 70)
        print(f"[{total}/{len(series_dirs)}] Processing series:")
        print(f"  {series_uid[:60]}...")

        # Find DICOM files in this series
        dicom_files = list(series_dir.glob("*.dcm"))

        if not dicom_files:
            print(f"  ✗ No DICOM files found, skipping...")
            failed += 1
            failed_series.append((series_uid, "No DICOM files"))
            continue

        num_dicoms = len(dicom_files)
        print(f"  Found {num_dicoms} DICOM files")

        # Get patient ID from DICOM metadata
        patient_id = get_patient_id_from_dicom(series_dir)
        if patient_id:
            print(f"  Patient ID: {patient_id}")
            # Use patient ID for output directory
            out_dir = output_base / patient_id / series_uid[:30]  # Truncate long UID
        else:
            print(f"  Warning: Could not determine patient ID, using series UID")
            # Fallback to series UID
            out_dir = output_base / series_uid[:30]

        # Skip if already processed
        if out_dir.exists() and (out_dir / "meta.json").exists():
            num_slices = len(list(out_dir.glob("slice_*.npy")))
            print(f"  ⊙ Already processed ({num_slices} slices), skipping...")
            skipped += 1
            continue

        # Build command
        cmd = [
            sys.executable,  # Python interpreter
            str(prepare_script),
            "--source", "dicom",
            "--dicom_dir", str(series_dir),
            "--clip", "-1000", "1000",
            "--norm", "minmax",
            "--out", str(out_dir)
        ]

        # Run preparation
        print(f"  Processing...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout per series
            )

            # Check if output files were created
            if out_dir.exists() and (out_dir / "meta.json").exists():
                num_slices = len(list(out_dir.glob("slice_*.npy")))
                slice_size = sum(f.stat().st_size for f in out_dir.glob("slice_*.npy")) / (1024 ** 2)
                print(f"  ✓ Success: Created {num_slices} slices ({slice_size:.1f} MB)")
                success += 1
            else:
                print(f"  ✗ Failed: No output files created")
                failed += 1
                failed_series.append((series_uid[:40], "No output files"))

        except subprocess.TimeoutExpired:
            print(f"  ✗ Failed: Timeout (>5 minutes)")
            failed += 1
            failed_series.append((series_uid[:40], "Timeout"))
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed with error:")
            if e.stderr:
                error_msg = e.stderr[:200].replace('\n', ' ')
                print(f"    {error_msg}")
                failed_series.append((series_uid[:40], error_msg[:50]))
            else:
                failed_series.append((series_uid[:40], "Unknown error"))
            failed += 1
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed: {error_msg}")
            failed_series.append((series_uid[:40], error_msg[:50]))
            failed += 1

    # Summary
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total series: {total}")
    print(f"Successful: {success}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")

    if failed_series:
        print()
        print("Failed series:")
        for series_uid, reason in failed_series[:10]:  # Show first 10
            print(f"  - {series_uid}... : {reason}")
        if len(failed_series) > 10:
            print(f"  ... and {len(failed_series) - 10} more")

    print()
    print(f"Output directory: {output_base}")

    # Calculate total size
    if output_base.exists():
        total_size = sum(f.stat().st_size for f in output_base.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024 ** 3)
        print(f"Total output size: {total_size_gb:.2f} GB")

    # Count unique patients
    if output_base.exists():
        patient_dirs = [d for d in output_base.iterdir() if d.is_dir()]
        print(f"Unique patients processed: {len(patient_dirs)}")

    print()
    print("To check results, run:")
    print(f"  ls -lh {output_base}")
    print(f"  tree -L 3 {output_base} | head -30")
    print()
    print("To view a series metadata:")
    series_example = list(output_base.rglob("meta.json"))
    if series_example:
        print(f"  cat {series_example[0]} | python3 -m json.tool")


if __name__ == "__main__":
    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom not installed")
        print("Install it with: pip3 install pydicom")
        sys.exit(1)

    main()