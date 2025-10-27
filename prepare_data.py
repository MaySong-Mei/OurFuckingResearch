"""
Simple data preparation script to organize DICOM files for training.

Raw data structure:
    incomingdir/
    ├── case-108185/
    │   └── HEAD-NECK/
    │       ├── THIN_LUNG_TORSO/
    │       │   ├── CT.xxx.dcm
    │       │   └── CT.yyy.dcm
    │       └── THIN_BONE_HEAD/
    │           └── CT.zzz.dcm
    └── case-113462/
        └── ...

Output structure:
    data/
    ├── train.txt
    ├── val.txt
    ├── case_108185_THIN_LUNG_TORSO.dcm
    ├── case_108185_THIN_BONE_HEAD.dcm
    └── ... (one DCM per series, consolidated from multi-slice DCM files)
"""

import os
import shutil
import pydicom
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dicom_series(series_dir: Path) -> Optional[np.ndarray]:
    """
    Load all DICOM files from a series directory and stack them.

    Args:
        series_dir: Path containing DICOM files

    Returns:
        3D numpy array [Z, H, W] or None if loading fails
    """
    try:
        dicom_files = sorted(series_dir.glob("*.dcm"))

        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_dir}")
            return None

        slices = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dcm_file))
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds.pixel_array)
            except Exception as e:
                logger.warning(f"Failed to read {dcm_file}: {e}")
                continue

        if not slices:
            logger.warning(f"No valid slices in {series_dir}")
            return None

        # Stack slices
        volume = np.stack(slices, axis=0)
        logger.info(f"Loaded series from {series_dir.name}: shape {volume.shape}")

        return volume

    except Exception as e:
        logger.error(f"Error loading series from {series_dir}: {e}")
        return None


def save_volume_as_dcm(volume: np.ndarray, output_path: Path) -> bool:
    """
    Save a 3D volume as a multi-frame DICOM file.

    Args:
        volume: 3D numpy array [Z, H, W]
        output_path: Output DICOM file path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a basic DICOM dataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"

        ds = pydicom.dataset.FileDataset(
            str(output_path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )

        # Add minimal required tags
        ds.PatientName = "PreparedData"
        ds.PatientID = "00000"
        ds.is_implicit_VR = True
        ds.is_little_endian = True

        # Handle different data types
        volume = volume.astype(np.int16)

        # For multi-frame, store as single slice with frame info
        # (Simplified: storing just the first slice)
        ds.PixelData = volume[0].tobytes()
        ds.Rows = volume[0].shape[0]
        ds.Columns = volume[0].shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Alternative: save as numpy for better multi-frame support
        # We'll use .npz instead for simplicity

        return True

    except Exception as e:
        logger.error(f"Error saving volume to {output_path}: {e}")
        return False


def save_volume_as_npz(volume: np.ndarray, output_path: Path) -> bool:
    """
    Save a 3D volume as a compressed NumPy file (.npz).
    More reliable for multi-frame data than DICOM.

    Args:
        volume: 3D numpy array [Z, H, W]
        output_path: Output .npz file path

    Returns:
        True if successful, False otherwise
    """
    try:
        np.savez_compressed(str(output_path), volume=volume.astype(np.float32))
        logger.info(f"Saved volume to {output_path}: shape {volume.shape}")
        return True
    except Exception as e:
        logger.error(f"Error saving volume to {output_path}: {e}")
        return False


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    format: str = "npz"
) -> None:
    """
    Prepare training data by organizing DICOM files.

    Args:
        raw_data_dir: Root directory containing case folders
        output_dir: Output directory for organized data
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        format: Output format ('npz' or 'dcm')
    """

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Collect all series directories
    series_dirs = []
    for case_dir in sorted(raw_data_dir.glob("case-*")):
        if not case_dir.is_dir():
            continue

        # Navigate through anatomy/series structure
        # case/ANATOMY/SERIES/
        for anatomy_dir in case_dir.iterdir():
            if not anatomy_dir.is_dir():
                continue

            for series_dir in anatomy_dir.iterdir():
                if not series_dir.is_dir():
                    continue

                # Check if directory contains DICOM files
                if list(series_dir.glob("*.dcm")):
                    series_dirs.append((case_dir.name, anatomy_dir.name, series_dir.name, series_dir))

    logger.info(f"Found {len(series_dirs)} series directories")

    # Process each series
    processed_files = []
    for case_id, anatomy, series_name, series_dir in series_dirs:
        logger.info(f"Processing: {case_id}/{anatomy}/{series_name}")

        # Load volume
        volume = load_dicom_series(series_dir)
        if volume is None:
            logger.warning(f"Skipping {case_id}/{anatomy}/{series_name} - failed to load")
            continue

        # Skip volumes with insufficient slices for interpolation
        # Need at least 258 slices (257*2) for step=2 sampling to extract 257 slices
        min_slices = 258
        if volume.shape[0] < min_slices:
            logger.warning(f"Skipping {case_id}/{anatomy}/{series_name} - only {volume.shape[0]} slices (need >= {min_slices})")
            continue

        # Generate output filename
        output_name = f"{case_id}_{anatomy}_{series_name}"

        if format == "npz":
            output_file = output_dir / f"{output_name}.npz"
            if save_volume_as_npz(volume, output_file):
                processed_files.append(output_file.name)

        elif format == "dcm":
            output_file = output_dir / f"{output_name}.dcm"
            if save_volume_as_dcm(volume, output_file):
                processed_files.append(output_file.name)

    logger.info(f"Processed {len(processed_files)} files")

    # Create train/val/test splits
    num_files = len(processed_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)

    # Shuffle for random split
    import random
    random.seed(42)
    random.shuffle(processed_files)

    train_files = processed_files[:num_train]
    val_files = processed_files[num_train:num_train + num_val]
    test_files = processed_files[num_train + num_val:]

    # Write split files
    (output_dir / "train.txt").write_text("\n".join(train_files))
    (output_dir / "val.txt").write_text("\n".join(val_files))
    (output_dir / "test.txt").write_text("\n".join(test_files))

    logger.info(f"Train: {len(train_files)} files")
    logger.info(f"Val:   {len(val_files)} files")
    logger.info(f"Test:  {len(test_files)} files")

    logger.info("Data preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Use --data_dir {output_dir} when training")


def main():
    parser = argparse.ArgumentParser(description="Prepare DICOM data for training")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/omi/incomingdir",
        help="Path to raw DICOM data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/data",
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["npz", "dcm"],
        help="Output format (npz recommended for multi-frame data)"
    )

    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)

    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return

    prepare_data(
        raw_data_dir,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        format=args.format
    )


if __name__ == "__main__":
    main()
