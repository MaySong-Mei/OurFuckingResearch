"""
Data loading and preprocessing for medical volumes
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def load_and_normalize_volume(file_path: Path, downsample: bool = True) -> np.ndarray:
    """
    Load a volume from file, normalize it, and optionally downsample.

    Args:
        file_path: Path to DICOM, NPY, or NPZ file
        downsample: If True, downsample H and W to half size to reduce memory

    Returns:
        Normalized volume as numpy array with shape [num_slices, H, W] or [num_slices, H//2, W//2]
    """
    file_path = Path(file_path)

    # Load volume
    if file_path.suffix == '.dcm':
        dicom_data = pydicom.dcmread(str(file_path))
        volume = dicom_data.pixel_array.astype(np.float32)
    elif file_path.suffix == '.npy':
        volume = np.load(file_path).astype(np.float32)
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        volume = data['volume'].astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Handle different dimensions
    if len(volume.shape) == 2:
        # Single slice - replicate it
        volume = np.stack([volume] * volume.shape[0], axis=0)
    elif len(volume.shape) == 4:
        # Multi-frame - take first timepoint
        volume = volume[0]

    # Normalize intensity to [0, 1]
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume = volume - vol_min

    # Downsample H and W to half size to reduce memory
    if downsample:
        # Use zoom with factors [1, 0.5, 0.5] to keep depth, halve H and W
        volume = zoom(volume, (1, 0.5, 0.5), order=1)

    return volume


class MedicalVolumeDataset(Dataset):
    """Dataset for loading medical imaging volumes from DICOM files"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train'
    ):
        """
        Args:
            data_dir: Directory containing DICOM files or processed volumes
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load file list
        self.files = self._load_file_list()

        logger.info(f"Loaded {len(self.files)} files for {split} split")

    def _load_file_list(self) -> List[Path]:
        """Load list of DICOM files or volume files"""
        split_file = self.data_dir / f"{self.split}.txt"

        if split_file.exists():
            # Load from split file
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
            files = [self.data_dir / fname for fname in file_names]
        else:
            # Search for DICOM files
            files = list(self.data_dir.glob("*.dcm"))
            if not files:
                files = list(self.data_dir.glob("**/*.dcm"))

            # Simple train/val split if no split file exists
            if self.split == 'train':
                files = files[:int(0.8 * len(files))]
            elif self.split == 'val':
                files = files[int(0.8 * len(files)):int(0.9 * len(files))]
            else:  # test
                files = files[int(0.9 * len(files)):]

        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single volume"""
        # Load volume
        file_path = self.files[idx]
        volume = self._load_volume(file_path)

        # Extract slices and ground truth (all slices)
        slices, ground_truth_slices = self._extract_slices(volume)

        # Convert to tensor
        slices_tensor = torch.from_numpy(slices).float()
        ground_truth_tensor = torch.from_numpy(ground_truth_slices).float()

        return {
            'slices': slices_tensor,  # [D, H, W]
            'ground_truth_slices': ground_truth_tensor,  # [2D-1, H, W]
            'file_path': str(file_path),
            'volume_shape': volume.shape
        }

    def _load_volume(self, file_path: Path) -> np.ndarray:
        """Load a 3D volume from DICOM or numpy file"""
        if file_path.suffix == '.dcm':
            return self._load_dicom(file_path)
        elif file_path.suffix == '.npy':
            return np.load(file_path)
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            return data['volume']
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_dicom(self, file_path: Path) -> np.ndarray:
        """Load DICOM file and extract 3D volume"""
        try:
            dicom_data = pydicom.dcmread(str(file_path))

            # Extract pixel array
            if hasattr(dicom_data, 'pixel_array'):
                volume = dicom_data.pixel_array

                # Handle different DICOM formats
                if len(volume.shape) == 2:
                    # Single slice - create dummy volume
                    volume = np.stack([volume] * self.num_slices, axis=0)
                elif len(volume.shape) == 3:
                    # Already 3D
                    pass
                elif len(volume.shape) == 4:
                    # Multi-frame with time - take first timepoint
                    volume = volume[0]

                return volume.astype(np.float32)
            else:
                raise ValueError(f"No pixel array found in DICOM: {file_path}")

        except Exception as e:
            logger.error(f"Error loading DICOM {file_path}: {e}")
            # Return dummy volume
            return np.zeros((self.num_slices, *self.img_size), dtype=np.float32)

    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1]"""
        # Clip outliers (optional)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)

        # Normalize to [0, 1]
        vol_min = volume.min()
        vol_max = volume.max()

        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        else:
            volume = volume - vol_min

        return volume

    def _extract_slices(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract slices from volume:
        1. Sampled slices: every 2nd slice
        2. Ground truth slices: all consecutive slices
        3. Downsample H and W to half size

        Returns:
            slices: [D, H//2, W//2] - sampled slices, downsampled
            ground_truth_slices: [2D-1, H//2, W//2] - all slices, downsampled
        """
        num_slices_available = volume.shape[0]

        # Sample with step of 2 (every other slice: 0, 2, 4, 6, ...)
        indices = np.arange(0, num_slices_available, 2)
        sampled_slices = volume[indices]
        num_sampled_slices = sampled_slices.shape[0]

        # Ground truth
        num_gt_slices = 2 * num_sampled_slices - 1
        ground_truth_slices = volume[:num_gt_slices]

        # Downsample H and W to half size to reduce memory
        # Use zoom with factors [1, 0.5, 0.5] to keep depth, halve H and W
        sampled_slices = zoom(sampled_slices, (1, 0.5, 0.5), order=1)
        ground_truth_slices = zoom(ground_truth_slices, (1, 0.5, 0.5), order=1)

        return sampled_slices, ground_truth_slices
