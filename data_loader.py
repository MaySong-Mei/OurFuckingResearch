"""
Data loading and preprocessing for medical volumes
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from scipy.ndimage import zoom
import json

logger = logging.getLogger(__name__)


class MedicalVolumeDataset(Dataset):
    """Dataset for loading medical imaging volumes from DICOM files"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_slices: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        transform=None,
        cache_data: bool = False,
        normalize: bool = True
    ):
        """
        Args:
            data_dir: Directory containing DICOM files or processed volumes
            split: 'train', 'val', or 'test'
            num_slices: Number of slices to extract from each volume
            img_size: Target size for each slice (H, W)
            transform: Optional transforms to apply
            cache_data: Whether to cache preprocessed data in memory
            normalize: Whether to normalize intensity values
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = transform
        self.cache_data = cache_data
        self.normalize = normalize

        # Load file list
        self.files = self._load_file_list()

        # Cache for preprocessed data
        self.cache = {} if cache_data else None

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
        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        # Load volume
        file_path = self.files[idx]
        volume = self._load_volume(file_path)

        # Preprocess
        volume = self._preprocess_volume(volume)

        # Extract slices (sampled: 0, 2, 4, ..., 256) and ground truth (all slices)
        slices, ground_truth_slices = self._extract_slices(volume)

        # Convert to tensor
        slices_tensor = torch.from_numpy(slices).float()
        ground_truth_tensor = torch.from_numpy(ground_truth_slices).float()

        # Apply transforms
        if self.transform is not None:
            slices_tensor = self.transform(slices_tensor)
            ground_truth_tensor = self.transform(ground_truth_tensor)

        sample = {
            'slices': slices_tensor,  # [N, H, W] - sampled slices (129 slices)
            'ground_truth_slices': ground_truth_tensor,  # [257, H, W] - all slices (0-256)
            'file_path': str(file_path),
            'volume_shape': volume.shape
        }

        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = sample

        return sample

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

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess volume: normalize, resize, etc."""
        # Normalize intensity
        if self.normalize:
            volume = self._normalize_intensity(volume)

        # Resize spatial dimensions if needed
        if volume.shape[-2:] != self.img_size:
            volume = self._resize_volume(volume, self.img_size)

        return volume

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

    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize volume to target spatial dimensions"""
        if len(volume.shape) == 3:
            num_slices, H, W = volume.shape
            zoom_factors = (1.0, target_size[0] / H, target_size[1] / W)
        elif len(volume.shape) == 2:
            H, W = volume.shape
            zoom_factors = (target_size[0] / H, target_size[1] / W)
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")

        resized = zoom(volume, zoom_factors, order=1)
        return resized

    def _extract_slices(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract slices from volume:
        1. Sampled slices: every 2nd slice (0, 2, 4, ..., 256) - 129 slices
        2. Ground truth slices: all consecutive slices (0-256) - 257 slices

        Returns:
            slices: [N, H, W] - sampled slices (129 slices with step=2)
            ground_truth_slices: [257, H, W] - all slices (0-256)
        """
        if len(volume.shape) == 2:
            # Single slice - replicate for both
            slices = np.stack([volume] * self.num_slices, axis=0)
            ground_truth_slices = np.stack([volume] * (self.num_slices * 2 - 1), axis=0)
            return slices, ground_truth_slices

        num_slices_available = volume.shape[0]

        if num_slices_available >= self.num_slices * 2:
            # Sample with step of 2 (every other slice: 0, 2, 4, 6, ...)
            indices = np.arange(0, num_slices_available, 2)
            indices = indices[:self.num_slices]
            slices = volume[indices]

            # Ground truth: all consecutive slices from 0 to min(256, num_slices_available-1)
            # For 129 sampled slices with step=2, we need 257 consecutive slices
            num_gt_slices = self.num_slices * 2 - 1  # 257 for 129 sampled slices
            if num_slices_available >= num_gt_slices:
                ground_truth_slices = volume[:num_gt_slices]
            else:
                # If not enough slices, pad with the last slice
                ground_truth_slices = volume.copy()
                pad_size = num_gt_slices - num_slices_available
                ground_truth_slices = np.pad(
                    ground_truth_slices,
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode='edge'
                )
        else:
            # Not enough slices
            raise ValueError(f"Not enough slices in volume: {num_slices_available} available, "
                             f"but {self.num_slices * 2} needed for step=2 sampling")

        return slices, ground_truth_slices


class SimpleDICOMDataset(Dataset):
    """Simple dataset for testing with a single DICOM file"""

    def __init__(
        self,
        dicom_path: str,
        num_slices: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        transform=None,
        normalize: bool = True
    ):
        """
        Args:
            dicom_path: Path to a single DICOM file
            num_slices: Number of slices to extract from the volume
            img_size: Target size for each slice (H, W)
            transform: Optional transforms to apply
            normalize: Whether to normalize intensity values
        """
        self.dicom_path = Path(dicom_path)
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

        if not self.dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        logger.info(f"Initialized SimpleDICOMDataset with {dicom_path}")

    def __len__(self) -> int:
        return 1  # Single volume, but we can treat it as multiple samples for testing

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get the single volume"""
        # Load DICOM file
        volume = self._load_dicom(self.dicom_path)

        # Preprocess
        if self.normalize:
            volume = self._normalize_intensity(volume)

        if volume.shape[-2:] != self.img_size:
            volume = self._resize_volume(volume, self.img_size)

        # Extract slices (sampled and ground truth)
        slices, ground_truth_slices = self._extract_slices(volume)

        # Convert to tensor
        slices_tensor = torch.from_numpy(slices).float()
        ground_truth_tensor = torch.from_numpy(ground_truth_slices).float()

        # Apply transforms
        if self.transform is not None:
            slices_tensor = self.transform(slices_tensor)
            ground_truth_tensor = self.transform(ground_truth_tensor)

        sample = {
            'slices': slices_tensor,  # [N, H, W] - sampled slices (129 slices)
            'ground_truth_slices': ground_truth_tensor,  # [257, H, W] - all slices (0-256)
            'file_path': str(self.dicom_path),
            'volume_shape': volume.shape
        }

        return sample

    def _load_dicom(self, file_path: Path) -> np.ndarray:
        """Load DICOM file and extract 3D volume"""
        try:
            dicom_data = pydicom.dcmread(str(file_path))

            # Extract pixel array
            if hasattr(dicom_data, 'pixel_array'):
                volume = dicom_data.pixel_array

                # Handle different DICOM formats
                if len(volume.shape) == 2:
                    # Single slice - create volume by replicating
                    logger.warning(f"Single slice detected, replicating to {self.num_slices} slices")
                    volume = np.stack([volume] * self.num_slices, axis=0)
                elif len(volume.shape) == 3:
                    # Already 3D - perfect
                    logger.info(f"3D volume loaded: shape {volume.shape}")
                elif len(volume.shape) == 4:
                    # Multi-frame with time - take first timepoint
                    logger.warning(f"4D volume detected, taking first timepoint")
                    volume = volume[0]

                return volume.astype(np.float32)
            else:
                raise ValueError(f"No pixel array found in DICOM: {file_path}")

        except Exception as e:
            logger.error(f"Error loading DICOM {file_path}: {e}")
            raise

    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1]"""
        # Clip outliers using percentiles
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

    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize volume to target spatial dimensions"""
        if len(volume.shape) == 3:
            num_slices, H, W = volume.shape
            zoom_factors = (1.0, target_size[0] / H, target_size[1] / W)
        elif len(volume.shape) == 2:
            H, W = volume.shape
            zoom_factors = (target_size[0] / H, target_size[1] / W)
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")

        resized = zoom(volume, zoom_factors, order=1)
        return resized

    def _extract_slices(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract slices from volume:
        1. Sampled slices: every 2nd slice (0, 2, 4, ..., 256) - 129 slices
        2. Ground truth slices: all consecutive slices (0-256) - 257 slices

        Returns:
            slices: [N, H, W] - sampled slices (129 slices with step=2)
            ground_truth_slices: [257, H, W] - all slices (0-256)
        """
        if len(volume.shape) == 2:
            # Single slice - replicate for both
            slices = np.stack([volume] * self.num_slices, axis=0)
            ground_truth_slices = np.stack([volume] * (self.num_slices * 2 - 1), axis=0)
            return slices, ground_truth_slices

        num_slices_available = volume.shape[0]

        if num_slices_available >= self.num_slices * 2:
            # Sample with step of 2 (every other slice: 0, 2, 4, 6, ...)
            indices = np.arange(0, num_slices_available, 2)
            indices = indices[:self.num_slices]
            slices = volume[indices]

            # Ground truth: all consecutive slices from 0 to min(256, num_slices_available-1)
            # For 129 sampled slices with step=2, we need 257 consecutive slices
            num_gt_slices = self.num_slices * 2 - 1  # 257 for 129 sampled slices
            if num_slices_available >= num_gt_slices:
                ground_truth_slices = volume[:num_gt_slices]
            else:
                # If not enough slices, pad with the last slice
                ground_truth_slices = volume.copy()
                pad_size = num_gt_slices - num_slices_available
                ground_truth_slices = np.pad(
                    ground_truth_slices,
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode='edge'
                )
        else:
            raise ValueError(f"Not enough slices in volume: {num_slices_available} available, "
                             f"but {self.num_slices * 2} needed for step=2 sampling")

        return slices, ground_truth_slices
