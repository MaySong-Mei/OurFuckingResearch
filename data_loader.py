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

        # Extract slices
        slices = self._extract_slices(volume)

        # Convert to tensor
        slices_tensor = torch.from_numpy(slices).float()

        # Apply transforms
        if self.transform is not None:
            slices_tensor = self.transform(slices_tensor)

        sample = {
            'slices': slices_tensor,
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

    def _extract_slices(self, volume: np.ndarray) -> np.ndarray:
        """Extract N slices from volume"""
        if len(volume.shape) == 2:
            # Single slice - replicate
            return np.stack([volume] * self.num_slices, axis=0)

        num_slices_available = volume.shape[0]

        if num_slices_available >= self.num_slices:
            # Uniformly sample slices
            indices = np.linspace(0, num_slices_available - 1, self.num_slices, dtype=int)
            slices = volume[indices]
        else:
            # Pad with zeros if not enough slices
            slices = np.zeros((self.num_slices, *self.img_size), dtype=np.float32)
            slices[:num_slices_available] = volume

        # Add channel dimension if needed
        if len(slices.shape) == 3:
            slices = slices[:, np.newaxis, :, :]  # [N, 1, H, W]
            slices = slices.squeeze(1)  # [N, H, W] for now

        return slices


class DICOMSeriesDataset(Dataset):
    """Dataset for loading complete DICOM series (multiple files per volume)"""

    def __init__(
        self,
        series_dir: str,
        split: str = 'train',
        num_slices: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        transform=None,
        normalize: bool = True
    ):
        """
        Args:
            series_dir: Directory containing subdirectories, each with a DICOM series
            split: 'train', 'val', or 'test'
            num_slices: Number of slices to use
            img_size: Target size for each slice
            transform: Optional transforms
            normalize: Whether to normalize intensity
        """
        self.series_dir = Path(series_dir)
        self.split = split
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

        # Find all series directories
        self.series_paths = self._load_series_list()

        logger.info(f"Loaded {len(self.series_paths)} DICOM series for {split} split")

    def _load_series_list(self) -> List[Path]:
        """Find all DICOM series directories"""
        series_dirs = [d for d in self.series_dir.iterdir() if d.is_dir()]

        # Simple train/val/test split
        total = len(series_dirs)
        if self.split == 'train':
            series_dirs = series_dirs[:int(0.8 * total)]
        elif self.split == 'val':
            series_dirs = series_dirs[int(0.8 * total):int(0.9 * total)]
        else:
            series_dirs = series_dirs[int(0.9 * total):]

        return series_dirs

    def __len__(self) -> int:
        return len(self.series_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a complete DICOM series"""
        series_path = self.series_paths[idx]

        # Load all DICOM files in series
        dicom_files = sorted(series_path.glob("*.dcm"))

        slices_list = []
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(str(dcm_file))
                if hasattr(dcm, 'pixel_array'):
                    slices_list.append(dcm.pixel_array.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to load {dcm_file}: {e}")

        # Stack slices into volume
        if len(slices_list) > 0:
            volume = np.stack(slices_list, axis=0)
        else:
            volume = np.zeros((self.num_slices, *self.img_size), dtype=np.float32)

        # Preprocess
        if self.normalize:
            volume = self._normalize_intensity(volume)

        if volume.shape[-2:] != self.img_size:
            volume = self._resize_volume(volume)

        # Extract/sample slices
        slices = self._extract_slices(volume)

        # Convert to tensor
        slices_tensor = torch.from_numpy(slices).float()

        if self.transform is not None:
            slices_tensor = self.transform(slices_tensor)

        return {
            'slices': slices_tensor,
            'series_path': str(series_path),
            'num_original_slices': len(slices_list)
        }

    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity to [0, 1]"""
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)

        vol_min = volume.min()
        vol_max = volume.max()

        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)

        return volume

    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize to target size"""
        N, H, W = volume.shape
        zoom_factors = (1.0, self.img_size[0] / H, self.img_size[1] / W)
        return zoom(volume, zoom_factors, order=1)

    def _extract_slices(self, volume: np.ndarray) -> np.ndarray:
        """Extract N slices uniformly"""
        num_available = volume.shape[0]

        if num_available >= self.num_slices:
            indices = np.linspace(0, num_available - 1, self.num_slices, dtype=int)
            return volume[indices]
        else:
            # Pad if needed
            slices = np.zeros((self.num_slices, *self.img_size), dtype=np.float32)
            slices[:num_available] = volume
            return slices
