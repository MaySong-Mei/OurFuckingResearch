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

class MedicalVolumeDataset(Dataset):
    """Dataset for loading medical imaging volumes from DICOM files"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_slices: int = 128
    ):
        """
        Args:
            data_dir: Directory containing DICOM files or processed volumes
            split: 'train', 'val', or 'test'
            max_slices: Maximum number of sampled slices to keep (to avoid OOM).
                       If volume has more sampled slices, discard the extras.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_slices = max_slices

        # Load file list
        self.files = self._load_file_list()

        logger.info(f"Loaded {len(self.files)} files for {split} split")
        logger.info(f"Max slices per volume: {self.max_slices}")

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
        # Clip outliers
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
        1. Sampled slices: 4 slices sampled from volume (for I3Net input)
        2. Ground truth slices: All slices from the volume

        Returns:
            slices: [4, H, W] - 4 sampled slices
            ground_truth_slices: [Z, H, W] - All ground truth slices
        """
        volume = volume.transpose(2, 0, 1) # [H, W, D] -> [D, H, W]
        # Downsampling
        volume = zoom(volume, (1.0, 0.5, 0.5), order=1)
        num_slices_available = volume.shape[0]

        indices = np.arange(0, num_slices_available, 2)[:4]  # Every 2nd slice, take first 4

        sampled_slices = volume[indices]  # [4, H, W]

        # Use all slices as ground truth
        ground_truth_slices = volume  # [Z, H, W] - all slices

        # Normalize intensities
        sampled_slices = self._normalize_intensity(sampled_slices)
        ground_truth_slices = self._normalize_intensity(ground_truth_slices)

        return sampled_slices, ground_truth_slices


class TVSRNDataset(Dataset):
    """
    Dataset for TVSRN training with special Z-axis mapping
    Input: 4 slices sampled at 5mm intervals [0, 5, 10, 15]
    Ground Truth: 10 consecutive slices [3, 4, ..., 12]

    Z-axis mapping:
    - Input slices: z_s=0, z_e=4 (span 15mm)
    - mask_z_s = z_s * 5 + 3 = 3
    - mask_z_e = (z_e - 1) * 5 - 2 = 13
    - GT range: [3:13) = 10 slices
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_slices: int = 128
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_slices = max_slices
        self.files = self._load_file_list()
        logger.info(f"[TVSRN] Loaded {len(self.files)} files for {split} split")

    def _load_file_list(self) -> List[Path]:
        """Load list of DICOM files or volume files"""
        split_file = self.data_dir / f"{self.split}.txt"

        if split_file.exists():
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
            files = [self.data_dir / fname for fname in file_names]
        else:
            files = list(self.data_dir.glob("*.dcm"))
            if not files:
                files = list(self.data_dir.glob("**/*.dcm"))

            if self.split == 'train':
                files = files[:int(0.8 * len(files))]
            elif self.split == 'val':
                files = files[int(0.8 * len(files)):int(0.9 * len(files))]
            else:
                files = files[int(0.9 * len(files)):]

        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        volume = self._load_volume(file_path)
        slices, ground_truth_slices = self._extract_tvsrn_slices(volume)

        slices_tensor = torch.from_numpy(slices).float()
        ground_truth_tensor = torch.from_numpy(ground_truth_slices).float()

        return {
            'slices': slices_tensor,
            'ground_truth_slices': ground_truth_tensor,
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
            if hasattr(dicom_data, 'pixel_array'):
                volume = dicom_data.pixel_array
                if len(volume.shape) == 2:
                    volume = np.stack([volume] * 128, axis=0)
                elif len(volume.shape) == 4:
                    volume = volume[0]
                return volume.astype(np.float32)
            else:
                raise ValueError(f"No pixel array found in DICOM: {file_path}")
        except Exception as e:
            logger.error(f"Error loading DICOM {file_path}: {e}")
            return np.zeros((128, 256, 256), dtype=np.float32)

    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1]"""
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        else:
            volume = volume - vol_min
        return volume

    def _extract_tvsrn_slices(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract slices for TVSRN training
        Input: 4 slices at indices [0, 5, 10, 15]
        Ground Truth: 10 slices at indices [3, 4, ..., 12]

        Returns:
            slices: [4, 256, 256] - input slices (resized to 256x256)
            ground_truth_slices: [10, 256, 256] - GT slices (resized to 256x256)

        Raises:
            IndexError: if not enough slices (automatically skipped by DataLoader)
        """
        volume = volume.transpose(2, 0, 1)  # [H, W, D] -> [D, H, W]

        num_slices = volume.shape[0]

        # Input: 4 slices at 5mm intervals [0, 5, 10, 15]
        input_indices = np.array([0, 5, 10, 15])

        # Skip if not enough slices
        if input_indices[-1] >= num_slices:
            raise IndexError(f"Not enough slices: need index 15, got {num_slices}")

        sampled_slices = volume[input_indices]  # [4, H, W]

        # Ground Truth: Z-axis mapping
        z_s = 0
        z_e = 4
        mask_z_s = z_s * 5 + 3      # 3
        mask_z_e = (z_e - 1) * 5 - 2  # 13

        # Skip if GT range is invalid
        if mask_z_e > num_slices:
            raise IndexError(f"Not enough slices for GT: need index {mask_z_e}, got {num_slices}")

        ground_truth_slices = volume[mask_z_s:mask_z_e]  # [10, H, W]

        sampled_slices = self._normalize_intensity(sampled_slices)
        ground_truth_slices = self._normalize_intensity(ground_truth_slices)

        # Resize to 256x256 for TVSRN
        target_size = 256
        H, W = sampled_slices.shape[1], sampled_slices.shape[2]
        if H != target_size or W != target_size:
            scale_h = target_size / H
            scale_w = target_size / W
            sampled_slices = zoom(sampled_slices, (1.0, scale_h, scale_w), order=1)
            ground_truth_slices = zoom(ground_truth_slices, (1.0, scale_h, scale_w), order=1)

        return sampled_slices, ground_truth_slices
