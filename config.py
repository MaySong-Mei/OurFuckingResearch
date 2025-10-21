"""
Configuration for training pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class Config:
    """Training configuration"""

    # Data settings
    data_dir: str = "./data"
    split_file: Optional[str] = None
    single_file_mode: bool = False  # Use a single DICOM file for training/testing
    single_dicom_path: Optional[str] = None  # Path to single DICOM file
    num_slices: int = 129  # Number of slices to sample (will be interpolated to 256)
    img_size: tuple = (256, 256)
    in_channels: int = 1
    num_classes: int = 2  # Background + foreground (adjust for multi-class)

    # Interpolation settings
    interpolation_factor: int = 2  # 2x interpolation (N -> 2N-1)
    interpolation_method: str = "rife"  # "rife", "film", or "simple"

    # Model settings
    segmentation_model: str = "vit"  # "unet", "attention_unet", "medsam", "vit"
    segmentation_checkpoint: Optional[str] = None
    freeze_segmentation: bool = True

    # Vision Transformer (TransUNet) settings
    vit_name: str = "ViT-B_16"  # "ViT-B_16", "ViT-B_32", "ViT-L_16", "R50-ViT-B_16", etc.
    vit_patches_size: int = 16  # Patch size for ViT
    n_skip: int = 3  # Number of skip connections (0-3)

    # MedSAM settings
    use_medsam: bool = False
    medsam_checkpoint: Optional[str] = None  # Path to MedSAM checkpoint
    medsam_model_type: str = "vit_b"  # "vit_b", "vit_l", "vit_h"
    medsam_auto_download: bool = False  # Auto-download checkpoint if not found

    # Training settings
    batch_size: int = 2  # Small batch size due to 3D volume memory
    num_epochs: int = 100
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    num_workers: int = 4

    # Loss weights (self-supervised: no ground truth required)
    lambda_consistency: float = 1.0  # Weight for multi-view consistency loss
    lambda_smoothness: float = 0.1  # Weight for smoothness regularization
    lambda_tv: float = 0.01  # Weight for total variation regularization
    consistency_loss_type: str = "dice"  # "dice", "ce", "mse", "combined"

    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10
    resume_from: Optional[str] = None

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    project_name: str = "medical-interpolation"
    experiment_name: Optional[str] = None

    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Validation
    val_interval: int = 1
    val_batch_size: int = 1

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    cache_data: bool = False

    # Misc
    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        """Validate and process configuration"""
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Validate paths
        if self.single_file_mode:
            if not self.single_dicom_path:
                raise ValueError("single_dicom_path must be specified when single_file_mode=True")
            if not Path(self.single_dicom_path).exists():
                raise ValueError(f"Single DICOM file does not exist: {self.single_dicom_path}")
        else:
            if self.data_dir and not Path(self.data_dir).exists():
                raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Validate parameters
        if self.interpolation_factor < 2:
            raise ValueError(f"Interpolation factor must be >= 2, got {self.interpolation_factor}")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")

    @property
    def transform(self):
        """Get data transform pipeline"""
        # Can be extended with augmentations
        return None

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Data-specific configuration"""
    dataset_name: str = "custom"
    data_format: str = "dicom"  # "dicom", "nifti", "numpy"
    modality: str = "CT"  # "CT", "MRI", "PET"
    anatomical_region: str = "chest"

    # Preprocessing
    clip_range: Optional[tuple] = None  # HU window for CT
    normalize_method: str = "percentile"  # "percentile", "zscore", "minmax"
    resample: bool = False
    target_spacing: Optional[tuple] = None
