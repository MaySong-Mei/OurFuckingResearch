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
    num_slices: int = 16
    img_size: tuple = (256, 256)
    in_channels: int = 1
    num_classes: int = 2  # Background + foreground (adjust for multi-class)

    # Interpolation settings
    interpolation_factor: int = 2  # 2x interpolation (N -> 2N-1)
    interpolation_method: str = "rife"  # "rife", "film", or "simple"

    # Model settings
    segmentation_model: str = "unet"  # "unet", "attention_unet", "medsam"
    segmentation_checkpoint: Optional[str] = None
    freeze_segmentation: bool = True

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

    # Loss weights
    lambda_consistency: float = 1.0
    lambda_smoothness: float = 0.1
    lambda_reconstruction: float = 1.0
    lambda_tv: float = 0.01
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
class RIFEConfig:
    """Configuration for RIFE interpolator"""
    scale: int = 2
    model_type: str = "v4.0"  # RIFE version
    ensemble: bool = False
    use_half: bool = False


@dataclass
class UNetConfig:
    """Configuration for U-Net segmentation"""
    in_channels: int = 1
    num_classes: int = 2
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_attention: bool = False
    dropout: float = 0.1


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


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = "experiment_001"
    description: str = ""

    # Sub-configs
    train: Config = field(default_factory=Config)
    model: UNetConfig = field(default_factory=UNetConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment tracking
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def save(self, path: str):
        """Save configuration to file"""
        import json

        config_dict = {
            'name': self.name,
            'description': self.description,
            'train': self.train.to_dict(),
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'tags': self.tags,
            'notes': self.notes
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = cls(
            name=config_dict['name'],
            description=config_dict['description']
        )

        config.train = Config.from_dict(config_dict['train'])
        config.model = UNetConfig(**config_dict['model'])
        config.data = DataConfig(**config_dict['data'])
        config.tags = config_dict.get('tags', [])
        config.notes = config_dict.get('notes', '')

        return config


# Default configurations for common scenarios
def get_default_config() -> Config:
    """Get default training configuration"""
    return Config()


def get_fast_debug_config() -> Config:
    """Get configuration for fast debugging"""
    return Config(
        batch_size=1,
        num_epochs=2,
        num_slices=8,
        img_size=(128, 128),
        log_interval=1,
        save_interval=1,
        num_workers=0,
        cache_data=False
    )


def get_high_quality_config() -> Config:
    """Get configuration for high-quality training"""
    return Config(
        batch_size=4,
        num_epochs=200,
        num_slices=32,
        img_size=(512, 512),
        learning_rate=5e-5,
        lambda_consistency=2.0,
        lambda_smoothness=0.2,
        lambda_reconstruction=2.0,
        mixed_precision=True,
        use_wandb=True
    )


def get_medsam_config() -> Config:
    """Get configuration using MedSAM for segmentation"""
    return Config(
        use_medsam=True,
        medsam_checkpoint="checkpoints/medsam_vit_b.pth",
        medsam_model_type="vit_b",
        medsam_auto_download=True,
        batch_size=2,
        num_epochs=100,
        num_slices=16,
        img_size=(256, 256),
        lambda_consistency=1.5,
        use_wandb=True
    )


def get_low_memory_config() -> Config:
    """Get configuration for limited GPU memory"""
    return Config(
        batch_size=1,
        num_slices=8,
        img_size=(256, 256),
        num_workers=2,
        mixed_precision=True,
        cache_data=False
    )
