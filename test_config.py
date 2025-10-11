"""
Simple configuration for testing with single DICOM file
"""

from config import Config


def get_test_config() -> Config:
    """Get configuration for testing with single DICOM file"""
    return Config(
        # Data settings - single file testing doesn't use data_dir
        data_dir=".",  # Current directory
        num_slices=8,  # Smaller for faster testing
        img_size=(256, 256),
        in_channels=1,
        num_classes=2,

        # Training settings - minimal for testing
        batch_size=1,
        num_epochs=1,  # Just 1 epoch for testing
        learning_rate=1e-4,
        num_workers=0,  # No workers for single file

        # Loss weights
        lambda_consistency=1.0,
        lambda_smoothness=0.1,
        lambda_reconstruction=1.0,

        # Checkpointing
        checkpoint_dir="./test_checkpoints",
        save_interval=1,

        # Logging
        log_interval=1,
        use_wandb=False,  # Disable wandb for testing

        # Hardware
        device="cuda",
        cache_data=False,

        # Segmentation
        use_medsam=False,  # Use simple U-Net
        segmentation_checkpoint=None,  # Random initialization for testing
    )


if __name__ == '__main__':
    # Print the config for verification
    config = get_test_config()
    print("Test Configuration:")
    print("=" * 80)
    for key, value in config.to_dict().items():
        print(f"{key:30s}: {value}")
    print("=" * 80)
