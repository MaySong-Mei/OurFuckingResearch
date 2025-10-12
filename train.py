"""
3D Medical Image Interpolation Training Pipeline
Self-Supervised Multi-View Consistency Approach

Uses RIFE for interpolation and U-Net for segmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import numpy as np

from data_loader import MedicalVolumeDataset, SimpleDICOMDataset
from models.IFNet import IFNet
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.multi_view import MultiViewExtractor
from losses import ConsistencyLoss, SmoothnessLoss, ReconstructionLoss
from config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Main training pipeline for self-supervised 3D interpolation"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.interpolator = IFNet().to(self.device)

        # Load frozen segmentation model using TransUNet's ViT_seg
        logger.info("Using Vision Transformer (TransUNet) for segmentation")

        # Get the ViT configuration
        vit_name = config.vit_name if hasattr(config, 'vit_name') else 'ViT-B_16'
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = config.num_classes

        # Set n_skip if provided (default to 3 for skip connections)
        if not hasattr(config_vit, 'n_skip'):
            config_vit.n_skip = config.n_skip if hasattr(config, 'n_skip') else 3

        # Set patch size
        vit_patches_size = config.vit_patches_size if hasattr(config, 'vit_patches_size') else 16
        config_vit.patches.size = (vit_patches_size, vit_patches_size)

        # Handle ResNet hybrid models
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(config.img_size[0] / vit_patches_size),
                int(config.img_size[1] / vit_patches_size)
            )
            # Add skip_channels for ResNet hybrid models
            if not hasattr(config_vit, 'skip_channels'):
                config_vit.skip_channels = [512, 256, 64, 16]
        else:
            # For non-ResNet models, ensure skip_channels exists (set to zeros)
            if not hasattr(config_vit, 'skip_channels'):
                config_vit.skip_channels = [0, 0, 0, 0]

        # Initialize the model
        self.segmentation = ViT_seg(
            config_vit,
            img_size=config.img_size[0],  # Assumes square images
            num_classes=config_vit.n_classes
        ).to(self.device)

        # Load pretrained ImageNet weights from config_vit.pretrained_path
        import os
        if hasattr(config_vit, 'pretrained_path') and config_vit.pretrained_path:
            pretrained_path = config_vit.pretrained_path
            logger.info(f"Loading pretrained ViT weights from: {pretrained_path}")
            if os.path.exists(pretrained_path):
                self.segmentation.load_from(weights=np.load(pretrained_path))
                logger.info("Pretrained ViT weights loaded successfully")
            else:
                logger.warning(f"Pretrained weights file not found: {pretrained_path}")
                logger.warning("Continuing with randomly initialized weights")
        else:
            logger.info("No pretrained weights specified, using random initialization")

        self.segmentation.eval()
        for param in self.segmentation.parameters():
            param.requires_grad = False

        # Multi-view extractor
        self.multi_view_extractor = MultiViewExtractor()

        # Loss functions
        self.consistency_loss = ConsistencyLoss(
            loss_type=config.consistency_loss_type
        )
        self.smoothness_loss = SmoothnessLoss()
        self.reconstruction_loss = ReconstructionLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.interpolator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )

        # Data loaders
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')

        # Metrics tracking
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config),
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def _create_data_loader(self, split: str) -> DataLoader:
        """Create data loader for train/val split"""
        if self.config.single_file_mode:
            # Use SimpleDICOMDataset for single file
            # In single file mode, we use the same file for train and val
            dataset = SimpleDICOMDataset(
                dicom_path=self.config.single_dicom_path,
                num_slices=self.config.num_slices,
                img_size=self.config.img_size,
                transform=self.config.transform,
                normalize=True
            )
        else:
            # Use MedicalVolumeDataset for multiple files
            dataset = MedicalVolumeDataset(
                data_dir=self.config.data_dir,
                split=split,
                transform=self.config.transform,
                cache_data=self.config.cache_data
            )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def interpolate_volume(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between slices to create denser volume using IFNet
        Takes 129 slices and interpolates to 256 slices

        Args:
            slices: [B, 129, H, W] - batch of 129 original slices

        Returns:
            interpolated: [B, 256, H, W] - batch of 256 interpolated slices
        """
        batch_size, num_slices, H, W = slices.shape

        # First use IFNet to do 2x interpolation: 129 -> 257 slices
        # For 2x interpolation: N' = 2*N - 1
        interpolated_slices = [slices[:, 0:1]]  # Keep dimension [B, 1, H, W]

        for i in range(num_slices - 1):
            # Get consecutive slice pairs
            frame0 = slices[:, i:i+1]  # [B, 1, H, W]
            frame1 = slices[:, i+1:i+2]  # [B, 1, H, W]

            # IFNet expects [B, 6, H, W] input (concatenated RGB images)
            # Convert grayscale to 3-channel by repeating
            frame0_rgb = frame0.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            frame1_rgb = frame1.repeat(1, 3, 1, 1)  # [B, 3, H, W]

            # Concatenate along channel dimension
            ifnet_input = torch.cat([frame0_rgb, frame1_rgb], dim=1)  # [B, 6, H, W]

            # Interpolate middle frame(s)
            with torch.set_grad_enabled(self.interpolator.training):
                # IFNet returns: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
                # merged is a list of 3 frames, merged[2] is the final refined output
                _, _, merged = self.interpolator(ifnet_input)
                middle_frame_rgb = merged[2]  # [B, 3, H, W]

                # Convert back to grayscale by averaging channels
                middle_frame = middle_frame_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            interpolated_slices.append(middle_frame)
            interpolated_slices.append(frame1)

        # Stack all slices: 129 -> 257 slices
        interpolated = torch.cat(interpolated_slices, dim=1)  # [B, 257, H, W]

        # Drop the last frame to get exactly 256 slices
        interpolated = interpolated[:, :256, :, :]  # [B, 256, H, W]

        return interpolated

    def compute_multi_view_segmentations(self, volume: torch.Tensor):
        """
        Compute segmentations for all three orthogonal views

        Args:
            volume: [B, N', H, W] interpolated volume

        Returns:
            seg_axial: [B, N', H, W, C] - axial segmentations
            seg_sagittal: [B, H, N', W, C] - sagittal segmentations
            seg_coronal: [B, W, N', H, C] - coronal segmentations
        """
        # Extract views
        axial_slices = volume  # [B, N', H, W]
        sagittal_slices = volume.permute(0, 2, 1, 3)  # [B, H, N', W]
        coronal_slices = volume.permute(0, 3, 1, 2)  # [B, W, N', H]

        # Debug logging
        logger.info(f"Volume shape: {volume.shape}")
        logger.info(f"Axial slices shape: {axial_slices.shape}")
        logger.info(f"Sagittal slices shape: {sagittal_slices.shape}")
        logger.info(f"Coronal slices shape: {coronal_slices.shape}")

        # Segment each view
        with torch.no_grad():
            seg_axial = self._segment_slices(axial_slices)
            seg_sagittal = self._segment_slices(sagittal_slices)
            seg_coronal = self._segment_slices(coronal_slices)

        return seg_axial, seg_sagittal, seg_coronal

    def _segment_slices(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Segment a batch of slices using TransUNet's ViT_seg
        Handles interpolation if slices are not 256x256

        Args:
            slices: [B, N, H, W] - grayscale slices

        Returns:
            segmentations: [B, N, H, W, C] - class probabilities/logits
        """
        B, N, H, W = slices.shape
        expected_size = self.config.img_size[0]  # 256

        # Debug logging
        logger.debug(f"_segment_slices input shape: [B={B}, N={N}, H={H}, W={W}]")

        # Reshape to process all slices at once
        slices_flat = slices.view(B * N, 1, H, W)  # [B*N, 1, H, W]

        # If spatial dimensions don't match expected size, resize
        if H != expected_size or W != expected_size:
            logger.debug(f"Resizing from {H}x{W} to {expected_size}x{expected_size}")
            slices_flat = torch.nn.functional.interpolate(
                slices_flat,
                size=(expected_size, expected_size),
                mode='bilinear',
                align_corners=False
            )  # [B*N, 1, expected_size, expected_size]

        # Segment - ViT_seg automatically converts 1-channel to 3-channel
        seg_flat = self.segmentation(slices_flat)  # [B*N, num_classes, expected_size, expected_size]

        # Resize back to original spatial dimensions if needed
        if H != expected_size or W != expected_size:
            seg_flat = torch.nn.functional.interpolate(
                seg_flat,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # [B*N, num_classes, H, W]

        # Reshape back
        C = seg_flat.shape[1]
        segmentations = seg_flat.view(B, N, C, H, W)  # [B, N, C, H, W]
        segmentations = segmentations.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]

        return segmentations

    def compute_loss(self, seg_axial, seg_sagittal, seg_coronal,
                     interpolated_volume, original_slices):
        """
        Compute total training loss

        Args:
            seg_axial, seg_sagittal, seg_coronal: Segmentations from different views
            interpolated_volume: Interpolated volume
            original_slices: Original input slices

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        # Remap sagittal and coronal to axial space for comparison
        seg_sagittal_remapped = seg_sagittal.permute(0, 2, 1, 3, 4)  # [B, N', H, W, C]
        seg_coronal_remapped = seg_coronal.permute(0, 2, 3, 1, 4)   # [B, N', H, W, C]

        # Consistency losses
        consistency_sag = self.consistency_loss(seg_axial, seg_sagittal_remapped)
        consistency_cor = self.consistency_loss(seg_axial, seg_coronal_remapped)
        consistency_total = (consistency_sag + consistency_cor) / 2

        # Smoothness loss on interpolated volume
        smoothness = self.smoothness_loss(interpolated_volume)

        # Reconstruction loss (preserve original slices)
        reconstruction = self.reconstruction_loss(interpolated_volume, original_slices)

        # Total weighted loss
        total_loss = (
            self.config.lambda_consistency * consistency_total +
            self.config.lambda_smoothness * smoothness +
            self.config.lambda_reconstruction * reconstruction
        )

        loss_dict = {
            'total': total_loss.item(),
            'consistency': consistency_total.item(),
            'consistency_sagittal': consistency_sag.item(),
            'consistency_coronal': consistency_cor.item(),
            'smoothness': smoothness.item(),
            'reconstruction': reconstruction.item()
        }

        return total_loss, loss_dict

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.interpolator.train()

        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            slices = batch['slices'].to(self.device)  # [B, N, H, W]

            # Forward pass
            self.optimizer.zero_grad()

            # 1. Interpolate volume
            interpolated_volume = self.interpolate_volume(slices)

            # 2. Generate multi-view segmentations
            seg_axial, seg_sagittal, seg_coronal = \
                self.compute_multi_view_segmentations(interpolated_volume)

            # 3. Compute losses
            loss, loss_dict = self.compute_loss(
                seg_axial, seg_sagittal, seg_coronal,
                interpolated_volume, slices
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.interpolator.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            # Logging
            epoch_losses.append(loss_dict)
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

            if self.config.use_wandb and self.global_step % self.config.log_interval == 0:
                wandb.log({f'train/{k}': v for k, v in loss_dict.items()},
                         step=self.global_step)

            self.global_step += 1

        # Average epoch metrics
        avg_loss = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validation loop"""
        self.interpolator.eval()

        val_losses = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            slices = batch['slices'].to(self.device)

            # Forward pass
            interpolated_volume = self.interpolate_volume(slices)
            seg_axial, seg_sagittal, seg_coronal = \
                self.compute_multi_view_segmentations(interpolated_volume)

            loss, loss_dict = self.compute_loss(
                seg_axial, seg_sagittal, seg_coronal,
                interpolated_volume, slices
            )

            val_losses.append(loss_dict)

        # Average validation metrics
        avg_val_loss = {
            k: sum(d[k] for d in val_losses) / len(val_losses)
            for k in val_losses[0].keys()
        }

        if self.config.use_wandb:
            wandb.log({f'val/{k}': v for k, v in avg_val_loss.items()},
                     step=self.global_step)

        return avg_val_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'interpolator_state_dict': self.interpolator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.config)
        }

        # Save latest
        save_path = Path(self.config.checkpoint_dir) / 'latest.pth'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        # Save best
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

        # Save periodic
        if epoch % self.config.save_interval == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss['total']:.4f}")

            # Validate
            val_loss = self.validate(epoch)
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss['total']:.4f}")

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            is_best = val_loss['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss['total']

            self.save_checkpoint(epoch, val_loss['total'], is_best)

        logger.info("Training complete!")
        if self.config.use_wandb:
            wandb.finish()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train 3D Medical Image Interpolation')

    # Single file mode
    parser.add_argument('--single_file', type=str, default=None,
                       help='Path to single DICOM file for training (enables single file mode)')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory (ignored in single file mode)')
    parser.add_argument('--num_slices', type=int, default=129,
                       help='Number of slices to extract (will be interpolated to 256)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (square)')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')

    # Loss weights
    parser.add_argument('--lambda_consistency', type=float, default=1.0,
                       help='Consistency loss weight')
    parser.add_argument('--lambda_smoothness', type=float, default=0.1,
                       help='Smoothness loss weight')
    parser.add_argument('--lambda_reconstruction', type=float, default=1.0,
                       help='Reconstruction loss weight')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')

    # Segmentation
    parser.add_argument('--use_medsam', action='store_true',
                       help='Use MedSAM for segmentation')
    parser.add_argument('--segmentation_checkpoint', type=str, default=None,
                       help='Path to segmentation checkpoint')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='medical-interpolation',
                       help='Wandb project name')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Create config based on single file mode or not
    if args.single_file:
        logger.info(f"Running in SINGLE FILE MODE with: {args.single_file}")
        config = Config(
            # Single file mode
            single_file_mode=True,
            single_dicom_path=args.single_file,
            data_dir=".",

            # Data settings
            num_slices=args.num_slices,
            img_size=(args.img_size, args.img_size),

            # Training settings
            batch_size=1,  # Force batch_size=1 for single file
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_workers=0,  # Force 0 workers for single file

            # Loss weights
            lambda_consistency=args.lambda_consistency,
            lambda_smoothness=args.lambda_smoothness,
            lambda_reconstruction=args.lambda_reconstruction,

            # Checkpointing
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,

            # Segmentation
            use_medsam=args.use_medsam,
            segmentation_checkpoint=args.segmentation_checkpoint,

            # Logging
            use_wandb=args.use_wandb,
            project_name=args.project_name,
            log_interval=1,  # Log every step for single file

            # Hardware
            device=args.device,
            cache_data=False,
        )
    else:
        logger.info("Running in NORMAL MODE with dataset directory")
        config = Config(
            # Normal mode
            single_file_mode=False,
            data_dir=args.data_dir,

            # Data settings
            num_slices=args.num_slices,
            img_size=(args.img_size, args.img_size),

            # Training settings
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,

            # Loss weights
            lambda_consistency=args.lambda_consistency,
            lambda_smoothness=args.lambda_smoothness,
            lambda_reconstruction=args.lambda_reconstruction,

            # Checkpointing
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,

            # Segmentation
            use_medsam=args.use_medsam,
            segmentation_checkpoint=args.segmentation_checkpoint,

            # Logging
            use_wandb=args.use_wandb,
            project_name=args.project_name,

            # Hardware
            device=args.device,
        )

    # Check CUDA availability
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.device = "cpu"

    # Print config summary
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    if config.single_file_mode:
        logger.info(f"Mode: Single File")
        logger.info(f"DICOM file: {config.single_dicom_path}")
    else:
        logger.info(f"Mode: Dataset Directory")
        logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Number of slices: {config.num_slices}")
    logger.info(f"Image size: {config.img_size}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of epochs: {config.num_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 80)

    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.train()


if __name__ == '__main__':
    main()
