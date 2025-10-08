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

from data_loader import MedicalVolumeDataset
from models.rife_interpolator import RIFEInterpolator
from models.unet_segmentation import UNetSegmentation
from models.medsam_segmentation import MedSAMSegmentation, load_medsam
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
        self.interpolator = RIFEInterpolator(
            scale=config.interpolation_factor
        ).to(self.device)

        # Load frozen segmentation model
        if config.use_medsam:
            logger.info("Using MedSAM for segmentation")
            self.segmentation = load_medsam(
                checkpoint_path=config.medsam_checkpoint,
                model_type=config.medsam_model_type,
                device=str(self.device),
                num_classes=config.num_classes,
                auto_download=config.medsam_auto_download
            )
        else:
            logger.info("Using U-Net for segmentation")
            self.segmentation = UNetSegmentation(
                in_channels=config.in_channels,
                num_classes=config.num_classes
            ).to(self.device)

            # Load checkpoint if provided
            if config.segmentation_checkpoint:
                logger.info(f"Loading segmentation checkpoint: {config.segmentation_checkpoint}")
                checkpoint = torch.load(config.segmentation_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.segmentation.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.segmentation.load_state_dict(checkpoint['state_dict'])
                else:
                    self.segmentation.load_state_dict(checkpoint)

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
        Interpolate between slices to create denser volume

        Args:
            slices: [B, N, H, W] - batch of N original slices

        Returns:
            interpolated: [B, N', H, W] - batch of N' interpolated slices
        """
        batch_size, num_slices, H, W = slices.shape

        # For 2x interpolation: N' = 2*N - 1
        interpolated_slices = [slices[:, 0]]

        for i in range(num_slices - 1):
            # Get consecutive slice pairs
            frame0 = slices[:, i:i+1]  # [B, 1, H, W]
            frame1 = slices[:, i+1:i+2]  # [B, 1, H, W]

            # Interpolate middle frame(s)
            with torch.set_grad_enabled(self.training):
                middle_frame = self.interpolator(frame0, frame1)

            interpolated_slices.append(middle_frame)
            interpolated_slices.append(frame1)

        # Stack all slices
        interpolated = torch.cat(interpolated_slices, dim=1)

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

        # Segment each view
        with torch.no_grad():
            seg_axial = self._segment_slices(axial_slices)
            seg_sagittal = self._segment_slices(sagittal_slices)
            seg_coronal = self._segment_slices(coronal_slices)

        return seg_axial, seg_sagittal, seg_coronal

    def _segment_slices(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Segment a batch of slices

        Args:
            slices: [B, N, H, W]

        Returns:
            segmentations: [B, N, H, W, C]
        """
        B, N, H, W = slices.shape

        # Reshape to process all slices at once
        slices_flat = slices.view(B * N, 1, H, W)

        # Segment
        seg_flat = self.segmentation(slices_flat)  # [B*N, C, H, W]

        # Reshape back
        C = seg_flat.shape[1]
        segmentations = seg_flat.view(B, N, C, H, W)
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
    config = Config()
    pipeline = TrainingPipeline(config)
    pipeline.train()


if __name__ == '__main__':
    main()
