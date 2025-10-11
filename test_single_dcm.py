"""
Test script for single DICOM file with IFNet interpolation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import numpy as np

from data_loader import SimpleDICOMDataset
from models.IFNet import IFNet
from models.unet_segmentation import UNetSegmentation
from utils.multi_view import MultiViewExtractor
from losses import ConsistencyLoss, SmoothnessLoss, ReconstructionLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTester:
    """Simple tester for single DICOM file"""

    def __init__(
        self,
        dicom_path: str,
        num_slices: int = 8,
        img_size: tuple = (256, 256),
        device: str = 'cuda'
    ):
        self.dicom_path = dicom_path
        self.num_slices = num_slices
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

        # Initialize models
        logger.info("Initializing IFNet interpolator...")
        self.interpolator = IFNet().to(self.device)
        self.interpolator.eval()  # Start in eval mode for testing

        logger.info("Initializing U-Net segmentation...")
        self.segmentation = UNetSegmentation(
            in_channels=1,
            num_classes=2
        ).to(self.device)
        self.segmentation.eval()
        for param in self.segmentation.parameters():
            param.requires_grad = False

        # Create dataset
        logger.info(f"Loading DICOM from: {dicom_path}")
        self.dataset = SimpleDICOMDataset(
            dicom_path=dicom_path,
            num_slices=num_slices,
            img_size=img_size,
            normalize=True
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False
        )

        # Loss functions
        self.consistency_loss = ConsistencyLoss(loss_type='dice')
        self.smoothness_loss = SmoothnessLoss()
        self.reconstruction_loss = ReconstructionLoss()

    def interpolate_volume(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between slices using IFNet

        Args:
            slices: [B, N, H, W] - batch of N original slices

        Returns:
            interpolated: [B, N', H, W] - batch of N' interpolated slices
        """
        batch_size, num_slices, H, W = slices.shape

        # For 2x interpolation: N' = 2*N - 1
        interpolated_slices = [slices[:, 0]]

        logger.info(f"Interpolating {num_slices - 1} frame pairs...")

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

            # Interpolate middle frame
            with torch.no_grad():
                # IFNet returns: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
                _, _, merged = self.interpolator(ifnet_input)
                middle_frame_rgb = merged[2]  # [B, 3, H, W]

                # Convert back to grayscale by averaging channels
                middle_frame = middle_frame_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

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
            seg_axial, seg_sagittal, seg_coronal
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

    def test_single_volume(self):
        """Test interpolation on single volume"""
        logger.info("=" * 80)
        logger.info("Starting test...")
        logger.info("=" * 80)

        for batch_idx, batch in enumerate(self.dataloader):
            slices = batch['slices'].to(self.device)  # [B, N, H, W]
            file_path = batch['file_path'][0]
            volume_shape = batch['volume_shape']

            logger.info(f"\nProcessing: {file_path}")
            logger.info(f"Original volume shape: {volume_shape}")
            logger.info(f"Input slices shape: {slices.shape}")
            logger.info(f"Slice value range: [{slices.min():.3f}, {slices.max():.3f}]")

            # 1. Interpolate volume
            logger.info("\n[Step 1/3] Interpolating volume with IFNet...")
            interpolated_volume = self.interpolate_volume(slices)
            logger.info(f"Interpolated volume shape: {interpolated_volume.shape}")
            logger.info(f"Interpolated value range: [{interpolated_volume.min():.3f}, {interpolated_volume.max():.3f}]")

            # 2. Generate multi-view segmentations
            logger.info("\n[Step 2/3] Generating multi-view segmentations...")
            seg_axial, seg_sagittal, seg_coronal = \
                self.compute_multi_view_segmentations(interpolated_volume)
            logger.info(f"Axial segmentation shape: {seg_axial.shape}")
            logger.info(f"Sagittal segmentation shape: {seg_sagittal.shape}")
            logger.info(f"Coronal segmentation shape: {seg_coronal.shape}")

            # 3. Compute losses
            logger.info("\n[Step 3/3] Computing losses...")

            # Remap sagittal and coronal to axial space for comparison
            seg_sagittal_remapped = seg_sagittal.permute(0, 2, 1, 3, 4)  # [B, N', H, W, C]
            seg_coronal_remapped = seg_coronal.permute(0, 2, 3, 1, 4)   # [B, N', H, W, C]

            # Consistency losses
            consistency_sag = self.consistency_loss(seg_axial, seg_sagittal_remapped)
            consistency_cor = self.consistency_loss(seg_axial, seg_coronal_remapped)
            consistency_total = (consistency_sag + consistency_cor) / 2

            # Smoothness loss
            smoothness = self.smoothness_loss(interpolated_volume)

            # Reconstruction loss
            reconstruction = self.reconstruction_loss(interpolated_volume, slices)

            logger.info(f"\nLoss breakdown:")
            logger.info(f"  Consistency (sagittal): {consistency_sag.item():.6f}")
            logger.info(f"  Consistency (coronal):  {consistency_cor.item():.6f}")
            logger.info(f"  Consistency (total):    {consistency_total.item():.6f}")
            logger.info(f"  Smoothness:             {smoothness.item():.6f}")
            logger.info(f"  Reconstruction:         {reconstruction.item():.6f}")

            # Total loss (with default weights)
            total_loss = consistency_total + 0.1 * smoothness + reconstruction
            logger.info(f"  Total loss:             {total_loss.item():.6f}")

        logger.info("\n" + "=" * 80)
        logger.info("Test completed successfully!")
        logger.info("=" * 80)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test IFNet with single DICOM file')
    parser.add_argument('--dicom_path', type=str,
                       default='./example.dcm',
                       help='Path to DICOM file')
    parser.add_argument('--num_slices', type=int, default=8,
                       help='Number of slices to extract')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (square)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create tester
    tester = SimpleTester(
        dicom_path=args.dicom_path,
        num_slices=args.num_slices,
        img_size=(args.img_size, args.img_size),
        device=args.device
    )

    # Run test
    tester.test_single_volume()


if __name__ == '__main__':
    main()
