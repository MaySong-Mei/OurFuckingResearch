"""
Multi-view extraction utilities for 3D volumes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiViewExtractor:
    """Extract orthogonal views from 3D volume"""

    def __init__(self):
        pass

    def extract_views(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract three orthogonal views from a 3D volume

        Args:
            volume: [B, N, H, W] - batch of volumes

        Returns:
            axial: [B, N, H, W] - axial slices (original)
            sagittal: [B, H, N, W] - sagittal slices
            coronal: [B, W, N, H] - coronal slices
        """
        axial = volume  # [B, N, H, W]
        sagittal = volume.permute(0, 2, 1, 3)  # [B, H, N, W]
        coronal = volume.permute(0, 3, 1, 2)   # [B, W, N, H]

        return axial, sagittal, coronal

    def remap_to_axial(
        self,
        sagittal_pred: torch.Tensor,
        coronal_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remap sagittal and coronal predictions back to axial coordinate system

        Args:
            sagittal_pred: [B, H, N, W, C] - predictions in sagittal view
            coronal_pred: [B, W, N, H, C] - predictions in coronal view

        Returns:
            sagittal_axial: [B, N, H, W, C] - sagittal remapped to axial
            coronal_axial: [B, N, H, W, C] - coronal remapped to axial
        """
        # Remap sagittal: [B, H, N, W, C] -> [B, N, H, W, C]
        sagittal_axial = sagittal_pred.permute(0, 2, 1, 3, 4)

        # Remap coronal: [B, W, N, H, C] -> [B, N, H, W, C]
        coronal_axial = coronal_pred.permute(0, 2, 3, 1, 4)

        return sagittal_axial, coronal_axial

    def align_predictions(
        self,
        axial_pred: torch.Tensor,
        sagittal_pred: torch.Tensor,
        coronal_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align all predictions to common coordinate system (axial)

        Args:
            axial_pred: [B, N, H, W, C]
            sagittal_pred: [B, H, N, W, C]
            coronal_pred: [B, W, N, H, C]

        Returns:
            Three tensors all in shape [B, N, H, W, C]
        """
        sagittal_aligned, coronal_aligned = self.remap_to_axial(
            sagittal_pred, coronal_pred
        )

        return axial_pred, sagittal_aligned, coronal_aligned


class VolumeInterpolator:
    """
    Helper class for volume interpolation operations
    """

    @staticmethod
    def interpolate_spatial(volume: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Interpolate spatial dimensions of volume

        Args:
            volume: [B, N, H, W]
            size: Target (H', W')

        Returns:
            interpolated: [B, N, H', W']
        """
        B, N, H, W = volume.shape

        # Reshape to [B*N, 1, H, W] for spatial interpolation
        volume_flat = volume.view(B * N, 1, H, W)

        # Interpolate
        interpolated_flat = F.interpolate(
            volume_flat,
            size=size,
            mode='bilinear',
            align_corners=True
        )

        # Reshape back
        interpolated = interpolated_flat.view(B, N, size[0], size[1])

        return interpolated

    @staticmethod
    def interpolate_depth(volume: torch.Tensor, num_slices: int) -> torch.Tensor:
        """
        Interpolate along depth dimension

        Args:
            volume: [B, N, H, W]
            num_slices: Target number of slices N'

        Returns:
            interpolated: [B, N', H, W]
        """
        B, N, H, W = volume.shape

        # Permute to [B, H, W, N] for depth interpolation
        volume_perm = volume.permute(0, 2, 3, 1)
        volume_perm = volume_perm.reshape(B, H * W, N)

        # Interpolate along last dimension
        volume_perm = volume_perm.unsqueeze(1)  # [B, 1, H*W, N]

        interpolated = F.interpolate(
            volume_perm,
            size=(H * W, num_slices),
            mode='bilinear',
            align_corners=True
        )

        # Reshape back
        interpolated = interpolated.squeeze(1)  # [B, H*W, N']
        interpolated = interpolated.view(B, H, W, num_slices)
        interpolated = interpolated.permute(0, 3, 1, 2)  # [B, N', H, W]

        return interpolated


class VolumeVisualizer:
    """Utilities for visualizing multi-view consistency"""

    @staticmethod
    def get_center_slices(volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get center slice from each view

        Args:
            volume: [B, N, H, W]

        Returns:
            axial_center: [B, H, W] - center axial slice
            sagittal_center: [B, N, W] - center sagittal slice
            coronal_center: [B, N, H] - center coronal slice
        """
        B, N, H, W = volume.shape

        # Center indices
        n_center = N // 2
        h_center = H // 2
        w_center = W // 2

        # Extract center slices
        axial_center = volume[:, n_center, :, :]  # [B, H, W]
        sagittal_center = volume[:, :, h_center, :]  # [B, N, W]
        coronal_center = volume[:, :, :, w_center]  # [B, N, H]

        return axial_center, sagittal_center, coronal_center

    @staticmethod
    def create_montage(axial: torch.Tensor, sagittal: torch.Tensor, coronal: torch.Tensor) -> torch.Tensor:
        """
        Create a montage visualization of three views

        Args:
            axial: [B, H, W]
            sagittal: [B, N, W]
            coronal: [B, N, H]

        Returns:
            montage: [B, H_total, W_total] combined visualization
        """
        B = axial.shape[0]

        # Resize all to same dimensions for visualization
        H = axial.shape[1]
        W = axial.shape[2]

        # Resize sagittal and coronal
        sagittal_resized = F.interpolate(
            sagittal.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

        coronal_resized = F.interpolate(
            coronal.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

        # Stack horizontally
        montage = torch.cat([axial, sagittal_resized, coronal_resized], dim=2)

        return montage


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice score between predictions

    Args:
        pred: [B, N, H, W, C] predictions
        target: [B, N, H, W, C] targets
        smooth: Smoothing factor

    Returns:
        dice: Scalar Dice score
    """
    # Apply softmax if needed
    if pred.dtype == torch.float32:
        pred = torch.softmax(pred, dim=-1)
    if target.dtype == torch.float32:
        target = torch.softmax(target, dim=-1)

    # Flatten spatial dimensions
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=0)
    pred_sum = pred_flat.sum(dim=0)
    target_sum = target_flat.sum(dim=0)

    # Dice score
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice.mean()


def calculate_consistency_metrics(
    axial_pred: torch.Tensor,
    sagittal_pred: torch.Tensor,
    coronal_pred: torch.Tensor
) -> dict:
    """
    Calculate consistency metrics between views

    Args:
        axial_pred: [B, N, H, W, C]
        sagittal_pred: [B, N, H, W, C] (already remapped)
        coronal_pred: [B, N, H, W, C] (already remapped)

    Returns:
        Dictionary of consistency metrics
    """
    metrics = {}

    # Axial-Sagittal consistency
    metrics['dice_axial_sagittal'] = calculate_dice_score(axial_pred, sagittal_pred)

    # Axial-Coronal consistency
    metrics['dice_axial_coronal'] = calculate_dice_score(axial_pred, coronal_pred)

    # Sagittal-Coronal consistency
    metrics['dice_sagittal_coronal'] = calculate_dice_score(sagittal_pred, coronal_pred)

    # Average consistency
    metrics['dice_average'] = (
        metrics['dice_axial_sagittal'] +
        metrics['dice_axial_coronal'] +
        metrics['dice_sagittal_coronal']
    ) / 3.0

    return metrics
