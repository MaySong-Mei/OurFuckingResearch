"""
Loss functions for self-supervised multi-view consistency training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConsistencyLoss(nn.Module):
    """
    Multi-view consistency loss between segmentation predictions
    """

    def __init__(self, loss_type: str = 'dice', weight: Optional[torch.Tensor] = None):
        """
        Args:
            loss_type: Type of consistency loss ('dice', 'ce', 'mse', 'combined')
            weight: Optional class weights for CE loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between two predictions

        Args:
            pred1: [B, N, H, W, C] - first prediction
            pred2: [B, N, H, W, C] - second prediction

        Returns:
            loss: Scalar consistency loss
        """
        if self.loss_type == 'dice':
            return self.dice_loss(pred1, pred2)
        elif self.loss_type == 'ce':
            return self.cross_entropy_loss(pred1, pred2)
        elif self.loss_type == 'mse':
            return self.mse_loss(pred1, pred2)
        elif self.loss_type == 'combined':
            dice = self.dice_loss(pred1, pred2)
            ce = self.cross_entropy_loss(pred1, pred2)
            return 0.5 * dice + 0.5 * ce
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def dice_loss(self, pred1: torch.Tensor, pred2: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Dice loss between two predictions

        Args:
            pred1, pred2: [B, N, H, W, C] predictions
            smooth: Smoothing factor

        Returns:
            loss: 1 - Dice coefficient
        """
        # Apply softmax to get probabilities
        pred1_prob = F.softmax(pred1, dim=-1)
        pred2_prob = F.softmax(pred2, dim=-1)

        # Flatten spatial dimensions
        pred1_flat = pred1_prob.reshape(-1, pred1.shape[-1])
        pred2_flat = pred2_prob.reshape(-1, pred2.shape[-1])

        # Calculate intersection and union
        intersection = (pred1_flat * pred2_flat).sum(dim=0)
        pred1_sum = pred1_flat.sum(dim=0)
        pred2_sum = pred2_flat.sum(dim=0)

        # Dice coefficient
        dice = (2.0 * intersection + smooth) / (pred1_sum + pred2_sum + smooth)

        # Return loss (1 - dice)
        return 1.0 - dice.mean()

    def cross_entropy_loss(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric cross-entropy loss between predictions

        Args:
            pred1, pred2: [B, N, H, W, C] predictions

        Returns:
            loss: Symmetric CE loss
        """
        # Get probabilities
        pred1_prob = F.softmax(pred1, dim=-1)
        pred2_prob = F.softmax(pred2, dim=-1)

        # Log probabilities
        pred1_log = F.log_softmax(pred1, dim=-1)
        pred2_log = F.log_softmax(pred2, dim=-1)

        # Symmetric KL divergence
        loss1 = -(pred2_prob * pred1_log).sum(dim=-1).mean()
        loss2 = -(pred1_prob * pred2_log).sum(dim=-1).mean()

        return 0.5 * (loss1 + loss2)

    def mse_loss(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        MSE loss between predictions

        Args:
            pred1, pred2: [B, N, H, W, C] predictions

        Returns:
            loss: MSE between predictions
        """
        pred1_prob = F.softmax(pred1, dim=-1)
        pred2_prob = F.softmax(pred2, dim=-1)

        return F.mse_loss(pred1_prob, pred2_prob)


class SmoothnessLoss(nn.Module):
    """
    Smoothness loss to ensure smooth transitions between interpolated slices
    """

    def __init__(self, order: int = 1):
        """
        Args:
            order: Order of derivative (1 or 2)
        """
        super().__init__()
        self.order = order

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss along depth dimension

        Args:
            volume: [B, N, H, W] interpolated volume

        Returns:
            loss: Smoothness loss
        """
        if self.order == 1:
            return self.first_order_smoothness(volume)
        elif self.order == 2:
            return self.second_order_smoothness(volume)
        else:
            raise ValueError(f"Order must be 1 or 2, got {self.order}")

    def first_order_smoothness(self, volume: torch.Tensor) -> torch.Tensor:
        """
        First-order smoothness (gradient magnitude)

        Args:
            volume: [B, N, H, W]

        Returns:
            loss: L1 norm of temporal gradient
        """
        # Gradient along depth dimension
        diff = volume[:, 1:] - volume[:, :-1]

        # L1 norm
        loss = torch.abs(diff).mean()

        return loss

    def second_order_smoothness(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Second-order smoothness (Laplacian)

        Args:
            volume: [B, N, H, W]

        Returns:
            loss: L1 norm of second derivative
        """
        # Second derivative along depth
        diff1 = volume[:, 1:] - volume[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]

        # L1 norm
        loss = torch.abs(diff2).mean()

        return loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss to preserve original slice information
    """

    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1', 'l2', or 'perceptual'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, interpolated_volume: torch.Tensor, original_slices: torch.Tensor) -> torch.Tensor:
        """
        Ensure interpolated volume preserves original slices

        Args:
            interpolated_volume: [B, N', H, W] interpolated volume
            original_slices: [B, N, H, W] original slices

        Returns:
            loss: Reconstruction loss
        """
        B, N_orig, H, W = original_slices.shape
        N_interp = interpolated_volume.shape[1]

        # For 2x interpolation, original slices are at indices 0, 2, 4, ...
        # Assuming N' = 2*N - 1
        if N_interp == 2 * N_orig - 1:
            # Extract original slice positions
            indices = torch.arange(0, N_interp, 2, device=interpolated_volume.device)
            reconstructed = interpolated_volume[:, indices]
        else:
            # General case: sample uniformly
            indices = torch.linspace(0, N_interp - 1, N_orig, device=interpolated_volume.device).long()
            reconstructed = interpolated_volume[:, indices]

        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(reconstructed, original_slices)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(reconstructed, original_slices)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            pred: [B, N, H, W, C] predictions
            target: [B, N, H, W, C] targets

        Returns:
            loss: Focal loss
        """
        # Get probabilities
        pred_prob = F.softmax(pred, dim=-1)
        target_prob = F.softmax(target, dim=-1)

        # Focal weight
        focal_weight = (1 - pred_prob) ** self.gamma

        # Cross entropy
        ce = -(target_prob * F.log_softmax(pred, dim=-1))

        # Focal loss
        loss = self.alpha * focal_weight * ce

        return loss.sum(dim=-1).mean()


class TotalVariationLoss(nn.Module):
    """
    Total variation loss for spatial smoothness
    """

    def __init__(self):
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss

        Args:
            volume: [B, N, H, W]

        Returns:
            loss: Total variation loss
        """
        # Spatial gradients
        diff_h = torch.abs(volume[:, :, 1:, :] - volume[:, :, :-1, :])
        diff_w = torch.abs(volume[:, :, :, 1:] - volume[:, :, :, :-1])

        # TV loss
        tv_loss = diff_h.mean() + diff_w.mean()

        return tv_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features (optional)
    """

    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        """
        Args:
            feature_extractor: Pre-trained model for feature extraction
        """
        super().__init__()
        self.feature_extractor = feature_extractor

        if feature_extractor is not None:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss

        Args:
            pred: [B, N, H, W] predictions
            target: [B, N, H, W] targets

        Returns:
            loss: Perceptual loss
        """
        if self.feature_extractor is None:
            # Fallback to L1 loss
            return F.l1_loss(pred, target)

        # Reshape for feature extraction
        B, N, H, W = pred.shape
        pred_flat = pred.view(B * N, 1, H, W)
        target_flat = target.view(B * N, 1, H, W)

        # Repeat to 3 channels if needed (for ImageNet models)
        if pred_flat.shape[1] == 1:
            pred_flat = pred_flat.repeat(1, 3, 1, 1)
            target_flat = target_flat.repeat(1, 3, 1, 1)

        # Extract features
        with torch.no_grad():
            feat_target = self.feature_extractor(target_flat)

        feat_pred = self.feature_extractor(pred_flat)

        # Compute loss
        loss = F.mse_loss(feat_pred, feat_target)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components
    """

    def __init__(
        self,
        lambda_consistency: float = 1.0,
        lambda_smoothness: float = 0.1,
        lambda_reconstruction: float = 1.0,
        lambda_tv: float = 0.01,
        consistency_type: str = 'dice'
    ):
        """
        Args:
            lambda_consistency: Weight for consistency loss
            lambda_smoothness: Weight for smoothness loss
            lambda_reconstruction: Weight for reconstruction loss
            lambda_tv: Weight for TV loss
            consistency_type: Type of consistency loss
        """
        super().__init__()

        self.lambda_consistency = lambda_consistency
        self.lambda_smoothness = lambda_smoothness
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_tv = lambda_tv

        self.consistency_loss = ConsistencyLoss(loss_type=consistency_type)
        self.smoothness_loss = SmoothnessLoss(order=1)
        self.reconstruction_loss = ReconstructionLoss(loss_type='l1')
        self.tv_loss = TotalVariationLoss()

    def forward(
        self,
        seg_axial: torch.Tensor,
        seg_sagittal: torch.Tensor,
        seg_coronal: torch.Tensor,
        interpolated_volume: torch.Tensor,
        original_slices: torch.Tensor
    ) -> tuple:
        """
        Compute total loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual components
        """
        # Consistency losses
        consistency_sag = self.consistency_loss(seg_axial, seg_sagittal)
        consistency_cor = self.consistency_loss(seg_axial, seg_coronal)
        consistency_total = (consistency_sag + consistency_cor) / 2

        # Smoothness loss
        smoothness = self.smoothness_loss(interpolated_volume)

        # Reconstruction loss
        reconstruction = self.reconstruction_loss(interpolated_volume, original_slices)

        # TV loss
        tv = self.tv_loss(interpolated_volume)

        # Total loss
        total = (
            self.lambda_consistency * consistency_total +
            self.lambda_smoothness * smoothness +
            self.lambda_reconstruction * reconstruction +
            self.lambda_tv * tv
        )

        loss_dict = {
            'total': total.item(),
            'consistency': consistency_total.item(),
            'smoothness': smoothness.item(),
            'reconstruction': reconstruction.item(),
            'tv': tv.item()
        }

        return total, loss_dict
