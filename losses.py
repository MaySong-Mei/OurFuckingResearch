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
