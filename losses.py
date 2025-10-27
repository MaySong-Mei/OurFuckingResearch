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


class InterpolationGroundTruthLoss(nn.Module):
    """
    Loss comparing interpolated slices with ground truth slices.

    From 129 sampled slices (0, 2, 4, ..., 256), we interpolate to get 128 intermediate slices
    (1, 3, 5, ..., 255). These are compared with the corresponding ground truth slices.
    """

    def __init__(self, loss_type: str = 'l1', use_ssim: bool = False):
        """
        Args:
            loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
            use_ssim: Whether to combine with SSIM loss for better perceptual quality
        """
        super().__init__()
        self.loss_type = loss_type
        self.use_ssim = use_ssim

    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Compute SSIM (Structural Similarity Index) loss
        SSIM is more perceptually meaningful than MSE

        Args:
            x, y: [B, N, H, W] tensors
            window_size: Gaussian window size (default 11)

        Returns:
            ssim_loss: 1 - SSIM (so lower is better)
        """
        # Flatten spatial and slice dimensions
        B, N, H, W = x.shape
        x_flat = x.reshape(B * N, 1, H, W)
        y_flat = y.reshape(B * N, 1, H, W)

        # Mean and variance
        mu1 = F.avg_pool2d(x_flat, window_size, padding=window_size//2)
        mu2 = F.avg_pool2d(y_flat, window_size, padding=window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(x_flat ** 2, window_size, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(y_flat ** 2, window_size, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(x_flat * y_flat, window_size, padding=window_size//2) - mu1_mu2

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim.mean()

    def forward(self, interpolated_volume: torch.Tensor,
                ground_truth_slices: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Compute interpolation ground truth loss

        Args:
            interpolated_volume: [B, 256, H, W] - full interpolated volume (0-256 indices)
            ground_truth_slices: [B, 257, H, W] - ground truth slices (0-256 indices)
            debug: Whether to print debug information

        Returns:
            loss: Scalar loss value
        """
        # Extract interpolated intermediate slices: indices 1, 3, 5, ..., 255 (128 slices)
        # These correspond to the gaps between sampled slices
        interpolated_intermediate = interpolated_volume[:, 1::2, :, :]  # [B, 128, H, W]

        # Extract ground truth intermediate slices: indices 1, 3, 5, ..., 255 (128 slices)
        ground_truth_intermediate = ground_truth_slices[:, 1::2, :, :]  # [B, 128, H, W]

        if debug:
            # Debug information
            print(f"\n=== InterpolationGroundTruthLoss Debug ===")
            print(f"Interpolated intermediate shape: {interpolated_intermediate.shape}")
            print(f"Ground truth intermediate shape: {ground_truth_intermediate.shape}")
            print(f"Interpolated - Min: {interpolated_intermediate.min():.6f}, Max: {interpolated_intermediate.max():.6f}, Mean: {interpolated_intermediate.mean():.6f}")
            print(f"Ground truth - Min: {ground_truth_intermediate.min():.6f}, Max: {ground_truth_intermediate.max():.6f}, Mean: {ground_truth_intermediate.mean():.6f}")

            # Compute element-wise differences
            diff = torch.abs(interpolated_intermediate - ground_truth_intermediate)
            print(f"Absolute difference - Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")

        # Compute base loss based on loss type
        if self.loss_type == 'l1':
            pixel_loss = F.l1_loss(interpolated_intermediate, ground_truth_intermediate)
        elif self.loss_type == 'l2':
            pixel_loss = F.mse_loss(interpolated_intermediate, ground_truth_intermediate)
        elif self.loss_type == 'smooth_l1':
            pixel_loss = F.smooth_l1_loss(interpolated_intermediate, ground_truth_intermediate)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Optionally add SSIM loss for better perceptual quality
        if self.use_ssim:
            ssim_loss = self._compute_ssim(interpolated_intermediate, ground_truth_intermediate)
            loss = 0.8 * pixel_loss + 0.2 * ssim_loss

            if debug:
                print(f"Pixel loss ({self.loss_type}): {pixel_loss.item():.6f}")
                print(f"SSIM loss: {ssim_loss.item():.6f}")
                print(f"Combined loss (0.8*pixel + 0.2*ssim): {loss.item():.6f}")
                print(f"=========================================\n")
        else:
            loss = pixel_loss

            if debug:
                print(f"Pixel loss ({self.loss_type}): {pixel_loss.item():.6f}")
                print(f"=========================================\n")

        return loss
