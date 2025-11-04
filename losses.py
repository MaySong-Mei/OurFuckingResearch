"""Loss functions for multi-view consistency training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConsistencyLoss(nn.Module):
    """Multi-view consistency loss (Dice)"""

    def __init__(self, loss_type: str = 'dice'):
        super().__init__()

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Compute Dice loss between two predictions"""
        pred1_prob = F.softmax(pred1, dim=-1)
        pred2_prob = F.softmax(pred2, dim=-1)

        print(f"pred_prob shape: {pred1_prob.shape}")

        # Flatten spatial dimensions
        pred1_flat = pred1_prob.reshape(-1, pred1.shape[-1])
        pred2_flat = pred2_prob.reshape(-1, pred2.shape[-1])

        # Dice coefficient
        smooth = 1e-6
        intersection = (pred1_flat * pred2_flat).sum(dim=0)
        pred1_sum = pred1_flat.sum(dim=0)
        pred2_sum = pred2_flat.sum(dim=0)
        dice = (2.0 * intersection + smooth) / (pred1_sum + pred2_sum + smooth)

        return 1.0 - dice.mean()


class SmoothnessLoss(nn.Module):
    """First-order smoothness loss for depth dimension"""

    def __init__(self, order: int = 1):
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute first-order smoothness loss with volume normalized to [0, 1]"""
        # Calculate differences between adjacent slices
        diff = volume[:, 1:] - volume[:, :-1]
        return torch.abs(diff).mean()


class SegmentationConsistencyWeighting(nn.Module):
    """Generate pixel-wise weights based on multi-view segmentation consistency"""

    def forward(self, prob_axial: torch.Tensor, prob_sagittal: torch.Tensor,
                prob_coronal: torch.Tensor) -> torch.Tensor:
        """
        Generate consistency weights from three views.
        Higher weight for inconsistent regions (where views disagree).

        Args:
            prob_axial: [B, N, H, W, C] - axial view probabilities
            prob_sagittal: [B, N, H, W, C] - sagittal view probabilities (remapped)
            prob_coronal: [B, N, H, W, C] - coronal view probabilities (remapped)

        Returns:
            weights: [B, N, H, W] - pixel-wise weights (normalized to [0.5, 1.0])
        """
        # Compute class predictions for each view
        class_axial = torch.argmax(prob_axial, dim=-1)  # [B, N, H, W]
        class_sag = torch.argmax(prob_sagittal, dim=-1)  # [B, N, H, W]
        class_cor = torch.argmax(prob_coronal, dim=-1)  # [B, N, H, W]

        # Measure disagreement: count how many views differ from majority vote
        majority_vote = torch.mode(
            torch.stack([class_axial, class_sag, class_cor], dim=-1),
            dim=-1
        ).values  # [B, N, H, W]

        disagreement = (
            (class_axial != majority_vote).float() +
            (class_sag != majority_vote).float() +
            (class_cor != majority_vote).float()
        ) / 3.0  # [B, N, H, W], range [0, 1]

        # Convert disagreement to weights: higher disagreement -> higher weight
        # Scale to [0.5, 1.0] to keep confident regions non-zero
        weights = 0.5 + 0.5 * disagreement

        return weights


class InterpolationGroundTruthLoss(nn.Module):
    """L1 loss with optional SSIM for interpolation quality"""

    def __init__(self, loss_type: str = 'l1', use_ssim: bool = False):
        super().__init__()
        self.use_ssim = use_ssim
        self.consistency_weighting = SegmentationConsistencyWeighting()

    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Compute SSIM loss"""
        B, N, H, W = x.shape
        x_flat = x.reshape(B * N, 1, H, W)
        y_flat = y.reshape(B * N, 1, H, W)

        mu1 = F.avg_pool2d(x_flat, window_size, padding=window_size//2)
        mu2 = F.avg_pool2d(y_flat, window_size, padding=window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(x_flat ** 2, window_size, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(y_flat ** 2, window_size, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(x_flat * y_flat, window_size, padding=window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim.mean()

    def _compute_ssim_metric(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> float:
        """Compute SSIM metric (higher is better)"""
        B, N, H, W = x.shape
        x_flat = x.reshape(B * N, 1, H, W)
        y_flat = y.reshape(B * N, 1, H, W)

        mu1 = F.avg_pool2d(x_flat, window_size, padding=window_size//2)
        mu2 = F.avg_pool2d(y_flat, window_size, padding=window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(x_flat ** 2, window_size, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(y_flat ** 2, window_size, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(x_flat * y_flat, window_size, padding=window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.mean().item()

    def _compute_psnr(self, x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
        """Compute PSNR metric"""
        mse = F.mse_loss(x, y)
        if mse == 0:
            return float('inf')
        psnr = 20 * math.log10(max_val / (torch.sqrt(mse).item()))
        return psnr

    def compute_metrics(self, interpolated_volume: torch.Tensor,
                       ground_truth_slices: torch.Tensor) -> dict:
        """Compute PSNR and SSIM metrics"""
        interpolated_intermediate = interpolated_volume[:, 1::2, :, :]
        ground_truth_intermediate = ground_truth_slices[:, 1::2, :, :]

        psnr = self._compute_psnr(interpolated_intermediate, ground_truth_intermediate)
        ssim = self._compute_ssim_metric(interpolated_intermediate, ground_truth_intermediate)

        return {'psnr': psnr, 'ssim': ssim}

    def forward(self, interpolated_volume: torch.Tensor,
                ground_truth_slices: torch.Tensor, weights: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        Compute interpolation loss with L1 and optional SSIM.

        Args:
            interpolated_volume: [B, N, H, W]
            ground_truth_slices: [B, N, H, W]
            weights: [B, N, H, W] - pixel-wise weights from consistency
        """
        interpolated_intermediate = interpolated_volume[:, 1::2, :, :]
        ground_truth_intermediate = ground_truth_slices[:, 1::2, :, :]

        # Compute per-pixel L1 loss
        pixel_loss = torch.abs(interpolated_intermediate - ground_truth_intermediate)

        # Apply consistency weights if provided
        if weights is not None:
            # Extract intermediate frame weights
            weight_intermediate = weights[:, 1::2, :, :]
            # Expand weights to match pixel_loss spatial dims
            pixel_loss = pixel_loss * weight_intermediate

        pixel_loss = pixel_loss.mean()

        if self.use_ssim:
            ssim_loss = self._compute_ssim(interpolated_intermediate, ground_truth_intermediate)
            loss = 0.8 * pixel_loss + 0.2 * ssim_loss
        else:
            loss = pixel_loss

        return loss
