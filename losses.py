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
        """Compute first-order smoothness loss"""
        diff = volume[:, 1:] - volume[:, :-1]
        return torch.abs(diff).mean()


class InterpolationGroundTruthLoss(nn.Module):
    """L1 loss with optional SSIM for interpolation quality"""

    def __init__(self, loss_type: str = 'l1', use_ssim: bool = False):
        super().__init__()
        self.use_ssim = use_ssim

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
                ground_truth_slices: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Compute interpolation loss with L1 and optional SSIM"""
        interpolated_intermediate = interpolated_volume[:, 1::2, :, :]
        ground_truth_intermediate = ground_truth_slices[:, 1::2, :, :]

        pixel_loss = F.l1_loss(interpolated_intermediate, ground_truth_intermediate)

        if self.use_ssim:
            ssim_loss = self._compute_ssim(interpolated_intermediate, ground_truth_intermediate)
            loss = 0.8 * pixel_loss + 0.2 * ssim_loss
        else:
            loss = pixel_loss

        return loss
