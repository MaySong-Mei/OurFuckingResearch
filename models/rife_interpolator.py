"""
RIFE-based interpolator for medical image slices
Simplified implementation adapted for medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(self.conv2(out))
        return F.leaky_relu(out + residual, 0.2, inplace=True)


class FlowEstimator(nn.Module):
    """Estimate optical flow between two frames"""

    def __init__(self, in_channels=2):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, 7, 1, 3),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ResBlock(128),
            ResBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1)  # Output: 2-channel flow (x, y)
        )

    def forward(self, img0, img1):
        """
        Estimate optical flow from img0 to img1

        Args:
            img0, img1: [B, 1, H, W]

        Returns:
            flow: [B, 2, H, W] optical flow
        """
        x = torch.cat([img0, img1], dim=1)
        features = self.encoder(x)
        flow = self.decoder(features)
        return flow


class RefineNet(nn.Module):
    """Refine interpolated frame"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32),  # 3 = img0 + img1 + warped
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ResBlock(128),
            ResBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img0, img1, warped):
        """
        Refine interpolated frame

        Args:
            img0, img1: [B, 1, H, W] input frames
            warped: [B, 1, H, W] warped frame

        Returns:
            refined: [B, 1, H, W] refined frame
        """
        x = torch.cat([img0, img1, warped], dim=1)
        features = self.encoder(x)
        refined = self.decoder(features)
        return refined


class RIFEInterpolator(nn.Module):
    """
    RIFE-inspired interpolator for medical images
    Simplified version for slice interpolation
    """

    def __init__(self, scale=2):
        """
        Args:
            scale: Interpolation scale (typically 2 for frame doubling)
        """
        super().__init__()
        self.scale = scale

        # Flow estimation networks
        self.flow_estimator = FlowEstimator(in_channels=2)

        # Refinement network
        self.refine_net = RefineNet()

    def warp(self, img, flow):
        """
        Warp image according to optical flow

        Args:
            img: [B, C, H, W] image
            flow: [B, 2, H, W] optical flow

        Returns:
            warped: [B, C, H, W] warped image
        """
        B, C, H, W = img.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )

        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Add flow to grid
        sample_grid = grid + flow

        # Normalize to [-1, 1] for grid_sample
        sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / (W - 1) - 1.0
        sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / (H - 1) - 1.0

        # Transpose to [B, H, W, 2] format required by grid_sample
        sample_grid = sample_grid.permute(0, 2, 3, 1)

        # Warp image
        warped = F.grid_sample(
            img,
            sample_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped

    def forward(self, img0, img1, timestep=0.5):
        """
        Interpolate between two consecutive slices

        Args:
            img0: [B, 1, H, W] first slice
            img1: [B, 1, H, W] second slice
            timestep: Temporal position (0.5 for middle frame)

        Returns:
            interpolated: [B, 1, H, W] interpolated slice
        """
        # Estimate bidirectional optical flow
        flow_01 = self.flow_estimator(img0, img1)  # Flow from img0 to img1
        flow_10 = self.flow_estimator(img1, img0)  # Flow from img1 to img0

        # Scale flows by timestep
        flow_0t = flow_01 * timestep
        flow_1t = flow_10 * (1 - timestep)

        # Warp images to timestep t
        img0_warped = self.warp(img0, flow_0t)
        img1_warped = self.warp(img1, flow_1t)

        # Simple blending
        img_blend = (1 - timestep) * img0_warped + timestep * img1_warped

        # Refine blended frame
        img_refined = self.refine_net(img0, img1, img_blend)

        # Final output with residual
        interpolated = img_blend + img_refined * 0.1

        return interpolated


class SimpleInterpolator(nn.Module):
    """
    Simpler learning-based interpolator (fallback option)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(2, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img0, img1, timestep=0.5):
        """
        Simple interpolation

        Args:
            img0, img1: [B, 1, H, W]
            timestep: Not used in this version

        Returns:
            interpolated: [B, 1, H, W]
        """
        x = torch.cat([img0, img1], dim=1)
        features = self.encoder(x)
        interpolated = self.decoder(features)
        return interpolated


class UNetInterpolator(nn.Module):
    """
    U-Net based interpolator with skip connections
    """

    def __init__(self, in_channels=2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128, stride=2)
        self.enc3 = ConvBlock(128, 256, stride=2)
        self.enc4 = ConvBlock(256, 512, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
        )

        # Decoder
        self.dec4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(512, 128, 4, 2, 1)  # 512 = 256 + 256 (skip)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)   # 256 = 128 + 128 (skip)

        self.final = nn.Sequential(
            ConvBlock(128, 64),  # 128 = 64 + 64 (skip)
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img0, img1, timestep=0.5):
        """
        U-Net interpolation with skip connections

        Args:
            img0, img1: [B, 1, H, W]
            timestep: Not used

        Returns:
            interpolated: [B, 1, H, W]
        """
        x = torch.cat([img0, img1], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = F.leaky_relu(self.dec4(b), 0.2)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = F.leaky_relu(self.dec3(d4), 0.2)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = F.leaky_relu(self.dec2(d3), 0.2)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.final(d2)

        return out
