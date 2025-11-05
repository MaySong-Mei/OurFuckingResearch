"""
RIFE (IFNet) Interpolation Adapter
Wraps IFNet to provide 4-input -> 7-output medical image interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .IFNet import IFNet


class RIFEInterpolator(nn.Module):
    """
    Adapter for IFNet (RIFE backbone)
    Takes 4 grayscale medical images and produces 7 interpolated slices

    Args:
        upscale: upscaling factor (default: 2, kept for interface compatibility)
        device: device to use
    """

    def __init__(self, upscale=2, device='cuda'):
        super(RIFEInterpolator, self).__init__()
        self.device = device
        self.upscale = upscale

        self.model = IFNet().to(device)
        self.scale = [4, 2, 1]  # Multi-scale inference

    def _interpolate_pair(self, img_a, img_b, timestep=0.5):
        """
        Interpolate between two grayscale images using IFNet

        Args:
            img_a: [B, 1, H, W] - first image
            img_b: [B, 1, H, W] - second image
            timestep: interpolation position (0.5 for middle frame)

        Returns:
            interpolated: [B, 1, H, W] - interpolated frame
        """
        # Convert grayscale to RGB by replicating channels
        img_a_rgb = img_a.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        img_b_rgb = img_b.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # Concatenate as [B, 6, H, W]
        x = torch.cat([img_a_rgb, img_b_rgb], dim=1)

        # IFNet forward pass
        flow_list, mask, merged_list = self.model(x, scale=self.scale, timestep=timestep)

        # merged_list[2] is the finest resolution interpolated frame
        # Convert back to grayscale by averaging RGB channels
        interpolated_rgb = merged_list[2]  # [B, 3, H, W]
        interpolated = interpolated_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        return interpolated

    def forward(self, x):
        """
        Interpolate 4 sampled slices to 7 output slices
        Pattern: [s0, s1, s2, s3] -> [s0, m01, s1, m12, s2, m23, s3]

        Args:
            x: [B, 4, H, W] - four consecutive grayscale frames

        Returns:
            output: [B, 7, H, W] - interpolated slices
        """
        B, C, H, W = x.shape

        # Extract individual slices
        s0 = x[:, 0:1, :, :]  # [B, 1, H, W]
        s1 = x[:, 1:2, :, :]  # [B, 1, H, W]
        s2 = x[:, 2:3, :, :]  # [B, 1, H, W]
        s3 = x[:, 3:4, :, :]  # [B, 1, H, W]

        # Interpolate between consecutive pairs
        m01 = self._interpolate_pair(s0, s1)  # [B, 1, H, W]
        m12 = self._interpolate_pair(s1, s2)  # [B, 1, H, W]
        m23 = self._interpolate_pair(s2, s3)  # [B, 1, H, W]

        # Concatenate: [s0, m01, s1, m12, s2, m23, s3]
        output = torch.cat([s0, m01, s1, m12, s2, m23, s3], dim=1)  # [B, 7, H, W]

        # Ensure output is in valid range [0, 1]
        output = torch.clamp(output, 0, 1)

        return output

    def parameters(self):
        """Return model parameters for optimizer"""
        return self.model.parameters()

    def train(self, mode=True):
        """Set training mode"""
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return self

    def state_dict(self):
        """Get state dict"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict"""
        return self.model.load_state_dict(state_dict, strict=strict)
