"""
I3Net Interpolation Adapter
Wraps FUSE_RDN from meta_multi.py to provide 2-frame interpolation
"""

import torch
import torch.nn as nn
from .meta_multi import FUSE_RDN


class SaintInterpolator(nn.Module):
    """
    Adapter for FUSE_RDN from meta_multi.py
    Takes 2 grayscale images and produces 1 interpolated frame

    Args:
        upscale: upscaling factor (not used, kept for interface compatibility)
        device: device to use
    """

    def __init__(self, upscale=2, device='cuda'):
        super(SaintInterpolator, self).__init__()
        self.device = device
        self.upscale = upscale

        # Create args for smaller FUSE_RDN to fit in memory
        class Args:
            scale = [2]
            G0 = 16  # Reduced from 32 to save GPU memory
            RDNkSize = 3
            RDNconfig = 'C'  # C config: D=4, C=6, G=12

        self.args = Args()
        self.model = FUSE_RDN(self.args).to(device)

    def forward(self, x):
        """
        Interpolate between two frames

        Args:
            x: [B, 2, H, W] - two consecutive grayscale frames

        Returns:
            output: [B, 1, H, W] - interpolated middle frame
        """
        # FUSE_RDN expects [B, 2, H, W] - two frames
        # Forward through FUSE_RDN
        # Use no_grad to save memory during inference
        if not self.training:
            with torch.no_grad():
                output = self.model(x)  # [B, 1, H, W]
        else:
            output = self.model(x)  # [B, 1, H, W]

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
