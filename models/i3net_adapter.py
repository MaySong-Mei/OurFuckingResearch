"""
I3Net Interpolation Adapter
Wraps I3Net from basic_model.py to provide 4-input -> 7-output interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_model import I3Net


class I3NetInterpolator(nn.Module):
    """
    Adapter for I3Net
    Takes 4 grayscale images and produces 7 interpolated frames

    Args:
        upscale: upscaling factor (default: 2, not used but kept for interface compatibility)
        device: device to use
    """

    def __init__(self, upscale=2, device='cuda', checkpoint_path=None):
        super(I3NetInterpolator, self).__init__()
        self.device = device
        self.upscale = upscale

        # Create args for I3Net
        class Args:
            pass

        self.args = Args()
        self.args.n_feats = 64
        self.args.kernel_size = 3
        self.args.num_blocks = 16
        self.args.res_scale = 1
        self.args.lr_slice_patch = 4  # Input: 4 slices
        self.args.hr_slice_patch = (self.args.lr_slice_patch - 1) * upscale + 1  # Output: 7 slices
        self.args.head_num = 1
        self.args.win_num_sqrt = 16
        self.args.window_size = 16
        self.args.upscale = upscale

        self.model = I3Net(self.args).to(device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            import os
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded I3Net checkpoint from {checkpoint_path}")
            else:
                print(f"⚠ Warning: I3Net checkpoint not found at {checkpoint_path}")
        else:
            print("⚠ Warning: No I3Net checkpoint provided, using randomly initialized model")

    def forward(self, x):
        """
        Interpolate 4 sampled slices to 7 output slices

        Args:
            x: [B, 4, H, W] - four consecutive grayscale frames

        Returns:
            output: [B, 7, H, W] - interpolated slices (s0, m01, s1, m12, s2, m23, s3)
        """
        B, C, H, W = x.shape
        original_size = (H, W)

        # # Downsample to 256x256 if needed (I3Net is optimized for this size)
        # if H != self.target_size or W != self.target_size:
        #     x_resized = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        # else:
        #     x_resized = x

        # Convert format: [B, 4, 256, 256] -> [B, 256, 256, 4]
        # I3Net expects [B, H, W, D] format
        x_i3net = x.permute(0, 2, 3, 1).contiguous()  # [B, 256, 256, 4]

        # Forward through I3Net

        output_i3net = self.model(x_i3net)  # [B, 256, 256, 7]

        # Convert back to [B, 7, 256, 256]
        output = output_i3net.permute(0, 3, 1, 2).contiguous()  # [B, 7, 256, 256]

        # # Upsample back to original size if needed
        # if original_size != (self.target_size, self.target_size):
        #     output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)

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
