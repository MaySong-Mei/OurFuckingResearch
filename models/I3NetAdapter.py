"""
I3Net Adapter for OurFuckingResearch Pipeline
Wraps I3Net model to replace IFNet for medical image interpolation
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path


def _load_i3net_model():
    """Lazily load I3Net model only when needed"""
    # Add parent directory to path so that I3Net can be imported as a module
    parent_path = str(Path(__file__).parent.parent.parent)

    # Ensure parent path is in sys.path at the beginning
    # Remove it first if it exists anywhere
    sys.path = [p for p in sys.path if p != parent_path]
    # Insert at the front
    sys.path.insert(0, parent_path)

    try:
        # Import I3Net
        from I3Net.model_zoo.i3net.basic_model import I3Net
        return I3Net
    except ImportError as e:
        # Add detailed error message
        raise ImportError(f"Failed to import I3Net: {e}") from e


class I3NetInterpolator(nn.Module):
    """
    Wrapper around I3Net to interpolate medical image slices

    Replaces IFNet in the pipeline while maintaining interface compatibility
    """

    def __init__(self, upscale=2, device='cuda'):
        """
        Initialize I3Net model

        Args:
            upscale: Upsampling factor (default: 2)
            device: Device to run model on (default: 'cuda')
        """
        super(I3NetInterpolator, self).__init__()
        self.device = device
        self.upscale = upscale

        # Load I3Net lazily
        I3Net = _load_i3net_model()

        # Create I3Net arguments
        self.args = argparse.Namespace(
            upscale=upscale,
            n_feats=64,
            kernel_size=3,
            res_scale=1,
            num_blocks=16,
            lr_slice_patch=2,  # 2 input slices
            hr_slice_patch=(2 - 1) * upscale + 1,  # output slices
            head_num=1,
            win_num_sqrt=16,
            window_size=16,
            n_size=256
        )

        # Initialize I3Net model
        self.model = I3Net(self.args).to(device)

    def forward(self, x):
        """
        Forward pass for slice pair interpolation

        Args:
            x: [B, 2, H, W] - Two consecutive grayscale slices
               (replaces IFNet's expected [B, 6, H, W] RGB concatenated format)

        Returns:
            middle_frame: [B, 1, H, W] - Interpolated middle frame
                         (maintains IFNet's output format for compatibility)
        """
        batch_size, num_channels, height, width = x.shape

        # Verify input format
        assert num_channels == 2, f"Expected [B, 2, H, W], got [B, {num_channels}, H, W]"

        # Convert from [B, 2, H, W] to I3Net format [B, H, W, 2]
        x_i3net = x.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Forward pass through I3Net
        # Output: [B, H, W, M] where M = (2-1)*upscale + 1
        output = self.model(x_i3net)  # [B, H, W, 3] for upscale=2

        # Convert back to [B, M, H, W] format
        output = output.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Extract middle frame (index 1 for upscale=2)
        # I3Net preserves input slices at positions 0 and 2, interpolates at position 1
        middle_frame = output[:, 1:2, :, :]  # [B, 1, H, W]

        return middle_frame

    def train(self, mode=True):
        """Set model to training mode"""
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return super().eval()

    def parameters(self):
        """Return model parameters for optimizer"""
        return self.model.parameters()

    def named_parameters(self, recurse=True):
        """Return named model parameters"""
        return self.model.named_parameters(recurse=recurse)

    def state_dict(self):
        """Get model state dict"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict"""
        return self.model.load_state_dict(state_dict)
